"""Shared expanded-catalog request and strict ID lookup; no model/network/GPU."""
from __future__ import annotations
import base64,json
from dataclasses import dataclass
from pathlib import Path
from history_client import STATE_SCHEMA,canonical,sha,make_frame
from executed_history import validate_records,digest

HERE=Path(__file__).resolve().parent
CAMERAS=('cam_high','cam_left_wrist','cam_right_wrist')
PAST_CAMERAS=('cam_high',)
PROTOCOL_CONFIG_ID='mainhist93_tok65536_v1'
MAX_PAST_TIMEPOINTS=93
MAX_TOTAL_IMAGES=96
MAX_TOKENS=65536
HTTP_TIMEOUT_SECONDS=600
RETRY_BUDGET_SECONDS=1850
PRIVATE_RPC_TIMEOUT_SECONDS=2100
MODES=('expanded_catalog_id',)
PROTOCOLS={'expanded_catalog_id':PROTOCOL_CONFIG_ID}
CATALOG_SHA='97734cb3db4e92fcf62beec96d635896d09d8f23c99c646170bf7736f8b48600'
INSTRUCTION='Sort the objects by category into the three baskets.'

class SubtaskAbstention(RuntimeError):
    """Compatibility exception only; no NO_MATCH output is admitted here."""

def unique_pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('Duplicate JSON key')
        result[k]=v
    return result

def strict_json(raw):
    def reject(value):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=unique_pairs,parse_constant=reject)

def load_catalog():
    raw=(HERE/'ACTIVE_EXPANDED_CATALOG.json').read_bytes()
    if sha(raw)!=CATALOG_SHA:raise ValueError('Frozen 288 catalog changed')
    catalog=strict_json(raw);items=catalog['candidates']
    if len(items)!=288 or len({item['id'] for item in items})!=288:raise ValueError('Require all288 unique IDs')
    return catalog

def resolve(raw, catalog=None):
    catalog=load_catalog() if catalog is None else catalog
    selected=strict_json(raw)
    if type(selected)is not dict or set(selected)!={'subtask_id'}:raise ValueError('Only single subtask_id JSON allowed')
    matches=[c for c in catalog['candidates'] if type(selected['subtask_id'])is str and c['id']==selected['subtask_id']]
    if len(matches)!=1:raise ValueError('Unknown/null/NO_MATCH ID forbidden; no substitute')
    item=matches[0];context={'subtask':item['text']}
    return context,{'selected_id':item['id'],'catalog_sha256':CATALOG_SHA,'exact_text_utf8_sha256':sha(item['text'].encode()),
        'raw_selection':selected,'resolution_kind':'exact_expanded288_catalog_lookup','host_fallback_used':False,
        'semantic_correctness_guaranteed':False}

def history_source_frame(index):
    if type(index)is not int or index<0:raise ValueError('Nonnegative source-grid index required')
    return 5*index//2


def is_history_source_frame(frame):
    if type(frame)is not int or frame<0:return False
    k=2*frame//5
    return frame in (history_source_frame(k),history_source_frame(k+1))


@dataclass(frozen=True)
class HistoryPoint:
    episode_id:str
    frame:int
    views:tuple
    state14:tuple


def make_history_point(episode_id,frame,state14,pngs):
    if not is_history_source_frame(frame):raise ValueError('History capture must be on actual floor(2.5*k) source-frame grid')
    if set(pngs)!=set(CAMERAS):raise ValueError('History point requires exactly3 real cameras')
    views=tuple(make_frame(episode_id,frame,state14,pngs[camera])for camera in CAMERAS)
    if any(v.state14!=views[0].state14 for v in views):raise ValueError('Triplet requires same measured State14')
    return HistoryPoint(episode_id,frame,views,views[0].state14)


def validate_complete_source_history(points,episode_id,current_frame):
    if any(not isinstance(p,HistoryPoint)for p in points):raise ValueError('Explicit actual captured history-point bundles required')
    if [p.frame for p in points]!=[history_source_frame(i)for i in range(len(points))]:
        raise ValueError('History must contain every actual10Hz-grid capture since0, without missing/extra/replan-only points')
    if any(not isinstance(p,HistoryPoint)or p.episode_id!=episode_id or p.frame>current_frame for p in points):
        raise ValueError('History must be current-episode actual source points up to current only')
    for p in points:
        if len(p.views)!=3 or any(v.frame!=p.frame or v.episode_id!=p.episode_id or v.state14!=p.state14 for v in p.views):
            raise ValueError('History point camera/frame/episode/State14 binding differs')
    if not points or history_source_frame(len(points))<=current_frame:
        raise ValueError('Actual10Hz history capture is missing before current API query')


def uniform_past(points):
    """At most93 timepoints sampled over ALL strictly past source points; payload MAIN only."""
    if len(points)<=93:return list(points)
    indices=[i*(len(points)-1)//92 for i in range(93)]
    if len(set(indices))!=93 or indices[0]!=0 or indices[-1]!=len(points)-1:raise AssertionError('Uniform93 endpoints/uniqueness failed')
    return [points[i]for i in indices]


def build_request(*,episode_id,frame,state14,pngs,instruction,past_frames,executed_subtask_history=None,catalog=None):
    if instruction!=INSTRUCTION:raise ValueError('Exact full original Instruction required')
    if type(frame)is not int or frame<0 or frame%8:raise ValueError('Actual replan frame0,8,... required')
    if set(pngs)!=set(CAMERAS):raise ValueError('Exactly current MAIN/left/right RGB required')
    current=[make_frame(episode_id,frame,state14,pngs[camera]) for camera in CAMERAS]
    validate_complete_source_history(past_frames,episode_id,frame)
    past=uniform_past([p for p in past_frames if p.frame<frame])
    catalog=load_catalog() if catalog is None else catalog
    executed=validate_records(executed_subtask_history,episode_id,frame,catalog)
    items=[{'id':item['id'],'text':item['text']}for item in catalog['candidates']]
    prompt=(HERE/'selector_prompt.md').read_text()
    content=[{'type':'text','text':canonical({'instruction':instruction,'current_frame':frame,'fps':25,'state_schema':STATE_SCHEMA,
        'past_cameras':list(PAST_CAMERAS),'current_cameras':list(CAMERAS),'cadence_execution_frames':8,
        'history_capture_source_fps':25,'history_capture_frame_rule':'floor(2.5*k), k=0,1,...; approximately10Hz',
        'maximum_total_images':MAX_TOTAL_IMAGES,'maximum_past_timepoints':MAX_PAST_TIMEPOINTS,'history_selection':'uniform over all strictly past10Hz source points; preserve earliest/latest; MAIN image only per selected timepoint',
        'current_camera_alignment':'All current cameras and State14 are from the same actual SDK observation.',
        'executed_subtask_history':executed,'latest_executed_subtask_record':executed[-1]if executed else None,
        'executed_history_semantics':'Actual instructions followed by executed actions, not object identity/completion truth; 1-based inclusive action ranges.'})}]
    observations=[]
    ordered=[(point.views[0],'cam_high')for point in past]+list(zip(current,CAMERAS))
    for f,camera in ordered:
        metadata={'frame':f.frame,'camera':camera,'is_current':f.frame==frame,'state14':list(f.state14),'time_seconds':f.time_seconds}
        content.extend([{'type':'text','text':canonical(metadata)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(f.image_png).decode()}}])
        observations.append({**metadata,'image_sha256':f.image_sha256,'original_camera_chw_sha256':f.camera_chw_sha256,'state_sha256':f.state_sha256})
    messages=[{'role':'system','content':prompt},{'role':'user','content':[{'type':'text','text':canonical({'candidates':items,
        'candidate_source':'All100training episodes color-explicit known descriptions paired with white/blue/red; derived pairs are not test labels.',
        'selection_rule':'Exactly one listed ID; NO_MATCH/null forbidden.'})}]},{'role':'user','content':content}]
    request={'model':'__PROVIDER__','messages':messages,'temperature':0,'max_tokens':MAX_TOKENS,'thinking':{'type':'enabled'},'reasoning_effort':'high','response_format':{'type':'json_object'}}
    proof={'schema':'ecot.mainhist93_tok65536.request.v1','protocol_config_id':PROTOCOL_CONFIG_ID,'episode_id':episode_id,'current_frame':frame,'fps':25,'cadence_execution_frames':8,
        'catalog_sha256':CATALOG_SHA,'prompt_sha256':sha(prompt.encode()),'messages_sha256':sha(canonical(messages).encode()),
        'observations':observations,'history_frames':[p.frame for p in past],'context_fields':['subtask'],
        'total_image_count':len(observations),'past_image_count':len(past),'current_image_count':3,
        'all_captured_source_frames':[p.frame for p in past_frames],'all_strictly_past_source_frame_count':sum(p.frame<frame for p in past_frames),
        'history_sampling_frame_rule':'floor(2.5*k)','past_cameras':list(PAST_CAMERAS),'current_cameras':list(CAMERAS),
        'history_views_per_point':1,'captured_history_views_per_point':3,'max_tokens':MAX_TOKENS,
        'selector_timeout_seconds':HTTP_TIMEOUT_SECONDS,'selector_retry_budget_seconds':RETRY_BUDGET_SECONDS,
        'policy_rpc_timeout_seconds':PRIVATE_RPC_TIMEOUT_SECONDS,'maximum_total_images':MAX_TOTAL_IMAGES,'maximum_past_timepoints':MAX_PAST_TIMEPOINTS,
        'prior_generated_text_sent':bool(executed),'only_actual_executed_prior_text_sent':True,
        'executed_subtask_history':executed,'executed_subtask_history_sha256':digest(executed),
        'executed_subtask_history_count':len(executed),'latest_executed_subtask_record':executed[-1]if executed else None,
        'GT_or_human_review_in_model_input':False}
    if (not len(observations)==len(past)+3<=MAX_TOTAL_IMAGES or any(p.frame>=frame for p in past)
            or any(o['camera']!='cam_high'for o in observations if not o['is_current'])):
        raise AssertionError('Strictpast MAIN/current3 maximum96 image budget failed')
    return request,proof,current[0]
