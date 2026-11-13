"""Committed actual execution memory, independent of generation/network/policy calls."""
from __future__ import annotations
from copy import deepcopy
import hashlib,json
from pi05_policy.physics_clock import verify_clock_delta

SCHEMA='ecot.actual_executed_subtask_history.v1'
ACK_SCHEMA='ecot.actual_executed_subtask_ack.v1'
INSTRUCTION='Sort the objects by category into the three baskets.'
RECORD_FIELDS={'observation_frame','subtask_id','subtask','executed_action_start','executed_action_end'}

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def validate_records(records,episode_id,frame,catalog):
    if not isinstance(episode_id,str)or not episode_id:raise ValueError('Actual episode required')
    if type(frame)is not int or not 0<=frame<=1096 or frame%8:raise ValueError('Native replan frame required')
    if type(records)is not list or len(records)!=frame//8 or len(records)>137:
        raise ValueError('Complete actual executed history required before every replan')
    by_id={item['id']:item['text'] for item in catalog['candidates']}
    for index,row in enumerate(records):
        f=8*index
        if (type(row)is not dict or set(row)!=RECORD_FIELDS
                or any(type(row[k])is not int for k in ('observation_frame','executed_action_start','executed_action_end'))
                or row['observation_frame']!=f or row['executed_action_start']!=f+1 or row['executed_action_end']!=f+8
                or type(row['subtask_id'])is not str or by_id.get(row['subtask_id'])!=row['subtask']):
            raise ValueError('History must cover exact real actions1..current with exact catalog ID/text')
    return deepcopy(records)

class ExecutedSubtaskHistory:
    def __init__(self,catalog):
        self.catalog=catalog;self.reset(None)
    def reset(self,episode_id):
        if episode_id is not None and(type(episode_id)is not str or not episode_id):raise ValueError('Actual episode ID required')
        self.episode_id=episode_id;self.records=[];self.ack_source_hashes=[];self.last_clock=None
    def for_request(self,episode_id,frame):
        if self.episode_id!=episode_id:raise ValueError('Cross-episode executed history forbidden')
        records=validate_records(self.records,episode_id,frame,self.catalog)
        return {'executed_subtask_history':records,'executed_subtask_history_sha256':digest(records),
            'executed_subtask_history_count':len(records),'latest_executed_subtask_record':deepcopy(records[-1])if records else None,
            'executed_subtask_ack_source_sha256':list(self.ack_source_hashes),
            'executed_subtask_ack_prefix_sha256':digest(self.ack_source_hashes)}
    def commit(self,ack,*,expected_injection,expected_actions,expected_clock):
        if type(ack)is not dict or set(ack)!={'schema','injection','execution_clocks','chunk_end_reason'}or ack.get('schema')!=ACK_SCHEMA:
            raise ValueError('Exact actual execution ACK required')
        # Reject duplicates instead of silently creating a second history entry.
        injection=ack['injection']
        if type(injection)is not dict or canonical(injection)!=canonical(expected_injection):
            raise ValueError('ACK must match exact current actual policy injection')
        f=injection.get('execution_frame')
        if (injection.get('episode_id')!=self.episode_id or type(f)is not int or f!=8*len(self.records)
                or f>1096 or injection.get('actual_injected')is not True or injection.get('checkpoint_step')!=5000
                or injection.get('training_composition_byte_equal')is not True):
            raise ValueError('ACK episode/frame/actual injection differs or was already committed')
        context=injection.get('context');lineage=injection.get('lineage',{})
        selected=lineage.get('selected_id');by_id={item['id']:item['text'] for item in self.catalog['candidates']}
        if (type(context)is not dict or set(context)!={'subtask'}or type(selected)is not str
                or by_id.get(selected)!=context['subtask']or context['subtask']is None
                or injection.get('context_json')!=canonical(context)
                or injection.get('prompt')!=INSTRUCTION+' Current subtask: '+context['subtask']
                or injection.get('prompt_sha256')!=hashlib.sha256(injection['prompt'].encode()).hexdigest()
                or lineage.get('resolution',{}).get('selected_id')!=selected):
            raise ValueError('ACK exact ID/text/prompt binding differs')
        rows=ack['execution_clocks'];reason=ack['chunk_end_reason']
        if (type(rows)is not list or not 1<=len(rows)<=8 or len(expected_actions)!=8
                or f+len(rows)>1100 or reason not in ('full_chunk','native_terminal','engineering_stop')
                or len(rows)<8 and reason=='full_chunk'):
            raise ValueError('ACK needs actual full8 or terminal/engineering partial execution')
        previous=expected_clock
        if self.last_clock is not None:verify_clock_delta(self.last_clock,previous,0,'ACK_between_chunks')
        for offset,row in enumerate(rows,1):
            if (type(row)is not dict or set(row)!={'episode_id','execution_frame','before','after','native_take_action_count','action'}
                    or row['episode_id']!=self.episode_id or type(row['execution_frame'])is not int
                    or row['execution_frame']!=f+offset or type(row['native_take_action_count'])is not int
                    or row['native_take_action_count']!=f+offset
                    or canonical(row['action'])!=canonical(expected_actions[offset-1])):
                raise ValueError('ACK actual action/frame/native count differs from returned command')
            verify_clock_delta(previous,row['before'],0,'ACK_between_actions')
            verify_clock_delta(row['before'],row['after'],10,'ACK_actual_action25Hz')
            previous=row['after']
        record={'observation_frame':f,'subtask_id':selected,'subtask':context['subtask'],
            'executed_action_start':f+1,'executed_action_end':f+len(rows)}
        source_sha=digest(ack);self.records.append(record);self.ack_source_hashes.append(source_sha);self.last_clock=deepcopy(previous)
        return {'schema':SCHEMA,'event':'actual_executed_subtask_committed','episode_id':self.episode_id,
            'record':deepcopy(record),'record_sha256':digest(record),'ack_source_sha256':source_sha,
            'history_count':len(self.records),'history_sha256':digest(self.records),
            'ack_source_prefix_sha256':digest(self.ack_source_hashes),'last_executed_frame':record['executed_action_end'],
            'policy_called':False,'RNG_advanced':False}
