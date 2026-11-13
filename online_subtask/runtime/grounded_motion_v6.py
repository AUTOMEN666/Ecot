"""A bounded release observation exception, never a fabricated closed grasp.

The first of at most two adjacent eight-action opening windows requires a real
previous ready/closed report for the same locked goal. No label oscillation can
restart that goal's burst. Re-arming after a complete visual drop/regrasp chain
is deliberately not implemented here; it must not be inferred from text.

Two opening observations authorize at most two ADDITIONAL execute8 blocks
(16 actions/.64s). The initial opening observation is already eight actions
after the last closed-ready report: worst-case time from that report through
the second authorized block is24 actions/.96s, not a measured release duration.
Actual V5/203 frame112->120 has the required prior-ready facts. V5/201
frame104->112 does not (prior release_ready=no) and remains rejected.
"""
from copy import deepcopy

import grounded_motion as original
from grounded_motion_v5 import TEMPORAL_CHAIN,unique_secured_actor
from release_grounded_ssr_protocol import MOTION_INTERPRETATION
from ssr_protocol import digest

MAX_OPENING_WINDOWS=2
RELEASE_VERBS=['hold','open','release']
GOAL_FIELDS=('subtask_id','instance_id','category','basket_color','selected_at')


def proof_binding(proof):
    """Program-supplied current captures and exact history, never LLM frames."""
    frame=proof.get('current_frame')
    original.require(type(frame)is int and frame>=0 and frame%8==0,
        'Release phase needs an actual eight-action observation frame')
    rows=proof.get('observations')
    original.require(type(rows)is list and all(type(r)is dict for r in rows),
        'Actual observation proof required for release phase')
    current=[r for r in rows if r.get('is_current')is True]
    original.require(len(current)==3 and {r.get('camera')for r in current}==
        {'cam_high','cam_left_wrist','cam_right_wrist'}and all(r.get('frame')==frame for r in current),
        'Release phase requires three actual same-frame current captures')
    original.require(all(type(r.get('image_sha256'))is str and len(r['image_sha256'])==64
        and set(r['image_sha256'])<=set('0123456789abcdef')for r in current),
        'Release phase current capture SHA binding required')
    return {k:deepcopy(proof.get(k))for k in
        ('episode_id','current_frame','observations','executed_subtask_history')}


def _goal_key(goal):
    original.require(type(goal)is dict and all(k in goal for k in GOAL_FIELDS),
        'Release progression needs the actual selected goal identity')
    return digest({k:goal[k]for k in GOAL_FIELDS})


def _held_at_ready(arm,obj,goal,*,gripper):
    return (type(arm)is dict and type(obj)is dict
        and arm.get('holding')=='identified'
        and arm.get('held_instance_id')==arm.get('interaction_instance_id')==goal['instance_id']
        and arm.get('gripper_state')==gripper and arm.get('visibility')=='clear'
        and arm.get('object_secured')=='yes'and arm.get('relation')=='over_basket'
        and arm.get('observed_basket')==goal['basket_color']and arm.get('release_ready')=='yes'
        and obj.get('instance_id')==goal['instance_id']and obj.get('identity_confidence')=='clear'
        and obj.get('category_confidence')=='clear'and obj.get('category')==goal['category']
        and obj.get('location')=='held'and obj.get('released')is not True)


def derive_release_transitions(evidence,*,before,proof,transition_state):
    """Return pending phase+budget; caller commits only after scene admission.

    transition_state is episode-local durable program memory. It contains no
    planned/expert action and never rewrites evidence. The usual guard still
    validates the complete actual execution/exposure prefix before commit.
    """
    binding=proof_binding(proof)
    state=deepcopy(transition_state)
    episode,frame=binding['episode_id'],binding['current_frame']
    original.require(type(episode)is str and bool(episode)and before.get('episode_id')==episode,
        'Release observation episode differs from current scene')
    original.require(type(state)is dict and set(state)=={'episode_id','bursts'}
        and state['episode_id']==episode and type(state['bursts'])is dict,
        'Episode-local durable release budget required')
    arms=evidence.get('arms',{})
    candidates=[name for name,arm in arms.items()if arm.get('object_secured')=='yes'
        and arm.get('gripper_state')=='transitioning']
    if not candidates:return {},state
    original.require(len(candidates)==1,'Only one identified release-progress hand may use the exception')
    name=candidates[0];other='right'if name=='left'else'left'
    original.require(name in{'left','right'}and before.get('current_frame')==frame-8,
        'Release progression needs adjacent actual observations eight actions apart')
    goal=before.get('active_goal');key=_goal_key(goal)
    original.require(evidence.get('goal_correction')is None and not evidence.get('corrections'),
        'A release-progress exception cannot change a goal or object interpretation')
    objects={o['instance_id']:o for o in evidence.get('objects',[])}
    obj=objects.get(goal['instance_id']);prior_obj=before.get('current_visible_objects',{}).get(goal['instance_id'])
    original.require(_held_at_ready(arms[name],obj,goal,gripper='transitioning')
        and arms[name].get('motion_trend')=='release',
        'Current release progress needs same identified supported target opening over its ready basket')
    prior=before.get('arms',{}).get(name)
    for reports in(arms,before.get('arms',{})):
        original.require(reports.get(other,{}).get('holding')=='empty'
            and reports[other].get('held_instance_id')is None and reports[other].get('visibility')=='clear'
            and reports[other].get('object_secured')in('no','unknown'),
            'Release progression requires the other hand visibly empty in both reports')
    for baskets in(evidence.get('baskets',{}),before.get('baskets',{})):
        basket=baskets.get(goal['basket_color'],{})
        original.require(basket.get('visibility')=='clear'and basket.get('unknown_contents')is False,
            'Release progression needs the same clearly observed unblocked basket')
    prior_burst=state['bursts'].get(key)
    if prior_burst is None:
        original.require(_held_at_ready(prior,prior_obj,goal,gripper='closed'),
            'First release-progress window needs previous actual closed secure ready evidence')
        count=1;start=frame
    else:
        original.require(prior_burst['arm']==name and prior_burst['last_frame']==frame-8
            and prior_burst['windows']<MAX_OPENING_WINDOWS
            and prior_burst['last_evidence_sha256']==digest({'arms':before['arms'],
                'objects':list(before['current_visible_objects'].values()),'baskets':before['baskets']}),
            'Release-progress burst exhausted, interrupted or changed; label oscillation cannot restart it')
        original.require(_held_at_ready(prior,prior_obj,goal,gripper='transitioning')
            and prior.get('motion_trend')=='release',
            'Second release-progress window needs the immediately previous admitted opening report')
        count=prior_burst['windows']+1;start=prior_burst['start_frame']
    current_digest=digest({'arms':arms,'objects':evidence['objects'],'baskets':evidence['baskets']})
    state['bursts'][key]={'arm':name,'windows':count,'start_frame':start,'last_frame':frame,
        'last_evidence_sha256':current_digest}
    transition={'arm':name,'instance_id':goal['instance_id'],'basket_color':goal['basket_color'],
        'category':goal['category'],'subtask_id':goal['subtask_id'],'goal_key':key,
        'previous_frame':frame-8,'current_frame':frame,'episode_id':episode,
        'window_index':count,'opening_windows':count,'maximum_opening_windows':MAX_OPENING_WINDOWS,
        'maximum_additional_execution_steps':MAX_OPENING_WINDOWS*8,
        'maximum_additional_execution_seconds':MAX_OPENING_WINDOWS*8/25,
        'maximum_steps_from_last_closed_ready_through_final_block':(MAX_OPENING_WINDOWS+1)*8,
        'initial_closed_ready_frame':start-8,'current_evidence_sha256':digest(evidence),
        'proof_binding_sha256':digest(binding),'previous_scene_sha256':digest(before),
        'allowed_verbs':list(RELEASE_VERBS),'observed_gripper_state_preserved':True,
        'holding_identity_or_grasp_support_promoted':False,'new_goal_or_completion_admitted':False}
    return {name:transition},state


def action_envelopes(evidence,target,*,contact_transition=None,known_release_transitions=None):
    transitions=known_release_transitions or {}
    envelope=original.action_envelopes(evidence,target,contact_transition=contact_transition,
        known_release_transitions=transitions)
    entailed=[]
    for name,item in envelope['by_arm'].items():
        if name in transitions:
            original.require(set(item['allowed_verbs'])==set(RELEASE_VERBS),
                'Known release progress must admit only open/release/hold')
        elif 'release'in item['allowed_verbs']:
            item['allowed_verbs']=sorted(set(item['allowed_verbs'])|{'open'})
            item['grounds'].append('opening realizes the already admitted release; readiness unchanged')
            entailed.append(name)
    envelope.update(motion_interpretation=MOTION_INTERPRETATION,
        release_open_entailments=entailed,release_readiness_relaxed=False,
        known_release_transitions=deepcopy(transitions),maximum_release_opening_windows=MAX_OPENING_WINDOWS)
    return envelope


def validate_plan(context,envelope):
    from negative_action_guard import is_negated
    original.require(envelope.get('motion_interpretation')==MOTION_INTERPRETATION,
        'Explicit v6 release/negative interpretation envelope required')
    original.require(TEMPORAL_CHAIN.search(context['action']['summary'])is None,
        'Action Summary contains explicit temporal chaining; one immediate action required')
    before=deepcopy(context);override=unique_secured_actor(context,envelope)
    result=original.validate_plan(context,envelope,acting_arms=override,negation_detector=is_negated)
    original.require(context==before,'Public annotations changed during validation')
    return dict(result,motion_interpretation=MOTION_INTERPRETATION,
        actor_resolution='unique_current_secured_holder_and_single_reasoning_arm'if override is not None else'explicit_summary_actor',
        actor_override_from_current_evidence=sorted(override)if override is not None else None,
        summary_temporal_screen='explicit_then_after_that_followed_by_only',
        known_release_transitions=deepcopy(envelope['known_release_transitions']),
        release_readiness_relaxed=False,negative_action_constraints='bounded_explicit_negation_v1')
