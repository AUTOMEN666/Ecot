"""Scoped v5 interpretation of physical release and a unique secured actor.

No public annotation is rewritten. These rules address specific grounded
false positives; the temporal screen is not a general English/accuracy grader.
"""
from copy import deepcopy
import re

import grounded_motion as original
from phase_grounded_ssr_protocol import MOTION_INTERPRETATION

ACTOR_INFERENCE_ACTIONS={'lift','carry','lower','release'}
TEMPORAL_CHAIN=re.compile(r'\bthen\b|\bafter\s+that\b|\bfollowed\s+by\b',re.I)
SIDE_WORD=re.compile(r'\b(?:left|right|both)\b',re.I)
HYPHENATED_REASON_ARM=re.compile(r'\b(left|right|both)-(?:arms?|grippers?|hands?|fingers|grasps?)\b',re.I)


def action_envelopes(evidence,target,*,contact_transition=None):
    """Opening is a physical release action only where release already passed."""
    envelope=original.action_envelopes(evidence,target,contact_transition=contact_transition)
    entailed=[]
    for name,item in envelope['by_arm'].items():
        if 'release' in item['allowed_verbs']:
            item['allowed_verbs']=sorted(set(item['allowed_verbs'])|{'open'})
            item['grounds'].append('physical gripper opening realizes an already admitted current release; no readiness prerequisite is relaxed')
            entailed.append(name)
    envelope.update(motion_interpretation=MOTION_INTERPRETATION,
        release_open_entailments=entailed,
        release_readiness_relaxed=False)
    return envelope


def unique_secured_actor(context,envelope):
    """Return a narrow actor override, never a default/side/intent guess.

    Any second clearly secure holder blocks omission resolution, including a
    different held instance. Explicit Summary actors always use the old path.
    """
    summary=context['action']['summary']
    if original.mentioned_arms(summary):
        return None
    if SIDE_WORD.search(summary):
        # Never reinterpret an unparsed 'right-arm' (or a spatial side word)
        # as a missing actor that can silently be replaced by the other hand.
        return None
    match=re.match(r'^([A-Za-z]+)\b',summary)
    if not match or match[1].lower()not in ACTOR_INFERENCE_ACTIONS:
        return None
    reason_arms=original.mentioned_arms(context['reasoning'])
    for name in HYPHENATED_REASON_ARM.findall(context['reasoning']):
        reason_arms.update({'left','right'}if name.lower()=='both'else{name.lower()})
    if len(reason_arms)!=1:
        return None
    holders=[]
    for name,item in envelope['by_arm'].items():
        arm=item['observed']
        if (arm['holding']=='identified'and arm['held_instance_id']is not None
                and arm['interaction_instance_id']==arm['held_instance_id']
                and arm['object_secured']=='yes'and arm['gripper_state']=='closed'
                and arm['visibility']=='clear'):
            holders.append(name)
    if len(holders)!=1 or set(holders)!=reason_arms:
        return None
    holder=holders[0]
    if envelope['by_arm'][holder]['observed']['held_instance_id']!=envelope['target_instance_id']:
        return None
    return {holder}


def validate_plan(context,envelope):
    """Preserve text and all original motion/category/state/basket checks."""
    original.require(envelope.get('motion_interpretation')==MOTION_INTERPRETATION,
        'Explicit v5 motion interpretation envelope required')
    original.require(TEMPORAL_CHAIN.search(context['action']['summary'])is None,
        'Action Summary contains explicit temporal chaining; one immediate action required')
    before=deepcopy(context)
    override=unique_secured_actor(context,envelope)
    result=original.validate_plan(context,envelope,acting_arms=override)
    original.require(context==before,'Public annotations changed during validation')
    return dict(result,motion_interpretation=MOTION_INTERPRETATION,
        actor_resolution='unique_current_secured_holder_and_single_reasoning_arm'if override is not None else'explicit_summary_actor',
        actor_override_from_current_evidence=sorted(override)if override is not None else None,
        summary_temporal_screen='explicit_then_after_that_followed_by_only',
        release_readiness_relaxed=False)
