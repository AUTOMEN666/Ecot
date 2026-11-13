"""V6 motion/negation plus a separately bound finite past-release predicate."""
from copy import deepcopy

import grounded_motion as original
import grounded_motion_v6 as v6
from grounded_motion_v5 import TEMPORAL_CHAIN,unique_secured_actor
from negative_action_guard import is_negated
from temporal_grounded_ssr_protocol import MOTION_INTERPRETATION


def action_envelopes(evidence,target,**kwargs):
    result=v6.action_envelopes(evidence,target,**kwargs)
    result['motion_interpretation']=MOTION_INTERPRETATION
    return result


def validate_plan(context,envelope,*,completed_release_predicate):
    original.require(envelope.get('motion_interpretation')==MOTION_INTERPRETATION,
        'Explicit v7 current-holding/completed-clause envelope required')
    original.require(callable(completed_release_predicate),'Current-proof-bound completed-release predicate required')
    original.require(TEMPORAL_CHAIN.search(context['action']['summary'])is None,
        'Action Summary contains explicit temporal chaining; one immediate action required')
    before=deepcopy(context);override=unique_secured_actor(context,envelope);recognized=[]
    def completed(text,index):
        result=completed_release_predicate(text,index)
        original.require(type(result)is bool,'Completed-release predicate must return an exact boolean')
        if result:recognized.append({'action_start_index':index,'scope':'actual_summary_terminal_completed_release_clause'})
        return result
    result=original.validate_plan(context,envelope,acting_arms=override,
        negation_detector=is_negated,completed_release_predicate=completed)
    original.require(context==before,'Public annotations changed during validation')
    return dict(result,motion_interpretation=MOTION_INTERPRETATION,
        actor_resolution='unique_current_secured_holder_and_single_reasoning_arm'if override is not None else'explicit_summary_actor',
        actor_override_from_current_evidence=sorted(override)if override is not None else None,
        summary_temporal_screen='explicit_then_after_that_followed_by_only',
        known_release_transitions=deepcopy(envelope['known_release_transitions']),
        release_readiness_relaxed=False,negative_action_constraints='bounded_explicit_negation_v1',
        completed_release_clause_contract='current_fact_bound_terminal_after_release_v1',
        completed_release_clause_mentions=recognized,
        completion_or_visual_truth_certified=False)
