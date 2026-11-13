"""Explicit revision for current-fact/local-uncertainty and short-action grounding."""
from pathlib import Path
from ssr_protocol import CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,compose_prompt,load_reference,validate_context

CONFIG_ID='mainhist93_tok65536_grounded_ssr_v2'
NAMESPACE='seed1_native50_grounded_ssr_mainhist93_65536_20261113_v2'
SCENE_PROTOCOL='current_observation_local_uncertainty_v2'
MOTION_PROTOCOL='current_grasp_prerequisites_v1'

def validate_spec(spec):
    from ssr_protocol import validate_spec as common
    result=common(spec,config_id=CONFIG_ID,namespace=NAMESPACE)
    if (spec.get('grounded_action_constraints')is not True
            or spec.get('scene_evidence_protocol')!=SCENE_PROTOCOL
            or spec.get('motion_evidence_protocol')!=MOTION_PROTOCOL):
        raise ValueError('Explicit current-fact/short-action protocol required')
    return dict(result,grounded_action_constraints=True,scene_evidence_protocol=SCENE_PROTOCOL,
                motion_evidence_protocol=MOTION_PROTOCOL)
