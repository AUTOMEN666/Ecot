"""Explicit v5 motion interpretation; v4 image/schema/science retained."""
from enum_grounded_ssr_protocol import (
    SCENE_PROTOCOL,MOTION_PROTOCOL,RESPONSE_FORMAT_MODE,SCHEMA_FACTORY_SHA,
    BASE_SCHEMA_FACTORY_SHA,RESPONSE_SCHEMA_FACTORY_SHA256,
    RESPONSE_SCHEMA_BASE_FACTORY_SHA256,validate_schema_factories)
from ssr_protocol import (CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,
    compose_prompt,load_reference,validate_context)

CONFIG_ID='mainhist93_tok65536_grounded_strict_ssr_v5'
NAMESPACE='seed1_native50_grounded_strict_ssr_mainhist93_65536_20261113_v5'
MOTION_INTERPRETATION='release_open_and_unique_secured_actor_v1'


def validate_spec(spec):
    from ssr_protocol import validate_spec as common
    result=common(spec,config_id=CONFIG_ID,namespace=NAMESPACE)
    pins=validate_schema_factories()
    if (spec.get('grounded_action_constraints')is not True
            or spec.get('scene_evidence_protocol')!=SCENE_PROTOCOL
            or spec.get('motion_evidence_protocol')!=MOTION_PROTOCOL
            or spec.get('response_format_mode')!=RESPONSE_FORMAT_MODE
            or spec.get('motion_interpretation')!=MOTION_INTERPRETATION
            or any(spec.get(k)!=v for k,v in pins.items())):
        raise ValueError('Explicit v5 release/actor interpretation and unchanged dual-schema grounded SSR required')
    return dict(result,grounded_action_constraints=True,
        scene_evidence_protocol=SCENE_PROTOCOL,motion_evidence_protocol=MOTION_PROTOCOL,
        response_format_mode=RESPONSE_FORMAT_MODE,motion_interpretation=MOTION_INTERPRETATION,**pins)
