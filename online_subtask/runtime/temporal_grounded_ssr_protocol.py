"""V7 finite current-fact completion clause, unchanged full SSR schema/science."""
from hashlib import sha256
from pathlib import Path

from enum_grounded_ssr_protocol import (
    SCENE_PROTOCOL,MOTION_PROTOCOL,RESPONSE_FORMAT_MODE,SCHEMA_FACTORY_SHA,
    BASE_SCHEMA_FACTORY_SHA,RESPONSE_SCHEMA_FACTORY_SHA256,
    RESPONSE_SCHEMA_BASE_FACTORY_SHA256,validate_schema_factories)
from ssr_protocol import (CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,
    compose_prompt,load_reference,validate_context)

CONFIG_ID='mainhist93_tok65536_grounded_strict_ssr_v7'
NAMESPACE='seed1_native50_grounded_strict_ssr_mainhist93_65536_20261113_v7'
MOTION_INTERPRETATION='current_holding_and_completed_release_clauses_v1'
COMPLETED_RELEASE_HELPER_SHA256='405314deb3048797f0a5133eb5c212abeb391623de92fc4eb73ce42f550f661d'


def validate_spec(spec):
    from ssr_protocol import validate_spec as common
    result=common(spec,config_id=CONFIG_ID,namespace=NAMESPACE)
    pins=validate_schema_factories()
    if (spec.get('grounded_action_constraints')is not True
            or spec.get('scene_evidence_protocol')!=SCENE_PROTOCOL
            or spec.get('motion_evidence_protocol')!=MOTION_PROTOCOL
            or spec.get('response_format_mode')!=RESPONSE_FORMAT_MODE
            or spec.get('motion_interpretation')!=MOTION_INTERPRETATION
            or any(spec.get(k)!=v for k,v in pins.items())
            or sha256((Path(__file__).resolve().parent/'temporal_action_clause.py').read_bytes()).hexdigest()
                !=COMPLETED_RELEASE_HELPER_SHA256):
        raise ValueError('Explicit v7 current-completion interpretation, exact helper and unchanged dual-schema SSR required')
    return dict(result,grounded_action_constraints=True,
        scene_evidence_protocol=SCENE_PROTOCOL,motion_evidence_protocol=MOTION_PROTOCOL,
        response_format_mode=RESPONSE_FORMAT_MODE,motion_interpretation=MOTION_INTERPRETATION,**pins)
