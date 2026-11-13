"""Explicit enum-category strict SSR revision; vocabulary is not visual truth."""
import hashlib
from pathlib import Path

from grounded_ssr_protocol import SCENE_PROTOCOL, MOTION_PROTOCOL
from ssr_protocol import (CONTEXT_FIELDS, HORIZON_STEPS, REFERENCE_SHA,
                          compose_prompt, load_reference, validate_context)

CONFIG_ID = 'mainhist93_tok65536_grounded_strict_ssr_v4'
NAMESPACE = 'seed1_native50_grounded_strict_ssr_mainhist93_65536_20261113_v4'
RESPONSE_FORMAT_MODE = 'json_schema_strict'
SCHEMA_FACTORY_SHA = 'b66d2d5d89bf6d0ef88d93636d1464c1f11ac0534d54f912dc0a06dee47f4d37'
BASE_SCHEMA_FACTORY_SHA = '0504bf0830ec16f220eb0e1fb46fd5f56e9f7593c23f433ed6d778562d31c94b'
RESPONSE_SCHEMA_FACTORY_SHA256 = SCHEMA_FACTORY_SHA
RESPONSE_SCHEMA_BASE_FACTORY_SHA256 = BASE_SCHEMA_FACTORY_SHA
HERE = Path(__file__).resolve().parent

def validate_schema_factories():
    for filename, expected in (('grounded_response_schema_v2.py', SCHEMA_FACTORY_SHA),
                               ('grounded_response_schema.py', BASE_SCHEMA_FACTORY_SHA)):
        actual = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen category-token/base schema factory bytes differ: ' + filename)
    return {'response_schema_factory_sha256': SCHEMA_FACTORY_SHA,
            'response_schema_base_factory_sha256': BASE_SCHEMA_FACTORY_SHA}

def validate_spec(spec):
    from ssr_protocol import validate_spec as common
    result = common(spec, config_id=CONFIG_ID, namespace=NAMESPACE)
    pins = validate_schema_factories()
    if (spec.get('grounded_action_constraints') is not True
            or spec.get('scene_evidence_protocol') != SCENE_PROTOCOL
            or spec.get('motion_evidence_protocol') != MOTION_PROTOCOL
            or spec.get('response_format_mode') != RESPONSE_FORMAT_MODE
            or any(spec.get(key) != value for key, value in pins.items())):
        raise ValueError('Explicit category-token/dual-schema/current-fact SSR protocol required')
    return dict(result, grounded_action_constraints=True,
        scene_evidence_protocol=SCENE_PROTOCOL, motion_evidence_protocol=MOTION_PROTOCOL,
        response_format_mode=RESPONSE_FORMAT_MODE, **pins)
