"""Explicit full-schema grounded SSR revision; no format fallback or repair."""
import hashlib
from pathlib import Path

from grounded_ssr_protocol import SCENE_PROTOCOL, MOTION_PROTOCOL
from ssr_protocol import (CONTEXT_FIELDS, HORIZON_STEPS, REFERENCE_SHA,
                          compose_prompt, load_reference, validate_context)

CONFIG_ID = 'mainhist93_tok65536_grounded_strict_ssr_v3'
NAMESPACE = 'seed1_native50_grounded_strict_ssr_mainhist93_65536_20261113_v3'
RESPONSE_FORMAT_MODE = 'json_schema_strict'
RESPONSE_SCHEMA_FACTORY_SHA256 = '0504bf0830ec16f220eb0e1fb46fd5f56e9f7593c23f433ed6d778562d31c94b'
SCHEMA_FACTORY_SHA = RESPONSE_SCHEMA_FACTORY_SHA256
HERE = Path(__file__).resolve().parent

def validate_schema_factory():
    """Bind byte-identical factory used by independently validated real probes."""
    actual = hashlib.sha256((HERE / 'grounded_response_schema.py').read_bytes()).hexdigest()
    if actual != RESPONSE_SCHEMA_FACTORY_SHA256:
        raise ValueError('Frozen full response schema factory bytes differ')
    return actual

def validate_spec(spec):
    from ssr_protocol import validate_spec as common
    result = common(spec, config_id=CONFIG_ID, namespace=NAMESPACE)
    actual = validate_schema_factory()
    if (spec.get('grounded_action_constraints') is not True
            or spec.get('scene_evidence_protocol') != SCENE_PROTOCOL
            or spec.get('motion_evidence_protocol') != MOTION_PROTOCOL
            or spec.get('response_format_mode') != RESPONSE_FORMAT_MODE
            or spec.get('response_schema_factory_sha256') != RESPONSE_SCHEMA_FACTORY_SHA256):
        raise ValueError('Explicit strict-schema/current-fact/short-action SSR protocol required')
    return dict(result, grounded_action_constraints=True,
        scene_evidence_protocol=SCENE_PROTOCOL, motion_evidence_protocol=MOTION_PROTOCOL,
        response_format_mode=RESPONSE_FORMAT_MODE, response_schema_factory_sha256=actual)
