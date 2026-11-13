"""Independent JSON-schema probe factories; never import production or call API.

These transport contracts do not replace current visual/history/motion guards.
They are candidates for an explicit new compatibility probe, not a fallback.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

SCHEMA_REVISION = 'ecot.classify_objects.grounded_ssr.schema_probe.v1'
BASKETS = ('white', 'blue', 'red')
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
RELATIONS = ('separated', 'approaching', 'aligned', 'enclosing', 'contact',
             'held', 'over_basket', 'released', 'retreat', 'unknown')
MOTION_TRENDS = ('approach', 'align', 'close', 'lift', 'transport', 'lower',
                 'release', 'retract', 'stationary', 'unknown')
ACTIONS = ('approach', 'align', 'open', 'close', 'lift', 'carry', 'lower',
           'release', 'retract', 'hold')
COLOR_WORDS = ('white', 'black', 'blue', 'red', 'green', 'yellow', 'orange',
               'purple', 'pink', 'gold', 'silver', 'beige', 'cyan', 'brown', 'gray', 'grey')

def exact_object(properties, description=None):
    value = {'type': 'object', 'properties': properties,
             'required': list(properties), 'additionalProperties': False}
    if description:
        value['description'] = description
    return value

def enum(values, *, nullable=False):
    values = list(values)
    return {'type': ['string', 'null'] if nullable else 'string',
            'enum': values + [None] if nullable else values}

def instance(*, nullable=False):
    return {'type': ['string', 'null'] if nullable else 'string',
            'pattern': r'^O[1-9][0-9]*(?![\s\S])'}

def category():
    # Same lowercase singular syntax and explicit generic/color exclusions as
    # the current guard. This cannot determine whether a visual kind is true.
    colors = '|'.join(COLOR_WORDS)
    pattern = ('^(?!(?:unknown|object|item|thing|toy|device)$)'
               '(?!(?:[a-z]+_)*(?:' + colors + ')(?:_|$))[a-z]+(?:_[a-z]+)*(?![\\s\\S])')
    return {'type': ['string', 'null'], 'pattern': pattern}

def text(limit, *, nullable=False):
    return {'type': ['string', 'null'] if nullable else 'string', 'minLength': 1,
            'maxLength': limit,
            'pattern': r'^[^\u0000-\u001F]*[^\s\u0000-\u001F][^\u0000-\u001F]*(?![\s\S])'}

def descriptor_array(descriptors):
    return {'type': 'array', 'items': enum(descriptors), 'uniqueItems': True}

def evidence_cameras():
    return {'type': 'array', 'items': enum(CAMERAS), 'minItems': 1,
            'maxItems': len(CAMERAS), 'uniqueItems': True}

def domains(catalog):
    if not isinstance(catalog, dict) or not isinstance(catalog.get('candidates'), list):
        raise ValueError('A real catalogue object with candidates is required')
    rows = catalog['candidates']
    ids = [row.get('id') for row in rows]
    descriptors = [row.get('descriptor_id') for row in rows]
    if (len(ids) != 288 or len(set(ids)) != 288
            or any(not isinstance(x, str) or not re.fullmatch(r'U[0-9]{4}', x) for x in ids)
            or any(not isinstance(x, str) or not re.fullmatch(r'D[0-9]{3}', x) for x in descriptors)):
        raise ValueError('Require the exact288 unique candidate ID domain and real descriptor keys')
    return sorted(ids), sorted(set(descriptors))

def capability_schema():
    """Small transport-only shape; successful API return is not full support."""
    return exact_object({'probe': enum(['schema_compatibility_only']),
                         'acknowledged': {'type': 'boolean'}},
                        'Minimal schema compatibility probe; no scene or policy result.')

def observer_schema(catalog):
    candidates, descriptors = domains(catalog)
    confidence = enum(['clear', 'uncertain'])
    basket = exact_object({'visibility': enum(['clear', 'unknown']),
                           'unknown_contents': {'type': 'boolean'}})
    object_row = exact_object({
        'instance_id': instance(), 'identity_confidence': deepcopy(confidence),
        'category': category(), 'category_confidence': deepcopy(confidence),
        'descriptor_ids': descriptor_array(descriptors),
        'descriptor_confidence': deepcopy(confidence),
        'location': enum(['table', 'held', 'white', 'blue', 'red', 'unknown']),
        'released': {'type': ['boolean', 'null']}})
    arm = exact_object({
        'holding': enum(['empty', 'identified', 'unknown']),
        'held_instance_id': instance(nullable=True),
        'gripper_state': enum(['open', 'closed', 'transitioning', 'unknown']),
        'interaction_instance_id': instance(nullable=True), 'relation': enum(RELATIONS),
        'object_secured': enum(['yes', 'no', 'unknown']),
        'observed_basket': enum(BASKETS, nullable=True),
        'release_ready': enum(['yes', 'no', 'unknown']),
        'visibility': enum(['clear', 'occluded']), 'motion_trend': enum(MOTION_TRENDS)})
    correction = exact_object({
        'instance_id': instance(), 'previous_category': category(),
        'previous_descriptor_ids': descriptor_array(descriptors),
        'reason': text(800), 'evidence_cameras': evidence_cameras()})
    goal_correction = exact_object({
        'previous_subtask_id': enum(candidates), 'actual_instance_id': instance(),
        'reason': text(800), 'evidence_cameras': evidence_cameras(),
        # Current guard accepts an omitted OR explicit null destination. Strict
        # schema uses required-nullable, never drops a field after generation.
        'corrected_basket': enum(BASKETS, nullable=True)})
    return exact_object({
        'scene_usable': {'type': 'boolean'},
        'baskets': exact_object({key: deepcopy(basket) for key in BASKETS}),
        'objects': {'type': 'array', 'items': object_row, 'maxItems': 64},
        'corrections': {'type': 'array', 'items': correction, 'maxItems': 64},
        'goal_correction': {'anyOf': [goal_correction, {'type': 'null'}]},
        'arms': exact_object({'left': deepcopy(arm), 'right': deepcopy(arm)})},
        'CURRENT fallible visual facts only. Host history/frames and API wrappers are not output fields.')

def selector_schema(catalog, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    """Exact public response shape, not resolved context and not private thinking.

    Prompt-supported null Reasoning/Summary is an explicit stop, never admission
    to pi0.5. Set allow_explicit_abstention=False for the accepted-only contract.
    """
    candidates, _ = domains(catalog)
    if allowed_candidate_ids is not None:
        requested = list(allowed_candidate_ids)
        if (not requested or len(set(requested)) != len(requested)
                or any(x not in candidates for x in requested)):
            raise ValueError('Allowed IDs must be nonempty, unique and from the exact catalogue')
        candidates = sorted(requested)
    summary = text(400, nullable=allow_explicit_abstention)
    # Bound the first imperative to the same case-insensitive verb set as
    # grounded_motion. Actor, phase and English contradictions remain host checks.
    variants = [''.join('[' + c.upper() + c.lower() + ']' for c in verb) for verb in ACTIONS]
    summary['pattern'] = r'^(?:' + '|'.join(variants) + r')\b[^\u0000-\u001F]*(?![\s\S])'
    return exact_object({'subtask_id': enum(candidates),
        'reasoning': text(1200, nullable=allow_explicit_abstention),
        'action': exact_object({'summary': summary})},
        'Exact candidate ID with public prospective guidance. Null public text stops before policy inference.')

def response_format(kind, catalog=None, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    factories = {'capability': lambda: capability_schema(),
                 'observer': lambda: observer_schema(catalog),
                 'selector': lambda: selector_schema(catalog, allowed_candidate_ids,
                     allow_explicit_abstention=allow_explicit_abstention)}
    if kind not in factories:
        raise ValueError('kind must be capability, observer or selector')
    return {'type': 'json_schema', 'json_schema': {
        'name': 'ecot_' + kind + '_probe_v1', 'schema': factories[kind](), 'strict': True}}

def capability_response_format():
    return response_format('capability')

def observer_response_format(catalog):
    return response_format('observer', catalog)

def selector_response_format(catalog, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    return response_format('selector', catalog, allowed_candidate_ids,
                           allow_explicit_abstention=allow_explicit_abstention)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['capability', 'observer', 'selector'], default='capability')
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--allowed-id', action='append')
    parser.add_argument('--accepted-only', action='store_true', help='Selector text cannot be null')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.kind != 'capability' and not args.catalog:
        parser.error('--catalog required for observer/selector')
    catalog = json.loads(args.catalog.read_text()) if args.catalog else None
    value = response_format(args.kind, catalog, args.allowed_id,
                            allow_explicit_abstention=not args.accepted_only)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(raw)
    else:
        print(raw, end='')

if __name__ == '__main__':
    main()
