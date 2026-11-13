"""Versioned category-token observer contract; original selector is unchanged.

The tokens constrain annotator language, not current SDK/GT classes. Neither a
descriptor ID nor its wording determines the visually observed category here.
"""
from copy import deepcopy
import grounded_response_schema as base

SCHEMA_REVISION = 'ecot.classify_objects.grounded_ssr.category_tokens.v2'
CATEGORY_TOKENS = ('pen', 'watch', 'camera', 'plush_toy', 'toy_vehicle',
                   'block', 'box', 'ring', 'bracelet', 'pouch', 'strap',
                   'tool', 'bottle_cap', 'can', 'tracker')

def category():
    return base.enum(CATEGORY_TOKENS, nullable=True)

def observer_schema(catalog):
    schema = base.observer_schema(catalog)
    schema['properties']['objects']['items']['properties']['category'] = category()
    schema['properties']['corrections']['items']['properties']['previous_category'] = category()
    return schema

def observer_response_format(catalog):
    value = base.observer_response_format(catalog)
    value['json_schema']['schema'] = observer_schema(catalog)
    return value

def selector_schema(catalog, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    return base.selector_schema(catalog, allowed_candidate_ids,
        allow_explicit_abstention=allow_explicit_abstention)

def selector_response_format(catalog, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    return base.selector_response_format(catalog, allowed_candidate_ids,
        allow_explicit_abstention=allow_explicit_abstention)

def response_format(kind, catalog=None, allowed_candidate_ids=None, *, allow_explicit_abstention=True):
    if kind == 'observer':
        return observer_response_format(catalog)
    if kind == 'selector':
        return selector_response_format(catalog, allowed_candidate_ids,
            allow_explicit_abstention=allow_explicit_abstention)
    if kind == 'capability':
        return deepcopy(base.capability_response_format())
    raise ValueError('kind must be capability, observer or selector')
