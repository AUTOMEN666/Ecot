"""Strict schema/category-token observer with unchanged causal scene guards."""
from pathlib import Path

import grounded_response_schema_v2
from grounded_ssr_provider import GroundedSSRProvider
from enum_grounded_ssr_protocol import CONFIG_ID, validate_schema_factories
from subtask_guard import SubtaskConstraintError

class EnumGroundedSSRProvider(GroundedSSRProvider):
    protocol_config_id = CONFIG_ID
    observer_prompt_path = Path(__file__).resolve().parent / 'grounded_ssr_observer_prompt_v4.md'

    def observer_response_format(self):
        validate_schema_factories()
        return grounded_response_schema_v2.observer_response_format(self.catalog)

    def selector_response_format(self, allowed_ids):
        validate_schema_factories()
        return grounded_response_schema_v2.selector_response_format(self.catalog, allowed_ids)

    def validate_observation_format(self, evidence):
        # Reject the original response, never turn descriptor/phrase/SDK labels
        # into a category. Format-valid tokens still require visual grounding.
        if type(evidence) is not dict or type(evidence.get('objects')) is not list:
            raise SubtaskConstraintError('Category-token observation requires current object records')
        for index, obj in enumerate(evidence['objects']):
            if type(obj) is not dict or 'category' not in obj:
                raise SubtaskConstraintError('Explicit current category token/null required')
            value = obj['category']
            if value is not None and (type(value) is not str or value not in grounded_response_schema_v2.CATEGORY_TOKENS):
                raise SubtaskConstraintError('Category must be an exact vocabulary token or null at objects[' + str(index) + ']')
            if value is None and obj.get('category_confidence') != 'uncertain':
                raise SubtaskConstraintError('Null category requires explicit uncertain confidence')
        corrections = evidence.get('corrections')
        if type(corrections) is not list:
            raise SubtaskConstraintError('Explicit current correction list required')
        for row in corrections:
            if type(row) is not dict or 'previous_category' not in row:
                raise SubtaskConstraintError('Correction must retain exact previous category token/null')
            old = row['previous_category']
            if old is not None and (type(old) is not str or old not in grounded_response_schema_v2.CATEGORY_TOKENS):
                raise SubtaskConstraintError('Previous category must be an exact vocabulary token or null')
