"""Full-schema grounded observer/selector using unchanged v2 causal guards."""
import grounded_response_schema
from grounded_ssr_provider import GroundedSSRProvider
from strict_grounded_ssr_protocol import CONFIG_ID, validate_schema_factory

class StrictGroundedSSRProvider(GroundedSSRProvider):
    protocol_config_id = CONFIG_ID

    def observer_response_format(self):
        validate_schema_factory()
        return grounded_response_schema.observer_response_format(self.catalog)

    def selector_response_format(self, allowed_ids):
        validate_schema_factory()
        return grounded_response_schema.selector_response_format(self.catalog, allowed_ids)
