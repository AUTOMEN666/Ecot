"""V5 release/actor interpretation with unchanged v4 observer/selector schema."""
from enum_grounded_ssr_provider import EnumGroundedSSRProvider
from grounded_motion_v5 import action_envelopes,validate_plan
from multiview_protocol import canonical,strict_json
from phase_grounded_ssr_protocol import CONFIG_ID,MOTION_INTERPRETATION


class PhaseGroundedSSRProvider(EnumGroundedSSRProvider):
    protocol_config_id=CONFIG_ID
    motion_interpretation=MOTION_INTERPRETATION

    def history_metadata(self,request,proof):
        request,proof=super().history_metadata(request,proof)
        # Added before either observer/selector is built, so actual request
        # bytes and both role proofs bind the same explicit interpretation.
        meta=strict_json(request['messages'][-1]['content'][0]['text'])
        meta['motion_interpretation']=MOTION_INTERPRETATION
        request['messages'][-1]['content'][0]['text']=canonical(meta)
        proof['motion_interpretation']=MOTION_INTERPRETATION
        return request,proof

    def make_action_envelopes(self,evidence,target,**kwargs):
        return action_envelopes(evidence,target,**kwargs)

    def validate_public_plan(self,context,envelope):
        return validate_plan(context,envelope)
