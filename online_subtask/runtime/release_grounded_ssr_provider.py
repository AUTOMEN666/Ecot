"""V6 current release progression, exact public text and actual proof binding."""
from copy import deepcopy
from pathlib import Path

from enum_grounded_ssr_provider import EnumGroundedSSRProvider
from grounded_ssr_provider import GroundedSSRProvider
from grounded_motion import validate_arms
from grounded_motion_v6 import GOAL_FIELDS,action_envelopes,derive_release_transitions,proof_binding,validate_plan
from multiview_protocol import canonical,strict_json
from release_grounded_ssr_protocol import CONFIG_ID,MOTION_INTERPRETATION
from ssr_protocol import digest
from subtask_guard import SubtaskConstraintError


class ReleaseGroundedSSRProvider(EnumGroundedSSRProvider):
    protocol_config_id=CONFIG_ID
    motion_interpretation=MOTION_INTERPRETATION
    observer_prompt_path=Path(__file__).resolve().parent/'grounded_ssr_observer_prompt_v6.md'

    def reset_context_state(self):
        super().reset_context_state()
        self._release_transition_state={'episode_id':getattr(self,'episode_id',None),'bursts':{}}
        self._release_pending=None

    def history_metadata(self,request,proof):
        # Call the shared metadata implementation directly, not v5's marker.
        request,proof=GroundedSSRProvider.history_metadata(self,request,proof)
        meta=strict_json(request['messages'][-1]['content'][0]['text'])
        meta['motion_interpretation']=MOTION_INTERPRETATION
        request['messages'][-1]['content'][0]['text']=canonical(meta)
        proof['motion_interpretation']=MOTION_INTERPRETATION
        return request,proof

    def validate_current_arms(self,evidence,proof):
        self._release_pending=None
        before=self.guard.snapshot();episode=proof['episode_id']
        state=getattr(self,'_release_transition_state',None)
        if state is None:
            if before.get('current_frame')is not None:
                raise SubtaskConstraintError('Release budget missing after episode observations began')
            state={'episode_id':episode,'bursts':{}}
        transitions,after=derive_release_transitions(evidence,before=before,proof=proof,transition_state=state)
        result=validate_arms(evidence['arms'],evidence['objects'],known_release_transitions=transitions)
        self._release_pending={'episode_id':episode,'frame':proof['current_frame'],
            'evidence_sha256':digest(evidence),'proof_binding':proof_binding(proof),
            'before':before,'after':after,'transitions':transitions}
        return result

    def make_action_envelopes(self,evidence,target,**kwargs):
        pending=self._release_pending
        current=self.guard.snapshot()
        if (pending is None or pending['episode_id']!=current['episode_id']
                or pending['frame']!=current['current_frame']or pending['evidence_sha256']!=digest(evidence)
                or current['arms']!=evidence['arms']or current['current_visible_objects']!=
                    {o['instance_id']:o for o in evidence['objects']}):
            raise SubtaskConstraintError('Release phase evidence was not bound before actual scene admission')
        transitions=pending['transitions']
        if transitions:
            goal=pending['before']['active_goal']
            actual=current['active_goal']
            if (not actual or any(actual.get(k)!=goal[k]for k in GOAL_FIELDS)
                    or any(target.get(k)!=goal[k]for k in('instance_id','category','basket_color'))):
                raise SubtaskConstraintError('Release transition cannot switch the locked target')
        envelope=action_envelopes(evidence,target,known_release_transitions=transitions,**kwargs)
        # Scene guard already verified the actual complete history/prefix. Do
        # not spend opening budget for a failed observation or failed scene.
        self._release_transition_state=deepcopy(pending['after'])
        return envelope

    def validate_public_plan(self,context,envelope):
        return validate_plan(context,envelope)
