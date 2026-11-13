"""V7 actual-selector proof binding for finite completed-release clauses.

The release budget, observer/schema, full history and actual ACK machinery are
inherited. A sealed copied selector proof is staged before parent selection;
the same context/evidence/envelope and authorized scene are rechecked on every
public-plan/injection validation. No evidence or public string is rewritten.
"""
from copy import deepcopy
from pathlib import Path

from grounded_motion import require
from grounded_motion_v6 import proof_binding
from grounded_motion_v7 import validate_plan
from multiview_protocol import canonical,strict_json
from release_grounded_ssr_provider import ReleaseGroundedSSRProvider
from ssr_protocol import digest
from temporal_action_clause import is_completed_release_clause
from temporal_grounded_ssr_protocol import CONFIG_ID,MOTION_INTERPRETATION


class TemporalGroundedSSRProvider(ReleaseGroundedSSRProvider):
    protocol_config_id=CONFIG_ID
    motion_interpretation=MOTION_INTERPRETATION
    observer_prompt_path=Path(__file__).resolve().parent/'grounded_ssr_observer_prompt_v7.md'

    def _clear_temporal_binding(self):
        self._temporal_binding=None;self._temporal_binding_sha256=None
        self._temporal_authorized_state=None;self._temporal_authorized_state_sha256=None
        self._temporal_phase=None

    def reset_context_state(self):
        super().reset_context_state();self._clear_temporal_binding()

    def history_metadata(self,request,proof):
        self._clear_temporal_binding()
        request,proof=super().history_metadata(request,proof)
        meta=strict_json(request['messages'][-1]['content'][0]['text'])
        meta['motion_interpretation']=MOTION_INTERPRETATION
        request['messages'][-1]['content'][0]['text']=canonical(meta)
        proof['motion_interpretation']=MOTION_INTERPRETATION
        return request,proof

    def make_action_envelopes(self,evidence,target,**kwargs):
        envelope=super().make_action_envelopes(evidence,target,**kwargs)
        envelope['motion_interpretation']=MOTION_INTERPRETATION
        return envelope

    def _stage_temporal_binding(self,context,resolution,proof):
        ident=resolution['selected_id'];scene=self.guard.snapshot()
        require(proof.get('generator_protocol')==proof.get('protocol_config_id')==CONFIG_ID
            and proof.get('motion_interpretation')==MOTION_INTERPRETATION
            and proof.get('call_role')=='subtask_selector'
            and proof.get('episode_id')==getattr(self,'episode_id',None)==scene['episode_id']
            and proof.get('current_frame')==scene['current_frame'],
            'Temporal clause needs the actual same-frame v7 selector proof')
        proof_binding(proof)
        decision=proof['grounded_decision'];evidence=proof['current_visual_observation']
        require(decision==self.guard.decision and decision['episode_id']==proof['episode_id']
            and decision['frame']==proof['current_frame']and ident in decision['allowed_ids']
            and ident in decision['targets']and ident in self._motion_envelopes,
            'Temporal clause selector target differs from actual guard admission')
        require(proof.get('current_visual_observation_sha256')==digest(evidence)
            and scene['arms']==evidence['arms']and scene['baskets']==evidence['baskets']
            and scene['current_visible_objects']=={o['instance_id']:o for o in evidence['objects']}
            and proof.get('candidate_scene_sha256')==digest(scene)==self._candidate_scene_sha
            and proof.get('grounded_motion_options_sha256')==digest(self._motion_envelopes),
            'Temporal clause evidence/envelope differs from actual current scene')
        require(context.get('subtask')==self.guard.by_id[ident]['text'],
            'Temporal clause context differs from exact selected candidate text')
        target=dict(deepcopy(decision['targets'][ident]),subtask_id=ident)
        return {'episode_id':proof['episode_id'],'frame':proof['current_frame'],'selected_id':ident,
            'selector_proof':deepcopy(proof),'selector_proof_sha256':digest(proof),
            'evidence':deepcopy(evidence),'target':target,'scene_before_selection_sha256':digest(scene),
            'context_sha256':digest(context),'envelope_sha256':digest(self._motion_envelopes[ident]),
            'all_motion_envelopes_sha256':digest(self._motion_envelopes)}

    def validate_selection(self,context,resolution,proof):
        self._clear_temporal_binding()
        try:
            binding=self._stage_temporal_binding(context,resolution,proof)
            self._temporal_binding=deepcopy(binding);self._temporal_binding_sha256=digest(binding)
            self._temporal_phase='selecting'
            super().validate_selection(context,resolution,proof)
            state={'binding_sha256':self._temporal_binding_sha256,
                'episode_id':self.episode_id,'frame':proof['current_frame'],'selected_id':resolution['selected_id'],
                'scene_sha256':digest(self.guard.snapshot()),'guard_authorization_sha256':digest(self.guard.authorization),
                'SSR_authorization_sha256':digest(self._ssr_authorization)}
            self._temporal_authorized_state=deepcopy(state);self._temporal_authorized_state_sha256=digest(state)
            self._temporal_phase='authorized'
            proof['temporal_completed_release_binding_sha256']=self._temporal_binding_sha256
            proof['temporal_actual_selector_proof_sha256']=binding['selector_proof_sha256']
            proof['temporal_binding_source']='actual_current_selector_proof_and_guard_target'
        except Exception:
            self._clear_temporal_binding()
            raise

    def _validate_temporal_binding(self,context,envelope):
        binding=getattr(self,'_temporal_binding',None);seal=getattr(self,'_temporal_binding_sha256',None)
        require(type(binding)is dict and type(seal)is str and digest(binding)==seal,
            'Temporal clause binding missing or mutated')
        scene=self.guard.snapshot();ident=binding['selected_id'];proof=binding['selector_proof']
        require(getattr(self,'stopped',None)is None
            and getattr(self,'episode_id',None)==scene['episode_id']==binding['episode_id']==proof['episode_id']
            and scene['current_frame']==binding['frame']==proof['current_frame']
            and digest(context)==binding['context_sha256']
            and digest(envelope)==binding['envelope_sha256']
            and ident in self._motion_envelopes
            and digest(self._motion_envelopes[ident])==binding['envelope_sha256']
            and digest(self._motion_envelopes)==binding['all_motion_envelopes_sha256']
            and digest(proof)==binding['selector_proof_sha256']
            and proof['grounded_decision']==self.guard.decision,
            'Temporal clause context/proof/envelope/frame/episode changed')
        evidence=binding['evidence'];target=binding['target']
        require(evidence==proof['current_visual_observation']
            and digest(evidence)==proof['current_visual_observation_sha256']
            and scene['arms']==evidence['arms']and scene['baskets']==evidence['baskets']
            and scene['current_visible_objects']=={o['instance_id']:o for o in evidence['objects']}
            and target==dict(proof['grounded_decision']['targets'][ident],subtask_id=ident),
            'Temporal clause current evidence or physical target changed')
        authorized=getattr(self,'_temporal_authorized_state',None)
        if getattr(self,'_temporal_phase',None)=='selecting':
            require(authorized is None and getattr(self,'_temporal_authorized_state_sha256',None)is None
                and digest(scene)==binding['scene_before_selection_sha256'],
                'Temporal clause selection scene changed before authorization')
        else:
            require(getattr(self,'_temporal_phase',None)=='authorized'
                and type(authorized)is dict and digest(authorized)==self._temporal_authorized_state_sha256
                and authorized['binding_sha256']==seal
                and authorized['episode_id']==binding['episode_id']and authorized['frame']==binding['frame']
                and authorized['selected_id']==ident and authorized['scene_sha256']==digest(scene)
                and authorized['guard_authorization_sha256']==digest(self.guard.authorization)
                and authorized['SSR_authorization_sha256']==digest(self._ssr_authorization),
                'Temporal clause authorized scene or authorization changed')
        return binding

    def _validate_injection_unlocked(self,context,lineage,*,episode_id,frame):
        super()._validate_injection_unlocked(context,lineage,episode_id=episode_id,frame=frame)
        ident=lineage['selected_id']
        binding=self._validate_temporal_binding(context,self._motion_envelopes[ident])
        require(lineage.get('temporal_completed_release_binding_sha256')==self._temporal_binding_sha256
            and lineage.get('temporal_actual_selector_proof_sha256')==binding['selector_proof_sha256']
            and lineage.get('temporal_binding_source')=='actual_current_selector_proof_and_guard_target'
            and all(lineage.get(key)==value for key,value in binding['selector_proof'].items()),
            'Temporal clause actual selector lineage differs from sealed current proof')

    def validate_public_plan(self,context,envelope):
        binding=self._validate_temporal_binding(context,envelope)
        def completed(text,index):
            # Recheck on each exact token; the helper is not the negation hook.
            current=self._validate_temporal_binding(context,envelope)
            return is_completed_release_clause(text,index,context=context,
                evidence=current['evidence'],target=current['target'],
                proof=current['selector_proof'],envelope=envelope)
        result=validate_plan(context,envelope,completed_release_predicate=completed)
        self._validate_temporal_binding(context,envelope)
        return dict(result,temporal_completed_release_binding_sha256=self._temporal_binding_sha256,
            temporal_actual_selector_proof_sha256=binding['selector_proof_sha256'],
            temporal_current_evidence_sha256=digest(binding['evidence']))
