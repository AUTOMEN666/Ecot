"""Current visual facts -> instance admission -> grounded prospective public SSR.

Reuses the existing actual image/history/transport/ACK implementation. A model
failure stops without output repair, semantic redraw, policy inference or ACK.
"""
from copy import deepcopy
from pathlib import Path

from grounded_motion import action_envelopes,validate_arms,validate_plan
from grounded_scene_guard import GroundedSceneGuard
from grounded_ssr_protocol import CONFIG_ID,SCENE_PROTOCOL,MOTION_PROTOCOL
from multiview_protocol import canonical,sha,strict_json
from rpc_lifecycle import exclusive_rpc
from ssr_protocol import CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,digest,load_reference,validate_context
from ssr_provider import SSRSteadyProvider
from steady_provider import SteadySubtaskProvider
from subtask_guard import SubtaskConstraintError

HERE=Path(__file__).resolve().parent

class GroundedSSRProvider(SSRSteadyProvider):
    protocol_config_id=CONFIG_ID
    observer_prompt_path=HERE/'grounded_ssr_observer_prompt.md'
    history_mode='actual_execution'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guard=GroundedSceneGuard(self.catalog)
        self._motion_envelopes={};self._candidate_scene_sha=None
    def reset_context_state(self):
        super().reset_context_state();self._motion_envelopes={};self._candidate_scene_sha=None

    def observer_response_format(self):
        return {'type':'json_object'}

    def selector_response_format(self,allowed_ids):
        return {'type':'json_object'}

    def validate_observation_format(self,evidence):
        # Versioned providers can add stricter format admission, without
        # rewriting model observations or changing the visual scene guard.
        pass

    def validate_current_arms(self,evidence,proof):
        return validate_arms(evidence['arms'],evidence['objects'])

    def make_action_envelopes(self,evidence,target,**kwargs):
        return action_envelopes(evidence,target,**kwargs)

    def validate_public_plan(self,context,envelope):
        return validate_plan(context,envelope)

    def history_metadata(self,request,proof):
        request=deepcopy(request);proof=deepcopy(proof)
        meta=strict_json(request['messages'][-1]['content'][0]['text'])
        replay=self.history_mode=='expert_replay'
        if self.history_mode not in('expert_replay','actual_execution'):raise ValueError('Explicit history mode required')
        meta.update(history_mode=self.history_mode,expert_motion_conditioned_on_annotations=False if replay else None,
            actual_policy_injection_or_ACK=False if replay else None,
            executed_history_semantics=(
                'Prior GENERATED Subtasks followed by exposure of recorded expert transitions. The expert motion was NOT conditioned on these annotations. This is NOT policy execution or a policy ACK. Selection/exposure is not object identity, grasp or completion truth.'
                if replay else 'Prior actual policy-injected Subtasks followed by logged executed actions with verified ACKs. These instructions do not prove object identity, contact, grasp or completion.'))
        meta['current_camera_alignment']='Current three synchronized RGB views and observed State14 at the actual source frame.'
        request['messages'][-1]['content'][0]['text']=canonical(meta)
        proof.update(history_mode=self.history_mode,only_actual_executed_prior_text_sent=not replay,
            actual_policy_injection_or_ACK=False if replay else None,
            expert_motion_conditioned_on_annotations=False if replay else None)
        return request,proof

    def prepare_selection(self,request,proof,*,started,unchanged):
        self._ssr_authorization=None;self._motion_envelopes={};self._candidate_scene_sha=None
        request,proof=self.history_metadata(request,proof)
        scene_before=self.guard.snapshot();state_sha=digest(scene_before)
        observer=deepcopy(request);observer_prompt=self.observer_prompt_path.read_text()
        observer['messages'][0]['content']=observer_prompt
        descriptors={c['descriptor_id']:c['object_phrase']for c in self.catalog['candidates']}
        observer['messages'][1]['content'][0]['text']=canonical({'descriptors':[
            {'descriptor_id':key,'text':text}for key,text in descriptors.items()],
            'allowed_descriptor_ids':sorted(descriptors),
            'descriptor_ids_output_rule':'Copy descriptor_id KEYS (D###), never the text or candidate U### IDs.',
            'registry_is_fallible_visual_memory_not_GT':True,'report_current_facts_only':True})
        meta=strict_json(observer['messages'][-1]['content'][0]['text'])
        meta.update(scene_registry=scene_before,current_report_frame=proof['current_frame'],
            model_must_not_generate_frame_numbers=True,
            actual_supplied_image_frames=sorted({x['frame']for x in proof['observations']}))
        observer['messages'][-1]['content'][0]['text']=canonical(meta)
        observer['response_format']=deepcopy(self.observer_response_format())
        def observer_unchanged():
            unchanged()
            if digest(self.guard.snapshot())!=state_sha:raise SubtaskConstraintError('Scene changed during visual observation')
        observer_proof=dict(proof,schema='ecot.classify.grounded_SSR.current_visual_observation.v2',
            protocol_config_id=self.protocol_config_id,generator_protocol=self.protocol_config_id,call_role='scene_observer',
            scene_evidence_protocol=SCENE_PROTOCOL,motion_evidence_protocol=MOTION_PROTOCOL,
            prompt_sha256=sha(observer_prompt.encode()),scene_registry_sha256=state_sha,
            model_supplied_frame_arrays=False,response_format=deepcopy(observer['response_format']),
            response_format_sha256=digest(observer['response_format']))
        evidence,receipt=SteadySubtaskProvider.observer_query(self,observer,observer_proof,
            started=started,unchanged=observer_unchanged)
        observer_unchanged();self.validate_observation_format(evidence)
        self.validate_current_arms(evidence,observer_proof)
        self.observer_receipt=receipt
        decision=self.guard.observe(evidence,episode_id=proof['episode_id'],frame=proof['current_frame'],
            available_frames=[x['frame']for x in proof['observations']],
            executed_records=proof['executed_subtask_history'])
        self._candidate_scene_sha=digest(self.guard.snapshot())
        for ident,target in decision['targets'].items():
            self._motion_envelopes[ident]=self.make_action_envelopes(evidence,target,
                contact_transition=decision.get('known_contact_transition'))
        envelopes={}
        for ident,target in decision['targets'].items():
            key=target['instance_id']+':'+target['basket_color'];full=self._motion_envelopes[ident]
            envelopes[key]={'instance_id':target['instance_id'],'basket_color':target['basket_color'],
                'by_arm':{arm:row['allowed_verbs']for arm,row in full['by_arm'].items()}}
        selector_prompt=(HERE/'grounded_ssr_selector_prompt.md').read_text()
        request['messages'][0]['content']=selector_prompt
        allowed=set(decision['allowed_ids'])
        request['response_format']=deepcopy(self.selector_response_format(sorted(allowed)))
        request['messages'][1]['content'][0]['text']=canonical({
            'allowed_candidates':[{'id':c['id'],'text':c['text'],
                'instance_id':decision['targets'][c['id']]['instance_id'],
                'basket_color':decision['targets'][c['id']]['basket_color'],
                'motion_key':decision['targets'][c['id']]['instance_id']+':'+decision['targets'][c['id']]['basket_color']}
                for c in self.catalog['candidates']if c['id']in allowed],
            'original_catalogue_count':288,'mode':decision['mode'],'locked_id':decision.get('locked_id'),
            'category_to_basket':decision['category_to_basket'],'basket_to_category':decision['basket_to_category'],
            'current_visual_observations':evidence,'current_observations_are_fallible_not_GT':True,
            'motion_options':envelopes,'host_repairs_model_text':False,
            'training_reference':deepcopy(load_reference()),'reference_only_not_current_truth':True,
            'prospective_action_window':{'steps':HORIZON_STEPS,'hz':25,'execute_before_reobserve':8,'expert_future_actions_available':False},
            'history_mode':self.history_mode,'selection_rule':'Exactly one allowed ID; immediate Summary verb must be admitted for its stated arm. No semantic retry, replacement or repair.'})
        proof.update(schema='ecot.classify.grounded_SSR.request.v2',protocol_config_id=self.protocol_config_id,
            generator_protocol=self.protocol_config_id,call_role='subtask_selector',context_fields=CONTEXT_FIELDS,
            response_format=deepcopy(request['response_format']),response_format_sha256=digest(request['response_format']),
            observer_request_sha256=receipt['request_sha256'],observer_response_sha256=receipt['response_sha256'],
            observer_prompt_sha256=receipt['prompt_sha256'],prompt_sha256=sha(selector_prompt.encode()),
            messages_sha256=sha(canonical(request['messages']).encode()),training_reference_sha256=REFERENCE_SHA,
            scene_evidence_protocol=SCENE_PROTOCOL,motion_evidence_protocol=MOTION_PROTOCOL,
            prospective_horizon_steps=HORIZON_STEPS,current_visual_observation=evidence,
            current_visual_observation_sha256=digest(evidence),grounded_decision=decision,
            current_frame_stamp_source='actual synchronized observation/proof; never model timestamps',
            grounded_motion_options_sha256=digest(self._motion_envelopes),
            candidate_scene_sha256=self._candidate_scene_sha,shared_observer_selector_budget_seconds=1850,
            future_test_information_supplied=False,SSR_generated_then_discarded=False)
        return request,proof

    def validate_selection(self,context,resolution,proof):
        validate_context(context);ident=resolution['selected_id']
        if (self._candidate_scene_sha is None or digest(self.guard.snapshot())!=self._candidate_scene_sha
                or proof.get('candidate_scene_sha256')!=self._candidate_scene_sha
                or proof.get('grounded_motion_options_sha256')!=digest(self._motion_envelopes)
                or ident not in self._motion_envelopes):
            raise SubtaskConstraintError('Current grounded candidate/phase evidence changed before selection')
        motion=self.validate_public_plan(context,self._motion_envelopes[ident])
        admission=self.guard.select(ident);proof['steady_guard']=admission
        self.guard.validate_injection({'subtask':context['subtask']},
            dict(proof,selected_id=ident,resolution=resolution),episode_id=self.episode_id,frame=proof['current_frame'])
        auth={'episode_id':self.episode_id,'frame':proof['current_frame'],'selected_id':ident,
            'context_sha256':digest(context),'reference_sha256':REFERENCE_SHA,
            'motion_envelope_sha256':digest(self._motion_envelopes[ident]),'motion_contract':motion}
        self._ssr_authorization=deepcopy(auth);proof['SSR_authorization']=auth
        proof['grounded_motion_authorization']=motion

    @exclusive_rpc
    def validate_injection(self,context,lineage,*,episode_id,frame):
        # Parent binds all three exact public fields, frame, episode and current
        # guard evidence. Revalidate motion too, without re-generating any text.
        self._validate_injection_unlocked(context,lineage,episode_id=episode_id,frame=frame)
        ident=lineage['selected_id']
        if ident not in self._motion_envelopes or digest(self._motion_envelopes[ident])!=self._ssr_authorization['motion_envelope_sha256']:
            raise SubtaskConstraintError('Motion evidence changed after authorization')
        self.validate_public_plan(context,self._motion_envelopes[ident])
