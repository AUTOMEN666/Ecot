"""Steady causal scene admission followed by a joint exact-ID/public-SSR call."""
from copy import deepcopy
from pathlib import Path

from executed_history import ExecutedSubtaskHistory
from multiview_protocol import canonical,sha
from rpc_lifecycle import exclusive_rpc
from steady_provider import SteadySubtaskProvider
from subtask_guard import SubtaskConstraintError
from ssr_protocol import (CONFIG_ID,CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,compose_prompt,
                          digest,load_reference,resolve_ssr,validate_context)

HERE=Path(__file__).resolve().parent

class SSRExecutedHistory(ExecutedSubtaskHistory):
    def __init__(self,catalog,checkpoint_step):
        if type(checkpoint_step)is not int or checkpoint_step<=0:raise ValueError('Actual SSR EMA step required')
        self.checkpoint_step=checkpoint_step
        super().__init__(catalog)
    def validate_context(self,context):validate_context(context)
    def prompt_for_context(self,context):
        from multiview_protocol import INSTRUCTION
        return compose_prompt(INSTRUCTION,context)
    def commit(self,ack,**kwargs):
        result=super().commit(ack,**kwargs)
        context=deepcopy(ack['injection']['context'])
        result.update(actual_SSR_context=context,actual_SSR_context_sha256=digest(context),
                      actual_SSR_prompt_sha256=ack['injection']['prompt_sha256'])
        return result

class SSRSteadyProvider(SteadySubtaskProvider):
    protocol_config_id=CONFIG_ID
    context_fields=CONTEXT_FIELDS
    def __init__(self,*args,checkpoint_step,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference=load_reference()
        self.executed_history=SSRExecutedHistory(self.catalog,checkpoint_step)
    def reset_context_state(self):self._ssr_authorization=None

    def observer_query(self,request,proof,**kwargs):
        # Query frames (e.g.16) need not lie on the floor(2.5*k) image-history
        # grid. Registry timestamps therefore are NOT automatically image evidence
        # in this request. Make the exact supplied frame domain explicit, without
        # filtering/repairing any model response or changing the persistent guard.
        request=deepcopy(request)
        allowed=sorted({row['frame']for row in proof['observations']})
        request['messages'][0]['content']+=('\nFRAME EVIDENCE CONTRACT: admissible_image_evidence_frames='+canonical(allowed)+
            '. Every evidence_frames, placement_frames and unplaced_frames value must be from this exact list. '
            'The scene_registry can contain older query-frame timestamps whose images are absent now; '
            'do not copy those timestamps into any evidence array. Reassess using only the supplied images.\n')
        from multiview_protocol import strict_json
        metadata=strict_json(request['messages'][-1]['content'][0]['text'])
        metadata.update(admissible_image_evidence_frames=allowed,
            registry_timestamps_are_not_current_request_image_evidence=True)
        request['messages'][-1]['content'][0]['text']=canonical(metadata)
        proof=dict(proof,protocol_config_id=CONFIG_ID,generator_protocol=CONFIG_ID,
                   schema='ecot.classify.steady_ssr.observer.v1',
                   frame_evidence_contract_revision='explicit_supplied_frame_list_v1',
                   admissible_image_evidence_frames=allowed,
                   prompt_sha256=sha(request['messages'][0]['content'].encode()))
        return super().observer_query(request,proof,**kwargs)

    def prepare_selection(self,request,proof,**kwargs):
        self._ssr_authorization=None
        request,proof=super().prepare_selection(request,proof,**kwargs)
        request['messages'][0]['content']=(HERE/'ssr_selector_prompt.md').read_text()
        from multiview_protocol import strict_json
        control=strict_json(request['messages'][1]['content'][0]['text'])
        control.update(training_reference=deepcopy(load_reference()),reference_only_not_current_truth=True,
                       causal_scene_evidence=self.guard.snapshot(),scene_evidence_is_fallible_not_GT=True,
                       prospective_action_window={'steps':HORIZON_STEPS,'hz':25,'expert_future_actions_available':False})
        request['messages'][1]['content'][0]['text']=canonical(control)
        proof.update(schema='ecot.classify.steady_ssr.request.v1',protocol_config_id=CONFIG_ID,
                     generator_protocol=CONFIG_ID,call_role='subtask_selector',context_fields=CONTEXT_FIELDS,
                     prompt_sha256=sha((HERE/'ssr_selector_prompt.md').read_bytes()),
                     messages_sha256=sha(canonical(request['messages']).encode()),
                     training_reference_sha256=REFERENCE_SHA,prospective_horizon_steps=HORIZON_STEPS,
                     observer_prompt_sha256=self.observer_receipt['prompt_sha256'],
                     frame_evidence_contract_revision=self.observer_receipt['frame_evidence_contract_revision'],
                     SSR_generated_then_discarded=False,future_test_information_supplied=False)
        return request,proof

    def resolve_selection(self,text):return resolve_ssr(text,self.catalog)

    def validate_selection(self,context,resolution,proof):
        validate_context(context)
        admission=self.guard.select(resolution['selected_id']);proof['steady_guard']=admission
        self.guard.validate_injection({'subtask':context['subtask']},
            dict(proof,selected_id=resolution['selected_id'],resolution=resolution),
            episode_id=self.episode_id,frame=proof['current_frame'])
        auth={'episode_id':self.episode_id,'frame':proof['current_frame'],'selected_id':resolution['selected_id'],
              'context_sha256':digest(context),'reference_sha256':REFERENCE_SHA}
        self._ssr_authorization=deepcopy(auth);proof['SSR_authorization']=auth

    def _validate_injection_unlocked(self,context,lineage,*,episode_id,frame):
        validate_context(context)
        auth=self._ssr_authorization
        if (self.stopped is not None or auth is None or auth!=lineage.get('SSR_authorization')
                or auth['episode_id']!=episode_id or auth['frame']!=frame
                or auth['context_sha256']!=digest(context)or lineage.get('context_sha256')!=digest(context)):
            raise SubtaskConstraintError('SSR text/episode/frame changed after generation authorization')
        self.guard.validate_injection({'subtask':context['subtask']},lineage,episode_id=episode_id,frame=frame)

    @exclusive_rpc
    def validate_injection(self,context,lineage,*,episode_id,frame):
        return self._validate_injection_unlocked(context,lineage,episode_id=episode_id,frame=frame)
