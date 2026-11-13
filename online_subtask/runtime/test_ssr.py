"""Causal public-SSR generation, original serialization and real ACK method wiring.

Policy/weights/transport are explicit fixtures, never real GPU/native acceptance.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import numpy as np

from multiview_protocol import CAMERAS,INSTRUCTION,canonical,is_history_source_frame,load_catalog
from prepare_ssr_spec import build
from rpc_lifecycle import RPCBusyError
from ssr_protocol import (CONFIG_ID,NAMESPACE,REFERENCE_SHA,SSR_WRAPPER,compose_prompt,load_reference,
                          resolve_ssr,serialize_context,validate_spec)
from ssr_provider import SSRSteadyProvider
from test_executed_history import ACK_SCHEMA,clock
from test_executed_history_integration import response
from test_history96 import state,triplet
from test_steady_runtime import obs
from test_subtask_guard import obj,scene

def public_reply(frame=0,ident='U0012'):
    return {'subtask_id':ident,'reasoning':'Keep the right gripper open while approaching and aligning with the pen; close once contact is supported.',
            'action':{'summary':'Approach the pen to prepare a grasp.' if frame==0 else 'Align the open gripper with the pen before closing.'}}

def capture_to(provider,frame):
    first=provider.frames[-1].frame+1 if provider.frames else 0
    for n in range(first,frame+1):
        if is_history_source_frame(n):
            provider.capture_history(triplet(n),INSTRUCTION,state14=state(n),episode_id=provider.episode_id,frame=n)

def transport(request,proof):
    if proof['call_role']=='scene_observer':return response(canonical(scene(obj(frame=proof['current_frame']))))
    return response(canonical(public_reply(proof['current_frame'])))

def acknowledge(model,commands,frame,count=8,reason='full_chunk'):
    injection=model.current_injection()
    return model.acknowledge_executed_subtask({'schema':ACK_SCHEMA,'injection':injection,'chunk_end_reason':reason,
        'execution_clocks':[{'episode_id':model.episode_id,'execution_frame':frame+i+1,
            'before':clock(frame+i),'after':clock(frame+i+1),'native_take_action_count':frame+i+1,
            'action':{k:v.tolist()for k,v in action.items()}}for i,action in enumerate(commands[:count])]})

@contextmanager
def runtime(tmp,wire=transport,events=None,step=17392):
    support=types.ModuleType('support');support.__path__=[]
    task=types.ModuleType('support.task_contract');task.INSTRUCTION=INSTRUCTION;task.TASK='classify_objects'
    weights=types.ModuleType('support.weights_contract');weights.validate_weights=lambda b:None
    training=types.ModuleType('strict_prompt')
    # Independent literal from the source-bound original training formatter.
    wrapper=('\nCurrent-frame action summary, subtask, and reasoning annotations: '
        'use the current observations, task, and supplied annotations. Missing or null fields are unspecified. '
        'No text response is requested.\nContext: ')
    training.compose_training_prompt=lambda instruction,text:instruction+wrapper+text
    class Random:
        key=staticmethod(lambda seed:np.array([seed],np.int32))
        key_data=staticmethod(lambda key:key)
        split=staticmethod(lambda key:(key,key+1))
        normal=staticmethod(lambda key,shape:np.zeros(shape,np.float32))
    class Policy:
        def __init__(self):self._rng=np.array([42],np.int32);self.observations=[]
        def infer(self,observation,noise):
            if events is not None:events.append('policy')
            self.observations.append(deepcopy(observation));self._rng+=1
            return {'actions':np.zeros((50,14),np.float32)}
    binding={'step':step,'weights_mode':'checkpoint','inference_params':'EMA','weights_identity_sha256':'fixture-only',
             'checkpoint':'fixture-only','checkpoint_metadata':{'identity':{'max_token_len':512}},'train_config_path':'fixture-only'}
    path=Path(tmp)/'binding.json';path.write_text(json.dumps(binding))
    spec=build(path,parent_root=tmp,audit_path=Path(tmp)/NAMESPACE/'actions.jsonl')
    with patch.dict(sys.modules,{'support':support,'support.task_contract':task,'support.weights_contract':weights,'strict_prompt':training}):
        import model_server,ssr_binding,subtask_client
        provider=SSRSteadyProvider(Path(tmp)/'requests',selector_kind='deepseek',transport=wire,checkpoint_step=step)
        policy=Policy()
        with patch.object(ssr_binding,'load_weights',return_value=binding), \
             patch.object(subtask_client,'png_current_rgb',lambda image:triplet(0)[CAMERAS[0]]):
            model=model_server.create_model(spec,policy=policy,jax_module=types.SimpleNamespace(random=Random),provider=provider)
            yield model,policy,provider,spec

class SSRContracts(unittest.TestCase):
    def test_registry_query_timestamp_is_not_admitted_as_an_absent_history_image(self):
        seen=[]
        def wire(request,proof):seen.append((deepcopy(request),deepcopy(proof)));return transport(request,proof)
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,wire,step=19566)as(model,policy,p,spec):
            model.reset()
            for frame in [0,8,16,24]:
                capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
            request,proof=[row for row in seen if row[1]['current_frame']==24 and row[1]['call_role']=='scene_observer'][0]
            allowed=[0,2,5,7,10,12,15,17,20,22,24]
            self.assertEqual(proof['admissible_image_evidence_frames'],allowed)
            metadata=json.loads(request['messages'][-1]['content'][0]['text'])
            self.assertEqual(metadata['admissible_image_evidence_frames'],allowed)
            self.assertEqual(metadata['scene_registry']['active_goal']['last_unplaced_at'],16)
            self.assertNotIn(16,allowed)
            self.assertIn('do not copy those timestamps',request['messages'][0]['content'])

    def test_actual_training_formatter_golden_examples_are_byte_identical(self):
        golden=json.loads((Path(__file__).parents[1]/'provenance/SSR_TRAINING_FORMAT_REFERENCE.json').read_text())
        self.assertEqual(golden['sample_count'],7);self.assertTrue(golden['actual_external_training_formatter_compared'])
        for row in golden['rows']:self.assertEqual(compose_prompt(INSTRUCTION,row['context']),row['expected_training_prompt'])

    def test_exact_ID_and_public_fields_no_extra_output_or_repair(self):
        catalog=load_catalog();value=public_reply();context,resolution=resolve_ssr(canonical(value),catalog)
        self.assertEqual(context['subtask'],next(c['text']for c in catalog['candidates']if c['id']=='U0012'))
        self.assertEqual(context['action'],value['action']);self.assertEqual(context['reasoning'],value['reasoning'])
        for broken in [dict(value,extra='bad'),dict(value,subtask_id='UNKNOWN'),dict(value,reasoning=None),
                       dict(value,action={'summary':None}),dict(value,action={'summary':'a','coordinates':[1,2]})]:
            with self.subTest(broken=broken),self.assertRaises(ValueError):resolve_ssr(canonical(broken),catalog)

    def test_real_training_reference_not_generated_examples_or_test_future(self):
        ref=load_reference();self.assertEqual(len(ref['examples']),7);self.assertTrue(ref['reference_only'])
        self.assertTrue(all(x['episode']==200 and x['frame']<=745 for x in ref['examples']))
        with tempfile.TemporaryDirectory()as tmp:
            seen=[]
            def wire(request,proof):seen.append((deepcopy(request),deepcopy(proof)));return transport(request,proof)
            p=SSRSteadyProvider(tmp,selector_kind='deepseek',transport=wire,checkpoint_step=19566)
            p.reset('fixture');capture_to(p,0);context,proof=p.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            request,meta=seen[1];control=json.loads(request['messages'][1]['content'][0]['text'])
            self.assertEqual(control['training_reference'],ref);self.assertFalse(control['prospective_action_window']['expert_future_actions_available'])
            self.assertIn('Anticipate a short action window',request['messages'][0]['content'])
            self.assertEqual(meta['training_reference_sha256'],REFERENCE_SHA);self.assertEqual(proof['context_fields'],['subtask','action.summary','reasoning'])
            self.assertEqual(len(p.executed_history.records),0)

    def test_generation_then_actual_policy_contains_all_three_fields_then_ACK_and_next_history(self):
        events=[];requests=[]
        def wire(request,proof):events.append(proof['call_role']);requests.append((deepcopy(request),deepcopy(proof)));return transport(request,proof)
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,wire,events,step=19566)as(model,policy,p,spec):
            model.reset()
            for frame in [0,8]:
                capture_to(p,frame);model.update_obs(obs(frame));commands=model.get_action()
                self.assertEqual(events[-3:],['scene_observer','subtask_selector','policy'])
                injected=model.current_injection();actual_prompt=policy.observations[-1]['prompt']
                self.assertEqual(actual_prompt,compose_prompt(INSTRUCTION,injected['context']))
                self.assertIn(public_reply(frame)['action']['summary'],actual_prompt)
                self.assertIn(public_reply(frame)['reasoning'],actual_prompt)
                self.assertEqual(set(policy.observations[-1]),{'state','images','prompt'})
                self.assertEqual(set(policy.observations[-1]['images']),set(CAMERAS))
                self.assertEqual(len(p.executed_history.records),frame//8)
                rng=policy._rng.copy();receipt=acknowledge(model,commands,frame)
                self.assertTrue(np.array_equal(rng,policy._rng));self.assertEqual(receipt['actual_SSR_context'],injected['context'])
            last=requests[-1][1]
            self.assertEqual(last['executed_subtask_history_count'],1)
            self.assertEqual(last['executed_subtask_history'][0]['executed_action_end'],8)
            self.assertEqual(model.metadata['source_training_dropout_probability'],0)

    def test_unsupported_annotation_stops_without_policy_semantic_retry_or_history(self):
        calls=[]
        def wire(request,proof):
            calls.append(proof['call_role'])
            if proof['call_role']=='scene_observer':return transport(request,proof)
            return response(canonical(dict(public_reply(),reasoning=None)))
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,wire)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0))
            with self.assertRaises(ValueError):model.get_action()
            self.assertEqual(calls,['scene_observer','subtask_selector']);self.assertEqual(policy.observations,[])
            self.assertEqual(p.executed_history.records,[])
            with self.assertRaises(Exception):model.get_action()
            self.assertEqual(len(calls),2)

    def test_full_context_mutation_rejected_before_policy_and_reset_clears_authorization(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=SSRSteadyProvider(tmp,selector_kind='deepseek',transport=transport,checkpoint_step=17392)
            p.reset('one');capture_to(p,0);context,lineage=p.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='one',frame=0)
            for field in ['reasoning','summary']:
                bad=deepcopy(context)
                if field=='summary':bad['action']['summary']='Different public action.'
                else:bad[field]='Different public reasoning.'
                with self.assertRaises(Exception):p.validate_injection(bad,lineage,episode_id='one',frame=0)
            p.reset('two');self.assertIsNone(p._ssr_authorization)
            with self.assertRaises(Exception):p.validate_injection(context,lineage,episode_id='one',frame=0)

    def test_SSR_terminal_partial_is_recorded_and_cannot_be_replanned(self):
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));commands=model.get_action()
            receipt=acknowledge(model,commands,0,count=3,reason='native_terminal')
            self.assertEqual(receipt['record']['executed_action_end'],3)
            capture_to(p,8);model.update_obs(obs(8))
            with self.assertRaises(ValueError):model.get_action()
            self.assertEqual(len(policy.observations),1)

    def test_retry_has_identical_full_request_reference_and_no_phantom_history(self):
        seen=[]
        def wire(request,proof):
            if proof['call_role']=='scene_observer':return transport(request,proof)
            seen.append(canonical(request))
            if len(seen)==1:return response('',finish='length')
            return transport(request,proof)
        with tempfile.TemporaryDirectory()as tmp:
            p=SSRSteadyProvider(tmp,selector_kind='deepseek',transport=wire,checkpoint_step=17392)
            p.reset('fixture');capture_to(p,0)
            context,proof=p.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(seen[0],seen[1]);self.assertEqual(proof['attempt_count'],2)
            self.assertEqual(p.executed_history.records,[])

    def test_new_spec_requires_explicit_new_namespace_exact_budget_reference_and_checkpoint(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'weights.json';path.write_text('{"step":19566}')
            spec=build(path,parent_root=tmp,audit_path=Path(tmp)/NAMESPACE/'actions.jsonl')
            validate_spec(spec)
            for key,value in [('maximum_total_images',34),('max_tokens',16384),('reference_sha256','bad'),
                              ('step',None),('ecot_dropout_probability',0.5),('ssr_enabled',False)]:
                bad=deepcopy(spec);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):validate_spec(bad)

    def test_original_weights_identity_rejects_Subtask_only_and_dropout_checkpoint(self):
        from ssr_binding import validate_identity
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'configs').mkdir();path=root/'configs/train.json';path.write_text('{}')
            binding={'weights_mode':'checkpoint','inference_params':'EMA','step':19566,'ecot_variant':'summary_subtask_reasoning',
                'train_ecot':True,'eval_ecot':False,'train_config_path':str(path),
                'checkpoint_metadata':{'identity':{'train_ecot':True,'eval_ecot':False,'max_token_len':512}}}
            spec={'step':19566,'parent_root':tmp};config={'ecot_variant':'summary_subtask_reasoning',
                'ecot_dropout_probability':0.0,'ecot_retained_fields':['subtask','action.summary','reasoning']}
            validate_identity(binding,spec,config)
            for key,value in [('ecot_dropout_probability',0.5),('ecot_variant','subtask')]:
                bad=deepcopy(config);bad[key]=value
                with self.assertRaises(ValueError):validate_identity(binding,spec,bad)

    def test_lifecycle_is_exclusive_through_policy_inference(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));entered=threading.Event();release=threading.Event()
            original=policy.infer
            def blocking(*a,**k):entered.set();self.assertTrue(release.wait(10));return original(*a,**k)
            policy.infer=blocking
            with ThreadPoolExecutor(max_workers=1)as pool:
                future=pool.submit(model.get_action);self.assertTrue(entered.wait(10))
                try:
                    for call in [model.reset,model.current_injection,lambda:model.update_obs(obs(8))]:
                        with self.assertRaises(RPCBusyError):call()
                finally:release.set()
                self.assertEqual(len(future.result(timeout=10)),8)

    def test_full96_causal_request_preserves29_real_ACK_records_and_only_pastMAIN(self):
        requests=[]
        def wire(request,proof):
            if proof['current_frame']==232:requests.append((deepcopy(request),deepcopy(proof)))
            return transport(request,proof)
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,wire)as(model,policy,p,spec):
            model.reset()
            for frame in range(0,233,8):
                first=p.frames[-1].frame+1 if p.frames else 0
                before=policy._rng.copy()
                for source in range(first,frame+1):
                    if is_history_source_frame(source):model.record_history_observation(obs(source))
                self.assertTrue(np.array_equal(before,policy._rng))
                model.update_obs(obs(frame));commands=model.get_action();acknowledge(model,commands,frame)
            self.assertEqual(len(requests),2)
            for request,proof in requests:
                self.assertEqual(len(proof['observations']),96)
                self.assertEqual(proof['executed_subtask_history_count'],29)
                self.assertEqual(proof['executed_subtask_history'][-1]['executed_action_end'],232)
                past=[row for row in proof['observations']if row['frame']<232]
                self.assertEqual(len(past),93);self.assertTrue(all(row['camera']=='cam_high'for row in past))
                self.assertEqual(min(row['frame']for row in past),0)
                self.assertEqual(len([row for row in proof['observations']if row['frame']==232]),3)

    def test_proposal_without_execution_ACK_never_becomes_next_request_history(self):
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));model.get_action()
            self.assertEqual(p.executed_history.records,[])
            capture_to(p,8);model.update_obs(obs(8))
            with self.assertRaises(ValueError):model.get_action()
            self.assertEqual(len(policy.observations),1);self.assertEqual(p.executed_history.records,[])

    def test_training_serializer_mismatch_stops_before_policy(self):
        import ssr_binding
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0))
            with patch('strict_prompt.compose_training_prompt',return_value='wrong input'):
                with self.assertRaises(ValueError):model.get_action()
            self.assertEqual(policy.observations,[]);self.assertEqual(p.executed_history.records,[])

    def test_branch_CLI_requires_actual_SSR_mode_and_does_not_silently_run_Subtask_only(self):
        from model_server import validate_entry_mode
        from prepare_steady_spec import convert
        from test_steady_runtime import legacy_spec
        from subtask_guard import STEADY_NAMESPACE
        with tempfile.TemporaryDirectory()as tmp:
            old=convert(legacy_spec(tmp),Path(tmp)/STEADY_NAMESPACE/'actions.jsonl')
            with self.assertRaises(ValueError):validate_entry_mode(old)
            self.assertTrue(validate_entry_mode(old,True)['steady_subtask_enabled'])
            file=Path(tmp)/'binding.json';file.write_text('{"step":17392}')
            ssr=build(file,parent_root=tmp,audit_path=Path(tmp)/NAMESPACE/'actions.jsonl')
            self.assertTrue(validate_entry_mode(ssr)['ssr_enabled'])

    def test_public_annotation_is_not_API_private_thinking(self):
        def wire(request,proof):
            raw,headers=transport(request,proof)
            reply=json.loads(raw)
            reply['choices'][0]['message']['reasoning_content']='PRIVATE_THINKING_MUST_NOT_BE_IN_POLICY'
            return canonical(reply).encode(),headers
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,wire)as(model,policy,p,spec):
            model.reset();capture_to(p,0);model.update_obs(obs(0));model.get_action()
            self.assertNotIn('PRIVATE_THINKING',policy.observations[0]['prompt'])
            self.assertEqual(model.current_injection()['context']['reasoning'],public_reply()['reasoning'])

if __name__=='__main__':unittest.main()
