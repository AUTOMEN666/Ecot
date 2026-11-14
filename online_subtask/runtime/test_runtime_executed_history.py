"""CPU runtime wiring with an explicit fake policy/JAX/transport; no GPU claims."""
from copy import deepcopy
import sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_executed_history import clock,ACK_SCHEMA
from test_history96 import triplet,state
from test_executed_history_integration import response
from multiview_protocol import CAMERAS,INSTRUCTION,PROTOCOLS,CATALOG_SHA
from multiview_provider import MultiviewHistoryProvider

class RuntimeExecutedHistory(unittest.TestCase):
    def test_ACK_wiring_freezes_policy_and_history_only_enters_generator(self):
        support=types.ModuleType('support');support.__path__=[]
        task=types.ModuleType('support.task_contract');task.INSTRUCTION=INSTRUCTION;task.TASK='classify_objects'
        weights=types.ModuleType('support.weights_contract');weights.validate_weights=lambda *a,**k:None
        data=types.ModuleType('subtask_data');data.compose_prompt=lambda instruction,subtask,enabled:instruction+' Current subtask: '+subtask
        class Random:
            key=staticmethod(lambda seed:np.array([seed],dtype=np.int32))
            key_data=staticmethod(lambda key:key)
            split=staticmethod(lambda key:(key,key+1))
            normal=staticmethod(lambda key,shape:np.zeros(shape,np.float32))
        class Policy:
            _rng=np.array([42],dtype=np.int32)
            def __init__(self):self.observations=[]
            def infer(self,obs,noise):
                self.observations.append(deepcopy(obs));self._rng+=1
                return {'actions':np.zeros((50,14),dtype=np.float32)}
        def obs(frame):return {'state':state(frame),'images':{c:np.zeros((480,640,3),np.uint8)for c in CAMERAS},
            'instruction':INSTRUCTION,'frame_index':frame,'physics_clock':clock(frame)}
        binding={'step':10000,'variant':'stack_bowls_subtask_online','parent_root':'FIXTURE_PARENT','training_identity':{'eval_ecot':False,'train_ecot':True},
            'binding_sha256':'fixture-only','export_path':'fixture-only','inference_params':{},'parameter_item_tree_sha256':'fixture-only'}
        with tempfile.TemporaryDirectory()as tmp,patch.dict(sys.modules,{'support':support,'support.task_contract':task,
                'support.weights_contract':weights,'subtask_data':data}):
            import model_server,subtask_client
            policy=Policy();provider=MultiviewHistoryProvider(Path(tmp)/'requests',selector_kind='deepseek',transport=lambda r,p:response())
            binding['parent_root']=str(Path(tmp).resolve())
            audit=Path(tmp)/'stack_bowls_random_seed1_online30_ema10000_v1/audit.jsonl'
            spec={'parent_root':tmp,'variant':'stack_bowls_subtask_online','task':'stack_bowls_random','training_task':'stack_bowls','source_training_eval_ecot':False,'cadence_execution_frames':8,
                'protocol_mode':'expanded_catalog_id','catalog_sha256':CATALOG_SHA,'generator_protocol':PROTOCOLS['expanded_catalog_id'],
                'audit_path':str(audit),'step':10000,'condition':'online_subtask','effective_eval_ecot':True,'noise_seed':42,
                'instruction':INSTRUCTION,'export_binding':'fixture-only','provider':'deepseek',
                'executed_subtask_history_enabled':True,'executed_history_protocol':'ecot.actual_executed_subtask_history.v1',
                'config_id':'stack_bowls_random_mainhist93_tok65536_ema10000_v1','protocol_config_id':'stack_bowls_random_mainhist93_tok65536_ema10000_v1','maximum_past_timepoints':93,'max_tokens':65536,'past_history_cameras':['cam_high'],'maximum_total_images':96,
                'selector_timeout_seconds':600,'selector_retry_budget_seconds':1850,'policy_rpc_timeout_seconds':2100}
            with patch.object(model_server,'read',lambda path:binding),patch.object(model_server,'validate_binding',lambda *a,**k:None),\
                    patch.object(subtask_client,'png_current_rgb',lambda image:triplet(0)[CAMERAS[0]]):
                model=model_server.create_model(spec,policy=policy,jax_module=types.SimpleNamespace(random=Random),provider=provider)
                model.reset();episode=model.episode_id
                model.record_history_observation(obs(0));model.update_obs(obs(0));commands=model.get_action();actual=model.current_injection()
                self.assertEqual(provider.executed_history.records,[])
                source={'schema':ACK_SCHEMA,'injection':actual,'execution_clocks':[
                    {'episode_id':episode,'execution_frame':i+1,'before':clock(i),'after':clock(i+1),
                    'native_take_action_count':i+1,'action':{k:v.tolist()for k,v in action.items()}}for i,action in enumerate(commands)],
                    'chunk_end_reason':'full_chunk'}
                rng=policy._rng.copy();current=model.current
                receipt=model.acknowledge_executed_subtask(source)
                self.assertEqual(receipt['last_executed_frame'],8);self.assertTrue(np.array_equal(rng,policy._rng))
                self.assertIs(model.current,current);self.assertEqual(model.frame_index,0);self.assertEqual(model.predicted_at,0)
                with self.assertRaises(ValueError):model.acknowledge_executed_subtask(source)
                for frame in (2,5,7):model.record_history_observation(obs(frame))
                model.update_obs(obs(8));model.get_action()
                self.assertEqual(model.current_injection()['lineage']['executed_subtask_history_count'],1)
                self.assertEqual(len(policy.observations),2)
                for actual_obs in policy.observations:
                    self.assertEqual(set(actual_obs),{'state','images','prompt'})
                    self.assertNotIn('executed_subtask_history',actual_obs['prompt'])
                    self.assertEqual(actual_obs['prompt'],INSTRUCTION+' Current subtask: '+actual['context']['subtask'])
                model.reset();self.assertNotEqual(model.episode_id,episode)
                self.assertEqual(provider.executed_history.for_request(model.episode_id,0)['executed_subtask_history'],[])
                self.assertIsNone(model.latest_commands);self.assertIsNone(model.latest_injection)

if __name__=='__main__':unittest.main()
