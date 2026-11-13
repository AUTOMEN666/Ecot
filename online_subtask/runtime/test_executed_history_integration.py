"""Offline transport and native-loop fixtures; not real model/simulation results."""
from copy import deepcopy
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_executed_history import ack,actions,clock,injection,ExecutedSubtaskHistory,CATALOG,canonical
from test_history96 import triplet,state
from multiview_protocol import INSTRUCTION,is_history_source_frame
from multiview_provider import MultiviewHistoryProvider,DEEPSEEK_MODEL

def response(text='{"subtask_id":"U0001"}',finish='stop'):
    return canonical({'model':DEEPSEEK_MODEL,'id':'offline-fixture',
        'choices':[{'finish_reason':finish,'message':{'content':text}}]}).encode(),{}
def capture_to(provider,end):
    for n in range(end+1):
        if is_history_source_frame(n)and (not provider.frames or n>provider.frames[-1].frame):
            provider.capture_history(triplet(n),INSTRUCTION,state14=state(n),episode_id='fixture',frame=n)

class ProviderHistoryIntegration(unittest.TestCase):
    def test_response_only_not_committed_and_missing_ACK_rejects_next_API(self):
        calls=[]
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=lambda r,p:(calls.append(r)or response()))
            provider.reset('fixture');capture_to(provider,0)
            provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(provider.executed_history.records,[])
            capture_to(provider,8)
            with self.assertRaises(ValueError):provider.generate(triplet(8),INSTRUCTION,state14=state(8),episode_id='fixture',frame=8)
            self.assertEqual(len(calls),1)
    def test_identical_empty_length_retry_has_exact_committed_history_once(self):
        calls=[];counts={}
        def transport(request,proof):
            f=proof['current_frame'];counts[f]=counts.get(f,0)+1
            calls.append((deepcopy(request),deepcopy(proof)))
            if f==8 and counts[f]==1:return response('',finish='length')
            return response()
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=transport)
            provider.reset('fixture');capture_to(provider,0)
            _,proof=provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            actual=injection(0,lineage=proof);source=ack(0,actual_injection=actual)
            provider.acknowledge_executed_subtask(source,expected_injection=actual,expected_actions=actions(),expected_clock=clock(0))
            capture_to(provider,8)
            _,last=provider.generate(triplet(8),INSTRUCTION,state14=state(8),episode_id='fixture',frame=8)
            self.assertEqual(counts,{0:1,8:2});self.assertEqual(canonical(calls[1][0]),canonical(calls[2][0]))
            for _,p in calls[1:]:
                self.assertEqual(p['executed_subtask_history_count'],1)
                self.assertEqual(p['executed_subtask_history'],provider.executed_history.records)
                self.assertEqual(p['executed_subtask_ack_source_sha256'],provider.executed_history.ack_source_hashes)
            self.assertEqual(last['attempt_count'],2);self.assertEqual(len(provider.executed_history.records),1)
            self.assertEqual(len((Path(tmp)/'executed_subtask_history.jsonl').read_text().splitlines()),1)
            content=calls[-1][0]['messages'][-1]['content'][0]['text']
            self.assertIn('executed_subtask_history',content)
            self.assertNotIn('execution_clocks',content);self.assertNotIn('chunk_end_reason',content)
            provider.reset('new')
            self.assertEqual(provider.executed_history.for_request('new',0)['executed_subtask_history'],[])
    def test_invalid_semantic_selection_no_retry_or_history(self):
        calls=[]
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=lambda r,p:(calls.append(r)or response('{"subtask_id":"NO_MATCH"}')))
            provider.reset('fixture');capture_to(provider,0)
            with self.assertRaises(ValueError):provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(len(calls),1);self.assertEqual(provider.executed_history.records,[])
    def test_transport_retry_same_request_and_no_proposal_duplicate(self):
        calls=[]
        def transport(request,proof):
            calls.append(deepcopy(request))
            if len(calls)==1:raise TimeoutError('offline transient fixture')
            return response()
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=transport)
            provider.reset('fixture');capture_to(provider,0)
            _,proof=provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(len(calls),2);self.assertEqual(canonical(calls[0]),canonical(calls[1]))
            self.assertEqual(proof['executed_subtask_history'],[]);self.assertEqual(provider.executed_history.records,[])

class NativeLoopACKOrdering(unittest.TestCase):
    def test_real_loop_code_ACKs_after8_and_terminal_partial3_only(self):
        from pi05_policy import deploy
        memory=ExecutedSubtaskHistory(CATALOG);log=[];rows={}
        class Env:
            n=0;eval_seed=1;current_env_seed_map=[1];step_lim=1100
            @property
            def take_action_cnt(self):return [self.n]
            def get_obs(self):return {'fixture':self.n}
            def take_action(self,action):self.n+=1;log.append(('take_action',self.n))
            def is_episode_end(self):return self.n>=11
        env=Env()
        class Recorder:
            def __init__(self,env):pass
            def start(self,*args):pass
            def after_action(self,*args):pass
            def close(self,completed):return {'offline_fixture_only':True,'completed':completed}
        class Client:
            def call(self,func_name,**kwargs):
                if func_name=='reset':memory.reset('fixture');return
                if func_name=='acknowledge_executed_subtask':
                    source=kwargs['obs'];f=source['injection']['execution_frame'];log.append(('ACK',env.n))
                    self_outer.assertEqual(source['execution_clocks'][-1]['execution_frame'],env.n)
                    return memory.commit(source,expected_injection=injection(f),expected_actions=actions(),expected_clock=clock(f))
                if func_name=='record_terminal_observation':return
                raise AssertionError('Unexpected fixture RPC '+func_name)
        self_outer=self
        def snapshot(env,obs):return {'physics_clock':clock(env.n),'fixture_frame':env.n}
        def static(before,after):self.assertEqual(before,after)
        def append(name,row):rows.setdefault(name,[]).append(deepcopy(row))
        with patch.dict(os.environ,{'PI05_ENGINEERING_COMMAND_LIMIT':'0'}),\
                patch.object(deploy,'MainCameraAction25Hz',Recorder),patch.object(deploy,'read_physics_clock',lambda e:clock(e.n)),\
                patch.object(deploy,'snapshot',snapshot),patch.object(deploy,'verify_static',static),patch.object(deploy,'append',append),\
                patch.object(deploy,'capture_history',lambda *a,**k:None),\
                patch.object(deploy,'replan',lambda env,client,n:(actions(),injection(n))):
            deploy.eval_one_episode(env,Client())
        self.assertEqual([x for x in log if x[0]=='ACK'],[('ACK',8),('ACK',11)])
        self.assertEqual([r['executed_action_end']for r in memory.records],[8,11])
        self.assertEqual([r['ack_source']['chunk_end_reason']for r in rows['executed_subtask_ack.jsonl']],['full_chunk','native_terminal'])
        self.assertEqual(len(rows['execution_clock.jsonl']),11)
        self.assertEqual(len([r for r in rows['injected_text.jsonl']if r['event']=='actual_executed_action_text']),11)
        self.assertTrue(all(r['unchanged']and r['ACK_RPC_returned']for r in rows['executed_subtask_ack.jsonl']))

if __name__=='__main__':unittest.main()
