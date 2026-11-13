"""Real SDK adapter methods through model RPC/SSR/ACK with explicit CPU fixtures."""
import ast
from copy import deepcopy
import inspect
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from test_ssr import runtime
from test_steady_runtime import obs
from test_executed_history import clock

class SSRNativeLoop(unittest.TestCase):
    def test_actual_adapter_RPC_loop_executes8_then3_and_commits_three_fields_after_each(self):
        from pi05_ssr_policy import deploy
        events=[];rows={}
        class Env:
            n=0;eval_seed=1;current_env_seed_map=[0];step_lim=1100
            obs_manager=types.SimpleNamespace(render_for_capture=lambda:None)
            @property
            def take_action_cnt(self):return [self.n]
            def get_obs(self):return obs(self.n)
            def take_action(self,action):self.n+=1;events.append(('action',self.n))
            def is_episode_end(self):return self.n>=11
        class Recorder:
            def __init__(self,env):pass
            def start(self,*args):pass
            def after_action(self,*args):pass
            def close(self,completed):return {'explicit_CPU_fixture':True,'completed':completed}
        env=Env()
        def snapshot(e,_obs):return {'physics_clock':clock(e.n),'frame':e.n}
        with tempfile.TemporaryDirectory()as tmp,runtime(tmp,step=19566)as(model,policy,p,spec):
            class Client:
                def call(self,func_name,**kwargs):
                    events.append((func_name,env.n))
                    return getattr(model,func_name)(**kwargs)
            with patch.dict(os.environ,{'PI05_ENGINEERING_COMMAND_LIMIT':'0','PI05_SSR_CHECKPOINT_STEP':'19566'}), \
                patch.object(deploy,'MainCameraAction25Hz',Recorder), \
                patch.object(deploy,'read_physics_clock',lambda e:clock(e.n)), \
                patch.object(deploy,'snapshot',snapshot), \
                patch.object(deploy,'verify_static',lambda a,b:self.assertEqual(a,b)), \
                patch.object(deploy,'append',lambda name,row:rows.setdefault(name,[]).append(deepcopy(row))):
                deploy.eval_one_episode(env,Client())
            self.assertEqual([x for x in events if x[0]=='acknowledge_executed_subtask'],
                             [('acknowledge_executed_subtask',8),('acknowledge_executed_subtask',11)])
            self.assertEqual([x['executed_action_end']for x in p.executed_history.records],[8,11])
            self.assertEqual(len(policy.observations),2)
            self.assertEqual(len(rows['execution_clock.jsonl']),11)
            self.assertEqual(len(rows['executed_subtask_ack.jsonl']),2)
            self.assertTrue(all(r['unchanged']for r in rows['freeze.jsonl']))
            self.assertTrue(all(set(r['ack_source']['injection']['context'])=={'subtask','action','reasoning'}for r in rows['executed_subtask_ack.jsonl']))
            self.assertEqual(rows['terminal.jsonl'][0]['executed_commands'],11)

    def test_native_action_loop_is_identical_to_the_preserved_original(self):
        from pi05_policy import deploy as original
        from pi05_ssr_policy import deploy as ssr
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(original.eval_one_episode))),
                         ast.dump(ast.parse(inspect.getsource(ssr.eval_one_episode))))

if __name__=='__main__':unittest.main()
