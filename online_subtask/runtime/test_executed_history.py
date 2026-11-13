"""Offline fixtures only; no actual API/model/physics result is asserted."""
from copy import deepcopy
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from executed_history import ExecutedSubtaskHistory,ACK_SCHEMA,canonical,digest
from multiview_protocol import load_catalog,INSTRUCTION,PROTOCOLS

CATALOG=load_catalog()
def clock(frame):
    return {'schema':'robodojo_native_physics_clock_v1','context_physics_step_index':100+frame*10,
        'SDK_physics_step_counter':100+frame*10,'simulation_time_seconds':.4+frame*.04,
        'physics_dt_seconds':.004,'decimation':1,'collect_frequency_hz':25.,'control_interval_physics_steps':10,
        'source':'offline_fixture_only','read_only_no_simulation_step':True}
def actions():
    return [{'left_arm_joint_state':[i/10]*6,'left_ee_joint_state':[.2],
        'right_arm_joint_state':[i/20]*6,'right_ee_joint_state':[.8]}for i in range(8)]
def injection(frame,episode='fixture',selected='U0001',lineage=None):
    text=next(x['text']for x in CATALOG['candidates']if x['id']==selected)
    context={'subtask':text};prompt=INSTRUCTION+' Current subtask: '+text
    return {'episode_id':episode,'execution_frame':frame,'context':context,'context_json':canonical(context),
        'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'lineage':lineage or {'selected_id':selected,'resolution':{'selected_id':selected},'generator_protocol':PROTOCOLS['expanded_catalog_id']},
        'actual_injected':True,'checkpoint_step':5000,'training_composition_byte_equal':True}
def ack(frame,episode='fixture',count=8,reason='full_chunk',actual_injection=None):
    commands=actions()
    return {'schema':ACK_SCHEMA,'injection':actual_injection or injection(frame,episode),
        'execution_clocks':[{'episode_id':episode,'execution_frame':frame+i+1,'before':clock(frame+i),
            'after':clock(frame+i+1),'native_take_action_count':frame+i+1,'action':commands[i]}for i in range(count)],
        'chunk_end_reason':reason}
def commit(memory,frame,**kwargs):
    source=ack(frame,memory.episode_id,**kwargs)
    return memory.commit(source,expected_injection=source['injection'],expected_actions=actions(),expected_clock=clock(frame))

class ExecutedHistoryContracts(unittest.TestCase):
    def setUp(self):self.memory=ExecutedSubtaskHistory(CATALOG);self.memory.reset('fixture')
    def test_empty_then_exact_frame8_and16_prefix(self):
        self.assertEqual(self.memory.for_request('fixture',0)['executed_subtask_history'],[])
        # Merely constructing a successful response/injection cannot create memory.
        injection(0)
        with self.assertRaises(ValueError):self.memory.for_request('fixture',8)
        first=commit(self.memory,0)
        self.assertEqual(first['record'],{'observation_frame':0,'subtask_id':'U0001',
            'subtask':CATALOG['candidates'][0]['text'],'executed_action_start':1,'executed_action_end':8})
        self.assertEqual(self.memory.for_request('fixture',8)['executed_subtask_history_count'],1)
        commit(self.memory,8)
        proof=self.memory.for_request('fixture',16)
        self.assertEqual([r['observation_frame']for r in proof['executed_subtask_history']],[0,8])
        self.assertEqual(proof['latest_executed_subtask_record']['executed_action_end'],16)
        self.assertEqual(proof['executed_subtask_history_sha256'],digest(self.memory.records))
        self.assertEqual(proof['executed_subtask_ack_prefix_sha256'],digest(self.memory.ack_source_hashes))
        # Returned snapshots cannot mutate committed state.
        proof['executed_subtask_history'][0]['subtask']='tamper'
        self.assertEqual(self.memory.records[0]['subtask'],CATALOG['candidates'][0]['text'])
    def test_invalid_ACK_is_atomic_and_no_duplicates(self):
        modifications=[lambda s:s.update(schema='bad'),
            lambda s:s['injection'].update(episode_id='other'),lambda s:s['injection'].update(execution_frame=8),
            lambda s:s['injection']['lineage'].update(selected_id='U0002'),
            lambda s:s['injection']['context'].update(subtask='unlisted text'),
            lambda s:s['injection'].update(prompt='other prompt'),
            lambda s:s['execution_clocks'][2].update(episode_id='other'),
            lambda s:s['execution_clocks'][2].update(execution_frame=99),
            lambda s:s['execution_clocks'][2].update(native_take_action_count=99),
            lambda s:s['execution_clocks'][2]['action'].update(left_ee_joint_state=[.9]),
            lambda s:s['execution_clocks'][2]['after'].update(context_physics_step_index=999),
            lambda s:s['execution_clocks'][2]['after'].update(SDK_physics_step_counter=999),
            lambda s:s['execution_clocks'][2]['after'].update(simulation_time_seconds=99),
            lambda s:s['execution_clocks'].pop(),lambda s:s.update(extra='unapproved')]
        for change in modifications:
            source=ack(0);expected=deepcopy(source['injection']);change(source)
            with self.subTest(source=source):
                with self.assertRaises(ValueError):self.memory.commit(source,expected_injection=expected,expected_actions=actions(),expected_clock=clock(0))
                self.assertEqual(self.memory.records,[]);self.assertEqual(self.memory.ack_source_hashes,[])
        commit(self.memory,0)
        with self.assertRaises(ValueError):commit(self.memory,0)
        self.assertEqual(len(self.memory.records),1)
    def test_real_terminal_partial1100_max137_before_last_replan(self):
        for frame in range(0,1096,8):commit(self.memory,frame)
        proof=self.memory.for_request('fixture',1096)
        self.assertEqual(proof['executed_subtask_history_count'],137)
        self.assertEqual(proof['latest_executed_subtask_record']['executed_action_end'],1096)
        terminal=commit(self.memory,1096,count=4,reason='native_terminal')
        self.assertEqual(terminal['history_count'],138);self.assertEqual(terminal['last_executed_frame'],1100)
        self.assertEqual(terminal['record']['executed_action_start'],1097)
        with self.assertRaises(ValueError):self.memory.for_request('fixture',1104)
    def test_reset_clears_actual_records_and_clock(self):
        commit(self.memory,0);self.memory.reset('new')
        self.assertEqual(self.memory.for_request('new',0)['executed_subtask_history'],[])
        self.assertIsNone(self.memory.last_clock)
        with self.assertRaises(ValueError):self.memory.for_request('fixture',0)
        commit(self.memory,0)
    def test_partial_cannot_be_followed_by_next_replan(self):
        commit(self.memory,0,count=3,reason='native_terminal')
        with self.assertRaises(ValueError):self.memory.for_request('fixture',8)

if __name__=='__main__':unittest.main()
