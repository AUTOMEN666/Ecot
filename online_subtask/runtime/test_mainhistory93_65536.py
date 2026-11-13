"""CPU-only request/timeout fixtures; no actual API, GPU or episode results."""
from dataclasses import replace
import base64,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import multiview_protocol as protocol
from multiview_protocol import CAMERAS,INSTRUCTION,build_request,make_history_point,history_source_frame,canonical
from multiview_provider import MultiviewHistoryProvider
from test_history96 import triplet,state
from test_executed_history import ExecutedSubtaskHistory,CATALOG,commit
from test_executed_history_integration import response,capture_to
from rpc_contract import patch_rpc_text,validate_rpc_text,TIMEOUT_SECONDS


def source_points(frame,episode='fixture'):
    # Repeated static-camera RGB/state are explicit offline fixtures, not real observations.
    original=make_history_point(episode,0,state(0),triplet(0));points=[];index=0
    while history_source_frame(index)<=frame:
        f=history_source_frame(index)
        points.append(replace(original,frame=f,views=tuple(replace(v,frame=f,time_seconds=f/25)for v in original.views)))
        index+=1
    return points

def memory_at(frame):
    memory=ExecutedSubtaskHistory(CATALOG);memory.reset('fixture')
    for f in range(0,frame,8):commit(memory,f)
    return memory

class MainPayloadContracts(unittest.TestCase):
    def test_actual_body_sends_current3_but_no_past_wrist_images(self):
        frame=256;points=[];k=0
        while history_source_frame(k)<=frame:
            f=history_source_frame(k);points.append(make_history_point('fixture',f,state(f),triplet(f)));k+=1
        memory=memory_at(frame)
        request,proof,_=build_request(episode_id='fixture',frame=frame,state14=state(frame),pngs=triplet(frame),
            instruction=INSTRUCTION,past_frames=points,executed_subtask_history=memory.records)
        content=request['messages'][-1]['content'];first=json.loads(content[0]['text'])
        self.assertEqual(first['past_cameras'],['cam_high']);self.assertEqual(first['current_cameras'],list(CAMERAS))
        self.assertEqual((proof['past_image_count'],proof['current_image_count'],proof['total_image_count']),(93,3,96))
        expected=[points[i*(len(points)-1)//92].frame for i in range(93)]
        self.assertEqual(proof['history_frames'],expected);self.assertEqual(expected[0],0);self.assertEqual(expected[-1],255)
        payload_images=[base64.b64decode(block['image_url']['url'].split(',',1)[1])for block in content if block['type']=='image_url']
        self.assertEqual(len(payload_images),96)
        expected_images=[triplet(f)['cam_high']for f in proof['history_frames']]+[triplet(frame)[c]for c in CAMERAS]
        self.assertEqual(payload_images,expected_images)
        for point in points:
            for camera in CAMERAS[1:]:self.assertNotIn(triplet(point.frame)[camera],payload_images[:-3])
        for obs,png in zip(proof['observations'],payload_images):
            self.assertEqual(obs['image_sha256'],hashlib.sha256(png).hexdigest())
            self.assertEqual(obs['state14'],state(obs['frame']))
            self.assertEqual(obs['state_sha256'],hashlib.sha256(canonical(state(obs['frame'])).encode()).hexdigest())
            if not obs['is_current']:self.assertEqual(obs['camera'],'cam_high')
        self.assertEqual(request['max_tokens'],65536);self.assertEqual(proof['protocol_config_id'],'mainhist93_tok65536_v1')
    def test_uniform_source_capacity_boundaries92_93_94(self):
        for count in(92,93,94):
            with self.subTest(strictpast_source_count=count):
                points=list(range(count));selected=protocol.uniform_past(points)
                expected=points if count<=93 else[points[i*(count-1)//92]for i in range(93)]
                self.assertEqual(selected,expected);self.assertEqual(len(selected),min(count,93))
                self.assertEqual((selected[0],selected[-1]),(0,count-1))
    def test_actual_replan_payload_before_at_and_after_capacity(self):
        # Actual8-frame replans have90/93/96 strictpast10Hz source points here.
        for frame,source_count in((224,90),(232,93),(240,96)):
            with self.subTest(frame=frame):
                points=source_points(frame);strict=[p for p in points if p.frame<frame]
                self.assertEqual(len(strict),source_count)
                request,proof,_=build_request(episode_id='fixture',frame=frame,state14=state(0),pngs=triplet(0),
                    instruction=INSTRUCTION,past_frames=points,executed_subtask_history=memory_at(frame).records)
                expected=strict if len(strict)<=93 else[strict[i*(len(strict)-1)//92]for i in range(93)]
                self.assertEqual(proof['history_frames'],[p.frame for p in expected])
                self.assertEqual(proof['past_image_count'],min(source_count,93))
                self.assertEqual(proof['total_image_count'],min(source_count,93)+3)
                self.assertEqual([o['camera']for o in proof['observations'][-3:]],list(CAMERAS))
                self.assertTrue(all(o['camera']=='cam_high'and o['frame']<frame for o in proof['observations'][:-3]))
    def test_last_replan_keeps137_full_executed_records_with_only96_images(self):
        frame=1096;memory=memory_at(frame)
        request,proof,_=build_request(episode_id='fixture',frame=frame,state14=state(0),pngs=triplet(0),instruction=INSTRUCTION,
            past_frames=source_points(frame),executed_subtask_history=memory.records)
        sent=json.loads(request['messages'][-1]['content'][0]['text'])
        self.assertEqual(sent['executed_subtask_history'],memory.records)
        self.assertEqual(proof['executed_subtask_history_count'],137)
        self.assertEqual(sent['latest_executed_subtask_record']['executed_action_end'],1096)
        self.assertEqual(proof['total_image_count'],96)
        self.assertTrue(all(f<1096 for f in proof['history_frames']))
        strict=[p for p in source_points(frame)if p.frame<frame]
        self.assertEqual(proof['history_frames'],[strict[i*(len(strict)-1)//92].frame for i in range(93)])
        self.assertEqual(proof['history_frames'][0],0);self.assertEqual(proof['history_frames'][-1],1095)
    def test_prompt_diff_is_only_visual_context_description(self):
        old=Path(__file__).resolve().parent.parent/'tests/fixtures/mainhistory31_selector_prompt.md'
        expected=old.read_text().replace('up to31 uniformly sampled strictly past MAIN-camera RGB views',
            'up to93 uniformly sampled strictly past MAIN-camera RGB views').replace('at most34 images total','at most96 images total')
        actual=(Path(__file__).parent/'selector_prompt.md').read_text()
        self.assertEqual(actual,expected);self.assertIn('Return ONLY JSON {"subtask_id":"one listed ID"}.',actual)

class BoundedRequestContracts(unittest.TestCase):
    def run_empty_attempts(self,attempt_cost):
        calls=[];now=[0.]
        def transport(request,proof):
            calls.append((canonical(request),dict(proof)));now[0]+=attempt_cost
            return response('',finish='length')
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=transport);provider.reset('fixture');capture_to(provider,0)
            with patch('multiview_provider.time.monotonic',side_effect=lambda:now[0]):
                with self.assertRaises(ValueError):provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            failures=[json.loads(p.read_text())for p in Path(tmp).glob('*_failure.json')]
            self.assertEqual(provider.executed_history.records,[]);self.assertEqual(provider.last_frame,-8)
        return calls,failures
    def test_three_identical600_second_attempts_fit1850_budget(self):
        calls,failures=self.run_empty_attempts(600)
        self.assertEqual(len(calls),3);self.assertEqual(len(failures),3);self.assertEqual(len({body for body,_ in calls}),1)
        for body,proof in calls:
            self.assertEqual(json.loads(body)['max_tokens'],65536)
            self.assertEqual(proof['maximum_identical_request_attempts'],3)
            self.assertEqual(proof['RPC_retry_budget_seconds'],1850)
            self.assertEqual(proof['unchanged_transport_timeout_seconds'],600)
            self.assertEqual(proof['policy_rpc_timeout_seconds'],2100)
        self.assertFalse(max(failures,key=lambda r:r['attempt_index'])['retry_scheduled'])
    def test_budget_blocks_a_third_attempt_without_changing_request(self):
        calls,failures=self.run_empty_attempts(650)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0][0],calls[1][0])
        last=max(failures,key=lambda r:r['attempt_index'])
        self.assertTrue(last['retry_budget_exhausted']);self.assertFalse(last['retry_scheduled'])
    def test_first_HTTP_is_not_started_after_preparation_exhausts_budget(self):
        calls=[]
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=lambda r,p:calls.append(r))
            provider.reset('fixture');capture_to(provider,0)
            with patch('multiview_provider.time.monotonic',side_effect=[0.,1300.]):
                with self.assertRaisesRegex(TimeoutError,'preparation exhausted'):
                    provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            self.assertEqual(calls,[]);self.assertEqual(provider.last_frame,-8)
            admission=json.loads(next(Path(tmp).glob('*_retry_budget_exhausted.json')).read_text())
            self.assertEqual(admission['attempt_request_ids'],[]);self.assertEqual(admission['next_attempt_not_started'],1)
            self.assertFalse(admission['policy_called']);self.assertFalse(admission['actual_injected'])
    def test_exact_HTTP600_is_enforced(self):
        for timeout in (300,480,601,True):
            with tempfile.TemporaryDirectory()as tmp:
                with self.assertRaises(ValueError):MultiviewHistoryProvider(tmp,selector_kind='deepseek',timeout_seconds=timeout)
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek')
            self.assertEqual(provider.timeout,600)
    def test_private_RPC_patch_is2100_and_preserves_other_constructor_arguments(self):
        original="""class Eval:
 def initialize(self):
  self.client=WsModelClient(self.model,host=self.host,ws_ping_timeout_s=self.deploy_cfg.get('ws_ping_timeout_s',20.0),port=self.port)
"""
        patched,proof=patch_rpc_text(original)
        self.assertEqual(TIMEOUT_SECONDS,2100)
        self.assertEqual(proof['adapter'],'private_sdk_rpc2100_v1')
        self.assertEqual(validate_rpc_text(patched)['request_timeout_seconds'],2100)
        self.assertEqual(validate_rpc_text(patched)['ws_ping_timeout_seconds'],2100)
        self.assertIn('host=self.host',patched);self.assertIn('port=self.port',patched)
        self.assertTrue(proof['only_timeout_keywords_changed']);self.assertFalse(proof['shared_SDK_modified'])

if __name__=='__main__':unittest.main()
