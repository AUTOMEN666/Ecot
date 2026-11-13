"""Offline contract fixtures only: no API, actual model or simulator result."""
import json,struct,sys,tempfile,unittest,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multiview_protocol import (CAMERAS,INSTRUCTION,history_source_frame,is_history_source_frame,uniform_past,
    make_history_point,build_request,load_catalog,resolve,canonical)
from multiview_provider import MultiviewHistoryProvider,DEEPSEEK_MODEL
from test_executed_history import ack,actions,clock,injection

def png(frame,camera):
    def chunk(tag,raw):return struct.pack('>I',len(raw))+tag+raw+struct.pack('>I',zlib.crc32(tag+raw)&0xffffffff)
    row=bytes((frame%256,CAMERAS.index(camera)*80,20))*640
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',640,480,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+row)*480))+chunk(b'IEND',b'')
def triplet(frame):return {camera:png(frame,camera)for camera in CAMERAS}
def state(frame):return [frame/100+j/1000 for j in range(14)]

class MainHistory96Contracts(unittest.TestCase):
    def test_source_grid_is_exact_rational_not8frame_cache(self):
        self.assertEqual([history_source_frame(i)for i in range(10)],[0,2,5,7,10,12,15,17,20,22])
        self.assertEqual([n for n in range(25)if is_history_source_frame(n)],[0,2,5,7,10,12,15,17,20,22])
        for n in [1,3,4,6,8,9,True,-1]:self.assertFalse(is_history_source_frame(n))

    def test_uniform_all_past93_preserves_endpoints(self):
        self.assertEqual(uniform_past(list(range(93))),list(range(93)))
        out=uniform_past(list(range(101)))
        self.assertEqual(len(out),93);self.assertEqual(len(set(out)),93);self.assertEqual(out[0],0);self.assertEqual(out[-1],100)
        self.assertLess(out[1],70) # Reject a93-most-recent sliding window.

    def test_actual_capture_separate_from_replan_full256_fixture(self):
        calls=[]
        def transport(request,proof):
            calls.append((request,proof))
            body={'model':DEEPSEEK_MODEL,'id':'fixture-only-'+str(proof['current_frame']),
                'choices':[{'finish_reason':'stop','message':{'content':'{"subtask_id":"U0001"}'}}]}
            return canonical(body).encode(),{}
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=transport);provider.reset('fixture-episode')
            for n in range(257):
                if is_history_source_frame(n):provider.capture_history(triplet(n),INSTRUCTION,state14=state(n),episode_id='fixture-episode',frame=n)
                if n%8==0:
                    context,proof=provider.generate(triplet(n),INSTRUCTION,state14=state(n),episode_id='fixture-episode',frame=n)
                    self.assertEqual(context,{'subtask':load_catalog()['candidates'][0]['text']})
                    self.assertLessEqual(proof['total_image_count'],96);self.assertEqual(proof['current_image_count'],3)
                    self.assertEqual(proof['past_image_count'],len(proof['history_frames']))
                    self.assertTrue(all(f<n for f in proof['history_frames']))
                    self.assertEqual(proof['observations'][-3:],[x for x in proof['observations']if x['is_current']])
                    self.assertEqual([x['camera']for x in proof['observations'][-3:]],list(CAMERAS))
                    self.assertTrue(all(x['state14']==state(x['frame'])for x in proof['observations']))
                    self.assertEqual([x['camera']for x in proof['observations'][:-3]],['cam_high']*len(proof['history_frames']))
                    self.assertEqual(proof['prior_generated_text_sent'],n>0)
                    self.assertEqual(proof['executed_subtask_history_count'],n//8)
                    self.assertEqual([r['observation_frame']for r in proof['executed_subtask_history']],list(range(0,n,8)))
                    actual=injection(n,'fixture-episode',lineage=proof)
                    source=ack(n,'fixture-episode',actual_injection=actual)
                    provider.acknowledge_executed_subtask(source,expected_injection=actual,expected_actions=actions(),expected_clock=clock(n))
                    self.assertEqual(request_config:=calls[-1][0]['thinking'],{'type':'enabled'})
                    self.assertEqual(calls[-1][0]['reasoning_effort'],'high');self.assertEqual(calls[-1][0]['max_tokens'],65536)
            self.assertEqual([p['current_frame']for _,p in calls],list(range(0,257,8)))
            self.assertEqual(len(provider.frames),103)
            self.assertEqual([p.frame for p in provider.frames],[history_source_frame(i)for i in range(103)])
            last=calls[-1][1];self.assertEqual(last['total_image_count'],96);self.assertEqual(last['history_frames'][0],0)
            self.assertEqual(last['history_frames'][-1],255);self.assertEqual(len(last['history_frames']),93)
            # Point80 is captured on the10Hz grid but never duplicated into API80 past.
            at80=next(p for _,p in calls if p['current_frame']==80);self.assertNotIn(80,at80['history_frames'])
            self.assertEqual(len([p for _,p in calls if p['current_frame']<256]),32)

    def test_missing_or_replan_only_capture_rejected(self):
        def never(*args):raise AssertionError('Transport must not be called')
        with tempfile.TemporaryDirectory()as tmp:
            provider=MultiviewHistoryProvider(tmp,selector_kind='deepseek',transport=never);provider.reset('fixture')
            with self.assertRaises(ValueError):provider.generate(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            provider.capture_history(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            with self.assertRaises(ValueError):provider.capture_history(triplet(8),INSTRUCTION,state14=state(8),episode_id='fixture',frame=8)
            with self.assertRaises(ValueError):provider.capture_history(triplet(0),INSTRUCTION,state14=state(0),episode_id='fixture',frame=0)
            with self.assertRaises(ValueError):provider.capture_history(triplet(2),INSTRUCTION,state14=state(2),episode_id='wrong',frame=2)

    def test_all288_and_no_fallback_or_extra_fields(self):
        for c in load_catalog()['candidates']:self.assertEqual(resolve(canonical({'subtask_id':c['id']}))[0]['subtask'],c['text'])
        for raw in ['null','{"subtask_id":null}','{"subtask_id":"NO_MATCH"}','{"subtask_id":"U0000"}',
                    '{"subtask_id":"U0001","reasoning":"x"}','{"subtask_id":"U0001","subtask_id":"U0002"}']:
            with self.assertRaises(ValueError):resolve(raw)

if __name__=='__main__':unittest.main()
