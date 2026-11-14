"""Synthetic artifact fixtures ONLY. Never evidence of actual native acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from acceptance import audit_case, verify_engineering, SCHEMA
from export_binding import sha
from layout_manifest import create_manifest
from multiview_protocol import CATALOG_SHA
from task_contract import TASK, PROTOCOL_CONFIG_ID
from test_stack_bowls_contracts import fixture_sdk, frozen, tool
from test_executed_history import clock, injection, ack, actions, ExecutedSubtaskHistory, CATALOG


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data)+'\n'); return path


def fixture(root):
    sdk = root/'sdk'; fixture_sdk(sdk)
    manifest = root/'manifest.json'; create_manifest(sdk, manifest); digest = sha(manifest)
    output = root/'case'; output.mkdir()
    plan = write(output/'sim_launch.json', dict(schema='ecot.stack_bowls_random.native_case_plan.v1',
        task=TASK, phase='engineering', layout_id=0, manifest_sha256=digest))
    native_exit = write(output/'NATIVE_EXIT.json', dict(schema='ecot.stack_bowls.actual_native_exit.v1',
        exit_code=0, timed_out=False, phase='engineering', layout_id=0,
        plan_path=str(plan), plan_sha256=sha(plan)))
    p = injection(0); p['lineage'].update(catalog_sha256=CATALOG_SHA,
        protocol_config_id=PROTOCOL_CONFIG_ID, current_frame=0, history_frames=[])
    source = ack(0, count=1, reason='native_terminal', actual_injection=p)
    memory = ExecutedSubtaskHistory(CATALOG); memory.reset('fixture')
    receipt = memory.commit(source, expected_injection=p, expected_actions=actions(), expected_clock=clock(0))
    write(output/'execution_clock.jsonl', source['execution_clocks'][0])
    with (output/'injected_text.jsonl').open('w') as out:
        out.write(json.dumps(dict(event='actual_injection', **p))+'\n')
        out.write(json.dumps(dict(event='actual_executed_action_text', execution_action=0,
            observed_outcome_frame=1, injection_execution_frame=0, episode_id='fixture', context=p['context'],
            prompt=p['prompt'], prompt_sha256=p['prompt_sha256']))+'\n')
    freeze = dict(before=frozen(), after=frozen())
    write(output/'freeze.jsonl', freeze); write(output/'history_capture_freeze.jsonl', freeze)
    write(output/'executed_subtask_ack.jsonl', dict(**freeze, ACK_RPC_returned=True, ack_source=source, receipt=receipt))
    policy = root/'policy.jsonl'
    commands14 = [a['left_arm_joint_state']+a['left_ee_joint_state']+a['right_arm_joint_state']+a['right_ee_joint_state'] for a in actions()]
    # These fixture arrays deliberately use exact float32-representable values.
    import numpy as np
    from pi05_policy.native_model import to_commands
    from pi05_policy.freeze import plain
    commands14 = np.zeros((50,14), np.float32)
    commands8, executed8 = to_commands(commands14)
    source['execution_clocks'][0]['action'] = plain(commands8[0])
    memory.reset('fixture'); receipt = memory.commit(source, expected_injection=p, expected_actions=plain(commands8), expected_clock=clock(0))
    write(output/'execution_clock.jsonl', source['execution_clocks'][0])
    write(output/'executed_subtask_ack.jsonl', dict(**freeze, ACK_RPC_returned=True, ack_source=source, receipt=receipt))
    write(policy, dict(event='inference', episode_id='fixture', executed_action_step=0, context=p['context'],
        prompt=p['prompt'], binding=dict(step=10000, effective_eval_ecot=True, source_training_eval_ecot=False),
        raw_predicted_absolute_joint14=commands14.tolist(),
        executed_first8_absolute_joint14=executed8.tolist(), physics_clock=clock(0)))
    original = root/'original-sdk-result'; original.mkdir()
    result = write(original/'_result.json', dict(eval_time=1, success_rate=0, score=0,
        details={'0':dict(layout_id=0, success=False, score=0.)}))
    video = root/'fixture-video'; video.write_bytes(b'explicit synthetic fixture; NOT an MP4')
    encoded = root/'fixture-overlay'; encoded.write_bytes(b'not an actual encoded movie')
    frames = write(root/'frames.jsonl', {'fixture_only':True})
    decoded = write(root/'decode.json', dict(fully_decoded=True, encoded_video_fully_decoded=True,
        frame_count=2, fps=25, actual_text_timeline_verified=True,
        injection_log_sha256=sha(output/'injected_text.jsonl'), source_video_sha256=sha(video),
        output_video=str(encoded), output_video_sha256=sha(encoded)))
    write(output/'terminal.jsonl', dict(executed_commands=1, actual_environment_terminal=True,
        engineering_stop=False, engineering_command_limit=0, native_step_limit=800, scene_seed=1,
        layout_id=0, engineering_probe=True, episode_id='fixture', main_camera_action25hz=dict(
            status='complete', frame_count=2, video_path=str(video), frames_path=str(frames), frames_sha256=sha(frames),
            original_SDK_save_dir=str(original))))
    return dict(output=output, policy_audit=policy, layout_manifest=manifest, manifest_sha256=digest,
                native_exit=native_exit, phase='engineering', layout_id=0, result_path=result, decode_receipt=decoded)


class CaseArtifactContracts(unittest.TestCase):
    def test_failed_SDK_goal_is_valid_engineering_integration_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp)); result = audit_case(**args)
            self.assertTrue(result['passed']); self.assertFalse(result['SDK_success'])
            path = write(Path(tmp)/'engineering.json', result)
            self.assertEqual(verify_engineering(path, args['manifest_sha256'])['phase'], 'engineering')
            args['result_path'].write_text('{}')
            with self.assertRaises(ValueError): verify_engineering(path, args['manifest_sha256'])

    def test_interruptions_missing_actions_and_forged_ACK_are_rejected(self):
        for failure in ('exit', 'clock', 'ack', 'text', 'video'):
            with tempfile.TemporaryDirectory() as tmp:
                args = fixture(Path(tmp)); out = args['output']
                if failure == 'exit':
                    d = json.loads(args['native_exit'].read_text()); d['exit_code'] = 1; write(args['native_exit'], d)
                elif failure == 'clock': (out/'execution_clock.jsonl').write_text('')
                elif failure == 'ack':
                    p = out/'executed_subtask_ack.jsonl'; d = json.loads(p.read_text())
                    d['ack_source']['execution_clocks'][0]['action']['left_ee_joint_state'] = [.7]; write(p, d)
                elif failure == 'text':
                    p = out/'injected_text.jsonl'; rows = [json.loads(s) for s in p.read_text().splitlines()]
                    rows[-1]['prompt'] = 'unexecuted text'; p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                else:
                    d = json.loads(args['decode_receipt'].read_text()); d['encoded_video_fully_decoded'] = False
                    write(args['decode_receipt'], d)
                with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                    audit_case(**args)

    def test_formal_summary_requires30_unique_cases_no_engineering(self):
        module = tool('summarize_random30')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); evidence = root/'evidence'; evidence.write_text('fixture only')
            common = dict(schema=SCHEMA, passed=True, phase='formal', task=TASK, scene_seed=1,
                step=10000, protocol_config_id=PROTOCOL_CONFIG_ID, manifest_sha256='fixture only',
                SDK_success=False, SDK_score=0., source_sha256={str(evidence):sha(evidence)})
            paths = [write(root/f'{i}.json', dict(common, layout_id=i, episode_id=f'fixture-{i}')) for i in range(30)]
            self.assertEqual(module.summarize(paths)['completed_episodes'], 30)
            with self.assertRaises(ValueError): module.summarize(paths[:29])
            with self.assertRaises(ValueError): module.summarize(paths[:29]+[paths[0]])
            d = json.loads(paths[0].read_text()); d['phase'] = 'engineering'; write(paths[0], d)
            with self.assertRaises(ValueError): module.summarize(paths)


if __name__ == '__main__': unittest.main()
