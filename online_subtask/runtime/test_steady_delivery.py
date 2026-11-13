"""Durability, provenance and complete mocked multi-goal Follow ECoT wiring."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from multiview_protocol import INSTRUCTION, canonical, sha, load_catalog, is_history_source_frame
from steady_provider import SteadySubtaskProvider
from subtask_guard import SubtaskConstraintError, STEADY_CONFIG_ID
from test_executed_history import ACK_SCHEMA, clock
from test_executed_history_integration import capture_to, response
from test_history96 import triplet, state
from test_steady_provider import acknowledge
from test_steady_runtime import runtime, obs
from test_subtask_guard import obj, scene


def answer(request, proof):
    return response(canonical(scene(obj()))) if proof['call_role'] == 'scene_observer' else response('{"subtask_id":"U0012"}')


class SteadyDeliveryContracts(unittest.TestCase):
    def test_failed_later_ACK_preserves_the_previously_durable_log_prefix(self):
        def transport(request, proof):
            if proof['call_role'] == 'scene_observer':
                return response(canonical(scene(obj(frame=proof['current_frame']))))
            return response('{"subtask_id":"U0012"}')
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            _, first = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            acknowledge(p, 0, first)
            path = Path(tmp) / 'executed_subtask_history.jsonl'; before = path.read_bytes()
            capture_to(p, 8)
            _, second = p.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            with patch('steady_provider.os.fsync', side_effect=OSError('Offline persistence failure')):
                with self.assertRaises(OSError):
                    acknowledge(p, 8, second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(p.executed_history.records), 1)
            self.assertEqual(list(Path(tmp).glob('*_execution_history.tmp')), [])

    def test_ACK_persistence_failure_cannot_publish_history_or_allow_more_generation(self):
        real_fsync = os.fsync
        for failed_sync in (1, 2):  # File sync before replace, directory sync after replace.
            with self.subTest(failed_sync=failed_sync), tempfile.TemporaryDirectory() as tmp:
                p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=answer)
                p.reset('fixture'); capture_to(p, 0)
                _, proof = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
                syncs = []
                def failing_sync(fd):
                    syncs.append(fd)
                    if len(syncs) == failed_sync:
                        raise OSError('Offline persistence failure')
                    return real_fsync(fd)
                with patch('steady_provider.os.fsync', side_effect=failing_sync):
                    with self.assertRaisesRegex(OSError, 'Offline persistence failure'):
                        acknowledge(p, 0, proof)
                self.assertEqual(p.executed_history.records, [])
                self.assertEqual(p.executed_history.ack_source_hashes, [])
                self.assertIsNone(p.executed_history.last_clock)
                self.assertEqual(p.stopped['stage'], 'execution_ACK_persistence')
                stop = json.loads(next(Path(tmp).glob('*_execution_history_stop.json')).read_text())
                self.assertIsNone(stop['SDK_outcome'])
                self.assertFalse(stop['history_published'])
                path = Path(tmp) / 'executed_subtask_history.jsonl'
                if failed_sync == 1:
                    self.assertFalse(path.exists())
                else:
                    self.assertEqual(json.loads(path.read_text())['record']['executed_action_end'], 8)
                with self.assertRaisesRegex(SubtaskConstraintError, 'already stopped'):
                    p.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)

    def test_invalid_ACK_is_rejected_without_blocking_a_correct_actual_ACK(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=answer)
            p.reset('fixture'); capture_to(p, 0)
            _, proof = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            with self.assertRaises(ValueError):
                p.acknowledge_executed_subtask({}, expected_injection={}, expected_actions=[], expected_clock={})
            self.assertIsNone(p.stopped)
            self.assertEqual(p.executed_history.records, [])
            acknowledge(p, 0, proof)
            row = json.loads((Path(tmp) / 'executed_subtask_history.jsonl').read_text())
            self.assertEqual(p.executed_history.records, [row['record']])
            self.assertEqual(row['record']['executed_action_end'], 8)

    def test_observer_artifacts_identify_the_actual_steady_prompt_and_registry(self):
        captured = []
        def transport(request, proof):
            captured.append((request, proof))
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            request, proof = captured[0]
            registry = json.loads(request['messages'][-1]['content'][0]['text'])['scene_registry']
            self.assertEqual(proof['protocol_config_id'], STEADY_CONFIG_ID)
            self.assertEqual(proof['generator_protocol'], STEADY_CONFIG_ID)
            self.assertEqual(proof['prompt_sha256'], sha(request['messages'][0]['content'].encode()))
            self.assertEqual(proof['scene_registry_sha256'], sha(canonical(registry).encode()))
            self.assertEqual(proof['messages_sha256'], sha(canonical(request['messages']).encode()))

    def test_failed_observation_does_not_mislabel_the_previous_decision_as_current(self):
        def transport(request, proof):
            if proof['current_frame'] == 8:
                return response(canonical(scene(obj(location='red', frame=8, placements=[8]),
                    obj('O2', 'camera', 'D003', 'red', frame=8, placements=[8]))))
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            _, first = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            acknowledge(p, 0, first); capture_to(p, 8)
            with self.assertRaises(SubtaskConstraintError):
                p.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            stop = json.loads((Path(tmp) / 'steady_decision_stops.jsonl').read_text())
            self.assertIsNone(stop['decision'])
            self.assertEqual(stop['last_admitted_decision']['frame'], 0)
            self.assertEqual(stop['observer_receipt']['current_frame'], 8)
            self.assertIsNone(stop['SDK_outcome'])

    def test_stopped_provider_rejects_previously_valid_authorization(self):
        def transport(request, proof):
            if proof['current_frame'] == 8:
                raise ValueError('Offline observer failure')
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            context, first = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            acknowledge(p, 0, first); capture_to(p, 8)
            with self.assertRaisesRegex(ValueError, 'Offline observer failure'):
                p.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            with self.assertRaisesRegex(SubtaskConstraintError, 'Stopped episode'):
                p.validate_injection(context, first, episode_id='fixture', frame=0)

    def test_admitted_current_state_and_clock_do_not_alias_caller_buffers(self):
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, answer) as (model, policy, provider):
            model.reset(); incoming = obs(0)
            incoming['state'] = np.asarray(incoming['state'], dtype=np.float32)
            model.record_history_observation(incoming); model.update_obs(incoming)
            expected_state, expected_clock = model.current['state'].copy(), dict(model.clock)
            incoming['state'][:] = 123
            incoming['images']['cam_high'][:] = 255
            incoming['physics_clock'].clear()
            model.get_action()
            self.assertTrue(np.array_equal(policy.observations[0]['state'], expected_state))
            self.assertEqual(model.clock, expected_clock)
            self.assertEqual(int(policy.observations[0]['images']['cam_high'].max()), 0)

    def test_policy_input_transforms_cannot_mutate_the_stored_actual_observation(self):
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, answer) as (model, policy, provider):
            model.reset(); model.record_history_observation(obs(0)); model.update_obs(obs(0))
            expected = model.current['state'].copy(); infer = policy.infer
            def mutating_infer(observation, noise):
                observation['state'][:] = 77
                observation['images']['cam_high'][:] = 55
                return infer(observation, noise=noise)
            policy.infer = mutating_infer
            model.get_action()
            self.assertTrue(np.array_equal(model.current['state'], expected))
            self.assertEqual(int(model.current['images']['cam_high'].max()), 0)

    def test_three_goal_mock_episode_preserves_IDs_through_grasp_and_switches_only_after_placement(self):
        catalog = load_catalog()
        watch = next(c['descriptor_id'] for c in catalog['candidates'] if 'watch' in c['object_phrase'])
        watch_id = next(c['id'] for c in catalog['candidates'] if c['descriptor_id'] == watch and c['basket_color'] == 'blue')
        ids = ['U0012', 'U0007', watch_id]
        goals = [('pen', 'D004', 'red'), ('camera', 'D003', 'white'), ('watch', watch, 'blue')]
        calls = []
        def transport(request, proof):
            frame = proof['current_frame']; calls.append((frame, proof['call_role']))
            if proof['call_role'] == 'subtask_selector':
                return response(canonical({'subtask_id': ids[frame // 16]}))
            available = sorted({o['frame'] for o in proof['observations']})
            objects = []
            for index, (category, desc, basket) in enumerate(goals):
                start, finish = 16 * index, 16 * (index + 1)
                if frame >= finish:
                    placements = [frame] if frame != finish else [available[-2], frame]
                    objects.append(obj('O' + str(index + 1), category, desc, basket, frame=frame, placements=placements))
                elif frame == start + 8:
                    objects.append(obj('O' + str(index + 1), category, desc, 'held', frame=frame, released=False))
                else:
                    objects.append(obj('O' + str(index + 1), category, desc, frame=frame))
            return response(canonical(scene(*objects)))
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, transport) as (model, policy, p):
            model.reset(); actual_ids = []
            for frame in range(0, 49, 8):
                first_capture = p.frames[-1].frame + 1 if p.frames else 0
                for capture in range(first_capture, frame + 1):
                    if is_history_source_frame(capture):
                        model.record_history_observation(obs(capture))
                model.update_obs(obs(frame))
                if frame == 48:
                    with self.assertRaisesRegex(SubtaskConstraintError, 'No unambiguous legal'):
                        model.get_action()
                    break
                commands = model.get_action(); injection = model.current_injection()
                actual_ids.append(injection['lineage']['selected_id'])
                ack = dict(schema=ACK_SCHEMA, injection=injection, chunk_end_reason='full_chunk', execution_clocks=[
                    dict(episode_id=model.episode_id, execution_frame=frame + i + 1,
                         before=clock(frame + i), after=clock(frame + i + 1),
                         native_take_action_count=frame + i + 1,
                         action={k: v.tolist() for k, v in action.items()}) for i, action in enumerate(commands)])
                model.acknowledge_executed_subtask(ack)
            self.assertEqual(actual_ids, [ident for ident in ids for _ in range(2)])
            self.assertEqual(len(policy.observations), 6)
            self.assertEqual([r['subtask_id'] for r in p.executed_history.records], actual_ids)
            self.assertEqual(p.guard.completed, {'O1', 'O2', 'O3'})
            self.assertEqual(p.guard.category_to_basket, {'pen': 'red', 'camera': 'white', 'watch': 'blue'})
            self.assertEqual(sum(role == 'scene_observer' for _, role in calls), 7)
            self.assertEqual(sum(role == 'subtask_selector' for _, role in calls), 6)
            for observation in policy.observations:
                self.assertEqual(set(observation), {'images', 'state', 'prompt'})
                self.assertIn(observation['prompt'], [INSTRUCTION + ' Current subtask: ' + next(c['text'] for c in catalog['candidates'] if c['id'] == ident) for ident in ids])


if __name__ == '__main__':
    unittest.main()
