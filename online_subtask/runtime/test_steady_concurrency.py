"""Controlled worker-thread overlap, with no network, GPU or simulator."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
import tempfile
import unittest

from multiview_protocol import INSTRUCTION, canonical
from rpc_lifecycle import RPCBusyError
from steady_provider import SteadySubtaskProvider
from test_executed_history_integration import capture_to, response
from test_history96 import triplet, state
from test_steady_runtime import runtime, obs
from test_subtask_guard import obj, scene


def answer(request, proof):
    if proof['call_role'] == 'scene_observer':
        return response(canonical(scene(obj())))
    return response('{"subtask_id":"U0012"}')


class SteadyConcurrency(unittest.TestCase):
    def test_model_rejects_overlapping_RPCS_during_API_wait_without_poisoning_episode(self):
        entered, release = Event(), Event()
        calls = []
        def transport(request, proof):
            calls.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Offline thread fixture timed out')
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, transport) as (model, policy, provider):
            model.reset(); episode = model.episode_id
            model.record_history_observation(obs(0)); model.update_obs(obs(0))
            current, rng = model.current, policy._rng.copy()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(model.get_action)
                try:
                    self.assertTrue(entered.wait(5))
                    operations = [model.reset, lambda: model.update_obs(obs(8)),
                        lambda: model.record_history_observation(obs(2)), model.get_action,
                        model.current_injection, lambda: model.acknowledge_executed_subtask({}),
                        lambda: model.record_terminal_observation({})]
                    for operation in operations:
                        with self.subTest(operation=operation), self.assertRaises(RPCBusyError):
                            operation()
                    self.assertEqual(model.episode_id, episode)
                    self.assertIs(model.current, current)
                    self.assertEqual(model.frame_index, 0)
                    self.assertEqual(policy._rng.tolist(), rng.tolist())
                    self.assertIsNone(provider.stopped)
                    self.assertEqual(calls, ['scene_observer'])
                finally:
                    release.set()
                self.assertEqual(len(pending.result(5)), 8)
            self.assertEqual(len(policy.observations), 1)
            self.assertEqual(model.current_injection()['episode_id'], episode)
            self.assertEqual(model.predicted_at, 0)
            model.reset()
            self.assertNotEqual(model.episode_id, episode)
            self.assertIsNone(provider.stopped)
            self.assertEqual(provider.executed_history.records, [])

    def test_model_also_protects_inference_after_provider_has_returned(self):
        entered, release = Event(), Event()
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, answer) as (model, policy, provider):
            model.reset(); episode = model.episode_id
            model.record_history_observation(obs(0)); model.update_obs(obs(0))
            infer = policy.infer
            def blocked_infer(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Offline thread fixture timed out')
                return infer(*args, **kwargs)
            policy.infer = blocked_infer
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(model.get_action)
                try:
                    self.assertTrue(entered.wait(5))
                    for operation in (model.reset, lambda: model.update_obs(obs(8)), model.get_action):
                        with self.assertRaises(RPCBusyError):
                            operation()
                    self.assertEqual(provider.last_frame, 0)
                    self.assertEqual(model.episode_id, episode)
                finally:
                    release.set()
                self.assertEqual(len(pending.result(5)), 8)
            actual = model.current_injection()
            self.assertEqual((actual['episode_id'], actual['execution_frame']), (episode, 0))
            self.assertEqual(len(policy.observations), 1)

    def test_direct_provider_overlap_is_rejected_before_stop_latch_or_API_attempt(self):
        entered, release = Event(), Event()
        calls = []
        def transport(request, proof):
            calls.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Offline thread fixture timed out')
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            provider = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            provider.reset('fixture'); capture_to(provider, 0)
            def generate():
                return provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(generate)
                try:
                    self.assertTrue(entered.wait(5))
                    before = deepcopy(provider.guard.snapshot())
                    operations = [lambda: provider.reset('new-episode'), generate,
                        lambda: provider.capture_history(triplet(2), INSTRUCTION,
                            state14=state(2), episode_id='fixture', frame=2),
                        lambda: provider.acknowledge_executed_subtask({}, expected_injection={},
                            expected_actions=[], expected_clock={}),
                        lambda: provider.validate_injection({}, {}, episode_id='fixture', frame=0)]
                    for operation in operations:
                        with self.assertRaises(RPCBusyError):
                            operation()
                    self.assertEqual(provider.guard.snapshot(), before)
                    self.assertEqual(provider.episode_id, 'fixture')
                    self.assertIsNone(provider.stopped)
                    self.assertEqual(calls, ['scene_observer'])
                finally:
                    release.set()
                context, proof = pending.result(5)
            provider.validate_injection(context, proof, episode_id='fixture', frame=0)
            self.assertEqual(calls, ['scene_observer', 'subtask_selector'])
            self.assertEqual(provider.executed_history.records, [])
            provider.reset('new-episode')
            self.assertIsNone(provider.stopped)

    def test_model_gate_is_released_when_inference_raises(self):
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, answer) as (model, policy, provider):
            model.reset(); episode = model.episode_id
            model.record_history_observation(obs(0)); model.update_obs(obs(0))
            def fail(*args, **kwargs):
                raise RuntimeError('Offline inference failure')
            policy.infer = fail
            with self.assertRaisesRegex(RuntimeError, 'Offline inference failure'):
                model.get_action()
            model.reset()
            self.assertNotEqual(model.episode_id, episode)
            self.assertIsNone(provider.stopped)
            self.assertEqual(provider.guard.objects, {})

    def test_delayed_old_episode_request_cannot_latch_the_new_episode(self):
        calls = []
        def transport(request, proof):
            calls.append(proof['call_role'])
            return answer(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            provider = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            provider.reset('old-episode'); provider.reset('fixture'); capture_to(provider, 0)
            with self.assertRaisesRegex(ValueError, 'same-episode'):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='old-episode', frame=0)
            self.assertIsNone(provider.stopped)
            self.assertEqual(provider.guard.episode_id, 'fixture')
            self.assertEqual(calls, [])
            _, proof = provider.generate(triplet(0), INSTRUCTION,
                                        state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(proof['selected_id'], 'U0012')


if __name__ == '__main__':
    unittest.main()
