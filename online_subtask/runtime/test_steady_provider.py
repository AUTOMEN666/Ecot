"""Actual provider/ACK wiring with explicit offline fake HTTP responses."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from executed_history import digest
from multiview_protocol import INSTRUCTION, canonical
from steady_provider import SteadySubtaskProvider
from subtask_guard import SubtaskConstraintError, STEADY_CONFIG_ID
from test_executed_history import ack, actions, clock, injection
from test_executed_history_integration import response, capture_to
from test_history96 import triplet, state
from test_subtask_guard import obj, scene


def acknowledge(provider, frame, lineage, selected='U0012'):
    actual = injection(frame, selected=selected, lineage=lineage)
    source = ack(frame, actual_injection=actual)
    provider.acknowledge_executed_subtask(source, expected_injection=actual,
                                        expected_actions=actions(), expected_clock=clock(frame))


class SteadyProviderContracts(unittest.TestCase):
    def test_actual_ACK_then_two_visual_placement_frames_allow_a_legal_goal_switch(self):
        def transport(request, proof):
            frame = proof['current_frame']
            if proof['call_role'] == 'scene_observer':
                evidence = scene(obj()) if frame == 0 else scene(
                    obj(location='red', frame=8, placements=[7, 8]), obj('O2', 'camera', 'D003', frame=8))
                return response(canonical(evidence))
            return response('{"subtask_id":"' + ('U0012' if frame == 0 else 'U0007') + '"}')
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            _, first = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            acknowledge(p, 0, first); capture_to(p, 8)
            context, second = p.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            self.assertEqual(second['steady_guard']['confirmed_completion']['subtask_id'], 'U0012')
            self.assertEqual(second['selected_id'], 'U0007')
            self.assertEqual(second['steady_guard']['category_to_basket'], {'pen': 'red'})
            acknowledge(p, 8, second, selected='U0007')
            self.assertEqual([r['subtask_id'] for r in p.executed_history.records], ['U0012', 'U0007'])

    def test_semantic_observer_error_is_not_redrawn(self):
        calls = []
        def transport(request, proof):
            calls.append(proof)
            return response('{"uncertain":false}')  # schema invalid, not a transient failure
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            with self.assertRaises(SubtaskConstraintError):
                p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(p.executed_history.records, [])

    def test_two_visual_requests_share_the_same_causal_images_and_only_ID_is_selected(self):
        calls = []
        def transport(request, proof):
            calls.append((deepcopy(request), deepcopy(proof)))
            if proof.get('call_role') == 'scene_observer':
                return response(canonical(scene(obj(frame=proof['current_frame']))))
            return response('{"subtask_id":"U0012"}')
        with tempfile.TemporaryDirectory() as tmp:
            provider = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            provider.reset('fixture')
            for frame in (0, 8, 16):
                capture_to(provider, frame)
                context, proof = provider.generate(triplet(frame), INSTRUCTION, state14=state(frame),
                                                   episode_id='fixture', frame=frame)
                provider.validate_injection(context, proof, episode_id='fixture', frame=frame)
                self.assertEqual(proof['protocol_config_id'], STEADY_CONFIG_ID)
                self.assertEqual(context, {'subtask': provider.guard.by_id['U0012']['text']})
                self.assertEqual(len(provider.executed_history.records), frame // 8)
                acknowledge(provider, frame, proof)
            for observer_call, selector_call in zip(calls[::2], calls[1::2]):
                a, b = observer_call[0], selector_call[0]
                images = lambda req: [c for c in req['messages'][-1]['content'] if c['type'] == 'image_url']
                self.assertEqual(images(a), images(b))
                self.assertLessEqual(len(images(a)), 96)
                self.assertNotIn('scene_registry', b['messages'][-1]['content'][0]['text'])
                for request in (a, b):
                    self.assertEqual(request['max_tokens'], 65536)
                    self.assertEqual(request['temperature'], 0)
                    self.assertEqual(request['thinking'], {'type': 'enabled'})
            allowed = json.loads(calls[3][0]['messages'][1]['content'][0]['text'])['allowed_candidates']
            self.assertEqual([c['id'] for c in allowed], ['U0012'])
            self.assertEqual(len(provider.executed_history.records), 3)

    def test_valid_catalogue_ID_that_jumps_goal_is_blocked_without_retry_or_commit(self):
        calls = []
        def transport(request, proof):
            calls.append(deepcopy(proof))
            if proof.get('call_role') == 'scene_observer':
                return response(canonical(scene(obj(frame=proof['current_frame']))))
            return response('{"subtask_id":"' + ('U0012' if proof['current_frame'] == 0 else 'U0001') + '"}')
        with tempfile.TemporaryDirectory() as tmp:
            provider = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            provider.reset('fixture'); capture_to(provider, 0)
            _, first = provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            acknowledge(provider, 0, first); capture_to(provider, 8)
            with self.assertRaisesRegex(SubtaskConstraintError, 'goal lock'):
                provider.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            self.assertEqual(len(calls), 4)
            self.assertEqual(provider.last_frame, 0)
            self.assertEqual(len(provider.executed_history.records), 1)
            stop = json.loads((Path(tmp) / 'steady_decision_stops.jsonl').read_text())
            self.assertIsNone(stop['SDK_outcome'])
            self.assertFalse(stop['policy_called'])
            with self.assertRaisesRegex(SubtaskConstraintError, 'already stopped'):
                provider.generate(triplet(8), INSTRUCTION, state14=state(8), episode_id='fixture', frame=8)
            self.assertEqual(len(calls), 4)  # RPC retries cannot become semantic redraws
            provider.reset('new-episode')
            self.assertIsNone(provider.stopped)

    def test_initial_basket_contents_block_an_incompatible_ID_on_the_first_frame(self):
        calls = []
        def transport(request, proof):
            calls.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer':
                return response(canonical(scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003'))))
            return response('{"subtask_id":"U0009"}')  # camera -> red, already occupied by pens
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            with self.assertRaises(SubtaskConstraintError):
                p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(calls, ['scene_observer', 'subtask_selector'])
            self.assertEqual(p.executed_history.records, [])

    def test_uncertain_observer_does_not_call_the_selector(self):
        calls = []
        def transport(request, proof):
            calls.append(proof)
            evidence = scene(obj()); evidence['uncertain'] = True
            return response(canonical(evidence))
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            with self.assertRaises(SubtaskConstraintError):
                p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(p.last_frame, -8)

    def test_observer_transport_retry_is_identical_and_does_not_add_history(self):
        calls = []; failed = [False]
        def transport(request, proof):
            calls.append((canonical(request), proof['call_role']))
            if proof['call_role'] == 'scene_observer':
                if not failed[0]:
                    failed[0] = True
                    raise TimeoutError('fixture transient')
                return response(canonical(scene(obj())))
            return response('{"subtask_id":"U0012"}')
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(len(calls), 3)
            self.assertEqual(p.executed_history.records, [])

    def test_observer_and_selector_share_budget_instead_of_resetting_timeouts(self):
        now = [0.]; calls = []
        def transport(request, proof):
            calls.append(proof['call_role']); now[0] += 700
            if proof['call_role'] == 'scene_observer':
                return response(canonical(scene(obj())))
            return response('', finish='length')
        with tempfile.TemporaryDirectory() as tmp, patch('steady_provider.time.monotonic', side_effect=lambda: now[0]):
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            with self.assertRaises(ValueError):
                p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            self.assertEqual(calls, ['scene_observer', 'subtask_selector'])

    def test_changed_authorization_or_context_cannot_reach_policy_injection(self):
        def transport(request, proof):
            return response(canonical(scene(obj()))) if proof['call_role'] == 'scene_observer' else response('{"subtask_id":"U0012"}')
        with tempfile.TemporaryDirectory() as tmp:
            p = SteadySubtaskProvider(tmp, selector_kind='deepseek', transport=transport)
            p.reset('fixture'); capture_to(p, 0)
            context, proof = p.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id='fixture', frame=0)
            for mutate in (lambda c, q: c.update(subtask='wrong'),
                           lambda c, q: q['steady_guard'].update(actual_selected_id='U0001')):
                c, q = deepcopy(context), deepcopy(proof); mutate(c, q)
                with self.assertRaises(SubtaskConstraintError):
                    p.validate_injection(c, q, episode_id='fixture', frame=0)


if __name__ == '__main__':
    unittest.main()
