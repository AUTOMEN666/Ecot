"""Fixture-only policy admission: blocked Subtasks must never call infer()."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from multiview_protocol import CAMERAS, INSTRUCTION, PROTOCOL_CONFIG_ID, CATALOG_SHA, canonical
from multiview_provider import MultiviewHistoryProvider
from steady_provider import SteadySubtaskProvider
from subtask_guard import SubtaskConstraintError, STEADY_CONFIG_ID, STEADY_NAMESPACE
from prepare_steady_spec import convert
from test_executed_history import clock, ACK_SCHEMA
from test_executed_history_integration import response
from test_history96 import triplet, state
from test_subtask_guard import obj, scene


def legacy_spec(tmp):
    return dict(parent_root=tmp, ablation_variant='A_subtask_online', isolated_multiview_probe=True,
        protocol_mode='expanded_catalog_id', catalog_sha256=CATALOG_SHA, generator_protocol=PROTOCOL_CONFIG_ID,
        audit_path=str(Path(tmp) / 'seed1_native50_mainhistory93_65536_20261113_v1/actions.jsonl'),
        step=5000, condition='online_subtask', effective_eval_ecot=True, noise_seed=42,
        instruction=INSTRUCTION, export_binding='fixture-only', provider='deepseek',
        executed_subtask_history_enabled=True, executed_history_protocol='ecot.actual_executed_subtask_history.v1',
        config_id=PROTOCOL_CONFIG_ID, protocol_config_id=PROTOCOL_CONFIG_ID, maximum_past_timepoints=93,
        max_tokens=65536, past_history_cameras=['cam_high'], maximum_total_images=96,
        selector_timeout_seconds=600, selector_retry_budget_seconds=1850, policy_rpc_timeout_seconds=2100)


def obs(frame):
    return dict(state=state(frame), images={c: np.zeros((480, 640, 3), np.uint8) for c in CAMERAS},
                instruction=INSTRUCTION, frame_index=frame, physics_clock=clock(frame))


@contextmanager
def runtime(tmp, transport, provider_class=SteadySubtaskProvider):
    support = types.ModuleType('support'); support.__path__ = []
    task = types.ModuleType('support.task_contract'); task.INSTRUCTION = INSTRUCTION; task.TASK = 'classify_objects'
    weights = types.ModuleType('support.weights_contract'); weights.validate_weights = lambda *a, **k: None
    data = types.ModuleType('subtask_data')
    data.compose_prompt = lambda instruction, subtask, enabled: instruction + ' Current subtask: ' + subtask
    class Random:
        key = staticmethod(lambda seed: np.array([seed], dtype=np.int32))
        key_data = staticmethod(lambda key: key)
        split = staticmethod(lambda key: (key, key + 1))
        normal = staticmethod(lambda key, shape: np.zeros(shape, np.float32))
    class Policy:
        def __init__(self):
            self._rng = np.array([42], dtype=np.int32)
            self.observations = []
        def infer(self, observation, noise):
            self.observations.append(deepcopy(observation)); self._rng += 1
            return {'actions': np.zeros((50, 14), dtype=np.float32)}
    binding = dict(step=5000, ablation_variant='A_subtask_online',
                   training_identity={'eval_ecot': True, 'train_ecot': True},
                   binding_sha256='fixture-only', export_path='fixture-only', inference_params={},
                   parameter_item_tree_sha256='fixture-only')
    with patch.dict(sys.modules, {'support': support, 'support.task_contract': task,
                                 'support.weights_contract': weights, 'subtask_data': data}):
        import model_server, subtask_client
        provider = provider_class(Path(tmp) / 'requests', selector_kind='deepseek', transport=transport)
        spec = convert(legacy_spec(tmp), Path(tmp) / STEADY_NAMESPACE / 'actions.jsonl')
        policy = Policy()
        with patch.object(model_server, 'read', lambda path: binding), \
             patch.object(model_server, 'validate_binding', lambda *a, **k: None), \
             patch.object(subtask_client, 'png_current_rgb', lambda image: triplet(0)[CAMERAS[0]]):
            model = model_server.create_model(spec, policy=policy,
                    jax_module=types.SimpleNamespace(random=Random), provider=provider)
            yield model, policy, provider


class SteadyRuntimeAdmission(unittest.TestCase):
    def test_actual_model_uses_steady_provider_and_preserves_policy_only_inputs_and_ACK(self):
        def transport(request, proof):
            return response(canonical(scene(obj(frame=proof['current_frame'])))) if proof['call_role'] == 'scene_observer' else response('{"subtask_id":"U0012"}')
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, transport) as (model, policy, provider):
            model.reset(); episode = model.episode_id
            for frame in (0, 8):
                for capture in range(frame + 1):
                    from multiview_protocol import is_history_source_frame
                    if is_history_source_frame(capture) and (not provider.frames or provider.frames[-1].frame < capture):
                        model.record_history_observation(obs(capture))
                model.update_obs(obs(frame)); commands = model.get_action(); actual = model.current_injection()
                self.assertEqual(actual['lineage']['selected_id'], 'U0012')
                self.assertEqual(model.metadata['protocol_config_id'], STEADY_CONFIG_ID)
                source = dict(schema=ACK_SCHEMA, injection=actual, chunk_end_reason='full_chunk', execution_clocks=[
                    dict(episode_id=episode, execution_frame=frame + i + 1,
                         before=clock(frame + i), after=clock(frame + i + 1),
                         native_take_action_count=frame + i + 1,
                         action={k: v.tolist() for k, v in action.items()}) for i, action in enumerate(commands)])
                rng = policy._rng.copy()
                model.acknowledge_executed_subtask(source)
                self.assertTrue(np.array_equal(rng, policy._rng))
            self.assertEqual(len(policy.observations), 2)
            for observation in policy.observations:
                self.assertEqual(set(observation), {'state', 'images', 'prompt'})
                self.assertNotIn('scene_registry', observation['prompt'])
                self.assertNotIn('category_to_basket', observation['prompt'])
            model.reset()
            self.assertEqual(provider.guard.category_to_basket, {})
            self.assertIsNone(provider.guard.active)
            self.assertEqual(provider.executed_history.records, [])

    def test_illegal_selector_output_never_reaches_policy(self):
        def transport(request, proof):
            if proof['call_role'] == 'scene_observer':
                return response(canonical(scene(obj(location='red', placements=[0]), obj('O2', 'camera', 'D003'))))
            return response('{"subtask_id":"U0009"}')
        with tempfile.TemporaryDirectory() as tmp, runtime(tmp, transport) as (model, policy, provider):
            model.reset(); model.record_history_observation(obs(0)); model.update_obs(obs(0))
            with self.assertRaises(SubtaskConstraintError):
                model.get_action()
            self.assertEqual(policy.observations, [])
            self.assertEqual(provider.executed_history.records, [])

    def test_legacy_provider_cannot_bypass_steady_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'cannot bypass'):
                with runtime(tmp, lambda *a: response(), provider_class=MultiviewHistoryProvider):
                    pass

    def test_explicit_mode_identity_and_new_namespace_are_mandatory(self):
        from model_server import validate_protocol_spec
        original = legacy_spec('/fixture')
        converted = convert(original, Path('/fixture') / STEADY_NAMESPACE / 'actions.jsonl')
        self.assertNotIn('steady_subtask_enabled', original)
        self.assertTrue(validate_protocol_spec(converted)['steady_subtask_enabled'])
        for key, value in [('steady_subtask_enabled', False), ('config_id', PROTOCOL_CONFIG_ID),
                           ('audit_path', original['audit_path']), ('provider', 'codex')]:
            wrong = dict(converted); wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol_spec(wrong)


if __name__ == '__main__':
    unittest.main()
