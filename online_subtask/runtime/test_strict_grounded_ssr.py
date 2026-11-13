"""Strict schema envelopes through real provider/policy/ACK methods, CPU only.

All HTTP, weights, policy outputs and SDK clocks are explicit behavior fixtures.
These checks establish no actual API/image perception/forecast/GPU acceptance.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import grounded_response_schema
import strict_grounded_ssr_protocol as strict_protocol
from grounded_ssr_provider import GroundedSSRProvider
from multiview_protocol import CAMERAS, INSTRUCTION, canonical, load_catalog
from prepare_ssr_spec import build
from ssr_protocol import compose_prompt
from strict_grounded_ssr_protocol import (CONFIG_ID, NAMESPACE, RESPONSE_FORMAT_MODE,
    RESPONSE_SCHEMA_FACTORY_SHA256, SCENE_PROTOCOL, MOTION_PROTOCOL, validate_spec)
from strict_grounded_ssr_provider import StrictGroundedSSRProvider
from subtask_guard import SubtaskConstraintError
from test_executed_history_integration import response
from test_grounded_ssr import facts, public
from test_history96 import state, triplet
from test_ssr import runtime, acknowledge, capture_to
from test_steady_runtime import obs

HERE = Path(__file__).resolve().parent

def strict_wire(request, proof):
    return response(canonical(facts() if proof['call_role'] == 'scene_observer' else public()))

@contextmanager
def strict_runtime(tmp, transport=strict_wire, events=None):
    # The independent old fixture supplies genuine formatter/weight contract
    # method boundaries but no real policy weights, API, GPU or simulator.
    with runtime(tmp, step=19566, events=events) as (_, policy, _, original):
        import model_server
        spec = build(original['weights_binding_file'], parent_root=original['parent_root'],
            audit_path=Path(tmp) / NAMESPACE / 'actions.jsonl', config_id=CONFIG_ID)
        provider = StrictGroundedSSRProvider(Path(tmp) / 'strict_requests', selector_kind='deepseek',
            checkpoint_step=19566, transport=transport)
        class Random:
            key = staticmethod(lambda seed: np.array([seed], np.int32))
            key_data = staticmethod(lambda key: key)
            split = staticmethod(lambda key: (key, key + 1))
            normal = staticmethod(lambda key, shape: np.zeros(shape, np.float32))
        model = model_server.create_model(spec, policy=policy,
            jax_module=types.SimpleNamespace(random=Random), provider=provider)
        yield model, policy, provider, spec

def factory_objects_are_exact(schema):
    if 'anyOf' in schema:
        for child in schema['anyOf']:
            factory_objects_are_exact(child)
    if schema.get('type') == 'object':
        assert schema['additionalProperties'] is False
        assert set(schema['required']) == set(schema['properties'])
        for child in schema['properties'].values():
            factory_objects_are_exact(child)
    if 'items' in schema:
        factory_objects_are_exact(schema['items'])

class StrictGroundedSSRContracts(unittest.TestCase):
    def test_factory_is_byte_exact_and_spec_requires_the_explicit_v3_binding(self):
        self.assertEqual(hashlib.sha256((HERE / 'grounded_response_schema.py').read_bytes()).hexdigest(),
                         RESPONSE_SCHEMA_FACTORY_SHA256)
        self.assertEqual(strict_protocol.validate_schema_factory(), RESPONSE_SCHEMA_FACTORY_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            binding = Path(tmp) / 'binding.json'; binding.write_text(json.dumps({'step': 19566}))
            # Test the protocol independently of the new builder dispatch.
            from ssr_protocol import NAMESPACE as OLD_NAMESPACE
            spec = build(binding, parent_root=tmp, audit_path=Path(tmp) / OLD_NAMESPACE / 'actions.jsonl')
            spec.update(config_id=CONFIG_ID, protocol_config_id=CONFIG_ID, generator_protocol=CONFIG_ID,
                audit_path=str(Path(tmp) / NAMESPACE / 'actions.jsonl'), grounded_action_constraints=True,
                scene_evidence_protocol=SCENE_PROTOCOL, motion_evidence_protocol=MOTION_PROTOCOL,
                response_format_mode=RESPONSE_FORMAT_MODE,
                response_schema_factory_sha256=RESPONSE_SCHEMA_FACTORY_SHA256)
            checked = validate_spec(spec)
            self.assertEqual(checked['config_id'], CONFIG_ID)
            for key, wrong in [('grounded_action_constraints', False),
                ('scene_evidence_protocol', 'wrong'), ('motion_evidence_protocol', 'wrong'),
                ('response_format_mode', 'json_object'), ('response_schema_factory_sha256', '0' * 64),
                ('max_tokens', 16384), ('execute_steps', 4), ('ecot_dropout_probability', 0.5),
                ('audit_path', str(Path(tmp) / OLD_NAMESPACE / 'actions.jsonl'))]:
                with self.subTest(field=key), self.assertRaises(ValueError):
                    validate_spec(dict(spec, **{key: wrong}))
            # A declared expected SHA cannot conceal changed actual factory bytes.
            altered = Path(tmp) / 'altered'; altered.mkdir()
            (altered / 'grounded_response_schema.py').write_bytes(b'EXPLICIT TAMPER FIXTURE')
            with patch.object(strict_protocol, 'HERE', altered), self.assertRaises(ValueError):
                validate_spec(spec)

    def test_two_real_provider_requests_use_exact_schema_and_only_admitted_selector_ids(self):
        seen = []
        def transport(request, proof):
            seen.append((deepcopy(request), deepcopy(proof))); return strict_wire(request, proof)
        with tempfile.TemporaryDirectory() as tmp:
            provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('strict_envelope_fixture'); capture_to(provider, 0)
            context, lineage = provider.generate(triplet(0), INSTRUCTION, state14=state(0),
                episode_id=provider.episode_id, frame=0)
            provider.validate_injection(context, lineage, episode_id=provider.episode_id, frame=0)
            self.assertEqual([meta['call_role'] for _, meta in seen], ['scene_observer', 'subtask_selector'])
            self.assertEqual(len(provider.executed_history.records), 0)  # proposals are not ACKs
            self.assertEqual(context['reasoning'], public()['reasoning'])
            self.assertEqual(context['action'], public()['action'])
            self.assertEqual(lineage['generator_protocol'], CONFIG_ID)
            self.assertEqual(StrictGroundedSSRProvider.__bases__, (GroundedSSRProvider,))
            observer, selector = seen[0][0], seen[1][0]
            allowed = json.loads(selector['messages'][1]['content'][0]['text'])['allowed_candidates']
            ids = [row['id'] for row in allowed]
            self.assertLess(len(ids), 288)
            self.assertEqual(observer['response_format'], grounded_response_schema.observer_response_format(provider.catalog))
            self.assertEqual(selector['response_format'], grounded_response_schema.selector_response_format(provider.catalog, ids))
            self.assertEqual(set(selector['response_format']['json_schema']['schema']['properties']['subtask_id']['enum']), set(ids))
            for request, proof in seen:
                self.assertEqual(proof['generator_protocol'], CONFIG_ID)
                self.assertEqual(request['model'], 'deepseek-v4-1-flash-261017')
                self.assertEqual(request['max_tokens'], 65536)
                self.assertEqual(request['thinking'], {'type': 'enabled'})
                self.assertEqual(request['reasoning_effort'], 'high')
                self.assertEqual(request['temperature'], 0)
                self.assertEqual(request['response_format']['type'], 'json_schema')
                self.assertIs(request['response_format']['json_schema']['strict'], True)
                factory_objects_are_exact(request['response_format']['json_schema']['schema'])

    def test_schema_HTTP_rejection_stops_before_selector_without_format_fallback(self):
        seen = []
        def transport(request, proof):
            seen.append((deepcopy(request), deepcopy(proof)))
            error = RuntimeError('deepseek HTTP 400')
            error.response_body = b'{"error":"schema unsupported explicit fixture"}'
            raise error
        with tempfile.TemporaryDirectory() as tmp:
            provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('strict_HTTP_failure'); capture_to(provider, 0)
            for _ in range(2):
                with self.assertRaises((RuntimeError, SubtaskConstraintError)):
                    provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0][1]['call_role'], 'scene_observer')
            self.assertEqual(seen[0][0]['response_format']['type'], 'json_schema')
            self.assertIsNotNone(provider.stopped); self.assertEqual(provider.executed_history.records, [])

    def test_transient_transport_retries_keep_identical_strict_schema_bodies_only(self):
        bodies = []
        def transport(request, proof):
            bodies.append(canonical(request)); raise ConnectionError('explicit transient fixture')
        with tempfile.TemporaryDirectory() as tmp:
            provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('strict_identical_retry_fixture'); capture_to(provider, 0)
            with self.assertRaises(ConnectionError):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(len(bodies), 3); self.assertEqual(len(set(bodies)), 1)
            self.assertTrue(all(json.loads(body)['response_format']['type'] == 'json_schema' for body in bodies))
            self.assertEqual(provider.executed_history.records, [])

    def test_extra_observer_key_stops_without_deletion_selector_or_history(self):
        seen = []
        def transport(request, proof):
            seen.append(proof['call_role']); evidence = facts(); evidence['type'] = 'json_object'
            return response(canonical(evidence))
        with tempfile.TemporaryDirectory() as tmp:
            provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('strict_extra_key_fixture'); capture_to(provider, 0)
            with self.assertRaises(SubtaskConstraintError):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(seen, ['scene_observer']); self.assertEqual(provider.executed_history.records, [])
            raw = next(Path(tmp).glob('observer_*_response.raw.json')).read_text()
            returned = json.loads(json.loads(raw)['choices'][0]['message']['content'])
            self.assertEqual(returned['type'], 'json_object')  # exact failed output retained

    def test_explicit_nullable_selector_text_is_terminal_never_host_patched(self):
        for field in ('reasoning', 'summary'):
            calls = []
            def transport(request, proof):
                calls.append(proof['call_role'])
                if proof['call_role'] == 'scene_observer': return response(canonical(facts()))
                value = public()
                if field == 'reasoning': value['reasoning'] = None
                else: value['action']['summary'] = None
                return response(canonical(value))
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
                provider.reset('strict_null_fixture'); capture_to(provider, 0)
                with self.assertRaises(ValueError):
                    provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
                self.assertEqual(calls, ['scene_observer', 'subtask_selector'])
                self.assertEqual(provider.executed_history.records, []); self.assertIsNotNone(provider.stopped)

    def test_unadmitted_catalogue_ID_is_not_replaced_by_an_allowed_ID(self):
        seen = []
        def transport(request, proof):
            seen.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer': return response(canonical(facts()))
            value = public(); value['subtask_id'] = 'U0288'; return response(canonical(value))
        with tempfile.TemporaryDirectory() as tmp:
            provider = StrictGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('strict_unadmitted_ID_fixture'); capture_to(provider, 0)
            with self.assertRaises(SubtaskConstraintError):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(seen, ['scene_observer', 'subtask_selector'])
            self.assertEqual(provider.executed_history.records, [])

    def test_three_fields_precede_policy_and_real_ACK_prefix_reaches_strict_96_image_requests(self):
        events = []; requests = []
        def transport(request, proof):
            events.append(proof['call_role']); requests.append((deepcopy(request), deepcopy(proof)))
            return strict_wire(request, proof)
        with tempfile.TemporaryDirectory() as tmp, strict_runtime(tmp, transport, events) as (model, policy, provider, spec):
            self.assertEqual(spec['config_id'], CONFIG_ID)
            self.assertEqual(spec['response_format_mode'], RESPONSE_FORMAT_MODE)
            self.assertEqual(spec['response_schema_factory_sha256'], RESPONSE_SCHEMA_FACTORY_SHA256)
            model.reset()
            for frame in range(0, 233, 8):
                capture_to(provider, frame); model.update_obs(obs(frame)); commands = model.get_action()
                self.assertEqual(events[-3:], ['scene_observer', 'subtask_selector', 'policy'])
                injection = model.current_injection()
                self.assertEqual(injection['context_protocol'], CONFIG_ID)
                self.assertEqual(injection['checkpoint_step'], 19566)
                self.assertEqual(policy.observations[-1]['prompt'], compose_prompt(INSTRUCTION, injection['context']))
                self.assertIn(public()['reasoning'], policy.observations[-1]['prompt'])
                self.assertIn(public()['action']['summary'], policy.observations[-1]['prompt'])
                self.assertEqual(set(policy.observations[-1]['images']), set(CAMERAS))
                self.assertEqual(len(provider.executed_history.records), frame // 8)
                self.assertEqual(injection['lineage']['executed_subtask_history_count'], frame // 8)
                acknowledge(model, commands, frame)
                self.assertEqual(len(provider.executed_history.records), frame // 8 + 1)
            self.assertEqual(len(policy.observations), 30)
            self.assertEqual([row['executed_action_end'] for row in provider.executed_history.records], list(range(8, 241, 8)))
            last = [(request, proof) for request, proof in requests if proof['current_frame'] == 232]
            self.assertEqual(len(last), 2)
            for request, proof in last:
                self.assertEqual(proof['total_image_count'], 96)
                self.assertEqual(proof['past_image_count'], 93)
                self.assertEqual(proof['current_image_count'], 3)
                self.assertTrue(all(row['camera'] == 'cam_high' and row['frame'] < 232
                    for row in proof['observations'] if not row['is_current']))
                self.assertEqual(request['response_format']['type'], 'json_schema')
                history = json.loads(request['messages'][-1]['content'][0]['text'])['executed_subtask_history']
                self.assertEqual(len(history), 29)
                self.assertEqual(history[-1]['executed_action_end'], 232)
                self.assertTrue(all(set(row) == {'observation_frame', 'subtask_id', 'subtask',
                    'executed_action_start', 'executed_action_end'} for row in history))

    def test_selector_failure_never_calls_policy_or_manufactures_ACK(self):
        calls = []; events = []
        def transport(request, proof):
            calls.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer': return response(canonical(facts()))
            value = public(); value['reasoning'] = None; return response(canonical(value))
        with tempfile.TemporaryDirectory() as tmp, strict_runtime(tmp, transport, events) as (model, policy, provider, spec):
            model.reset(); capture_to(provider, 0); model.update_obs(obs(0))
            with self.assertRaises(ValueError): model.get_action()
            self.assertEqual(calls, ['scene_observer', 'subtask_selector'])
            self.assertEqual(policy.observations, []); self.assertNotIn('policy', events)
            self.assertEqual(provider.executed_history.records, [])
            with self.assertRaises((ValueError, RuntimeError)): model.current_injection()

if __name__ == '__main__':
    unittest.main(verbosity=2)
