"""V4 token/schema/method integration fixtures, no API/model/GPU quality claim."""
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

import grounded_response_schema as original_factory
import grounded_response_schema_v2 as enum_factory
import enum_grounded_ssr_protocol as enum_protocol
from enum_grounded_ssr_protocol import (CONFIG_ID, NAMESPACE, SCHEMA_FACTORY_SHA,
    BASE_SCHEMA_FACTORY_SHA, RESPONSE_FORMAT_MODE, SCENE_PROTOCOL, MOTION_PROTOCOL, validate_spec)
from enum_grounded_ssr_provider import EnumGroundedSSRProvider
from grounded_motion import validate_arms, action_envelopes, validate_plan
from multiview_protocol import INSTRUCTION, canonical, load_catalog
from prepare_ssr_spec import build
from ssr_protocol import compose_prompt
from subtask_guard import SubtaskConstraintError
from test_executed_history_integration import response
from test_grounded_ssr import facts, public
from test_history96 import state, triplet
from test_ssr import runtime, capture_to, acknowledge
from test_steady_runtime import obs

HERE = Path(__file__).resolve().parent
# Exact saved episode202/frame24 PUBLIC observer document. No private thinking,
# source frames, SDK categories or future actions enter this regression fixture.
ACTUAL_202 = json.loads(r'''{"arms":{"left":{"gripper_state":"open","held_instance_id":null,"holding":"empty","interaction_instance_id":"O1","motion_trend":"approach","object_secured":"no","observed_basket":null,"relation":"approaching","release_ready":"no","visibility":"clear"},"right":{"gripper_state":"open","held_instance_id":null,"holding":"empty","interaction_instance_id":null,"motion_trend":"stationary","object_secured":"no","observed_basket":null,"relation":"unknown","release_ready":"no","visibility":"clear"}},"baskets":{"blue":{"unknown_contents":false,"visibility":"clear"},"red":{"unknown_contents":false,"visibility":"clear"},"white":{"unknown_contents":false,"visibility":"clear"}},"corrections":[],"goal_correction":null,"objects":[{"category":"black-and-brown plush doll","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D001"],"identity_confidence":"clear","instance_id":"O1","location":"table","released":null},{"category":"black-and-brown plush doll","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D020"],"identity_confidence":"clear","instance_id":"O2","location":"table","released":null},{"category":"pen","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D089"],"identity_confidence":"clear","instance_id":"O3","location":"table","released":null},{"category":"camera","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D009"],"identity_confidence":"clear","instance_id":"O4","location":"table","released":null},{"category":"camera","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D009"],"identity_confidence":"clear","instance_id":"O5","location":"table","released":null}],"scene_usable":true}''')
ACTUAL_202_SHA = '329d81a24cd01e2d99b43f3e20c3952c850c56475940e854b5ceb0140e29d8b2'
ACTUAL_203 = json.loads(r'''{"arms":{"left":{"gripper_state":"transitioning","held_instance_id":"O1","holding":"identified","interaction_instance_id":"O1","motion_trend":"close","object_secured":"no","observed_basket":null,"relation":"enclosing","release_ready":"no","visibility":"clear"},"right":{"gripper_state":"open","held_instance_id":null,"holding":"empty","interaction_instance_id":null,"motion_trend":"stationary","object_secured":"no","observed_basket":null,"relation":"separated","release_ready":"no","visibility":"clear"}},"baskets":{"blue":{"unknown_contents":false,"visibility":"clear"},"red":{"unknown_contents":false,"visibility":"clear"},"white":{"unknown_contents":false,"visibility":"clear"}},"corrections":[],"goal_correction":null,"objects":[{"category":"watch","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D070"],"identity_confidence":"clear","instance_id":"O1","location":"table","released":null},{"category":"watch","category_confidence":"uncertain","descriptor_confidence":"uncertain","descriptor_ids":["D019"],"identity_confidence":"uncertain","instance_id":"O2","location":"table","released":null},{"category":"block","category_confidence":"uncertain","descriptor_confidence":"uncertain","descriptor_ids":["D012"],"identity_confidence":"uncertain","instance_id":"O3","location":"table","released":null},{"category":"pen","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D004"],"identity_confidence":"clear","instance_id":"O4","location":"table","released":null},{"category":"pen","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D039"],"identity_confidence":"clear","instance_id":"O5","location":"table","released":null},{"category":"camera","category_confidence":"clear","descriptor_confidence":"clear","descriptor_ids":["D003"],"identity_confidence":"clear","instance_id":"O6","location":"table","released":null},{"category":"block","category_confidence":"uncertain","descriptor_confidence":"uncertain","descriptor_ids":["D006"],"identity_confidence":"uncertain","instance_id":"O7","location":"table","released":null}],"scene_usable":true}''')
ACTUAL_203_SHA = 'b4d469e4374e707581db13af65a12c1e29268a49b4bc71246f44cbd99b717d4a'

def wire(request, proof):
    return response(canonical(facts() if proof['call_role'] == 'scene_observer' else public()))

@contextmanager
def enum_runtime(tmp, transport=wire, events=None):
    with runtime(tmp, step=19566, events=events) as (_, policy, _, original):
        import model_server
        spec = build(original['weights_binding_file'], parent_root=original['parent_root'],
            audit_path=Path(tmp) / NAMESPACE / 'actions.jsonl', config_id=CONFIG_ID)
        provider = EnumGroundedSSRProvider(Path(tmp) / 'enum_requests', selector_kind='deepseek',
            checkpoint_step=19566, transport=transport)
        class Random:
            key = staticmethod(lambda seed: np.array([seed], np.int32))
            key_data = staticmethod(lambda key: key)
            split = staticmethod(lambda key: (key, key + 1))
            normal = staticmethod(lambda key, shape: np.zeros(shape, np.float32))
        model = model_server.create_model(spec, policy=policy,
            jax_module=types.SimpleNamespace(random=Random), provider=provider)
        yield model, policy, provider, spec

class EnumGroundedSSRContracts(unittest.TestCase):
    def test_exact15token_observer_changes_only_two_category_definitions_and_selector_unchanged(self):
        expected = ('pen', 'watch', 'camera', 'plush_toy', 'toy_vehicle', 'block', 'box',
                    'ring', 'bracelet', 'pouch', 'strap', 'tool', 'bottle_cap', 'can', 'tracker')
        self.assertEqual(enum_factory.CATEGORY_TOKENS, expected)
        catalogue = load_catalog(); old = original_factory.observer_response_format(catalogue)
        new = enum_factory.observer_response_format(catalogue)
        enum_category = {'type': ['string', 'null'], 'enum': list(expected) + [None]}
        self.assertEqual(new['json_schema']['schema']['properties']['objects']['items']['properties']['category'], enum_category)
        self.assertEqual(new['json_schema']['schema']['properties']['corrections']['items']['properties']['previous_category'], enum_category)
        old['json_schema']['schema']['properties']['objects']['items']['properties']['category'] = enum_category
        old['json_schema']['schema']['properties']['corrections']['items']['properties']['previous_category'] = enum_category
        self.assertEqual(old, new)
        for ids in (None, ['U0010', 'U0012']):
            self.assertEqual(enum_factory.selector_response_format(catalogue, ids),
                             original_factory.selector_response_format(catalogue, ids))
        self.assertEqual(hashlib.sha256((HERE / 'grounded_response_schema.py').read_bytes()).hexdigest(), BASE_SCHEMA_FACTORY_SHA)
        self.assertEqual(hashlib.sha256((HERE / 'grounded_response_schema_v2.py').read_bytes()).hexdigest(), SCHEMA_FACTORY_SHA)

    def test_dual_actual_factory_SHA_and_explicit_v4_spec_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding = Path(tmp) / 'binding.json'; binding.write_text(json.dumps({'step': 19566}))
            from ssr_protocol import NAMESPACE as OLD_NAMESPACE
            spec = build(binding, parent_root=tmp, audit_path=Path(tmp) / OLD_NAMESPACE / 'actions.jsonl')
            spec.update(config_id=CONFIG_ID, protocol_config_id=CONFIG_ID, generator_protocol=CONFIG_ID,
                audit_path=str(Path(tmp) / NAMESPACE / 'actions.jsonl'), grounded_action_constraints=True,
                scene_evidence_protocol=SCENE_PROTOCOL, motion_evidence_protocol=MOTION_PROTOCOL,
                response_format_mode=RESPONSE_FORMAT_MODE, response_schema_factory_sha256=SCHEMA_FACTORY_SHA,
                response_schema_base_factory_sha256=BASE_SCHEMA_FACTORY_SHA)
            self.assertEqual(validate_spec(spec)['config_id'], CONFIG_ID)
            for key, value in [('response_schema_factory_sha256', '0' * 64),
                ('response_schema_base_factory_sha256', '0' * 64), ('response_format_mode', 'json_object'),
                ('grounded_action_constraints', False), ('max_tokens', 16384)]:
                with self.subTest(field=key), self.assertRaises(ValueError): validate_spec(dict(spec, **{key: value}))
            altered = Path(tmp) / 'altered'; altered.mkdir()
            for name in ('grounded_response_schema.py', 'grounded_response_schema_v2.py'):
                for other in ('grounded_response_schema.py', 'grounded_response_schema_v2.py'):
                    (altered / other).write_bytes((HERE / other).read_bytes())
                (altered / name).write_bytes(b'EXPLICIT TAMPER FIXTURE')
                with self.subTest(factory=name), patch.object(enum_protocol, 'HERE', altered), self.assertRaises(ValueError):
                    validate_spec(spec)

    def test_actual202_both_category_phrases_rejected_unchanged_without_normalization(self):
        self.assertEqual(hashlib.sha256(canonical(ACTUAL_202).encode()).hexdigest(), ACTUAL_202_SHA)
        before = deepcopy(ACTUAL_202)
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=wire)
            for index in (0, 1):
                isolated = deepcopy(ACTUAL_202); isolated['objects'] = [deepcopy(ACTUAL_202['objects'][index])]
                with self.subTest(object=index), self.assertRaisesRegex(SubtaskConstraintError, 'exact vocabulary token'):
                    provider.validate_observation_format(isolated)
                self.assertEqual(isolated['objects'][0]['category'], 'black-and-brown plush doll')
            with self.assertRaises(SubtaskConstraintError): provider.validate_observation_format(ACTUAL_202)
        self.assertEqual(ACTUAL_202, before)

    def test_vocabulary_does_not_automatically_classify_by_descriptor_and_null_is_not_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=wire)
            for category in ('plush_toy', 'camera', 'block', None):
                evidence = facts(); evidence['objects'][0].update(category=category,
                    category_confidence='uncertain' if category is None else 'clear', descriptor_ids=['D006'])
                before = deepcopy(evidence)
                provider.validate_observation_format(evidence)
                self.assertEqual(evidence, before)  # D006 never forces category=block
            bad = facts(); bad['objects'][0].update(category=None, category_confidence='clear')
            with self.assertRaisesRegex(SubtaskConstraintError, 'uncertain confidence'):
                provider.validate_observation_format(bad)

    def test_unknown_type_with_known_descriptor_does_not_get_category_fallback_or_selector(self):
        calls = []
        def transport(request, proof):
            calls.append(proof['call_role']); value = facts()
            value['objects'][0].update(category=None, category_confidence='uncertain')
            return response(canonical(value))
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('enum_unknown_fixture'); capture_to(provider, 0)
            with self.assertRaises(SubtaskConstraintError):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(calls, ['scene_observer']); self.assertEqual(provider.executed_history.records, [])
            self.assertIsNotNone(provider.stopped)

    def test_legitimate_plush_token_goes_through_current_visual_admission_and_exact_roles_hashes(self):
        seen = []
        def transport(request, proof):
            seen.append((deepcopy(request), deepcopy(proof)))
            if proof['call_role'] == 'scene_observer':
                value = facts(); value['objects'][0].update(category='plush_toy', descriptor_ids=['D001'])
                return response(canonical(value))
            return response(canonical(dict(subtask_id='U0003',
                reasoning='The left gripper is open while approaching the visible plush toy; continue open-jaw alignment.',
                action={'summary': 'Approach the plush toy with the open left gripper for alignment.'})))
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('enum_plush_fixture'); capture_to(provider, 0)
            context, lineage = provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            provider.validate_injection(context, lineage, episode_id=provider.episode_id, frame=0)
            self.assertEqual(lineage['selected_id'], 'U0003'); self.assertEqual(lineage['generator_protocol'], CONFIG_ID)
            self.assertEqual(provider.executed_history.records, [])
            for request, proof in seen:
                self.assertEqual(proof['generator_protocol'], CONFIG_ID)
                self.assertEqual(proof['response_format'], request['response_format'])
                self.assertEqual(proof['response_format_sha256'], hashlib.sha256(canonical(request['response_format']).encode()).hexdigest())
                self.assertEqual(proof['messages_sha256'], hashlib.sha256(canonical(request['messages']).encode()).hexdigest())
                self.assertEqual(proof['request_sha256'], hashlib.sha256(canonical(request).encode()).hexdigest())
                self.assertEqual(request['response_format']['type'], 'json_schema')
                if proof['call_role'] == 'scene_observer':
                    self.assertEqual(request['messages'][0]['content'], (HERE / 'grounded_ssr_observer_prompt_v4.md').read_text())
                    self.assertEqual(request['response_format'], enum_factory.observer_response_format(provider.catalog))
                else:
                    ids = proof['grounded_decision']['allowed_ids']
                    self.assertEqual(request['response_format'], original_factory.selector_response_format(provider.catalog, ids))

    def test_new_prompt_retains_entire_old_prefix_and_explicit_full15_token_language_contract(self):
        raw = (HERE / 'grounded_ssr_observer_prompt_v4.md').read_bytes()
        original = (HERE / 'grounded_ssr_observer_prompt.md').read_bytes()
        self.assertTrue(raw.startswith(original))
        text = raw[len(original):].decode()
        for token in enum_factory.CATEGORY_TOKENS: self.assertIn(token, text)
        self.assertIn('category=null', text); self.assertIn('category_confidence=uncertain', text)
        self.assertIn('not a complete SDK', text); self.assertIn('same', text.lower())

    def test_all_three_fields_precede_policy_and_real_ACK_methods_supply_next_full_history(self):
        events = []; requests = []
        def transport(request, proof):
            events.append(proof['call_role']); requests.append((deepcopy(request), deepcopy(proof)))
            return wire(request, proof)
        with tempfile.TemporaryDirectory() as tmp, enum_runtime(tmp, transport, events) as (model, policy, provider, spec):
            self.assertEqual(spec['config_id'], CONFIG_ID)
            self.assertEqual(spec['response_schema_factory_sha256'], SCHEMA_FACTORY_SHA)
            self.assertEqual(spec['response_schema_base_factory_sha256'], BASE_SCHEMA_FACTORY_SHA)
            model.reset()
            for frame in (0, 8, 16):
                capture_to(provider, frame); model.update_obs(obs(frame)); commands = model.get_action()
                self.assertEqual(events[-3:], ['scene_observer', 'subtask_selector', 'policy'])
                injection = model.current_injection()
                self.assertEqual(injection['context_protocol'], CONFIG_ID)
                self.assertEqual(injection['checkpoint_step'], 19566)
                self.assertEqual(policy.observations[-1]['prompt'], compose_prompt(INSTRUCTION, injection['context']))
                self.assertEqual(injection['context']['reasoning'], public()['reasoning'])
                self.assertEqual(injection['context']['action'], public()['action'])
                self.assertEqual(len(provider.executed_history.records), frame // 8)
                acknowledge(model, commands, frame)
            self.assertEqual([r['executed_action_end'] for r in provider.executed_history.records], [8, 16, 24])
            self.assertEqual(len(policy.observations), 3)
            for request, proof in requests:
                self.assertEqual(proof['executed_subtask_history_count'], proof['current_frame'] // 8)
                self.assertTrue(all(set(r) == {'observation_frame', 'subtask_id', 'subtask',
                    'executed_action_start', 'executed_action_end'} for r in proof['executed_subtask_history']))

    def test_actual202_invalid_token_stops_before_policy_without_retry_fallback_or_ACK(self):
        calls = []; events = []
        def transport(request, proof):
            calls.append(proof['call_role']); return response(canonical(ACTUAL_202))
        with tempfile.TemporaryDirectory() as tmp, enum_runtime(tmp, transport, events) as (model, policy, provider, spec):
            model.reset(); capture_to(provider, 0); model.update_obs(obs(0))
            with self.assertRaisesRegex(SubtaskConstraintError, 'exact vocabulary token'): model.get_action()
            self.assertEqual(calls, ['scene_observer']); self.assertEqual(policy.observations, [])
            self.assertEqual(provider.executed_history.records, []); self.assertNotIn('policy', events)
            raw = next((Path(tmp) / 'enum_requests').glob('observer_*_response.raw.json')).read_text()
            saved = json.loads(json.loads(raw)['choices'][0]['message']['content'])
            self.assertEqual(saved, ACTUAL_202)
            with self.assertRaises(SubtaskConstraintError): model.get_action()
            self.assertEqual(calls, ['scene_observer'])

    def test_actual203_identified_hand_table_object_contradiction_still_stops_unchanged(self):
        self.assertEqual(hashlib.sha256(canonical(ACTUAL_203).encode()).hexdigest(), ACTUAL_203_SHA)
        calls = []; before = deepcopy(ACTUAL_203)
        def transport(request, proof):
            calls.append(proof['call_role']); return response(canonical(ACTUAL_203))
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            # All enum tokens are legal; the independent physical cross-field
            # guard must still reject, rather than promoting table to held.
            provider.validate_observation_format(ACTUAL_203)
            with self.assertRaisesRegex(SubtaskConstraintError, 'Held arm/object observations contradict'):
                validate_arms(ACTUAL_203['arms'], ACTUAL_203['objects'])
            provider.reset('enum_actual203_schema_fixture'); capture_to(provider, 0)
            with self.assertRaisesRegex(SubtaskConstraintError, 'Held arm/object observations contradict'):
                provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(calls, ['scene_observer']); self.assertEqual(provider.executed_history.records, [])
            saved = next(Path(tmp).glob('observer_*_response.raw.json')).read_text()
            self.assertEqual(json.loads(json.loads(saved)['choices'][0]['message']['content']), ACTUAL_203)
        self.assertEqual(ACTUAL_203, before)

    def test_visible_empty_table_enclosure_allows_close_but_transitioning_does_not_invent_support(self):
        seen = []
        def transport(request, proof):
            seen.append(proof['call_role'])
            if proof['call_role'] == 'scene_observer':
                value = facts('close'); value['objects'][0]['released'] = None
                return response(canonical(value))
            return response(canonical(public('Close')))
        with tempfile.TemporaryDirectory() as tmp:
            provider = EnumGroundedSSRProvider(tmp, selector_kind='deepseek', checkpoint_step=19566, transport=transport)
            provider.reset('enum_empty_enclosure_fixture'); capture_to(provider, 0)
            context, proof = provider.generate(triplet(0), INSTRUCTION, state14=state(0), episode_id=provider.episode_id, frame=0)
            self.assertEqual(context['action']['summary'], public('Close')['action']['summary'])
            self.assertEqual(seen, ['scene_observer', 'subtask_selector'])
            self.assertEqual(proof['current_visual_observation']['objects'][0]['location'], 'table')
            self.assertEqual(proof['current_visual_observation']['arms']['left']['holding'], 'empty')
            self.assertEqual(provider.executed_history.records, [])
        # Same enum token does not erase actual security/state restrictions.
        value = facts('close'); value['objects'][0]['released'] = None
        value['arms']['left'].update(gripper_state='transitioning', motion_trend='close')
        envelope = action_envelopes(value, dict(instance_id='O1', category='pen', basket_color='red'))
        self.assertEqual(envelope['by_arm']['left']['allowed_verbs'], ['hold'])
        for verb in ('Lift', 'Release'):
            invalid = public(verb); invalid.pop('subtask_id'); invalid['subtask'] = 'unused'
            with self.subTest(action=verb), self.assertRaises(SubtaskConstraintError): validate_plan(invalid, envelope)
        carry = {'subtask': 'unused', 'reasoning': 'Carry the pen toward the basket with the left arm.',
                 'action': {'summary': 'Carry the pen with the left gripper toward the red basket.'}}
        for security in ('no', 'unknown'):
            value['arms']['left']['object_secured'] = security
            unsupported = action_envelopes(value, dict(instance_id='O1', category='pen', basket_color='red'))
            with self.subTest(security=security), self.assertRaises(SubtaskConstraintError): validate_plan(carry, unsupported)
        value['arms']['left']['object_secured'] = 'no'
        # identified plus secured=no remains a valid supported-but-unsecured
        # contact state when the SAME object is actually located in the hand.
        value['objects'][0]['location'] = 'held'
        value['arms']['left'].update(holding='identified', held_instance_id='O1')
        validate_arms(value['arms'], value['objects'])
        envelope = action_envelopes(value, dict(instance_id='O1', category='pen', basket_color='red'))
        self.assertEqual(envelope['by_arm']['left']['allowed_verbs'], ['close', 'hold'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
