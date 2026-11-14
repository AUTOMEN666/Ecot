"""Explicit CPU fixtures, never API/GPU/native acceptance or scientific results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import export_binding as binding
from layout_manifest import create_manifest, validate_manifest, SOURCE_FILES
from model_server import validate_protocol_spec
from multiview_protocol import load_catalog, CATALOG_SHA, PROTOCOLS
from pi05_policy.freeze import SCHEMA as FREEZE_SCHEMA, verify_static
from task_contract import *
from test_executed_history import clock, injection

ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    spec = importlib.util.spec_from_file_location('_fixture_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture_sdk(root):
    """Minimal inspected SOURCE SHAPES, no actual SDK/physics execution."""
    for relative in SOURCE_FILES:
        p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# offline source fixture\n')
    layouts = root / 'Assets/Eval_Layout/RoboDojo/arx_x5/1'
    layouts.mkdir(parents=True)
    for index in range(30):
        (layouts / f'{TASK}_{index}.json').write_text(json.dumps({'offline_fixture_layout': index}))
    (root / 'XPolicyLab/policy').mkdir(parents=True)
    (root / 'third_party/curobo').mkdir(parents=True)
    (root / 'env/global_configs.py').write_text(
        "import os\nBENCHMARK='RoboDojo'\n"
        "ASSETS_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Assets'))\n")
    (root / 'env/seed_manager/seed_manager.py').write_text(
        'class SeedManager:\n    def fixture(self):\n'
        '        all_layout_ids = list(range(len(matching_files)))\n'
        '        data = load_json(file_path)\n')
    (root / 'src/eval_client/eval_env.py').write_text(
        "class Eval:\n    def initialize(self):\n"
        "        self.client=WsModelClient(self.model,host=self.host,ws_ping_timeout_s=self.deploy_cfg.get('ws_ping_timeout_s',20.0),port=self.port)\n"
        "        if self.fixture:\n            self.reward_manager.init_state()\n")


def frozen(frame=0):
    labels = ('bowl0', 'bowl1', 'bowl2', 'clutter17')
    return {'schema': FREEZE_SCHEMA,
            'object_inventory': {label: {'instance_name': label, 'object_type': 'rigid'} for label in labels},
            'object_poses': {label: [[0., 0., 0.], [1., 0., 0., 0.]] for label in labels},
            'measured_joint_gripper_state': [0.] * 14, 'physics_clock': clock(frame)}


class StackBowlsContracts(unittest.TestCase):
    def test_existing_H20_readback_hook_is_rehomed_once_without_foreign_support(self):
        from namespace_fiveseed import patch_readback_source
        original = ('class Eval:\n    def reset(self):\n'
            '            from support.probe_readback import install_private_reward_readback\n'
            '            if not hasattr(self, "_ervla_readback_audit"):\n'
            '                self._ervla_readback_audit = install_private_reward_readback(self)\n'
            '            self.reward_manager.init_state()\n')
        patched = patch_readback_source(original)
        self.assertNotIn('support.', patched)
        self.assertEqual(patched.count('install_private_reward_readback(self)'), 1)
        self.assertEqual(patched.count('self.reward_manager.init_state()'), 1)
        with self.assertRaises(ValueError):patch_readback_source(patched)
        with self.assertRaises(ValueError):patch_readback_source(original.replace('support.probe_readback','support.unreviewed'))
    def test_actual_catalogue_source_counts_and_original_words(self):
        c = load_catalog()
        self.assertEqual(len(c['candidates']), 297)
        self.assertEqual(sum(row['training_start_count'] for row in c['candidates']), 44674)
        self.assertEqual(len({row['text'] for row in c['candidates']}), 297)
        self.assertNotIn('basket', json.dumps(c['candidates']).lower())
        self.assertEqual(c['source_cache_sha256'], CONTEXT_CACHE_SHA)
        text = 'Pick up the rightmost bowl.'
        self.assertIn(text, [row['text'] for row in c['candidates']])
        self.assertEqual(compose_prompt(INSTRUCTION, text), INSTRUCTION + ' Current subtask: ' + text)
        for bad in (None, '', ' ' + text):
            with self.assertRaises(ValueError): compose_prompt(INSTRUCTION, bad)
        with self.assertRaises(ValueError): compose_prompt(INSTRUCTION, text, False)

    def test_training_identity_and_step_cannot_be_relabelled(self):
        spec = dict(task_name=TRAIN_TASK, train_samples_expected=44674, batch_size=32, fsdp_devices=8,
                    seed=42, warmup_steps=200, peak_lr=2.5e-5, ema_decay=.99, num_train_steps=15000,
                    ecot_variant='subtask', ecot_retained_fields=['subtask'], ecot_dropout_probability=0.,
                    train_ecot=True, eval_ecot=False, frozen_vlm=False, lora=False, max_token_len=512,
                    episode_indices=list(range(3000, 3100)))
        identity = dict(batch_size=32, fsdp_devices=8, seed=42, full_parameter_finetuning=True,
                        frozen_vlm=False, train_samples=44674, steps_per_epoch=1397, num_train_steps=15000,
                        train_ecot=True, eval_ecot=False, ecot_variant='subtask', ecot_retained_fields=['subtask'],
                        ecot_prompt_version='stack_bowls_subtask_only_natural_v1',
                        dataset_manifest_sha256=DATASET_MANIFEST_SHA, ecot_context_cache_sha256=CONTEXT_CACHE_SHA,
                        norm_sha256=NORM_SHA, parameter_inventory_sha256=INVENTORY_SHA, max_token_len=512,
                        config_sha256=binding.canonical_sha(spec))
        metadata = dict(identity=deepcopy(identity), completed_step=10000, inference_params='EMA', epoch=7,
                        next_batch_in_epoch=221)
        self.assertTrue(binding.validate_training_metadata(spec, identity, metadata))
        for key, value in [('task_name', 'classify_objects'), ('ecot_dropout_probability', .5),
                           ('batch_size', 64), ('eval_ecot', True), ('ecot_retained_fields', ['subtask', 'reasoning'])]:
            changed = deepcopy(spec); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                binding.validate_training_metadata(changed, identity, metadata)
        for key, value in [('completed_step', 5000), ('inference_params', 'raw'), ('next_batch_in_epoch', 220)]:
            changed = deepcopy(metadata); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                binding.validate_training_metadata(spec, identity, changed)
        with self.assertRaises(ValueError): binding.validate_training_metadata(spec, identity, metadata, 5000)
        self.assertFalse(spec['eval_ecot']); self.assertFalse(identity['eval_ecot'])

    def test_complete_scene_and_robot_must_freeze_including_clutter(self):
        before = frozen(); self.assertTrue(verify_static(before, deepcopy(before))['passed'])
        for field in ('clutter', 'bowl', 'joint', 'clock', 'missing_bowl'):
            after = deepcopy(before)
            if field in ('clutter', 'bowl'): after['object_poses']['clutter17' if field == 'clutter' else 'bowl0'][0][0] = .001
            elif field == 'joint': after['measured_joint_gripper_state'][0] = .001
            elif field == 'clock': after['physics_clock'] = clock(1)
            else: del after['object_inventory']['bowl0']; del after['object_poses']['bowl0']
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                verify_static(before, after)

    def test_task_only_model_and_128_action_shortcut_fail_closed(self):
        from pi05_policy.native_model import Model
        from pi05_policy.deploy import eval_one_episode
        from unittest.mock import patch
        import os
        with self.assertRaises(ValueError): Model({'audit_path':'/unused-fixture'})
        with patch.dict(os.environ, {'PI05_ENGINEERING_COMMAND_LIMIT': '128', 'PI05_EVAL_PHASE': 'engineering'}):
            with self.assertRaisesRegex(ValueError, 'Full native800'): eval_one_episode(object(), object())

    def test_manifest_unique30_native_source_pins_and_private_namespace(self):
        from prepare_native_sim import prepare
        from seed_namespace_validation import validate_namespace
        from rpc_contract import validate_rpc_namespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sdk = root / 'source'; fixture_sdk(sdk)
            mp = root / 'manifest.json'; manifest = create_manifest(sdk, mp); digest = binding.sha(mp)
            self.assertEqual(validate_manifest(mp, expected_sha256=digest)['episodes'], 30)
            a = types.SimpleNamespace(layout=29, phase='engineering', layout_manifest=mp,
                manifest_sha256=digest, engineering_acceptance=None,
                output=root / OUTPUT_NAMESPACE / 'engineering', sim_python=Path(sys.executable),
                port=19071, native_gpu_index=3, experience=None)
            source_bytes = {name: (sdk / name).read_bytes() for name in SOURCE_FILES}
            plan = prepare(a)
            validate_namespace(plan['namespace'], manifest); validate_rpc_namespace(plan['namespace'])
            self.assertEqual(source_bytes, {name: (sdk / name).read_bytes() for name in SOURCE_FILES})
            self.assertEqual(plan['namespace']['layout_ids'], [29])
            self.assertEqual(len(plan['namespace']['layout_pool']), 30)
            self.assertIn(TASK, plan['argv']); self.assertNotIn('CUDA_VISIBLE_DEVICES', plan['required_environment_overrides'])
            self.assertEqual(plan['original_SDK_action_limit'], 800)
            self.assertEqual(plan['argv'][plan['argv'].index('--device')+1], 'cuda:3')
            import argparse
            parser = argparse.ArgumentParser(allow_abbrev=False)
            parser.add_argument('--kit_args'); parser.add_argument('--device')
            parsed, _ = parser.parse_known_args(plan['argv'][2:])
            self.assertEqual(parsed.kit_args, '--/renderer/multiGpu/enabled=false')
            self.assertEqual(parsed.device, 'cuda:3')
            from preflight import validate_native_plan
            self.assertTrue(validate_native_plan(plan, a.output/'sim_launch.json')['passed'])
            changed = deepcopy(plan); changed['runtime_PYTHONPATH_prefix'] = [changed['runtime_PYTHONPATH_prefix'][0]]
            with self.assertRaises(ValueError): validate_native_plan(changed, a.output/'sim_launch.json')
            for field, value in [('phase', 'formal'), ('layout', 30)]:
                bad = deepcopy(a); setattr(bad, field, value)
                with self.assertRaises(ValueError): prepare(bad)
            scene = sdk / f'Assets/Eval_Layout/RoboDojo/arx_x5/1/{TASK}_0.json'
            scene.write_text('{}')
            with self.assertRaises(ValueError): validate_manifest(mp, expected_sha256=digest)
            with self.assertRaises(ValueError): validate_native_plan(plan, a.output/'sim_launch.json')
            scene.write_text(json.dumps({'offline_fixture_layout': 0}))
            task = sdk / SOURCE_FILES[-2]; task.write_text('# changed native task\n')
            with self.assertRaises(ValueError): validate_manifest(mp, expected_sha256=digest)

    def test_video_text_uses_executed_action_boundary_not_next_proposal(self):
        module = tool('render_subtask_video')
        rows = []
        plans = {0: injection(0), 8: injection(8, selected='U0002')}
        for frame, plan in plans.items(): rows.append({'event': 'actual_injection', **plan})
        for index in range(9):
            plan = plans[index // 8 * 8]
            rows.append(dict(event='actual_executed_action_text', execution_action=index,
                observed_outcome_frame=index+1, injection_execution_frame=index//8*8,
                episode_id='fixture', context=plan['context'], prompt=plan['prompt'], prompt_sha256=plan['prompt_sha256']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'text.jsonl'; path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            labels = module.timeline(path, 9)
            self.assertEqual(labels[:9], [plans[0]['context']['subtask']] * 9)
            self.assertEqual(labels[9], plans[8]['context']['subtask'])
            rows[-1]['prompt'] = plans[0]['prompt']; path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError): module.timeline(path, 9)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_actual_CPU_fixture_video_encode_decode_and_text_receipt(self):
        module = tool('render_subtask_video')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source.mp4'; plan = injection(0)
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=black:s=640x480:r=25', '-frames:v', '2', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', str(source)], check=True, capture_output=True, timeout=30)
            rows = [dict(event='actual_injection', **plan), dict(event='actual_executed_action_text',
                execution_action=0, observed_outcome_frame=1, injection_execution_frame=0, episode_id='fixture',
                context=plan['context'], prompt=plan['prompt'], prompt_sha256=plan['prompt_sha256'])]
            log = root/'text.jsonl'; log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            receipt = module.render(source, log, root/'text.mp4', root/'receipt.json')
            self.assertEqual(receipt['frame_count'], 2)
            self.assertTrue(receipt['encoded_video_fully_decoded'])
            self.assertFalse(receipt['model_or_API_called'])


if __name__ == '__main__': unittest.main()
