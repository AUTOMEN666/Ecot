"""Prepare one full native Random case; never starts GPU, API or simulator."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from export_binding import sha
from layout_manifest import validate_manifest
from namespace_fiveseed import build
from seed_namespace_validation import validate_namespace
from rpc_contract import patch_private_namespace, validate_rpc_namespace
from acceptance import verify_engineering
from task_contract import TASK, OUTPUT_NAMESPACE, LAYOUT_IDS, PROTOCOL_CONFIG_ID
from preflight import runtime_sources, native_import_roots
HERE = Path(__file__).resolve().parent


def prepare(a):
    if a.layout not in LAYOUT_IDS or a.phase not in ('engineering', 'formal'):
        raise ValueError('Exact existing Random30 layout and phase required')
    python = a.sim_python.resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Usable native Python on this host required')
    manifest = validate_manifest(a.layout_manifest, expected_sha256=a.manifest_sha256)
    if a.phase == 'formal':
        if a.engineering_acceptance is None:
            raise ValueError('An independent accepted full-case engineering witness is required first')
        verify_engineering(a.engineering_acceptance, a.manifest_sha256)
    elif a.engineering_acceptance is not None:
        raise ValueError('Engineering is separate; it must not reuse a formal acceptance label')
    out = a.output.resolve()
    if OUTPUT_NAMESPACE not in out.parts:
        raise ValueError('New isolated Stack Bowls output namespace required')
    out.mkdir(parents=True, exist_ok=False)
    alias = 'Pi05StackBowls_' + sha_text(str(out))[:16]
    ns = build(out / 'sdk', [a.layout], HERE / 'pi05_policy', alias, seed=1, manifest=manifest)
    patch_private_namespace(ns)
    validate_namespace(ns, manifest)
    validate_rpc_namespace(ns)
    sdk = Path(ns['private_source'])
    env = {'PI05_ONLINE_LANE_OUTPUT': str(out), 'PI05_EVAL_PHASE': a.phase,
           'PI05_ENGINEERING_COMMAND_LIMIT': '0', 'EVAL_NUM': '1',
           'ROBODOJO_SAVE_EVAL_VIDEOS': 'true', 'ROBODOJO_SAVE_EVAL_VIDEOS_MAX_LAYOUTS': '1',
           'ROBODOJO_RUN_ID': out.name, 'PYTHONDONTWRITEBYTECODE': '1'}
    argv = [str(a.sim_python), str(sdk / 'src/eval_client/main.py'),
            '--task_name', TASK, '--num_envs', '1', '--env_cfg_type', 'arx_x5',
            '--policy_name', alias, '--port', str(a.port), '--host', '127.0.0.1',
            '--seed', '1', '--additional_info', out.name, '--headless', '--enable_cameras',
            '--device_id', str(a.native_gpu_index), '--device', 'cuda:' + str(a.native_gpu_index),
            '--kit_args=--/renderer/multiGpu/enabled=false']
    if a.experience:
        argv += ['--experience', str(a.experience)]
    plan = {'schema': 'ecot.stack_bowls_random.native_case_plan.v1',
            'config_id': PROTOCOL_CONFIG_ID, 'phase': a.phase, 'task': TASK,
            'scene_seed': 1, 'layout_id': a.layout, 'noise_seed': 42,
            'manifest_path': str(a.layout_manifest.resolve()), 'manifest_sha256': a.manifest_sha256,
            'argv': argv, 'cwd': str(sdk), 'required_environment_overrides': env,
            'runtime_PYTHONPATH_prefix': native_import_roots(sdk),
            'runtime_source_sha256': runtime_sources(),
            'credential_not_required_by_native_child': True,
            'environment_must_be_unset': ['CUDA_VISIBLE_DEVICES', 'ECOT_AUTHORIZED_API_KEY'],
            'native_physical_gpu_index': a.native_gpu_index,
            'native_single_GPU_rendering': True,
            'required_resource_scope': 'All8 public UUID locks and fresh owner/renderer admission; no synthetic parking',
            'namespace': ns, 'engineering_command_limit': 0, 'original_SDK_action_limit': 800,
            'engineering_acceptance': str(a.engineering_acceptance.resolve()) if a.engineering_acceptance else None,
            'engineering_acceptance_sha256': sha(a.engineering_acceptance) if a.engineering_acceptance else None,
            'launched': False, 'GPU_admission_done': False,
            'runtime_environment_and_actual_renderer_validation_required': True}
    (out / 'sim_launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def parser(*, include_layout=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout-manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sim-python', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    if include_layout:
        p.add_argument('--layout', type=int, required=True)
    p.add_argument('--phase', choices=['engineering', 'formal'], required=True)
    p.add_argument('--engineering-acceptance', type=Path)
    p.add_argument('--native-gpu-index', type=int, choices=range(8), default=3)
    p.add_argument('--experience', type=Path)
    return p


def main():
    a = parser().parse_args()
    if not 1024 <= a.port <= 65535:
        raise ValueError('Unprivileged port required')
    d = prepare(a)
    print(json.dumps({'plan': str(a.output / 'sim_launch.json'), 'phase': d['phase'],
                      'layout': d['layout_id'], 'launched': False}))


if __name__ == '__main__':
    main()
