"""Prepare a verified EMA10000 online policy configuration. Never launches it."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from export_binding import read, validate_binding, create_binding, canonical_sha
from multiview_protocol import CATALOG_SHA, PROTOCOLS
from model_server import validate_protocol_spec
from import_contract import create_contract, validate_policy_environment
from task_contract import (TASK, TRAIN_TASK, INSTRUCTION, CHECKPOINT_STEP, VARIANT,
                           OUTPUT_NAMESPACE, PROTOCOL_CONFIG_ID)
HERE = Path(__file__).resolve().parent


def prepare(a):
    parent = a.parent_root.resolve(strict=True)
    binding_path = a.export_binding.resolve(strict=True)
    binding = read(binding_path)
    validate_binding(binding, full_hashes=True)
    if binding['step'] != CHECKPOINT_STEP or binding['parent_root'] != str(parent):
        raise ValueError('Exact actual Stack Bowls parent and EMA10000 required')
    python = a.policy_python.resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Usable policy Python on this host required')
    imports = create_contract(parent, a.sdk_root)
    policy_environment = validate_policy_environment(read(a.policy_env_json)) if a.policy_env_json else {}
    out = a.output.resolve()
    if OUTPUT_NAMESPACE not in out.parts:
        raise ValueError('Use the new isolated Stack Bowls namespace')
    out.mkdir(parents=True, exist_ok=False)
    spec = {
        'parent_root': str(parent), 'task': TASK, 'training_task': TRAIN_TASK,
        'variant': VARIANT, 'condition': 'online_subtask', 'step': CHECKPOINT_STEP,
        'export_binding': str(binding_path), 'noise_seed': 42, 'instruction': INSTRUCTION,
        'audit_path': str(out / 'actions.jsonl'), 'effective_eval_ecot': True,
        'source_training_eval_ecot': False, 'provider': 'deepseek',
        'protocol_mode': 'expanded_catalog_id', 'generator_protocol': PROTOCOLS['expanded_catalog_id'],
        'catalog_sha256': CATALOG_SHA, 'config_id': PROTOCOL_CONFIG_ID,
        'protocol_config_id': PROTOCOL_CONFIG_ID, 'max_tokens': 65536,
        'selector_timeout_seconds': 600, 'selector_retry_budget_seconds': 1850,
        'policy_rpc_timeout_seconds': 2100, 'past_history_cameras': ['cam_high'],
        'maximum_total_images': 96, 'maximum_past_timepoints': 93,
        'executed_subtask_history_enabled': True,
        'executed_history_protocol': 'ecot.actual_executed_subtask_history.v1',
        'cadence_execution_frames': 8,
        'runtime_import_contract': imports,
        'policy_environment': policy_environment,
        'policy_environment_sha256': canonical_sha(policy_environment),
    }
    validate_protocol_spec(spec)
    contract_path = out / 'runtime_import_contract.json'
    contract_path.write_text(json.dumps(imports, indent=2) + '\n')
    env = dict(os.environ, **policy_environment, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for key in ('ECOT_AUTHORIZED_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN', 'PYTHONPATH'):
        env.pop(key, None)
    checked = subprocess.run([str(a.policy_python), '-B', str(HERE / 'import_contract.py'),
                              '--contract', str(contract_path)], env=env, capture_output=True,
                             text=True, timeout=120)
    (out / 'CPU_IMPORT_CHECK.json').write_text(json.dumps({
        'exit_code': checked.returncode, 'stdout': checked.stdout, 'stderr': checked.stderr,
        'API_GPU_native_started': False}, indent=2) + '\n')
    if checked.returncode:
        raise ValueError('Policy CPU imports failed; inspect CPU_IMPORT_CHECK.json before launch')
    (out / 'model.json').write_text(json.dumps(spec, indent=2) + '\n')
    argv = [str(a.policy_python), str(HERE / 'model_server.py'),
            '--config', str(out / 'model.json'), '--port', str(a.port),
            '--ready-output', str(out / 'READY.json')]
    launch = {'schema': 'ecot.stack_bowls.policy_launch.v1', 'argv': argv,
              'cwd': str(HERE), 'config': str(out / 'model.json'),
              'required_environment_overrides': {
                  'OPENPI_DATA_HOME': str(parent / 'cache/openpi'),
                  'JAX_COMPILATION_CACHE_DIR': str(out / 'cache/jax'),
                  'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                  'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                  'PYTHONDONTWRITEBYTECODE': '1', 'WANDB_MODE': 'disabled'},
              'credential_environment_variable_name_only': 'ECOT_AUTHORIZED_API_KEY',
              'GPU_launch_performed': False,
              'runtime_environment_and_GPU_admission_required': True}
    launch['required_environment_overrides'].update(policy_environment)
    (out / 'policy_launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    seal = sub.add_parser('bind')
    seal.add_argument('--train-config', type=Path, required=True)
    seal.add_argument('--output', type=Path, required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('--parent-root', type=Path, required=True)
    plan.add_argument('--export-binding', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--policy-python', type=Path, required=True)
    plan.add_argument('--sdk-root', type=Path, required=True)
    plan.add_argument('--policy-env-json', type=Path,
                      help='Optional accepted per-policy LD_LIBRARY_PATH/PATH/XLA_FLAGS; never apply to native child')
    plan.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if args.operation == 'bind':
        result = create_binding(args.train_config, CHECKPOINT_STEP, args.output)
        print(json.dumps({'binding': str(args.output), 'step': result['step'],
                          'actual_payload_hashes_verified': True, 'launched': False}))
    else:
        if not 1024 <= args.port <= 65535:
            raise ValueError('Unprivileged port required')
        print(json.dumps(prepare(args)))


if __name__ == '__main__':
    main()
