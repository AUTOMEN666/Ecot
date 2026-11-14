"""Revalidate a prepared native plan immediately BEFORE its child is launched."""
from pathlib import Path
import os
from export_binding import sha
from task_contract import TASK, PROTOCOL_CONFIG_ID, OUTPUT_NAMESPACE, LAYOUT_IDS


def runtime_sources():
    root = Path(__file__).resolve().parent
    files = [p for p in root.rglob('*.py') if not p.name.startswith('test_')]
    files += [root / 'selector_prompt.md', root / 'ACTIVE_EXPANDED_CATALOG.json']
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def native_import_roots(sdk):
    sdk = Path(sdk)
    source = sdk / 'third_party/IsaacLab/source'
    if (not (source / 'isaaclab/isaaclab/app/app_launcher.py').is_file()
            or not (sdk / 'third_party/curobo').is_dir()):
        raise ValueError('Actual SDK IsaacLab and curobo source roots required')
    return [str(Path(__file__).resolve().parent), str(sdk), str(sdk / 'XPolicyLab'),
            str(sdk / 'third_party/curobo'), *[str(p) for p in sorted(source.iterdir()) if p.is_dir()]]


def validate_native_plan(plan, plan_path):
    from layout_manifest import validate_manifest
    from seed_namespace_validation import validate_namespace
    from rpc_contract import validate_rpc_namespace
    from acceptance import verify_engineering
    phase = plan.get('phase')
    if (plan.get('schema') != 'ecot.stack_bowls_random.native_case_plan.v1'
            or plan.get('config_id') != PROTOCOL_CONFIG_ID or plan.get('task') != TASK
            or plan.get('scene_seed') != 1 or plan.get('noise_seed') != 42
            or type(plan.get('layout_id')) is not int or plan['layout_id'] not in LAYOUT_IDS
            or phase not in ('engineering', 'formal') or plan.get('launched') is not False
            or plan.get('native_single_GPU_rendering') is not True
            or plan.get('engineering_command_limit') != 0 or plan.get('original_SDK_action_limit') != 800):
        raise ValueError('Prepared Stack Bowls Random/800/seed1/noise42 plan differs')
    folder = Path(plan_path).resolve(strict=True).parent
    if OUTPUT_NAMESPACE not in folder.parts:
        raise ValueError('Isolated Stack Bowls namespace required')
    if plan.get('runtime_source_sha256') != runtime_sources():
        raise ValueError('Runtime code, prompt or catalogue changed after preparation')
    manifest = validate_manifest(plan['manifest_path'], expected_sha256=plan['manifest_sha256'])
    namespace = plan['namespace']
    validate_namespace(namespace, manifest); validate_rpc_namespace(namespace)
    sdk = Path(namespace['private_source']).resolve(strict=True)
    if namespace['layout_ids'] != [plan['layout_id']] or Path(plan['cwd']).resolve() != sdk:
        raise ValueError('Actual private SDK/layout differs')
    alias = sdk / 'XPolicyLab/policy' / namespace['policy_alias']
    if not alias.is_symlink() or alias.resolve() != Path(__file__).resolve().parent / 'pi05_policy':
        raise ValueError('Private policy alias does not address this checked runtime')
    roots = native_import_roots(sdk)
    if plan.get('runtime_PYTHONPATH_prefix') != roots:
        raise ValueError('Private SDK and RPC import precedence differs')
    argv = plan['argv']; python = Path(argv[0])
    if not python.is_file() or not os.access(python, os.X_OK) or Path(argv[1]).resolve() != (sdk / 'src/eval_client/main.py').resolve():
        raise ValueError('Native interpreter or main entry is unavailable/different')
    for key, value in {'--task_name': TASK, '--num_envs': '1', '--seed': '1',
                       '--env_cfg_type': 'arx_x5', '--policy_name': namespace['policy_alias'],
                       '--host': '127.0.0.1', '--device_id': str(plan['native_physical_gpu_index']),
                       '--device': 'cuda:' + str(plan['native_physical_gpu_index'])}.items():
        if argv.count(key) != 1 or argv[argv.index(key) + 1] != value:
            raise ValueError('Native command protocol differs: ' + key)
    if [arg for arg in argv if arg.startswith('--kit_args')] != ['--kit_args=--/renderer/multiGpu/enabled=false']:
        raise ValueError('Single-GPU renderer arguments must use unambiguous argparse assignment')
    env = plan['required_environment_overrides']
    expected = {'PI05_ONLINE_LANE_OUTPUT': str(folder), 'PI05_EVAL_PHASE': phase,
                'PI05_ENGINEERING_COMMAND_LIMIT': '0', 'EVAL_NUM': '1'}
    if (any(env.get(k) != v for k, v in expected.items()) or 'CUDA_VISIBLE_DEVICES' in env
            or not {'CUDA_VISIBLE_DEVICES', 'ECOT_AUTHORIZED_API_KEY'} <= set(plan['environment_must_be_unset'])):
        raise ValueError('Native environment does not preserve the approved protocol')
    if phase == 'formal':
        path = plan.get('engineering_acceptance')
        if not path or sha(path) != plan.get('engineering_acceptance_sha256'):
            raise ValueError('Independent engineering receipt changed after preparation')
        verify_engineering(path, plan['manifest_sha256'])
    return {'passed': True, 'phase': phase, 'layout_id': plan['layout_id'],
            'manifest_sha256': plan['manifest_sha256'], 'runtime_revalidated': True,
            'GPU_admission_included': False}
