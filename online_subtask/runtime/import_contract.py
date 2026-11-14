"""Pin the ACTUALLY imported training/OpenPI/RPC source, without loading weights."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import re
import sys
from export_binding import read, sha, canonical_sha

SCHEMA = 'ecot.stack_bowls.runtime_imports.v1'


def validate_policy_environment(value):
    # Device ownership, Python source precedence and credentials must not be
    # overridden by an environment library-path file.
    allowed = {'LD_LIBRARY_PATH', 'PATH', 'XLA_FLAGS'}
    if (not isinstance(value, dict) or not set(value) <= allowed
            or any(not isinstance(v, str) or not v or '\n' in v or '\x00' in v for v in value.values())):
        raise ValueError('Only explicit policy library/compiler paths are permitted')
    if 'XLA_FLAGS' in value and not re.fullmatch(r'--xla_gpu_cuda_data_dir=/[^\s\x00]+', value['XLA_FLAGS']):
        raise ValueError('Library profile may set only the CUDA toolkit path, not model precision/algorithm flags')
    return dict(value)


def create_contract(parent, sdk_root):
    parent = Path(parent).resolve(strict=True)
    sdk = Path(sdk_root).resolve(strict=True)
    config = read(parent / 'configs/train.json')
    vendor = Path(config['vendor_root']).resolve(strict=True)
    roots = [parent / 'src', vendor / 'src', vendor / 'packages/openpi-client/src', sdk / 'XPolicyLab']
    if (parent / 'vendor/lerobot').is_dir():
        roots.append(parent / 'vendor/lerobot')
    for root in roots:
        if not root.is_dir():
            raise ValueError('Required policy import root missing: ' + str(root))
    files = {'strict_prompt': parent / 'src/strict_prompt.py',
             'client_server.ws.model_server': sdk / 'XPolicyLab/client_server/ws/model_server.py',
             'client_server.ws.model_client': sdk / 'XPolicyLab/client_server/ws/model_client.py',
             'openpi_client.base_policy': vendor / 'packages/openpi-client/src/openpi_client/base_policy.py'}
    for name in ('models.pi0_config', 'models.pi0', 'policies.policy_config', 'policies.policy',
                 'policies.aloha_policy', 'shared.normalize', 'training.config', 'transforms'):
        files['openpi.' + name] = vendor / 'src/openpi' / (name.replace('.', '/') + '.py')
    d = {'schema': SCHEMA, 'parent_root': str(parent), 'sdk_root': str(sdk),
         'pythonpath': [str(root.resolve()) for root in roots],
         'modules': {name: {'path': str(path.resolve(strict=True)), 'sha256': sha(path)}
                     for name, path in files.items()}}
    d['contract_sha256'] = canonical_sha(d)
    return d


def validate_contract(d):
    if (d.get('schema') != SCHEMA or canonical_sha({k: v for k, v in d.items()
            if k != 'contract_sha256'}) != d.get('contract_sha256')
            or create_contract(d['parent_root'], d['sdk_root']) != d):
        raise ValueError('Policy source/import contract changed')
    return d


def activate(d, *, import_modules=True):
    validate_contract(d)
    # Runtime first prevents a parent experiment's task_contract from shadowing
    # this task; training vendor precedes any environment/site-packages install.
    preferred = [str(Path(__file__).resolve().parent), *d['pythonpath']]
    sys.path[:] = preferred + [p for p in sys.path if p not in preferred]
    actual = {}
    for name, wanted in d['modules'].items():
        if not import_modules and name not in sys.modules:
            continue
        module = importlib.import_module(name) if import_modules else sys.modules[name]
        path = getattr(module, '__file__', None)
        if path is None or str(Path(path).resolve()) != wanted['path'] or sha(path) != wanted['sha256']:
            raise ValueError('Wrong actual imported source: ' + name)
        actual[name] = {'path': str(Path(path).resolve()), 'sha256': sha(path)}
    return {'contract_sha256': d['contract_sha256'], 'actual_imports': actual,
            'weights_loaded': False, 'API_GPU_native_started': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(activate(read(args.contract))))
