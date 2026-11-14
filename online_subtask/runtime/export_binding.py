"""Bind the approved Stack Bowls100 EMA10000 without rewriting training identity.

Read-only preparation hashes inference payloads against the independently sealed
formal checkpoint. No weights are exported, restored, recast or mutated here.
The original training eval_ecot=False is preserved; new evaluation is online.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import stat
import time
from task_contract import (TRAIN_TASK, INSTRUCTION, CHECKPOINT_STEP, TRAINING_BUDGET,
                           VARIANT, DATASET_MANIFEST_SHA, CONTEXT_CACHE_SHA, NORM_SHA,
                           INVENTORY_SHA, CHECKPOINT_SEAL_SHA)

SCHEMA = 'ecot.stack_bowls100.ema10000.inference_binding.v1'
STEPS = (CHECKPOINT_STEP,)
VARIANTS = (VARIANT,)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def fingerprint(path):
    s = Path(path).stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def regular_tree(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Regular committed parameter directory required')
    out = []
    for p in sorted(path.rglob('*')):
        mode = p.lstat().st_mode
        if stat.S_ISREG(mode):
            out.append(p)
        elif not stat.S_ISDIR(mode):
            raise ValueError('Nonregular parameter payload')
    if not out or not any(p.stat().st_size for p in out):
        raise ValueError('Empty parameter payload')
    return out


def validate_normalizer(path):
    d = read(path)
    if set(d) != {'norm_stats'} or set(d['norm_stats']) != {'state', 'actions'}:
        raise ValueError('Exact joint14 normalizer required')
    for item in d['norm_stats'].values():
        if set(item) != {'mean', 'std', 'q01', 'q99'}:
            raise ValueError('Missing normalizer arrays')
        for row in item.values():
            if (not isinstance(row, list) or len(row) != 14
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
                raise ValueError('Normalizer must contain finite14-vectors')
        if any(x < 0 for x in item['std']) or any(a > b for a, b in zip(item['q01'], item['q99'])):
            raise ValueError('Invalid normalizer statistics')
    return d


def validate_training_metadata(spec, identity, metadata, step=CHECKPOINT_STEP):
    if type(step) is not int or step != CHECKPOINT_STEP:
        raise ValueError('Only approved EMA10000 is admitted')
    expected_spec = {
        'task_name': TRAIN_TASK, 'train_samples_expected': 44674, 'batch_size': 32,
        'fsdp_devices': 8, 'seed': 42, 'warmup_steps': 200, 'peak_lr': 2.5e-5,
        'ema_decay': .99, 'num_train_steps': TRAINING_BUDGET,
        'ecot_variant': 'subtask', 'ecot_retained_fields': ['subtask'],
        'ecot_dropout_probability': 0., 'train_ecot': True, 'eval_ecot': False,
        'frozen_vlm': False, 'lora': False, 'max_token_len': 512,
    }
    if (any(spec.get(k) != v for k, v in expected_spec.items())
            or spec.get('episode_indices') != list(range(3000, 3100))):
        raise ValueError('Actual Stack Bowls100 Subtask-only training configuration differs')
    expected_identity = {
        'batch_size': 32, 'fsdp_devices': 8, 'seed': 42,
        'full_parameter_finetuning': True, 'frozen_vlm': False,
        'train_samples': 44674, 'steps_per_epoch': 1397,
        'num_train_steps': 15000, 'train_ecot': True, 'eval_ecot': False,
        'ecot_variant': 'subtask', 'ecot_retained_fields': ['subtask'],
        'ecot_prompt_version': 'stack_bowls_subtask_only_natural_v1',
        'dataset_manifest_sha256': DATASET_MANIFEST_SHA,
        'ecot_context_cache_sha256': CONTEXT_CACHE_SHA, 'norm_sha256': NORM_SHA,
        'parameter_inventory_sha256': INVENTORY_SHA, 'max_token_len': 512,
    }
    if any(identity.get(k) != v for k, v in expected_identity.items()):
        raise ValueError('Genuine training identity differs; never relabel another checkpoint')
    if identity.get('config_sha256') != canonical_sha(spec):
        raise ValueError('Resolved training configuration changed')
    if (metadata.get('identity') != identity or metadata.get('completed_step') != step
            or metadata.get('inference_params') != 'EMA'
            or (metadata.get('epoch'), metadata.get('next_batch_in_epoch')) != divmod(step, 1397)):
        raise ValueError('Checkpoint actual step, EMA or sampler identity differs')
    return True


def inspect(config_path, step):
    config_path = Path(config_path).resolve(strict=True)
    root = config_path.parent.parent
    if config_path != root / 'configs/train.json':
        raise ValueError('Actual parent configs/train.json required')
    spec = read(config_path)
    run = Path(spec['output_dir']).resolve(strict=True)
    checkpoint = run / 'checkpoints' / str(step)
    if checkpoint.is_symlink() or checkpoint.resolve(strict=True) != checkpoint:
        raise ValueError('Checkpoint may not redirect')
    identity_path = run / 'run_identity.json'
    identity = read(identity_path)
    metadata = read(checkpoint / 'resume_metadata.json')
    validate_training_metadata(spec, identity, metadata, step)
    if read(run / 'resolved_config.json') != spec:
        raise ValueError('Saved training configuration differs')

    commit = read(checkpoint / '_CHECKPOINT_METADATA')
    if (not commit.get('commit_timestamp_nsecs')
            or commit['commit_timestamp_nsecs'] < commit['init_timestamp_nsecs']
            or set(commit.get('item_handlers', {})) != {'assets', 'params', 'train_state'}):
        raise ValueError('Native checkpoint is not fully committed')
    seal_path = root / 'reports/continuous_monitor/checkpoints/10000/SEAL.json'
    if sha(seal_path) != CHECKPOINT_SEAL_SHA:
        raise ValueError('Independent approved10000 checkpoint seal changed')
    seal = read(seal_path)
    if (not seal.get('passed') or seal.get('actual_stored_step') != step
            or seal.get('run_identity_sha256') != sha(identity_path)
            or Path(seal['checkpoint']) != checkpoint):
        raise ValueError('Checkpoint seal does not bind the actual formal checkpoint')
    sealed = {row['path']: row for row in seal['files']}
    if len(sealed) != len(seal['files']):
        raise ValueError('Duplicate checkpoint file in seal')

    norm = Path(spec['norm_dir']) / 'norm_stats.json'
    checkpoint_norm = checkpoint / 'assets/arx_x5_sim/norm_stats.json'
    validate_normalizer(norm)
    if sha(norm) != NORM_SHA or sha(checkpoint_norm) != NORM_SHA:
        raise ValueError('Use the exact training and checkpoint normalizer')
    fixed = [config_path, identity_path, run / 'resolved_config.json',
             checkpoint / 'resume_metadata.json', checkpoint / '_CHECKPOINT_METADATA',
             checkpoint_norm, norm, seal_path]
    for item in ('params', 'train_state'):
        for rel in ('_METADATA', 'array_metadatas/process_0', 'manifest.ocdbt'):
            p = checkpoint / item / rel
            if not p.is_file() or str(p.relative_to(checkpoint)) not in sealed:
                raise ValueError('Missing committed model/Adam/EMA item metadata')
            fixed.append(p)
    for key, wanted in [('dataset_manifest', 'dataset_manifest_sha256'),
                        ('ecot_context_manifest', 'ecot_context_manifest_sha256'),
                        ('ecot_token_audit', 'ecot_token_audit_sha256')]:
        p = Path(spec[key])
        if sha(p) != identity.get(wanted):
            raise ValueError('Training data/context/token audit changed')
        fixed.append(p)
    context = read(spec['ecot_context_manifest'])
    cache = Path(context['cache'])
    if sha(cache) != CONTEXT_CACHE_SHA or context['language_values_modified'] is not False:
        raise ValueError('Original training Subtask projection changed')
    fixed.append(cache)
    inventory = run / 'fullparam_parameter_inventory.json'
    inv = read(inventory)
    if (inv.get('inventory_sha256') != INVENTORY_SHA
            or inv.get('trainable_parameters') != 3353433872 or inv.get('frozen_parameters') != 0):
        raise ValueError('Actual full-parameter selection differs')
    fixed.append(inventory)
    sources = identity.get('source_sha256', {})
    if not {'train.py', 'strict_prompt.py', 'masked_pi05.py', 'ecot_dataset.py'} <= set(sources):
        raise ValueError('Incomplete actual training source identity')
    for relative, digest in sources.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Nonlocal training source path')
        p = Path(spec['vendor_root']) / rel if relative.startswith(('src/', 'scripts/')) else root / 'src' / rel
        if sha(p) != digest:
            raise ValueError('Actual scientific source changed: ' + relative)
        fixed.append(p)
    params = regular_tree(checkpoint / 'params')
    for p in params:
        rel = str(p.relative_to(checkpoint))
        if rel not in sealed or p.stat().st_size != sealed[rel]['bytes']:
            raise ValueError('Inference parameter file is not in approved checkpoint seal')
    expected_params = {name for name in sealed if name.startswith('params/')}
    if expected_params != {str(p.relative_to(checkpoint)) for p in params}:
        raise ValueError('Inference parameter inventory differs from approved seal')
    paths = sorted(set(fixed + params))
    return {
        'schema': SCHEMA, 'parent_root': str(root), 'config_path': str(config_path),
        'export_path': str(checkpoint), 'source_checkpoint': str(checkpoint), 'step': step,
        'inference_params': 'EMA', 'inference_item': 'params', 'variant': VARIANT,
        'training_identity': identity, 'source_training_eval_ecot': False,
        'norm_stats_path': str(checkpoint_norm), 'norm_stats_sha256': NORM_SHA,
        'checkpoint_contains_full_train_state': True, 'inference_only': True,
        'parameter_item_relative_files': [str(p.relative_to(checkpoint / 'params')) for p in params],
        'checkpoint_seal_sha256': CHECKPOINT_SEAL_SHA, 'files': paths,
    }


def create_binding(config_path, step, output_path):
    info = inspect(config_path, step)
    paths = info.pop('files')
    before = {str(p): fingerprint(p) for p in paths}
    payload = {str(p): sha(p) for p in paths}
    if before != {str(p): fingerprint(p) for p in paths}:
        raise ValueError('Training/checkpoint files changed during binding')
    checkpoint = Path(info['export_path'])
    seal = read(Path(info['parent_root']) / 'reports/continuous_monitor/checkpoints/10000/SEAL.json')
    sealed = {row['path']: row['sha256'] for row in seal['files']}
    for p in paths:
        if p.is_relative_to(checkpoint) and payload[str(p)] != sealed[str(p.relative_to(checkpoint))]:
            raise ValueError('Actual checkpoint bytes differ from approved full-file seal')
    param = checkpoint / 'params'
    info.update(file_sha256=payload, file_fingerprints=before,
                parameter_item_tree_sha256=canonical_sha({
                    str(p.relative_to(param)): payload[str(p)] for p in paths if p.is_relative_to(param)}),
                created_unix=time.time(), actual_current_payload_hashes_verified=True,
                scope='Actual EMA10000 inference bytes/identity; no full tensor restore or native success claim.')
    info['binding_sha256'] = canonical_sha(info)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(info, out, indent=2, allow_nan=False)
        out.write('\n')
    return info


def validate_binding(binding, *, full_hashes=True):
    binding = read(binding) if isinstance(binding, (str, Path)) else binding
    if (binding.get('schema') != SCHEMA or canonical_sha({
            k: v for k, v in binding.items() if k != 'binding_sha256'}) != binding.get('binding_sha256')):
        raise ValueError('Checkpoint binding digest changed')
    current = inspect(binding['config_path'], binding['step'])
    paths = current.pop('files')
    for k, v in current.items():
        if binding.get(k) != v:
            raise ValueError('Bound identity changed: ' + k)
    if set(map(str, paths)) != set(binding['file_sha256']) or set(binding['file_fingerprints']) != set(binding['file_sha256']):
        raise ValueError('Bound inventory changed')
    param = Path(binding['export_path']) / 'params'
    if binding.get('actual_current_payload_hashes_verified') is not True or binding['parameter_item_tree_sha256'] != canonical_sha({
            str(p.relative_to(param)): binding['file_sha256'][str(p)] for p in paths if p.is_relative_to(param)}):
        raise ValueError('Binding inference payload hash tree differs')
    for p in paths:
        before = fingerprint(p)
        if before != binding['file_fingerprints'][str(p)]:
            raise ValueError('Bound file changed: ' + str(p))
        # Small metadata/source files are always hashed. Large tensor payloads
        # additionally get a complete rehash at explicit pre/postflight.
        is_tensor = p.is_relative_to(param) and '/d/' in str(p)
        if (full_hashes or not is_tensor) and sha(p) != binding['file_sha256'][str(p)]:
            raise ValueError('Bound file content changed: ' + str(p))
        if fingerprint(p) != before:
            raise ValueError('Bound file changed while validating')
    return {'accepted': True, 'full_hashes_verified': full_hashes,
            'step': binding['step'], 'binding_sha256': binding['binding_sha256'],
            'inference_params': 'EMA', 'source_training_eval_ecot': False}
