"""Separate official step-zero parameters from committed task EMA checkpoints.

Base bytes are hashed once during CPU preparation. The SHA-pinned receipt and
each file's device/inode/size/mtime/ctime are checked before and after sampling.
Nothing is written under the shared, read-only official base directory.
"""
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path

OFFICIAL_BASE = Path('/mnt/xzt/ecot/pi05_cover_blocks_100_no_ecot_current3_20261028/models/pi05_base')
OFFICIAL_REPO = 'gs://openpi-assets/checkpoints/pi05_base'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return {key: getattr(stat, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def normalization_semantic_sha(path):
    """Exact native-loader float64 bits, independent of JSON numeric spelling.

    No tolerance, rounding, clipping, key omission or dimension dropping is
    permitted. Each of the eight state/action vectors includes both grippers.
    """
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate normalization JSON key: '+key)
            result[key] = value
        return result

    data = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    if not isinstance(data, dict) or set(data) != {'norm_stats'}:
        raise ValueError('Normalization requires exactly the norm_stats root')
    stats = data['norm_stats']
    if not isinstance(stats, dict) or set(stats) != {'state', 'actions'}:
        raise ValueError('Normalization requires exactly state and actions')
    arrays = {}
    for key in ('state', 'actions'):
        row = stats[key]
        if not isinstance(row, dict) or set(row) != {'mean', 'std', 'q01', 'q99'}:
            raise ValueError('Normalization requires all four exact statistical fields')
        for field in ('mean', 'std', 'q01', 'q99'):
            values = row[field]
            if (not isinstance(values, list) or len(values) != 14
                    or any(type(v) is not float or not math.isfinite(v) for v in values)):
                raise ValueError('Normalization requires finite native-loader float64 arrays')
            # Prepared exact_stats().tolist() and native OpenPI save() both
            # emit JSON floats. Require those types explicitly: Python JSON
            # floats are binary64, matching all actual native-loaded arrays.
            # Reject integer/bool/string encodings rather than infer a dtype.
            bits = struct.pack('<14d', *values)
            arrays[key+'.'+field] = {'dtype': 'float64', 'shape': [14],
                'little_endian_bits_sha256': hashlib.sha256(bits).hexdigest()}
    return json_sha({'schema': 'openpi_norm_joint14_float64_bits_v1', 'arrays': arrays})


def verify_official_base(base, receipt_path):
    base, receipt_path = Path(base).resolve(strict=True), Path(receipt_path)
    if base != OFFICIAL_BASE or (base/'resume_metadata.json').exists() or (base/'model.safetensors').exists():
        raise ValueError('Base mode accepts only the original official JAX pi05_base root')
    manifest_path = base/'download_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('repo_id') != OFFICIAL_REPO or not manifest.get('files'):
        raise ValueError('Official pi05_base source manifest required')
    files = manifest['files']
    for relative in files:
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or not (base/p).is_file():
            raise ValueError('Invalid official base inventory')
    actual = {str(p.relative_to(base)) for p in (base/'params').rglob('*') if p.is_file()}
    if actual != {p for p in files if p.startswith('params/')}:
        raise ValueError('Official parameter inventory differs from download manifest')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('base') != str(base) or receipt.get('manifest_sha256') != sha(manifest_path):
            raise ValueError('Existing base verification receipt has different provenance')
        if receipt.get('file_sha256') != files:
            raise ValueError('Existing receipt file inventory changed')
        if receipt.get('fingerprints') != {p: fingerprint(base/p) for p in files}:
            raise ValueError('Official bytes changed since full hash verification')
    else:
        before = {p: fingerprint(base/p) for p in files}
        for relative, expected in sorted(files.items()):
            if sha(base/relative) != expected:
                raise ValueError('Official base checksum mismatch: '+relative)
        if before != {p: fingerprint(base/p) for p in files}:
            raise ValueError('Official base changed while hashing')
        receipt = {'schema': 'pi05_official_base_verified_v1', 'base': str(base),
                   'source': OFFICIAL_REPO, 'manifest_sha256': sha(manifest_path),
                   'file_sha256': files, 'fingerprints': before,
                   'params_tree_sha256': json_sha({p: files[p] for p in sorted(actual)}),
                   'actual_full_file_hashes_verified': True, 'gradient_updates': 0}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open('x') as f:
            json.dump(receipt, f, indent=2)
            f.write('\n')
    if receipt.get('actual_full_file_hashes_verified') is not True or receipt.get('gradient_updates') != 0:
        raise ValueError('Missing actual full-file base verification')
    return receipt



SMOKE_RUN_NAME = 'pi05_classify_objects100_own_ssr_full_v2_smoke'


def smoke_proof_binding(run_dir, checkpoint, identity, spec, contract):
    """Admit only the named step2 engineering run with its measured proof."""
    files = {name: run_dir/name for name in ('run_identity.json', 'resolved_config.json',
             'fullparam_update_receipt.json', 'fullparam_before.json', 'fullparam_after.json',
             'restore_check.json')}
    files['state_fingerprint.json'] = checkpoint/'state_fingerprint.json'
    records = {name: json.loads(path.read_text()) for name, path in files.items()}
    meta = json.loads((checkpoint/'resume_metadata.json').read_text())
    proof = records['fullparam_update_receipt.json']
    if (checkpoint.name != '2' or meta.get('completed_step') != 2
            or meta.get('identity') != identity
            or records['run_identity.json'] != identity or records['resolved_config.json'] != spec
            or proof.get('identity') != identity
            or meta.get('fullparam_update_receipt_sha256') != sha(files['fullparam_update_receipt.json'])):
        raise ValueError('Isolated smoke checkpoint lacks its exact step2 identity/update proof')
    recomputed = contract.compare_update_proof(records['fullparam_before.json'],
        records['fullparam_after.json'], proof.get('gradient_metrics', []))
    if (any(proof.get(k) != v for k, v in recomputed.items())
            or proof.get('before_snapshot_sha256') != sha(files['fullparam_before.json'])
            or proof.get('after_snapshot_sha256') != sha(files['fullparam_after.json'])):
        raise ValueError('Isolated smoke raw/EMA update evidence changed')
    restored, fingerprint = records['restore_check.json'], records['state_fingerprint.json']
    leaves = fingerprint.get('leaves', [])
    if (restored.get('passed') is not True or restored.get('completed_step') != 2
            or restored.get('identity') != identity
            or restored.get('fullparam_update_proof_verified') is not True
            or restored.get('fullparam_update_receipt_sha256') != sha(files['fullparam_update_receipt.json'])
            or restored.get('array_leaves') != 207 or len(leaves) != 207
            or fingerprint.get('tree_sha256') != json_sha(leaves)
            or restored.get('tree_sha256') != fingerprint.get('tree_sha256')):
        raise ValueError('Isolated smoke requires the matching 207-array cold restore')
    return {'engineering_smoke_only': True,
            'smoke_proof_files_sha256': {str(path): sha(path) for path in files.values()}}


def fullparam_checkpoint_binding(root, checkpoint, identity, spec):
    """Verify the exact trainable tree; frozen-VLM checkpoints cannot pass."""
    root, checkpoint = Path(root).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    contract_path = root/'src/fullparam_contract.py'
    source_sha = sha(contract_path)
    if identity.get('source_sha256', {}).get('fullparam_contract.py') != source_sha:
        raise ValueError('Checkpoint full-parameter contract source changed')
    module_spec = importlib.util.spec_from_file_location('evaluation_fullparam_contract', contract_path)
    contract = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(contract)
    contract.validate_checkpoint_identity(identity, spec)
    context_manifest = json.loads(Path(spec['ecot_context_manifest']).read_text())
    if (identity.get('ecot_context_manifest_sha256') != sha(spec['ecot_context_manifest'])
            or identity.get('ecot_context_cache_sha256') != context_manifest['cache_sha256']
            or identity.get('ecot_token_audit_sha256') != sha(spec['ecot_token_audit'])):
        raise ValueError('Checkpoint training annotation/token-audit identity changed')
    if identity.get('config_sha256') != json_sha(spec):
        raise ValueError('Checkpoint was trained with a different full-parameter config')
    run_dir = checkpoint.parent.parent
    formal_run = (root/'runs/seed42').resolve(strict=False)
    smoke_run = (root/'runs'/SMOKE_RUN_NAME).resolve(strict=False)
    if (checkpoint.parent.name != 'checkpoints'
            or Path(spec['output_dir']).resolve(strict=False) != formal_run
            or run_dir not in (formal_run, smoke_run) or not run_dir.is_relative_to(root)):
        raise ValueError('Full-parameter checkpoint must belong to the configured run or exact isolated smoke')
    smoke_binding = (smoke_proof_binding(run_dir, checkpoint, identity, spec, contract)
                     if run_dir == smoke_run else {})
    inventory_path = run_dir/'fullparam_parameter_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    audited = contract.inventory_from_rows(inventory.get('leaves', []))
    if any(inventory.get(k) != v for k, v in audited.items()):
        raise ValueError('Full-parameter inventory report is inconsistent')
    if identity['parameter_inventory_sha256'] != audited['inventory_sha256']:
        raise ValueError('Checkpoint trained parameter inventory differs from the run')
    return {'parameter_policy': contract.PARAMETER_POLICY,
            'parameter_inventory_sha256': audited['inventory_sha256'],
            'parameter_inventory_path': str(inventory_path),
            'parameter_inventory_file_sha256': sha(inventory_path),
            'fullparam_contract_path': str(contract_path),
            'fullparam_contract_sha256': source_sha,
            'train_config_path': str(root/'configs/train.json'),
            'train_config_sha256': sha(root/'configs/train.json'),
            'frozen_vlm': False, 'ecot_variant': 'summary_subtask_reasoning',
            'full_parameter_finetuning': True, 'train_ecot': True, 'eval_ecot': False,
            'trainable_parameters': audited['trainable_parameters'],
            'trainable_parameter_leaves': audited['trainable_leaves'], **smoke_binding}


def prepare_weights(mode, checkpoint, norm_path, root, receipt_path):
    root = Path(root).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    spec = json.loads((root/'configs/train.json').read_text())
    manifest_path = Path(spec['dataset_manifest'])
    manifest = json.loads(manifest_path.read_text())
    if mode == 'base':
        if norm_path is None or Path(norm_path).resolve(strict=True) != (root/'assets/arx_x5_sim/norm_stats.json').resolve(strict=True):
            raise ValueError('Base requires explicit normalization from this train-all experiment')
        receipt = verify_official_base(checkpoint, receipt_path)
        norm = Path(norm_path).resolve(strict=True)
        result = {'weights_mode': 'base', 'checkpoint': str(checkpoint), 'step': 0,
                  'gradient_updates': 0, 'inference_params': 'official_pretrained_params',
                  'base_source': OFFICIAL_REPO, 'base_params_tree_sha256': receipt['params_tree_sha256'],
                  'base_verification_receipt': str(receipt_path), 'base_verification_receipt_sha256': sha(receipt_path)}
    elif mode == 'checkpoint':
        if not checkpoint.is_relative_to(root) or norm_path is not None:
            raise ValueError('SFT mode requires this experiment checkpoint and its committed normalization')
        meta_path = checkpoint/'resume_metadata.json'
        meta = json.loads(meta_path.read_text())
        step = meta.get('completed_step')
        if type(step) is not int or step <= 0 or meta.get('inference_params') != 'EMA' or not (checkpoint/'params').is_dir():
            raise ValueError('Require exact positive-step committed EMA checkpoint')
        identity = meta.get('identity', {})
        if (identity.get('train_ecot') is not True or identity.get('eval_ecot') is not False
                or identity.get('max_token_len') != spec['max_token_len']):
            raise ValueError('Require this ECoT train-only checkpoint and exact token capacity')
        if identity.get('source_sha256', {}).get('strict_prompt.py') != sha(root/'src/strict_prompt.py'):
            raise ValueError('Checkpoint tokenizer adapter source changed')
        parameter_binding = fullparam_checkpoint_binding(root, checkpoint, identity, spec)
        norm = checkpoint/'assets/arx_x5_sim/norm_stats.json'
        result = {'weights_mode': 'checkpoint', 'checkpoint': str(checkpoint), 'step': step,
                  'inference_params': 'EMA', 'checkpoint_metadata_sha256': sha(meta_path),
                  'checkpoint_metadata': meta, **parameter_binding}
    else:
        raise ValueError('Explicit base or checkpoint weight mode required')
    if mode == 'base':
        # Preserve the accepted step-zero contract and its existing bindings.
        if sha(norm) != manifest.get('norm_stats_sha256'):
            raise ValueError('Normalization differs from this exact prepared target100 manifest')
    else:
        source_norm = (root/'assets/arx_x5_sim/norm_stats.json').resolve(strict=True)
        source_sha = sha(source_norm)
        if (source_sha != manifest.get('norm_stats_sha256')
                or meta.get('identity', {}).get('norm_sha256') != source_sha):
            raise ValueError('Checkpoint training normalization provenance differs from target100')
        semantic_sha = normalization_semantic_sha(source_norm)
        if normalization_semantic_sha(norm) != semantic_sha:
            raise ValueError('Checkpoint normalization numeric bits differ from target100')
        # Native OpenPI save() may spell the same float differently. Bind both
        # files exactly, plus their equal native-loader bits; never overwrite
        # a committed checkpoint or silently substitute the source file.
        result.update(source_norm_stats_path=str(source_norm),
                      source_norm_stats_sha256=source_sha,
                      norm_stats_semantic_sha256=semantic_sha,
                      normalization_semantics_schema='openpi_norm_joint14_float64_bits_v1')
    stats = json.loads(norm.read_text()).get('norm_stats', {})
    for key in ('state', 'actions'):
        for kind in ('mean', 'std', 'q01', 'q99'):
            values = stats.get(key, {}).get(kind, [])
            if len(values) != 14 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Require exact joint14 normalization')
    result.update(norm_stats_path=str(norm), norm_stats_sha256=sha(norm),
                  target_data_normalization=True, strict_no_target_data=False,
                  normalization_dataset_manifest_sha256=sha(manifest_path))
    result['weights_identity_sha256'] = json_sha(result)
    return result


def validate_weights(binding):
    payload = {k: v for k, v in binding.items() if k != 'weights_identity_sha256'}
    if json_sha(payload) != binding['weights_identity_sha256']:
        raise ValueError('Weight identity descriptor changed')
    if sha(binding['norm_stats_path']) != binding['norm_stats_sha256']:
        raise ValueError('Planned normalization changed')
    if binding['weights_mode'] == 'base':
        receipt = Path(binding['base_verification_receipt'])
        if sha(receipt) != binding['base_verification_receipt_sha256']:
            raise ValueError('Official base verification receipt changed')
        actual = verify_official_base(binding['checkpoint'], receipt)
        if actual['params_tree_sha256'] != binding['base_params_tree_sha256'] or binding['step'] != 0:
            raise ValueError('Base provenance/step mismatch')
    elif binding['weights_mode'] == 'checkpoint':
        config_path = Path(binding['train_config_path'])
        if sha(config_path) != binding['train_config_sha256']:
            raise ValueError('Pinned full-parameter training config changed')
        actual = fullparam_checkpoint_binding(config_path.parent.parent, binding['checkpoint'],
            binding['checkpoint_metadata']['identity'], json.loads(config_path.read_text()))
        if any(binding.get(k) != v for k, v in actual.items()):
            raise ValueError('Pinned full-parameter provenance changed')
        if sha(Path(binding['checkpoint'])/'resume_metadata.json') != binding['checkpoint_metadata_sha256']:
            raise ValueError('Committed EMA checkpoint metadata changed')
        if (binding.get('normalization_semantics_schema') != 'openpi_norm_joint14_float64_bits_v1'
                or sha(binding['source_norm_stats_path']) != binding['source_norm_stats_sha256']):
            raise ValueError('Pinned target100 source normalization changed')
        expected = binding['norm_stats_semantic_sha256']
        if (normalization_semantic_sha(binding['norm_stats_path']) != expected
                or normalization_semantic_sha(binding['source_norm_stats_path']) != expected):
            raise ValueError('Pinned normalization numeric bits changed')
    else:
        raise ValueError('Invalid weight mode')
