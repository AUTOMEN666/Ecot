"""Own-run native Orbax inference binding; no historical evaluation receipt.

The native checkpoint's params item contains the EMA exported by save_state.
The full train_state stays untouched. Every payload and application identity is
sealed before sampling and checked again after it. This is not a restore test.

The trainer atomically rewrites three descriptive run JSON files on resume.
Their original sealed byte hashes always remain mandatory; only their stat
fingerprints may differ. All checkpoint, source and other files remain strict.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys
import time

SCHEMA = 'pi05_classify_two_variants_20261112_native_ema_binding_v1'
STEPS = (5000,10000,15000,20000)
VARIANTS = ('A_subtask_online','B_ssr_dropout05_taskonly')
EXPECTED = {name: STEPS for name in VARIANTS}
FILE_VALIDATION_POLICY = 'exact_descriptive_run_metadata_content_only_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate JSON key: '+k)
            result[k]=v
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))

def fingerprint(path):
    s=Path(path).stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def file_validation_policy(run):
    """Derive the only stat exceptions from inspect's real spec output_dir.

    This exact policy is reconstructed during every validation. A binding or
    caller cannot mark a parameter/source/foreign path as descriptive metadata.
    """
    return {'scheme':FILE_VALIDATION_POLICY,
        'content_only_metadata_paths':sorted(str(Path(run)/name)for name in
            ('run_identity.json','resolved_config.json','fullparam_parameter_inventory.json')),
        'metadata_sha256_always_verified':True,
        'all_other_files':'sealed_stat_fingerprint_and_sha256_when_full_hashes_requested'}

def regular_tree(path):
    path=Path(path)
    if path.is_symlink() or not path.is_dir():raise ValueError('Native params must be a real directory')
    rows=[]
    for p in sorted(path.rglob('*')):
        mode=p.lstat().st_mode
        if stat.S_ISREG(mode):rows.append(p)
        elif not stat.S_ISDIR(mode):raise ValueError('Nonregular native parameter payload')
    if not rows or not any(p.stat().st_size for p in rows):raise ValueError('Empty native parameter payload')
    return rows

def _contract(root):
    sys.path.insert(0,str(root/'src'))
    name='_eval_subtask_dropout05_training_contract'
    spec=importlib.util.spec_from_file_location(name,root/'src/ablation_training_contract.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def validate_normalizer(path):
    data=read(path)
    if set(data)!= {'norm_stats'} or set(data['norm_stats'])!={'state','actions'}:
        raise ValueError('Actual joint14 normalizer root differs')
    for item in data['norm_stats'].values():
        if set(item)!={'mean','std','q01','q99'}:raise ValueError('Missing actual normalizer vectors')
        for values in item.values():
            if (not isinstance(values,list)or len(values)!=14 or
                    any(type(v)not in (int,float)or not math.isfinite(v)for v in values)):
                raise ValueError('Nonfinite or wrong-dimensional normalizer')
        if any(v<0 for v in item['std'])or any(a>b for a,b in zip(item['q01'],item['q99'])):
            raise ValueError('Invalid standard deviation or quantile order')
    return data

def validate_state_fingerprint(path,step):
    """Bind trainer's actual saved raw/EMA/Adam leaf hashes, no huge rehash."""
    data=read(path);leaves=data.get('leaves')
    if not isinstance(leaves,list)or not leaves or canonical_sha(leaves)!=data.get('tree_sha256'):
        raise ValueError('Saved full-state fingerprint tree is incomplete')
    paths=[]
    for row in leaves:
        name=row.get('path');digest=row.get('sha256');shape=row.get('shape')
        if (not isinstance(name,str)or not isinstance(digest,str)or len(digest)!=64 or
                any(ch not in '0123456789abcdef'for ch in digest)or
                not isinstance(shape,list)or any(type(n)is not int or n<0 for n in shape)):
            raise ValueError('Invalid saved full-state fingerprint leaf')
        paths.append(name)
    if (len(paths)!=len(set(paths))or not any('.params' in p for p in paths)
            or not any('.ema_params' in p for p in paths)or not any('.opt_state' in p for p in paths)):
        raise ValueError('Saved raw parameters, EMA and optimizer fingerprints required')
    step_rows=[r for r in leaves if r['path']=='.step']
    if len(step_rows)!=1 or step_rows[0]['shape']!=[]:raise ValueError('Actual saved step leaf missing')
    row=step_rows[0];dtype=row.get('dtype')
    if dtype not in ('int32','int64','uint32','uint64'):raise ValueError('Saved step dtype changed')
    width=4 if dtype.endswith('32')else 8
    if hashlib.sha256(step.to_bytes(width,'little',signed=not dtype.startswith('u'))).hexdigest()!=row['sha256']:
        raise ValueError('Full-state fingerprint actual step differs from metadata/config')
    return data

def inspect(config_path,step):
    config_path=Path(config_path).resolve(strict=True);root=config_path.parent.parent
    if config_path!=root/'configs/train.json':raise ValueError('Explicit variant configs/train.json required')
    if type(step)is not int or step not in STEPS:raise ValueError('Only approved 5000/10000/15000/20000 EMA milestones')
    spec=read(config_path);contract=_contract(root);contract.validate_spec(spec)
    run=Path(spec['output_dir']).resolve(strict=True)
    checkpoint=run/'checkpoints'/str(step)
    if checkpoint.is_symlink() or checkpoint.resolve(strict=True)!=checkpoint:raise ValueError('Checkpoint namespace may not redirect')
    identity=read(run/'run_identity.json');metadata=read(checkpoint/'resume_metadata.json')
    if read(run/'resolved_config.json')!=spec or identity.get('config_sha256')!=canonical_sha(spec):
        raise ValueError('Actual training config differs from evaluation config')
    requirements={
        'experiment_protocol':contract.PROTOCOL,'ablation_variant':spec['ablation_variant'],
        'seed':42,'train_ecot':True,'eval_ecot':spec['eval_ecot'],'max_token_len':512,
        'batch_size':32,'fsdp_devices':8,'full_parameter_finetuning':True,
        'frozen_vlm':False,'train_samples':69562,'steps_per_epoch':2174,
        'num_train_steps':43480,'training_end_step':20000,
        'ecot_retained_fields':spec['ecot_retained_fields'],'ecot_variant':spec['ecot_variant'],
        'prompt_version':spec['prompt_version'],'ecot_dropout_probability':spec['ecot_dropout_probability'],
        'dropout_mode':spec['dropout_mode'],'dropout_seed':42,
        'subtask_boundary_dropout':False,'subtask_action_mask':False,'episode_tail_mask':True,'text_ce_weight':0.0}
    if spec.get('ablation_variant')not in VARIANTS:raise ValueError('Only the two new independent variants')
    if any(identity.get(k)!=v for k,v in requirements.items()):raise ValueError('Actual own new training identity differs')
    from subtask_data import MASK_SCHEME, MASK_DOMAIN
    if (identity.get('ecot_dropout_mask_scheme')!=MASK_SCHEME or identity.get('ecot_dropout_mask_domain')!=list(MASK_DOMAIN)):
        raise ValueError('Actual training condition mask algorithm differs')
    if (metadata.get('identity')!=identity or metadata.get('completed_step')!=step
            or metadata.get('inference_params')!='EMA'
            or (metadata.get('epoch'),metadata.get('next_batch_in_epoch'))!=divmod(step,2174)):
        raise ValueError('Checkpoint metadata actual step/EMA/sampler identity mismatch')
    if identity.get('source_sha256')!=contract.source_hashes(root,spec):raise ValueError('Pinned training/native source changed')
    cpu=contract.validate_preflight(root,spec)
    if (identity.get('cpu_acceptance_sha256')!=sha(spec['preflight_cpu'])
            or identity.get('official_base_params_tree_sha256')!=cpu['base_params_tree_sha256']):
        raise ValueError('Own actual CPU/base identity differs')
    norm=Path(spec['norm_dir'])/'norm_stats.json'
    validate_normalizer(norm)
    manifest_keys={'dataset_manifest':'dataset_manifest_sha256',
                   'ecot_context_manifest':'ecot_context_manifest_sha256'}
    fixed=[config_path,run/'run_identity.json',run/'resolved_config.json',
           checkpoint/'resume_metadata.json',Path(spec['preflight_cpu']),norm]
    for key,wanted in manifest_keys.items():
        path=Path(spec[key])
        if sha(path)!=identity.get(wanted):raise ValueError('Training input changed: '+key)
        fixed.append(path)
    from subtask_data import source_context_cache_path
    context_cache=source_context_cache_path(spec['ecot_context_manifest'])
    if sha(context_cache)!=identity.get('ecot_context_cache_sha256'):
        raise ValueError('Actual source Subtask context cache changed')
    fixed.append(context_cache)
    data_audit=read(spec['data_audit'])
    if identity.get('subtask_projection_sha256')!=data_audit.get('subtask_projection_sha256'):
        raise ValueError('Actual original Subtask projection identity differs from CPU audit')
    if sha(norm)!=identity.get('norm_sha256'):raise ValueError('Shared actual normalizer changed')
    native=Path(spec['vendor_root'])/'src/openpi/training/checkpoints.py'
    # Native source is already SHA-bound above. Check the required export rule
    # explicitly, so a future native change cannot silently turn EMA into raw.
    native_text=native.read_text()
    if 'ema_params' not in native_text or 'params' not in native_text:raise ValueError('Native inference export rule unavailable')
    params=regular_tree(checkpoint/'params')
    if not (checkpoint/'_CHECKPOINT_METADATA').is_file():raise ValueError('Orbax checkpoint not committed')
    if not (checkpoint/'train_state').is_dir():raise ValueError('Full training checkpoint item absent')
    state_fingerprint_path=checkpoint/'state_fingerprint.json'
    state_fingerprint=validate_state_fingerprint(state_fingerprint_path,step)
    fixed.append(state_fingerprint_path)
    fullinv=run/'fullparam_parameter_inventory.json'
    inv=read(fullinv)
    imod=importlib.util.spec_from_file_location('_eval_fullparam_inventory',root/'src/fullparam_contract.py')
    inventory_module=importlib.util.module_from_spec(imod);imod.loader.exec_module(inventory_module)
    actual=inventory_module.inventory_from_rows(inv['leaves'])
    if (any(inv.get(k)!=v for k,v in actual.items())or
            actual['inventory_sha256']!=identity.get('parameter_inventory_sha256')):
        raise ValueError('Actual full-parameter inventory identity mismatch')
    fixed.append(fullinv)
    for key in identity['source_sha256']:
        fixed.append(Path(spec['vendor_root'])/key.removeprefix('vendor/') if key.startswith('vendor/') else root/key)
    paths=sorted(set(params+fixed))
    policy=file_validation_policy(run)
    if any(Path(p).is_symlink()or not Path(p).is_file()for p in policy['content_only_metadata_paths']):
        raise ValueError('Descriptive run metadata must remain regular files')
    return {'schema':SCHEMA,'parent_root':str(root),'config_path':str(config_path),
        'export_path':str(checkpoint),'source_checkpoint':str(checkpoint),'step':step,
        'inference_params':'EMA','inference_item':'params','training_identity':identity,
        'ablation_variant':spec['ablation_variant'],'norm_stats_path':str(norm),
        'norm_stats_sha256':sha(norm),'inference_only':True,
        'saved_full_state_fingerprint_sha256':sha(state_fingerprint_path),
        'saved_raw_EMA_optimizer_tree_sha256':state_fingerprint['tree_sha256'],
        'saved_full_state_fingerprint_recomputed_during_eval':False,
        'checkpoint_contains_full_train_state':True,'files':paths,
        'file_validation_policy':policy,
        'parameter_item_relative_files':[str(p.relative_to(checkpoint/'params'))for p in params]}

def create_binding(config_path,step,output_path):
    info=inspect(config_path,step);paths=info.pop('files')
    before={str(p):fingerprint(p)for p in paths};payload={str(p):sha(p)for p in paths}
    if before!={str(p):fingerprint(p)for p in paths}:raise ValueError('Checkpoint changed during byte seal')
    param_root=Path(info['export_path'])/'params'
    info.update(file_sha256=payload,file_fingerprints=before,
        parameter_item_tree_sha256=canonical_sha({str(p.relative_to(param_root)):payload[str(p)]for p in paths if p.is_relative_to(param_root)}),
        created_unix=time.time(),actual_current_payload_hashes_verified=True,
        scope='Own native checkpoint EMA bytes and actual metadata; no historical performance claim.')
    info['binding_sha256']=canonical_sha(info)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(info,f,indent=2,allow_nan=False);f.write('\n')
    return info

def validate_binding(binding,*,full_hashes=True):
    if isinstance(binding,(str,Path)):binding=read(binding)
    if binding.get('schema')!=SCHEMA or canonical_sha({k:v for k,v in binding.items()if k!='binding_sha256'})!=binding.get('binding_sha256'):
        raise ValueError('Own checkpoint binding digest changed')
    current=inspect(binding['config_path'],binding['step']);paths=current.pop('files')
    expected_fields=set(current)|{'file_sha256','file_fingerprints','parameter_item_tree_sha256',
        'created_unix','actual_current_payload_hashes_verified','scope','binding_sha256'}
    if set(binding)!=expected_fields:raise ValueError('Own checkpoint binding fields changed')
    if (set(map(str,paths))!=set(binding['file_sha256'])
            or set(binding['file_fingerprints'])!=set(binding['file_sha256'])):
        raise ValueError('Bound checkpoint/source inventory changed')
    for k,v in current.items():
        if binding.get(k)!=v:raise ValueError('Bound identity changed: '+k)
    param_root=Path(current['export_path'])/'params'
    parameter_tree=canonical_sha({str(p.relative_to(param_root)):binding['file_sha256'][str(p)]
        for p in paths if p.is_relative_to(param_root)})
    if binding['parameter_item_tree_sha256']!=parameter_tree or binding['actual_current_payload_hashes_verified']is not True:
        raise ValueError('Own checkpoint binding payload seal differs')
    metadata=set(current['file_validation_policy']['content_only_metadata_paths'])
    for p in paths:
        key=str(p);before=fingerprint(p)
        if key in metadata:
            # Even cheap pre-sampling checks must hash these three small JSON
            # files. A same-value/different-byte rewrite is a binding failure.
            if sha(p)!=binding['file_sha256'][key]or fingerprint(p)!=before:
                raise ValueError('Bound descriptive run metadata bytes changed: '+key)
        elif before!=binding['file_fingerprints'][key]or(full_hashes and sha(p)!=binding['file_sha256'][key]):
            raise ValueError('Bound checkpoint/source bytes changed: '+str(p))
    return {'accepted':True,'full_hashes_verified':full_hashes,'step':binding['step'],
            'descriptive_metadata_sha256_always_verified':True,
            'binding_sha256':binding['binding_sha256'],'inference_params':'EMA'}
