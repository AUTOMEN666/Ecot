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
import ast
import copy
import os
import socket

SCHEMA = 'pi05_classify_two_variants_20261112_native_ema_binding_v1'
STEPS = (5000,10000,15000,20000)
VARIANTS = ('A_subtask_online','B_ssr_dropout05_taskonly')
EXPECTED = {name: STEPS for name in VARIANTS}
FILE_VALIDATION_POLICY = 'exact_descriptive_run_metadata_content_only_v1'
HOST_RECEIPT_SHA256='614b05e846d42b7a43682d31ef5374596b6210f0cac0420428df52dea8d6e2ed'
HOST_BINDING_SHA256='5fb6fdbf3f9a7d1d55c18f71ca2320ec5c532c2c9451f3339f6659ab7b01b921'
HOST_BINDING_FILE_SHA256='8eee6d54af23643b3e3fe152cec16955eaa2451d3a71b21e9ac071c08591c26e'
HOST_DATA_AUDIT_SHA256='6cc7871bc1285a157655c5563f362c8151968ac647b43b2ebfc1530b3293506a'
HOSTNAME='is-ddqnbunddh3z4pac-devmachine-0'
HOST_FIELDS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
_HOST_CACHE=None

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


def _host_require(condition,message):
    if not condition:raise ValueError('Strict host attestation: '+message)


def _host_verified_json(path,expected_sha):
    before=fingerprint(path);raw=Path(path).read_bytes()
    _host_require(hashlib.sha256(raw).hexdigest()==expected_sha and fingerprint(path)==before,'pinned JSON bytes changed or unstable: '+str(path))
    def pairs(values):
        result={}
        for key,value in values:
            _host_require(key not in result,'duplicate pinned JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in()).throw(ValueError('Nonfinite pinned JSON '+value)))


def _host_coverage(receipt,binding,audit):
    """Validate exact original coverage; never creates a stat value or a seal."""
    _host_require(receipt.get('schema')=='ecot.original_bytes.host_stat_attestation.v1'and receipt.get('passed')is True,'wrong receipt schema/status')
    _host_require(receipt.get('hostname')==HOSTNAME==socket.gethostname(),'actual hostname differs')
    _host_require(receipt.get('original_binding_canonical_sha256')==binding.get('binding_sha256')==HOST_BINDING_SHA256,'original binding identity differs')
    _host_require(receipt.get('original_binding_file_sha256')==HOST_BINDING_FILE_SHA256
        and receipt.get('original_data_audit_sha256')==HOST_DATA_AUDIT_SHA256,'original seal byte pin differs')
    _host_require(receipt.get('original_parameter_item_tree_sha256')==binding['parameter_item_tree_sha256']
        and receipt.get('norm_stats_path')==binding['norm_stats_path']and receipt.get('norm_stats_sha256')==binding['norm_stats_sha256']
        and receipt.get('source_sha256')==binding['training_identity']['source_sha256']
        and receipt.get('original_training_identity')==binding['training_identity']
        and receipt.get('inference_params')==binding.get('inference_params')=='EMA'and receipt.get('step')==binding.get('step')==5000,'parameter/normalizer/source/EMA identity differs')
    metadata=set(binding['file_validation_policy']['content_only_metadata_paths'])
    _host_require(sorted(metadata)==receipt.get('original_content_only_metadata_paths')and len(metadata)==3,'original metadata exceptions differ')
    expected={('training_payload',row['path']):(row['sha256'],{k:row[k]for k in HOST_FIELDS})for row in audit['payload_bindings']}
    _host_require(len(expected)==400,'original400payload inventory differs')
    expected.update({('checkpoint_binding',p):(s,binding['file_fingerprints'][p])for p,s in binding['file_sha256'].items()})
    records=receipt.get('files');_host_require(isinstance(records,list)and len(records)==len(expected),'missing or extra receipt coverage')
    mapped={}
    for row in records:
        key=(row.get('domain'),row.get('path'))
        _host_require(key in expected and key not in mapped,'missing/duplicate/foreign covered path')
        expected_sha,sealed=expected[key]
        _host_require(row.get('sealed_sha256')==row.get('actual_sha256')==expected_sha and row.get('sealed_stat')==sealed,'original SHA/stat coverage differs: '+str(key))
        before,after=row.get('actual_stat_before'),row.get('actual_stat_after')
        _host_require(isinstance(after,dict)and set(after)==set(HOST_FIELDS)and before==after
            and row.get('unchanged_while_hashing')is True and row.get('sha_equal')is True,'unverified or unstable host stat')
        content_only=key[0]=='checkpoint_binding'and key[1]in metadata
        _host_require(content_only or all(after[k]==sealed[k]for k in HOST_FIELDS if k!='st_dev'),'non-device stat difference in receipt')
        _host_require(row.get('stat_policy')==('original_three_metadata_content_only'if content_only else'st_dev_only'),'unexpected stat exception policy')
        mapped[key]=row
    _host_require(set(mapped)==set(expected)and receipt.get('payload_files_verified')==400
        and receipt.get('checkpoint_binding_files_verified')==len(binding['file_sha256']),'incomplete original coverage')
    _host_require(receipt.get('allowed_non_metadata_stat_difference')==['st_dev']
        and receipt.get('all_file_before_after_and_final_stats_stable')is True
        and receipt.get('actual_host_stats_are_not_substituted_old_values')is True
        and receipt.get('old_files_or_seals_modified')is False,'receipt does not establish truthful strict device-only policy')
    return mapped


def _host_context():
    """Explicit path+raw-SHA pin, fixed real receipt, real host; otherwise strict."""
    global _HOST_CACHE
    path=os.environ.get('PI05_HOST_RECEIPT_PATH');pin=os.environ.get('PI05_HOST_RECEIPT_SHA256')
    if path is None and pin is None:return None
    _host_require(path is not None and pin==HOST_RECEIPT_SHA256,'explicit fixed receipt path and SHA required')
    receipt_path=Path(path);_host_require(receipt_path.is_absolute()and not receipt_path.is_symlink(),'receipt must be an absolute regular original file')
    _host_require(socket.gethostname()==HOSTNAME,'actual hostname differs')
    before=fingerprint(receipt_path);signature=(str(receipt_path),pin,tuple(before.items()),socket.gethostname())
    if _HOST_CACHE is not None and _HOST_CACHE['signature']==signature:
        _host_require(all(fingerprint(path)==observed for path,observed in _HOST_CACHE['bootstrap_stats'].items()),'original binding/config/audit changed after receipt admission')
        return _HOST_CACHE
    receipt=_host_verified_json(receipt_path,pin)
    binding_path=Path(receipt['original_binding_path'])
    binding=_host_verified_json(binding_path,HOST_BINDING_FILE_SHA256)
    _host_require(canonical_sha({k:v for k,v in binding.items()if k!='binding_sha256'})==binding.get('binding_sha256')==HOST_BINDING_SHA256,'original binding canonical digest changed')
    config_path=Path(binding['config_path'])
    config=_host_verified_json(config_path,binding['file_sha256'][str(config_path)]);audit_path=Path(config['data_audit'])
    audit=_host_verified_json(audit_path,HOST_DATA_AUDIT_SHA256);records=_host_coverage(receipt,binding,audit)
    _HOST_CACHE={'signature':signature,'receipt':receipt,'binding':binding,'audit_path':str(audit_path),'records':records,
        'bootstrap_stats':{str(p):fingerprint(p)for p in(binding_path,config_path,audit_path)}}
    return _HOST_CACHE


def _host_stat_allowed(path,sealed_stat,expected_sha,observed_stat,domain):
    context=_host_context()
    if context is None:return False
    key=(domain,str(path));_host_require(key in context['records'],'missing exact current-file coverage: '+str(path))
    row=context['records'][key]
    _host_require(row['sealed_stat']==sealed_stat and row['sealed_sha256']==expected_sha,'caller does not match original sealed SHA/stat')
    _host_require(set(observed_stat)==set(HOST_FIELDS)and observed_stat==fingerprint(path),'caller stat is not the actual current stat')
    _host_require(observed_stat==row['actual_stat_after'],'current file differs from fully hashed host observation: '+str(path))
    _host_require(all(observed_stat[k]==sealed_stat[k]for k in HOST_FIELDS if k!='st_dev'),'only original st_dev may differ')
    return True


def _host_payload_stat_allowed(row,actual_stat):
    return _host_stat_allowed(row['path'],{k:row[k]for k in HOST_FIELDS},row['sha256'],
        {k:getattr(actual_stat,k)for k in HOST_FIELDS},'training_payload')


def _host_adjusted_contract(module,root):
    context=_host_context()
    if context is None:return module
    _host_require(Path(context['binding']['parent_root']).resolve()==Path(root).resolve(),'original training root differs')
    source=Path(root)/'src/ablation_training_contract.py'
    expected=context['binding']['training_identity']['source_sha256']['src/ablation_training_contract.py']
    _host_require(sha(source)==expected,'original preflight source bytes changed')
    tree=ast.parse(source.read_text());original=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_preflight')
    adjusted=copy.deepcopy(original)
    def guards(function):
        return[n for n in ast.walk(function)if isinstance(n,ast.If)and any(isinstance(x,ast.Constant)
            and x.value=='Previously hashed numeric/image payload changed: 'for statement in n.body for x in ast.walk(statement))]
    candidates=guards(adjusted);_host_require(len(candidates)==1,'original payload stat guard shape changed')
    condition=candidates[0].test
    _host_require(isinstance(condition,ast.Call)and isinstance(condition.func,ast.Name)and condition.func.id=='any','original stat rejection condition changed')
    candidates[0].test=ast.BoolOp(op=ast.And(),values=[condition,ast.UnaryOp(op=ast.Not(),operand=ast.Call(
        func=ast.Name(id='_host_payload_stat_allowed',ctx=ast.Load()),args=[ast.Name(id='row',ctx=ast.Load()),ast.Name(id='stat',ctx=ast.Load())],keywords=[]))])
    restored=copy.deepcopy(adjusted);guards(restored)[0].test=copy.deepcopy(condition)
    _host_require(ast.dump(restored)==ast.dump(original),'unexpected non-stat preflight transformation')
    module.__dict__['_host_payload_stat_allowed']=_host_payload_stat_allowed
    compiled=ast.fix_missing_locations(ast.Module(body=[adjusted],type_ignores=[]))
    exec(compile(compiled,str(source)+'[isolated strict host-stat adapter]','exec'),module.__dict__)
    return module


def host_attestation_summary():
    context=_host_context()
    if context is None:return None
    return {'policy':'original_bytes_attested_st_dev_only_v1','receipt_sha256':HOST_RECEIPT_SHA256,
        'hostname':socket.gethostname(),'original_binding_sha256':HOST_BINDING_SHA256,
        'original_data_audit_sha256':HOST_DATA_AUDIT_SHA256,'covered_files':len(context['records']),
        'fingerprint_returns_actual_host_stat':True,'old_seals_or_training_sources_modified':False}

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
    return _host_adjusted_contract(mod,root)

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
    host_context=_host_context()
    if host_context is not None:_host_require(binding['binding_sha256']==HOST_BINDING_SHA256,'caller is not the exact pinned original binding')
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
        elif (before!=binding['file_fingerprints'][key]and not _host_stat_allowed(p,binding['file_fingerprints'][key],binding['file_sha256'][key],before,'checkpoint_binding'))or(full_hashes and sha(p)!=binding['file_sha256'][key]):
            raise ValueError('Bound checkpoint/source bytes changed: '+str(p))
    result={'accepted':True,'full_hashes_verified':full_hashes,'step':binding['step'],
            'descriptive_metadata_sha256_always_verified':True,
            'binding_sha256':binding['binding_sha256'],'inference_params':'EMA'}
    host=host_attestation_summary()
    if host is not None:result['host_stat_attestation']=host
    return result
