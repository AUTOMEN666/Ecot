"""Verify declared SSR source/protocol/reference bytes without models/network."""
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'online_subtask/provenance/SSR_IMPLEMENTATION_SOURCE_MAP.json').read_text());seen=set()
    for row in manifest['files']:
        rel=Path(row['path']);path=root/rel
        if rel.is_absolute()or '..'in rel.parts or str(rel)in seen:raise ValueError('Invalid SSR source path')
        if path.is_symlink()or not path.is_file()or sha(path)!=row['sha256']:raise ValueError('SSR source changed: '+str(rel))
        seen.add(str(rel))
    active='mainhist93_tok65536_grounded_strict_ssr_v4'
    if manifest['current_protocol']!=active:raise ValueError('Active grounded protocol differs from source map')
    for name in ['mainhist93_tok65536_steady_ssr_v1','mainhist93_tok65536_grounded_ssr_v2','mainhist93_tok65536_grounded_strict_ssr_v3',active]:
        config=json.loads((root/('online_subtask/configs/'+name+'.json')).read_text());gen=config['generator']
        pins=[('prompt','prompt_sha256'),('training_reference','training_reference_sha256')]
        if 'observer_prompt'in gen:pins.append(('observer_prompt','observer_prompt_sha256'))
        for key,pin in pins:
            if sha(root/'online_subtask'/gen[key])!=gen[pin]:raise ValueError('SSR prompt/reference mismatch: '+name)
        observations=config['observations'];policy=config['policy'];timeouts=config['timeouts']
        if (config['config_id']!=name or policy['ecot_dropout_probability']!=0.0 or gen['API_calls_per_replan']!=2
                or gen['max_tokens']!=65536 or gen['temperature']!=0 or gen['thinking']!={'type':'enabled'}
                or gen['reasoning_effort']!='high' or gen['catalog_count']!=288
                or observations['maximum_total_images']!=96 or observations['maximum_past_timepoints']!=93
                or observations['past_cameras']!=['cam_high'] or observations['state_dim']!=14
                or (policy['predict_steps'],policy['execute_steps'],policy['flow_steps'],policy['noise_seed'],policy['action_hz'],policy['action_limit'])!=(50,8,10,42,25,1100)
                or (timeouts['HTTP_absolute_seconds'],timeouts['shared_generation_budget_seconds'],timeouts['maximum_identical_request_attempts'],timeouts['policy_RPC_seconds'])!=(600,1850,3,2100)):
            raise ValueError('SSR contract differs: '+name)
    if (gen['response_format_mode']!='json_schema_strict' or gen['fallback_to_json_object']is not False
            or sha(root/'online_subtask'/gen['response_schema_factory'])!=gen['response_schema_factory_sha256']):
        raise ValueError('Strict API response schema factory or no-fallback contract differs')
    if sha(root/'online_subtask'/gen['response_schema_base_factory'])!=gen['response_schema_base_factory_sha256']:
        raise ValueError('Original strict schema base factory changed')
    if gen['descriptor_or_SDK_to_category_mapping']is not False or gen['category_null_for_unknown']is not True:
        raise ValueError('Category tokens cannot become an automatic visual truth mapping')
    if gen['host_rewrites_public_annotations']is not False or gen['quality_acceptance_passed']is not False:
        raise ValueError('Experimental quality boundaries changed')
    for row in manifest['unchanged_native_files']:
        if sha(root/row['path'])!=row['sha256']:raise ValueError('Original native action/camera/clock code changed')
    print(json.dumps({'SSR_source_files':len(seen),'native_files_unchanged':len(manifest['unchanged_native_files']),
        'training_reference_examples':gen['training_reference_examples'],'public_SSR_enabled':True,'active_protocol':active,
        'API_GPU_native_acceptance_claimed':False,'network_or_models_called':False}))

if __name__=='__main__':main()
