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
    config=json.loads((root/'online_subtask/configs/mainhist93_tok65536_steady_ssr_v1.json').read_text());gen=config['generator']
    for key,pin in [('prompt','prompt_sha256'),('training_reference','training_reference_sha256')]:
        if sha(root/'online_subtask'/gen[key])!=gen[pin]:raise ValueError('SSR prompt/reference mismatch')
    if (config['policy']['ecot_dropout_probability']!=0.0 or gen['API_calls_per_replan']!=2
            or gen['max_tokens']!=65536 or config['observations']['maximum_total_images']!=96):
        raise ValueError('SSR contract differs')
    for row in manifest['unchanged_native_files']:
        if sha(root/row['path'])!=row['sha256']:raise ValueError('Original native action/camera/clock code changed')
    print(json.dumps({'SSR_source_files':len(seen),'native_files_unchanged':len(manifest['unchanged_native_files']),
        'training_reference_examples':gen['training_reference_examples'],'public_SSR_enabled':True,
        'API_GPU_native_acceptance_claimed':False,'network_or_models_called':False}))

if __name__=='__main__':main()
