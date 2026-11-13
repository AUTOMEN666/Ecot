"""Exact historical C5000 seed1/layout0..49 bytes; no generated layouts."""
from pathlib import Path
from export_binding import read,sha,canonical_sha

def validate_manifest(path,*,check_files=True,expected_sha256=None):
    path=Path(path).resolve(strict=True)
    if expected_sha256 is not None and sha(path)!=expected_sha256:raise ValueError('Frozen manifest SHA differs')
    d=read(path)
    if any(d.get(k)!=v for k,v in {'schema':'ecot.original_C5000_seed1_native50.v1','task':'classify_objects','scene_seeds':[1],'layout_ids':list(range(50)),'episodes':50,'noise_seed':42}.items()):raise ValueError('Exact original seed1/native50 required')
    baseline=read(d['baseline_plan_path'])
    if sha(d['baseline_plan_path'])!=d['baseline_plan_sha256'] or (baseline['seed'],baseline['noise_seed'],baseline['step'],baseline['ablation_variant'])!=(1,42,5000,'C_no_ecot'):raise ValueError('Historical baseline plan differs')
    original=sorted([r for lane in baseline['lanes']for r in lane['namespace']['layouts']],key=lambda r:r['layout_id'])
    expected=[dict(r,seed=1)for r in original]
    if d['entries']!=expected or [r['layout_id']for r in expected]!=list(range(50)):raise ValueError('Historical baseline50 bytes/IDs differ')
    for r in expected:
        current=Path(d['sdk_source'])/'Assets/Eval_Layout/RoboDojo/arx_x5/1'/('classify_objects_'+str(r['layout_id'])+'.json')
        if current.resolve(strict=True)!=Path(r['path']).resolve(strict=True):raise ValueError('Original SDK asset path differs')
        if check_files and sha(r['path'])!=r['sha256']:raise ValueError('Original baseline layout changed')
    if d['common_layout_digest']!=canonical_sha([{k:r[k]for k in ('seed','layout_id','sha256')}for r in expected]):raise ValueError('Global50 digest differs')
    return d

def rows_for(manifest,seed,ids):
    if type(seed)is not int or seed!=1 or not ids or len(ids)!=len(set(ids))or any(type(i)is not int or i not in range(50)for i in ids):raise ValueError('Exact seed1/original50 subset required')
    table={r['layout_id']:r for r in manifest['entries']}
    return [dict(table[i])for i in ids]
