"""Prepare private original-SDK seed0/layout0 sim side of split CPU-policy/H20 test.

No policy/checkpoint needed on H20. Builds only a private source namespace and argv.
Existing H20 native Python/experience/library environment are supplied by Root.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from namespace_fiveseed import build
from seed_namespace_validation import validate_namespace
from rpc_contract import patch_private_namespace,validate_rpc_namespace
HERE=Path(__file__).resolve().parent

def prepare(a):
    source=a.sdk_root.resolve(strict=True);out=a.output.resolve()
    if 'online96_10hz_20261112'not in out.parts:raise ValueError('New isolated comparison namespace required')
    if a.seed!=0 or a.layout!=0:raise ValueError('Bounded split pilot fixed to unscreened original seed0/layout0')
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in range(20):
        p=source/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/f'classify_objects_{i}.json'
        rows.append({'seed':0,'layout_id':i,'path':str(p.resolve(strict=True)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'schema':'ecot.original_seed0_layout_pool.split_pilot.v1','task':'classify_objects','sdk_source':str(source),
        'scene_seeds':[0],'layout_ids':list(range(20)),'entries':rows,'native_100_claimed':False,'policy_or_outcome_screening':False}
    alias='Pi05Online8_'+hashlib.sha256(str(out).encode()).hexdigest()[:16]
    namespace=build(out/'sdk',[0],HERE/'pi05_policy',alias,seed=0,manifest=manifest)
    patch_private_namespace(namespace);validate_namespace(namespace,manifest);validate_rpc_namespace(namespace)
    sdk=Path(namespace['private_source'])
    env={'PI05_ONLINE_LANE_OUTPUT':str(out),'PI05_ENGINEERING_COMMAND_LIMIT':str(a.command_limit),
        'EVAL_NUM':'1','ROBODOJO_SAVE_EVAL_VIDEOS':'true','ROBODOJO_SAVE_EVAL_VIDEOS_MAX_LAYOUTS':'1',
        'ROBODOJO_RUN_ID':out.name,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':a.sim_gpu_uuid}
    argv=[str(a.sim_python),str(sdk/'src/eval_client/main.py'),'--task_name','classify_objects','--num_envs','1','--env_cfg_type','arx_x5',
        '--policy_name',alias,'--port',str(a.port),'--host','127.0.0.1','--seed','0','--additional_info',out.name,
        '--headless','--enable_cameras','--device_id','0']
    if a.experience:argv+=['--experience',str(a.experience)]
    plan={'schema':'ecot.h20_sim_split.expanded288_online96_10hz.v1','argv':argv,'cwd':str(sdk),'required_environment_overrides':env,
        'namespace':namespace,'layout_manifest':manifest,'launched':False,'GPU_admission_done':False,
        'additional_existing_native_environment_required':True,'policy_server_endpoint':'127.0.0.1:'+str(a.port),
        'policy_GPU_on_H20':False,'engineering_command_limit':a.command_limit,'engineering_stop_is_native_terminal':False}
    (out/'sim_launch.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'launch':str(out/'sim_launch.json'),'launched':False}))
    return plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--sdk-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-python',type=Path,required=True);p.add_argument('--sim-gpu-uuid',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--seed',type=int,choices=[0],default=0);p.add_argument('--layout',type=int,choices=[0],default=0)
    p.add_argument('--command-limit',type=int,choices=[0,128],default=128);p.add_argument('--experience',type=Path)
    a=p.parse_args()
    if not 1024<=a.port<=65535:raise ValueError('Unprivileged port required')
    prepare(a)
if __name__=='__main__':main()
