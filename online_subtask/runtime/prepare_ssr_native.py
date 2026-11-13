"""Prepare one private seed1 native SSR acceptance case; launches no GPU/API."""
import argparse,hashlib,json
from pathlib import Path
from export_binding import read,sha
from layout_manifest import validate_manifest
from namespace_fiveseed import build
from rpc_contract import patch_private_namespace,validate_rpc_namespace
from seed_namespace_validation import validate_namespace
from ssr_protocol import CONFIG_ID,NAMESPACE,validate_spec
HERE=Path(__file__).resolve().parent

def prepare(a):
    spec=read(a.model_config)
    from grounded_ssr_protocol import CONFIG_ID as GROUNDED_ID,NAMESPACE as GROUNDED_NAMESPACE,validate_spec as validate_grounded
    from strict_grounded_ssr_protocol import CONFIG_ID as STRICT_ID,NAMESPACE as STRICT_NAMESPACE,validate_spec as validate_strict
    from enum_grounded_ssr_protocol import CONFIG_ID as ENUM_ID,NAMESPACE as ENUM_NAMESPACE,validate_spec as validate_enum
    from phase_grounded_ssr_protocol import CONFIG_ID as PHASE_ID,NAMESPACE as PHASE_NAMESPACE,validate_spec as validate_phase
    from release_grounded_ssr_protocol import CONFIG_ID as RELEASE_ID,NAMESPACE as RELEASE_NAMESPACE,validate_spec as validate_release
    grounded=spec.get('config_id')==GROUNDED_ID
    strict=spec.get('config_id')==STRICT_ID
    if spec.get('config_id')==RELEASE_ID:validate_release(spec);protocol_namespace=RELEASE_NAMESPACE
    elif spec.get('config_id')==PHASE_ID:validate_phase(spec);protocol_namespace=PHASE_NAMESPACE
    elif spec.get('config_id')==ENUM_ID:validate_enum(spec);protocol_namespace=ENUM_NAMESPACE
    elif strict:validate_strict(spec);protocol_namespace=STRICT_NAMESPACE
    elif grounded:validate_grounded(spec);protocol_namespace=GROUNDED_NAMESPACE
    else:validate_spec(spec);protocol_namespace=NAMESPACE
    manifest=validate_manifest(a.layout_manifest)
    if not a.sim_python.is_file()or not a.sim_gpu_uuid.startswith('GPU-'):
        raise ValueError('Existing native interpreter and explicit physical GPU UUID required')
    out=a.output.resolve()
    if protocol_namespace not in out.parts:raise ValueError('Isolated SSR native namespace required')
    if a.command_limit!=128:raise ValueError('This engineering preparer permits exactly128 commands')
    out.mkdir(parents=True,exist_ok=False)
    alias='Pi05SSR_'+hashlib.sha256(str(out).encode()).hexdigest()[:16]
    namespace=build(out/'sdk',[a.layout],HERE/'pi05_ssr_policy',alias,seed=1,manifest=manifest)
    patch_private_namespace(namespace);validate_namespace(namespace,manifest);validate_rpc_namespace(namespace)
    sdk=Path(namespace['private_source'])
    env={'PI05_ONLINE_LANE_OUTPUT':str(out),'PI05_ENGINEERING_COMMAND_LIMIT':'128',
        'PI05_SSR_CHECKPOINT_STEP':str(spec['step']),'EVAL_NUM':'1',
        'PI05_SSR_PROTOCOL_ID':spec['config_id'],
        'ROBODOJO_SAVE_EVAL_VIDEOS':'true','ROBODOJO_SAVE_EVAL_VIDEOS_MAX_LAYOUTS':'1',
        'ROBODOJO_RUN_ID':out.name,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':a.sim_gpu_uuid}
    argv=[str(a.sim_python),str(sdk/'src/eval_client/main.py'),'--task_name','classify_objects',
        '--num_envs','1','--env_cfg_type','arx_x5','--policy_name',alias,'--port',str(a.port),
        '--host','127.0.0.1','--seed','1','--additional_info',out.name,'--headless','--enable_cameras','--device_id','0']
    if a.experience:argv+=['--experience',str(a.experience)]
    plan={'schema':'ecot.classify_objects.SSR.native_engineering_preparation.v1','config_id':spec['config_id'],
        'argv':argv,'cwd':str(sdk),'required_environment_overrides':env,'namespace':namespace,
        'model_config':str(a.model_config.resolve()),'model_config_sha256':sha(a.model_config),
        'layout_manifest':str(a.layout_manifest.resolve()),'layout_manifest_sha256':sha(a.layout_manifest),
        'checkpoint_step':spec['step'],'scene_seed':1,'layout':a.layout,'engineering_command_limit':128,
        'policy_adapter':'pi05_ssr_policy','policy_adapter_sha256':sha(HERE/'pi05_ssr_policy/deploy.py'),
        'launched':False,'GPU_admission_done':False,'SDK_outcome':None,'formal_episode':False,
        'existing_native_environment_and_platform_device_acceptance_required':True}
    (out/'sim_launch.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'launch':str(out/'sim_launch.json'),'launched':False,'checkpoint_step':spec['step']}))
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--layout-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-python',type=Path,required=True);p.add_argument('--sim-gpu-uuid',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--layout',type=int,choices=range(50),default=0)
    p.add_argument('--command-limit',type=int,choices=[128],default=128);p.add_argument('--experience',type=Path)
    a=p.parse_args()
    if not 1024<=a.port<=65535:raise ValueError('Unprivileged port required')
    prepare(a)
