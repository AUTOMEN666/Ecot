"""Prepare exact original A5000 CPU policy spec/argv; no process or GPU launched."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from export_binding import read,validate_binding
from multiview_protocol import INSTRUCTION,CATALOG_SHA,PROTOCOLS,PROTOCOL_CONFIG_ID
HERE=Path(__file__).resolve().parent

def prepare(a):
    parent=a.parent_root.resolve(strict=True);binding_path=a.export_binding.resolve(strict=True)
    binding=read(binding_path);validate_binding(binding,full_hashes=False)
    if binding['step']!=5000 or binding['ablation_variant']!='A_subtask_online':raise ValueError('Exact original own A EMA5000 required')
    out=a.output.resolve()
    if 'seed1_native50_mainhistory93_65536_20261113_v1'not in out.parts:raise ValueError('Explicit isolated output namespace required')
    out.mkdir(parents=True,exist_ok=False)
    spec={'parent_root':str(parent),'condition':'online_subtask','export_binding':str(binding_path),'step':5000,
        'ablation_variant':'A_subtask_online','noise_seed':42,'instruction':INSTRUCTION,'audit_path':str(out/'actions.jsonl'),
        'effective_eval_ecot':True,'provider':a.provider,'isolated_multiview_probe':True,'protocol_mode':'expanded_catalog_id',
        'generator_protocol':PROTOCOLS['expanded_catalog_id'],'catalog_sha256':CATALOG_SHA,'catalog_episode':None,
        'relay_url':a.relay_url,'selector_timeout_seconds':600,'selector_retry_budget_seconds':1850,'policy_rpc_timeout_seconds':2100,
        'config_id':PROTOCOL_CONFIG_ID,'protocol_config_id':PROTOCOL_CONFIG_ID,'max_tokens':65536,'past_history_cameras':['cam_high'],'maximum_total_images':96,'maximum_past_timepoints':93,
        'executed_subtask_history_enabled':True,'executed_history_protocol':'ecot.actual_executed_subtask_history.v1',
        'cpu_policy_test':True,'cadence_execution_frames':8,'native_actions_changed':False}
    (out/'model.json').write_text(json.dumps(spec,indent=2)+'\n')
    argv=[str(a.policy_python),'-B',str(HERE/'model_server.py'),'--config',str(out/'model.json'),'--port',str(a.port),'--ready-output',str(out/'ready.json')]
    # This is the allowlisted nonsecret override, never an inherited environment dump.
    env={'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled','OMP_NUM_THREADS':str(a.cpu_threads),
        'OPENBLAS_NUM_THREADS':str(a.cpu_threads),'MKL_NUM_THREADS':str(a.cpu_threads),
        'JAX_COMPILATION_CACHE_DIR':str(out/'cache/jax'),'OPENPI_DATA_HOME':str(parent/'cache/openpi')}
    vendor=Path(read(parent/'configs/train.json')['vendor_root'])
    sdk=a.sdk_root.resolve(strict=True)
    env['PYTHONPATH']=os.pathsep.join(map(str,[HERE,parent/'src',vendor/'src',vendor/'packages/openpi-client/src',vendor.parent/'lerobot',parent/'evaluation_native',sdk,sdk/'XPolicyLab']))
    plan={'schema':'ecot.cpu_policy_split.mainhist93_tok65536.v1','argv':argv,'cwd':str(parent),'environment_overrides':env,
        'inherited_environment_serialized':False,'launched':False,'policy_backend':'cpu','simulator_is_separate_H20_process':True,
        'binding_sha256':binding['binding_sha256'],'export_path':binding['export_path'],'new_policy_weights_or_decoding':False}
    (out/'policy_launch.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'spec':str(out/'model.json'),'launch':str(out/'policy_launch.json'),'launched':False}))
    return plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--export-binding',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--provider',choices=['deepseek','codex'],required=True)
    p.add_argument('--policy-python',type=Path,required=True);p.add_argument('--sdk-root',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--relay-url',default='http://127.0.0.1:28782/select');p.add_argument('--cpu-threads',type=int,default=8)
    a=p.parse_args()
    if not 1024<=a.port<=65535 or not 1<=a.cpu_threads<=32:raise ValueError('Bounded unprivileged port/CPU threads required')
    prepare(a)
if __name__=='__main__':main()
