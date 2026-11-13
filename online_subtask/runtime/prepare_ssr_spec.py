"""Prepare an explicit causal SSR run from a genuine noDropout weights binding.

This creates a new specification only. No API, model, simulator or GPU is started.
"""
import argparse
import json
from pathlib import Path
from export_binding import read
from multiview_protocol import CATALOG_SHA,INSTRUCTION
from ssr_protocol import CONFIG_ID,CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,validate_spec

def build(binding_file,*,parent_root,audit_path,config_id=CONFIG_ID):
    binding=read(binding_file)
    result=dict(config_id=config_id,protocol_config_id=config_id,generator_protocol=config_id,
        parent_root=str(Path(parent_root)),weights_binding_file=str(Path(binding_file)),step=binding.get('step'),
        steady_subtask_enabled=True,ssr_enabled=True,provider='deepseek',condition='online_ssr',effective_eval_ecot=True,
        task='classify_objects',instruction=INSTRUCTION,protocol_mode='expanded_catalog_id_with_ssr',
        isolated_multiview_probe=True,max_tokens=65536,maximum_past_timepoints=93,maximum_total_images=96,
        past_history_cameras=['cam_high'],selector_timeout_seconds=600,selector_retry_budget_seconds=1850,
        policy_rpc_timeout_seconds=2100,executed_subtask_history_enabled=True,
        executed_history_protocol='ecot.actual_executed_subtask_history.v1',context_fields=CONTEXT_FIELDS,
        ecot_dropout_probability=0.0,noise_seed=42,predict_steps=50,execute_steps=8,flow_steps=10,
        action_hz=25,action_limit=1100,reference_sha256=REFERENCE_SHA,prospective_horizon_steps=HORIZON_STEPS,
        catalog_sha256=CATALOG_SHA,audit_path=str(audit_path))
    if config_id==CONFIG_ID:validate_spec(result)
    else:
        from grounded_ssr_protocol import CONFIG_ID as GROUNDED_ID,SCENE_PROTOCOL,MOTION_PROTOCOL,validate_spec as grounded_validate
        from strict_grounded_ssr_protocol import CONFIG_ID as STRICT_ID,SCHEMA_FACTORY_SHA,validate_spec as strict_validate
        if config_id not in(GROUNDED_ID,STRICT_ID):raise ValueError('Unknown explicit SSR protocol')
        result.update(grounded_action_constraints=True,scene_evidence_protocol=SCENE_PROTOCOL,motion_evidence_protocol=MOTION_PROTOCOL)
        if config_id==STRICT_ID:
            result.update(response_format_mode='json_schema_strict',response_schema_factory_sha256=SCHEMA_FACTORY_SHA)
            strict_validate(result)
        else:grounded_validate(result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-binding',type=Path,required=True);p.add_argument('--parent-root',type=Path,required=True)
    p.add_argument('--audit-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=['steady_ssr_v1','grounded_ssr_v2','grounded_strict_ssr_v3'],default='grounded_strict_ssr_v3')
    args=p.parse_args()
    from grounded_ssr_protocol import CONFIG_ID as GROUNDED_ID
    from strict_grounded_ssr_protocol import CONFIG_ID as STRICT_ID
    spec=build(args.weights_binding,parent_root=args.parent_root,audit_path=args.audit_path,
               config_id={'steady_ssr_v1':CONFIG_ID,'grounded_ssr_v2':GROUNDED_ID,'grounded_strict_ssr_v3':STRICT_ID}[args.protocol])
    import sys
    sys.path[:0]=[str(args.parent_root/'src'),str(args.parent_root/'evaluation')]
    from ssr_binding import load_weights
    load_weights(spec)
    with args.output.open('x')as stream:json.dump(spec,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
