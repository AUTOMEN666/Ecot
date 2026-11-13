"""Prepare an explicit causal SSR run from a genuine noDropout weights binding.

This creates a new specification only. No API, model, simulator or GPU is started.
"""
import argparse
import json
from pathlib import Path
from export_binding import read
from multiview_protocol import CATALOG_SHA,INSTRUCTION
from ssr_protocol import CONFIG_ID,CONTEXT_FIELDS,HORIZON_STEPS,REFERENCE_SHA,validate_spec

def build(binding_file,*,parent_root,audit_path):
    binding=read(binding_file)
    result=dict(config_id=CONFIG_ID,protocol_config_id=CONFIG_ID,generator_protocol=CONFIG_ID,
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
    validate_spec(result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-binding',type=Path,required=True);p.add_argument('--parent-root',type=Path,required=True)
    p.add_argument('--audit-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();spec=build(args.weights_binding,parent_root=args.parent_root,audit_path=args.audit_path)
    import sys
    sys.path[:0]=[str(args.parent_root/'src'),str(args.parent_root/'evaluation')]
    from ssr_binding import load_weights
    load_weights(spec)
    with args.output.open('x')as stream:json.dump(spec,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
