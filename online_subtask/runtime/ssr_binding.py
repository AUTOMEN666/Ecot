"""Admit genuine existing noDropout SSR EMA weights without relabeling them."""
from pathlib import Path
from export_binding import read
from ssr_protocol import CONTEXT_FIELDS,compose_prompt,serialize_context

def validate_identity(binding,spec,config):
    identity=binding.get('checkpoint_metadata',{}).get('identity',{})
    if (binding.get('weights_mode')!='checkpoint' or binding.get('inference_params')!='EMA'
            or type(binding.get('step'))is not int or binding['step']!=spec['step']
            or binding.get('ecot_variant')!='summary_subtask_reasoning'
            or binding.get('train_ecot')is not True or binding.get('eval_ecot')is not False
            or identity.get('train_ecot')is not True or identity.get('eval_ecot')is not False
            or identity.get('max_token_len')!=512
            or config.get('ecot_variant')!='summary_subtask_reasoning'
            or type(config.get('ecot_dropout_probability'))not in (int,float)
            or config.get('ecot_dropout_probability')!=0.0
            or config.get('ecot_retained_fields')!=CONTEXT_FIELDS):
        raise ValueError('A genuine noDropout three-field SSR EMA checkpoint is required')
    root=Path(spec['parent_root']).resolve(strict=True)
    if Path(binding['train_config_path']).resolve(strict=True)!=root/'configs/train.json':
        raise ValueError('SSR external training root differs from the original sealed weight binding')
    return identity

def load_weights(spec):
    from support.weights_contract import validate_weights
    binding=read(spec['weights_binding_file'])
    # Keep the genuine source, normalization, metadata and parameter binding rules.
    validate_weights(binding)
    config=read(binding['train_config_path'])
    validate_identity(binding,spec,config)
    return binding

def actual_training_prompt(instruction,context):
    from strict_prompt import compose_training_prompt
    actual=compose_training_prompt(instruction,serialize_context(context))
    if actual!=compose_prompt(instruction,context):
        raise ValueError('SSR policy prompt differs byte-for-byte from the actual training serializer')
    return actual
