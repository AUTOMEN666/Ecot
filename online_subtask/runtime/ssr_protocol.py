"""Prospective three-field annotations with exact original SSR serialization."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from multiview_protocol import INSTRUCTION,canonical,resolve,strict_json

CONFIG_ID='mainhist93_tok65536_steady_ssr_v1'
NAMESPACE='seed1_native50_steady_ssr_mainhist93_65536_20261113_v1'
CONTEXT_FIELDS=['subtask','action.summary','reasoning']
HORIZON_STEPS=16
HERE=Path(__file__).resolve().parent
SSR_WRAPPER=('\nCurrent-frame action summary, subtask, and reasoning annotations: '
    'use the current observations, task, and supplied annotations. Missing or null fields are unspecified. '
    'No text response is requested.\nContext: ')
# Replaced with the exact source-bound file digest during preparation.
REFERENCE_SHA='45acaa8f26b03d85d16b2a49593b0895c7583e1a8a2a5ed8eaaa7958e97f247d'

def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def validate_context(context):
    if type(context)is not dict or set(context)!={'subtask','action','reasoning'}:
        raise ValueError('SSR requires exactly subtask, action.summary and reasoning')
    if type(context['action'])is not dict or set(context['action'])!={'summary'}:
        raise ValueError('SSR action must contain exactly summary')
    for name,value,limit in [('subtask',context['subtask'],512),('reasoning',context['reasoning'],1200),
                             ('action.summary',context['action']['summary'],400)]:
        if (type(value)is not str or not value.strip()or len(value)>limit
                or any(ord(char)<32 for char in value)):
            raise ValueError('Nonempty bounded public annotation text required: '+name)
    return deepcopy(context)

def serialize_context(context):
    context=validate_context(context)
    ordered={'subtask':context['subtask'],'action':{'summary':context['action']['summary']},'reasoning':context['reasoning']}
    return json.dumps(ordered,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def compose_prompt(instruction,context):
    if instruction!=INSTRUCTION:raise ValueError('Exact classify_objects instruction required')
    return instruction+SSR_WRAPPER+serialize_context(context)

def resolve_ssr(text,catalog):
    value=strict_json(text)
    if type(value)is not dict or set(value)!={'subtask_id','reasoning','action'}:
        raise ValueError('Exact public subtask_id/reasoning/action.summary response required')
    subtask,resolution=resolve(canonical({'subtask_id':value['subtask_id']}),catalog)
    context={'subtask':subtask['subtask'],'action':deepcopy(value['action']),'reasoning':value['reasoning']}
    validate_context(context)
    return context,dict(resolution,context_fields=CONTEXT_FIELDS,SSR_public_fields_preserved=True)

def load_reference():
    raw=(HERE/'ssr_training_reference.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA:raise ValueError('Training reference bytes changed')
    data=strict_json(raw)
    if (data.get('schema')!='ecot.training_reference.exact_examples.v1' or data.get('reference_only')is not True
            or data.get('trajectory_id')!='training_episode200' or len(data.get('examples',[]))!=7
            or data.get('source_training_projection_sha256')!='7ea5a9b2d3f7c0e05ad97f8880c3596a1deea6a52d1e9ee857b4f064351a8156'):
        raise ValueError('Source-bound original classify_objects training examples required')
    for item in data['examples']:
        if item.get('episode')!=200 or type(item.get('frame'))is not int or not 0<=item['frame']<=745:
            raise ValueError('Reference must be an actual training start, not an annotation-only tail')
        validate_context(item['context'])
    return data

def validate_spec(spec):
    required={'config_id':CONFIG_ID,'protocol_config_id':CONFIG_ID,'generator_protocol':CONFIG_ID,
        'steady_subtask_enabled':True,'ssr_enabled':True,'provider':'deepseek',
        'condition':'online_ssr','effective_eval_ecot':True,'task':'classify_objects','instruction':INSTRUCTION,
        'protocol_mode':'expanded_catalog_id_with_ssr','isolated_multiview_probe':True,
        'max_tokens':65536,'maximum_past_timepoints':93,'maximum_total_images':96,
        'past_history_cameras':['cam_high'],'selector_timeout_seconds':600,
        'selector_retry_budget_seconds':1850,'policy_rpc_timeout_seconds':2100,
        'executed_subtask_history_enabled':True,'executed_history_protocol':'ecot.actual_executed_subtask_history.v1',
        'context_fields':CONTEXT_FIELDS,'ecot_dropout_probability':0.0,'noise_seed':42,
        'predict_steps':50,'execute_steps':8,'flow_steps':10,'action_hz':25,'action_limit':1100,
        'reference_sha256':REFERENCE_SHA,'prospective_horizon_steps':HORIZON_STEPS}
    if any(spec.get(key)!=value for key,value in required.items()):raise ValueError('Explicit noDropout classify_objects SSR protocol differs')
    for key,value in required.items():
        if type(value)is bool and spec.get(key)is not value:raise ValueError('Explicit boolean required: '+key)
    if type(spec.get('ecot_dropout_probability'))not in (int,float):raise ValueError('Numeric dropout probability required')
    from multiview_protocol import CATALOG_SHA
    if spec.get('catalog_sha256')!=CATALOG_SHA:raise ValueError('Exact288 catalogue binding required')
    if NAMESPACE not in Path(spec.get('audit_path','')).parts:raise ValueError('New SSR run namespace required')
    if type(spec.get('step'))is not int or spec['step']<=0:raise ValueError('Explicit genuine EMA checkpoint step required')
    if not isinstance(spec.get('weights_binding_file'),str)or not spec['weights_binding_file']:
        raise ValueError('Genuine noDropout SSR weights binding file required')
    load_reference()
    return {'config_id':CONFIG_ID,'ssr_enabled':True,'steady_subtask_enabled':True,
            'maximum_past_timepoints':93,'maximum_total_images':96}
