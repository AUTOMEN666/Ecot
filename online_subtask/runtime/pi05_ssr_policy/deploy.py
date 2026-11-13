"""SSR native SDK adapter; action loop, clocks, recorder and freeze code retained."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from pi05_policy.physics_clock import read_physics_clock,verify_clock_delta
from executed_history import ACK_SCHEMA,digest
from pi05_policy.main_camera_action25hz import MainCameraAction25Hz
from pi05_policy.freeze import snapshot,verify_static,plain,canonical

from ssr_protocol import CONFIG_ID,compose_prompt,validate_context

INSTRUCTION='Sort the objects by category into the three baskets.'

def history_source_frame(index):
    return 5*index//2


def append(name,row):
    p=Path(os.environ['PI05_ONLINE_LANE_OUTPUT'])/name
    with p.open('a') as f:f.write(canonical(row).decode()+'\n')


def current_observation(obs,n,clock):
    if obs['instruction']!=INSTRUCTION:raise ValueError('Original task Instruction changed')
    vision=obs.get('images',obs.get('vision',{}));images={}
    aliases={'cam_high':('cam_high','cam_head','head_camera','top_camera'),
             'cam_left_wrist':('cam_left_wrist','left_camera'),
             'cam_right_wrist':('cam_right_wrist','right_camera')}
    for camera,names in aliases.items():
        key=next((name for name in names if name in vision),None)
        if key is None:raise ValueError('Missing current camera '+camera)
        value=vision[key]
        if isinstance(value,dict):value=value['color'] if 'color' in value else value['rgb']
        images[camera]=deepcopy(value)
    state=obs['state']
    if isinstance(state,dict):state={k:state[k] for k in ['left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state']}
    return {'state':deepcopy(state),'images':images,'instruction':INSTRUCTION,
            'frame_index':n,'physics_clock':deepcopy(clock)}


def replan(env,client,n):
    obs=env.get_obs();before=snapshot(env,obs);started=time.monotonic();injection=None
    try:
        client.call(func_name='update_obs',obs=current_observation(obs,n,read_physics_clock(env)))
        actions=client.call(func_name='get_action')
        injection=client.call(func_name='current_injection')
        validate_context(injection['context'])
        expected_step=int(os.environ['PI05_SSR_CHECKPOINT_STEP'])
        from grounded_ssr_protocol import CONFIG_ID as GROUNDED_ID
        from strict_grounded_ssr_protocol import CONFIG_ID as STRICT_ID
        expected_protocol=os.environ.get('PI05_SSR_PROTOCOL_ID',CONFIG_ID)
        if expected_protocol not in(CONFIG_ID,GROUNDED_ID,STRICT_ID):raise ValueError('Unknown explicit SSR SDK protocol')
        if (expected_step<=0 or injection['execution_frame']!=n or injection['actual_injected'] is not True
                or injection['checkpoint_step']!=expected_step or injection['training_composition_byte_equal'] is not True
                or injection.get('context_protocol')!=expected_protocol
                or injection.get('lineage',{}).get('generator_protocol')!=expected_protocol
                or injection['prompt']!=compose_prompt(INSTRUCTION,injection['context'])
                or len(actions)!=8):raise ValueError('Missing exact genuine SSR checkpoint/three-field injection')
        if hashlib.sha256(injection['prompt'].encode()).hexdigest()!=injection['prompt_sha256']:
            raise ValueError('Actual prompt hash differs')
    finally:
        after=snapshot(env,env.get_obs())
        append('freeze.jsonl',{'event':'online_selector_and_policy_wait','execution_frame':n,
            'episode_id':injection['episode_id']if injection is not None else None,'before':before,'after':after,'unchanged':canonical(before)==canonical(after),
            'wait_seconds':time.monotonic()-started,'request_returned':injection is not None})
        verify_static(before,after)
        verify_clock_delta(before['physics_clock'],after['physics_clock'],0,'blocking_selector_and_policy')
    return actions,injection


def capture_history(env,client,n,source_index):
    if n!=history_source_frame(source_index):raise ValueError('Actual10Hz source boundary required')
    before=snapshot(env,env.get_obs());started=time.monotonic();receipt=None
    try:
        # Native render/readback only. Never call take_action/sim.step here.
        env.obs_manager.render_for_capture()
        obs=env.get_obs()
        receipt=client.call(func_name='record_history_observation',obs=current_observation(obs,n,read_physics_clock(env)))
        if (receipt.get('history_frame')!=n or receipt.get('capture_count')!=source_index+1
                or receipt.get('policy_called')is not False or receipt.get('RNG_advanced')is not False):
            raise ValueError('History RPC did not preserve actual source frame/policy/RNG contract')
    finally:
        after=snapshot(env,env.get_obs())
        append('history_capture_freeze.jsonl',{'event':'actual_10hz_three_camera_history_capture','history_frame':n,
            'history_source_index':source_index,'episode_id':receipt.get('episode_id')if receipt else None,
            'before':before,'after':after,'unchanged':canonical(before)==canonical(after),'wait_seconds':time.monotonic()-started,
            'history_RPC_returned':receipt is not None,'policy_called':False,'RNG_advanced':False})
        verify_static(before,after)
        verify_clock_delta(before['physics_clock'],after['physics_clock'],0,'history_capture_render_readback_RPC')
    return receipt


def acknowledge_executed_subtask(env,client,injection,execution_clocks,reason):
    # This ACK is created ONLY after take_action and the real clock/text audit rows.
    source={'schema':ACK_SCHEMA,'injection':injection,'execution_clocks':execution_clocks,'chunk_end_reason':reason}
    before=snapshot(env,env.get_obs());started=time.monotonic();receipt=None
    try:
        receipt=client.call(func_name='acknowledge_executed_subtask',obs=source)
        if (receipt.get('ack_source_sha256')!=digest(source)or receipt.get('episode_id')!=injection['episode_id']
                or receipt.get('last_executed_frame')!=execution_clocks[-1]['execution_frame']
                or receipt.get('policy_called')is not False or receipt.get('RNG_advanced')is not False):
            raise ValueError('Executed-history ACK receipt differs from actual native source')
    finally:
        after=snapshot(env,env.get_obs())
        append('executed_subtask_ack.jsonl',{'event':'actual_executed_subtask_ack','ack_source':source,
            'ack_source_sha256':digest(source),'receipt':receipt,'before':before,'after':after,
            'unchanged':canonical(before)==canonical(after),'wait_seconds':time.monotonic()-started,
            'ACK_RPC_returned':receipt is not None,'policy_called':False,'RNG_advanced':False})
        verify_static(before,after)
        verify_clock_delta(before['physics_clock'],after['physics_clock'],0,'executed_subtask_ACK_RPC')
    return receipt


def eval_one_episode(TASK_ENV,model_client):
    env=TASK_ENV;recorder=MainCameraAction25Hz(env);completed=False
    engineering_limit=int(os.environ.get('PI05_ENGINEERING_COMMAND_LIMIT','128'))
    if engineering_limit not in (0,128):raise ValueError('Only full native0 or bounded128 command limit admitted')
    model_client.call(func_name='reset')
    try:
        first=read_physics_clock(env);n=0;replans=0;history_source_index=0
        recorder.start(env.get_obs(),0)
        verify_clock_delta(first,read_physics_clock(env),0,'initial_recording_capture')
        capture_history(env,model_client,0,history_source_index);history_source_index+=1
        while not env.is_episode_end()and(not engineering_limit or n<engineering_limit):
            actions,injection=replan(env,model_client,n);replans+=1
            append('injected_text.jsonl',{'event':'actual_injection',**injection})
            executed_chunk=[]
            for action in actions:
                before=read_physics_clock(env)
                env.take_action(action);after=read_physics_clock(env)
                verify_clock_delta(before,after,10,'native_action25Hz')
                n+=1;recorder.after_action(n)
                verify_clock_delta(after,read_physics_clock(env),0,'action_recording_capture')
                if n==history_source_frame(history_source_index):
                    capture_history(env,model_client,n,history_source_index);history_source_index+=1
                execution_clock={'episode_id':injection['episode_id'],'execution_frame':n,'before':before,'after':after,
                       'native_take_action_count':int(env.take_action_cnt[0]),'action':plain(action)}
                append('execution_clock.jsonl',execution_clock)
                executed_chunk.append(execution_clock)
                append('injected_text.jsonl',{'event':'actual_executed_action_text','execution_action':n-1,
                    'observed_outcome_frame':n,'injection_execution_frame':injection['execution_frame'],
                    'episode_id':injection['episode_id'],'context':injection['context'],'context_json':injection['context_json'],
                    'prompt':injection['prompt'],'prompt_sha256':injection['prompt_sha256']})
                if env.is_episode_end()or(engineering_limit and n>=engineering_limit):break
            reason='native_terminal'if env.is_episode_end()else'engineering_stop'if engineering_limit and n>=engineering_limit else'full_chunk'
            acknowledge_executed_subtask(env,model_client,injection,executed_chunk,reason)
        final=read_physics_clock(env)
        timing=verify_clock_delta(first,final,n*10,'whole_native_episode')
        terminal={'episode_id':injection['episode_id'],'executed_commands':n,'frame_index':n,'actual_replans':replans,
            'history_capture_count':history_source_index,'history_capture_frame_rule':'floor(2.5*k)',
            'actual_environment_terminal':bool(env.is_episode_end()),'engineering_stop':bool(engineering_limit and n>=engineering_limit and not env.is_episode_end()),'engineering_command_limit':engineering_limit,'physics_timing':timing,
            'physics_steps_during_generation':0,'actual_text_synchronized':True,
            'scene_seed':int(env.eval_seed),'layout_id':int(env.current_env_seed_map[0]),
            'native_step_limit':int(env.step_lim),'engineering_probe':True,
            'diagnostic_scope':'isolated_mainhist93_tok65536_actualexecutedhistory_bounded128_or_native_full',
            'generator_protocol':injection['lineage']['generator_protocol']}
        completed=True
    finally:recording=recorder.close(completed)
    terminal['main_camera_action25hz']=recording
    append('terminal.jsonl'if terminal['actual_environment_terminal']else'engineering_stop.jsonl',terminal)
    model_client.call(func_name='record_terminal_observation',obs=terminal)


def eval_one_episode_batch(TASK_ENV,model_client):
    raise ValueError('Only one native environment is admitted')
