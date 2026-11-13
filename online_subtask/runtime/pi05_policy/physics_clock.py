"""Read-only native simulation clocks; never advance or alter the simulator."""
import math
import operator

SCHEMA='robodojo_native_physics_clock_v1'


def read_physics_clock(env):
    backend=getattr(env.sim,'unwrapped',env.sim)
    context=backend.sim
    dt=float(context.get_physics_dt())
    declared_dt=float(env.dt)
    decimation=operator.index(backend.cfg.decimation)
    interval=float(env.obs_manager.collect_interval)
    frequency=float(env.obs_manager.collect_freq)
    index=operator.index(context.current_time_step_index)
    counter=operator.index(backend._sim_step_counter)
    seconds=float(context.current_time)
    if (not all(math.isfinite(x) for x in (dt,declared_dt,interval,frequency,seconds))
            or not math.isclose(dt,.004,rel_tol=0,abs_tol=1e-8)
            or not math.isclose(declared_dt,.004,rel_tol=0,abs_tol=1e-8)
            or decimation!=1 or interval!=10 or frequency!=25 or min(index,counter)<0):
        raise ValueError('Require actual250Hz physics,decimation1,10substeps per25Hz control')
    return {'schema':SCHEMA,'context_physics_step_index':index,'SDK_physics_step_counter':counter,
            'simulation_time_seconds':seconds,'physics_dt_seconds':dt,'decimation':decimation,
            'collect_frequency_hz':frequency,'control_interval_physics_steps':int(interval),
            'source':'SimulationContext.current_time/current_time_step_index;DirectRLEnv._sim_step_counter',
            'read_only_no_simulation_step':True}


def verify_clock_delta(before,after,expected_physics_steps,label):
    if any(row.get('schema')!=SCHEMA for row in (before,after)):
        raise ValueError('Actual native simulation clock descriptor required')
    for key in ('physics_dt_seconds','decimation','collect_frequency_hz','control_interval_physics_steps'):
        if before.get(key)!=after.get(key):raise ValueError('Simulation clock configuration changed')
    delta_index=after['context_physics_step_index']-before['context_physics_step_index']
    delta_counter=after['SDK_physics_step_counter']-before['SDK_physics_step_counter']
    seconds=after['simulation_time_seconds']-before['simulation_time_seconds']
    expected_seconds=expected_physics_steps*before['physics_dt_seconds']
    if (delta_index!=expected_physics_steps or delta_counter!=expected_physics_steps
            or not math.isfinite(seconds) or not math.isclose(seconds,expected_seconds,rel_tol=0,abs_tol=1e-5)):
        raise ValueError('Actual physics time differs from25Hz control contract: '+label)
    return {'label':label,'actual_context_physics_steps':delta_index,'actual_SDK_physics_steps':delta_counter,
            'actual_simulation_seconds':seconds,'expected_physics_steps':expected_physics_steps,
            'expected_simulation_seconds':expected_seconds,'passed':True}


def gripper_command_measurement(command,observation):
    """High-level target versus measured fingers, not interpolated subtargets."""
    measured=observation['measured_joint14'];audit=measured['measurement_audit'];result={}
    for side,column in (('left',6),('right',13)):
        value=float(command[side+'_ee_joint_state'][0])
        actual=audit[side];fingers=[float(x) for x in actual['finger_joint_metres']]
        if not math.isfinite(value) or not 0<=value<=1 or len(fingers)!=2 or not all(map(math.isfinite,fingers)):
            raise ValueError('Finite SDK-normalized gripper command and actual two fingers required')
        low,high=map(float,actual['scale_metres'])
        if (low,high)!=(-.01,.044) or actual['sign']!=1:
            raise ValueError('Unexpected X5 gripper physical scale')
        target=low+value*(high-low)
        result[side]={'command_gripper01':value,'SDK_high_level_target_metres_primary_mimic':[target,target],
                      'actual_finger_metres_primary_mimic':fingers,
                      'actual_measured_gripper01':float(measured['values'][column]),
                      'target_minus_actual_metres':[target-x for x in fingers],
                      'direction':'larger01_opens;smaller01_closes','historical_dataset_source_verified':False}
    return result


def verify_episode_timing(trace,commands,replans):
    if (not isinstance(trace,dict) or trace.get('schema')!='joint14_25hz_physics_trace_v1'
            or len(trace.get('commands',[]))!=commands
            or len(trace.get('replans',[]))!=replans or trace.get('read_only_instrumentation')is not True):
        raise ValueError('Missing complete actual physics trace')
    verify_clock_delta(trace['before_initial_capture'],trace['initial_clock'],0,'initial_capture')
    for i,row in enumerate(trace['commands'],1):
        if row['control_frame']!=i:raise ValueError('Missing actual control command timing')
        previous=trace['initial_clock'] if i==1 else trace['commands'][i-2]['after_capture']
        verify_clock_delta(previous,row['before'],0,'between_commands')
        verify_clock_delta(row['before'],row['after_take_action'],10,'command_%d'%i)
        verify_clock_delta(row['after_take_action'],row['after_capture'],0,'capture_%d'%i)
    if [row['control_frame'] for row in trace['replans']]!=list(range(0,commands,8)):
        raise ValueError('RF timing must follow actual8-command boundaries')
    for row in trace['replans']:
        verify_clock_delta(row['before'],row['after'],0,'RF_replan')
    total=verify_clock_delta(trace['initial_clock'],trace['final_clock'],commands*10,'whole_episode')
    if commands and trace['final_clock']!=trace['commands'][-1]['after_capture']:
        raise ValueError('Terminal time is not the final real control observation')
    return {'actual_commands':commands,'actual_replans':replans,**total,
            'captures_and_RF_do_not_advance_physics':True,'control_frequency_hz':25}
