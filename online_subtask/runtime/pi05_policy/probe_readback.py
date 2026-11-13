"""Private probe compatibility/readiness helpers; no shared SDK mutations."""
from types import MethodType
import numpy as np


def cpu_readback(value):
    if hasattr(value,'detach') and hasattr(value,'cpu'):
        return value.detach().cpu().numpy().copy()
    if isinstance(value,tuple):
        return tuple(cpu_readback(x) for x in value)
    if isinstance(value,list):
        return [cpu_readback(x) for x in value]
    if isinstance(value,dict):
        return {k:cpu_readback(v) for k,v in value.items()}
    return value


class RewardReadbackProxy:
    """Convert only reward parser's pose/point read outputs to NumPy.

    Physics/layout manager itself, positions, units, actions, thresholds and
    reward rules remain identical. The proxy is attached to one parser only.
    """
    def __init__(self,manager,audit):
        self._manager=manager
        self._audit=audit
    def __getattr__(self,name):
        return getattr(self._manager,name)
    def get_instance_pose(self,*args,**kwargs):
        self._audit['pose_readback_calls']+=1
        return cpu_readback(self._manager.get_instance_pose(*args,**kwargs))
    def get_functional_points(self,*args,**kwargs):
        self._audit['point_readback_calls']+=1
        return cpu_readback(self._manager.get_functional_points(*args,**kwargs))


def install_private_reward_readback(env):
    parser=env.reward_manager.func_parser
    original=parser.initialize
    audit={'scope':'one_probe_reward_parser_read_outputs_only','shared_SDK_modified':False,
           'pose_readback_calls':0,'point_readback_calls':0}
    def initialize(_self,current_env):
        original(current_env)
        _self.layout_manager=RewardReadbackProxy(_self.layout_manager,audit)
    parser.initialize=MethodType(initialize,parser)
    if hasattr(parser,'layout_manager'):
        parser.layout_manager=RewardReadbackProxy(parser.layout_manager,audit)
    return audit


def validate_interface_receipt(receipt,child_returncode):
    if child_returncode!=0:
        raise ValueError('Actual native simulator child did not exit0')
    if (receipt.get('schema')!='joint14_actual_interface_replay_v1'
            or receipt.get('status') not in {
                'passed_controlled_native_joint_interface',
                'passed_controlled_native_joint_interface_pending_process_exit'}
            or receipt.get('all_interface_checks_passed')is not True):
        raise ValueError('Child exit0 without completed actual interface checks is not acceptance')
    commands=receipt.get('commands',[])
    if len(commands)!=24 or [c.get('control_frame') for c in commands]!=list(range(1,25)):
        raise ValueError('Need all24 actual native command frames')
    gates={g['gate'] for g in receipt.get('gates',[])}
    expected={'isaac_started','official_scene_reset','native_reward_checks_registered','three_real_camera_frames_before',
              'native_joint_control24_commands','three_real_camera_frames_after','environment_closed'}
    if not expected<=gates or receipt.get('simulation_app_close_requested')is not True:
        raise ValueError('Missing actual scene/camera/control/close gates')
    for command in commands:
        for key in ('target_joint14','before_measured_joint14','after_measured_joint14'):
            value=np.asarray(command.get(key),dtype=np.float64)
            if value.shape!=(14,) or not np.isfinite(value).all():
                raise ValueError('Invalid actual interface command trace')
    return True
