"""Read-only freeze witness; scene labels never enter policy or DeepSeek inputs."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from .physics_clock import read_physics_clock

SCHEMA='stack_bowls_random_current_layout_freeze_v1'
# Exact LayoutManager OBJECT_CONFIG_TYPES and its separately registered fixtures.
# Enumerate every current record, including non-task objects such as camera_stand.
OBJECT_TYPES=('Rigid','Dynamic','Geometry','Articulation','Garment','Fluid')
FIXTURE_TYPES=('room','light','ground','table')


def plain(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):
        if not np.isfinite(value).all():raise ValueError('Nonfinite measured state')
        return value.copy().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return deepcopy(value)


def current_inventory(manager):
    """Read actual six-type records and cross-check the SDK instance registry."""
    records=manager.object_records_by_type
    if not isinstance(records,dict)or set(records)!=set(OBJECT_TYPES):
        raise ValueError('Cannot prove complete current SDK object-type enumeration')
    inventory={};names={}
    for kind in OBJECT_TYPES:
        rows=manager.get_layout_records(env_idx=0,object_type=kind)
        if not isinstance(rows,list):raise ValueError('Current SDK layout records must be lists')
        for row in rows:
            if not isinstance(row,dict):raise ValueError('Invalid current SDK object record')
            label=row.get('label');name=row.get('inst_name')
            if any(not isinstance(x,str)or not x.strip()or x!=x.strip()for x in (label,name)):
                raise ValueError('Every current object needs a nonempty label and instance')
            if label in inventory or name in names:raise ValueError('Duplicate current object label or instance')
            if manager.get_instance_name(env_idx=0,label=label)!=name:
                raise ValueError('Current label lookup disagrees with layout record')
            inventory[label]={'instance_name':name,'object_type':kind.lower()};names[name]=kind.lower()
    registry=manager.instance_type_by_env[0]
    if not isinstance(registry,dict):raise ValueError('Missing complete current SDK instance registry')
    registered={}
    for name,kind in registry.items():
        if not isinstance(name,str)or not name.strip()or not isinstance(kind,str):
            raise ValueError('Invalid current SDK instance registry entry')
        if kind in FIXTURE_TYPES:continue
        if kind not in {x.lower()for x in OBJECT_TYPES}:raise ValueError('Unknown current instance type')
        registered[name]=kind
    if registered!=names:raise ValueError('Current object records and instance registry are incomplete or inconsistent')
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory):
    if not isinstance(inventory,dict)or not inventory:raise ValueError('Missing actual current object inventory')
    names=[]
    for label,row in inventory.items():
        if(not isinstance(label,str)or not label.strip()or label!=label.strip()or not isinstance(row,dict)
                or set(row)!={'instance_name','object_type'}or not isinstance(row['instance_name'],str)
                or not row['instance_name'].strip()or row['instance_name']!=row['instance_name'].strip()
                or row['object_type']not in {x.lower()for x in OBJECT_TYPES}):
            raise ValueError('Invalid actual current object identity')
        names.append(row['instance_name'])
    if len(set(names))!=len(names):raise ValueError('Duplicate actual current object instance')
    for label in ('bowl0','bowl1','bowl2'):
        if label not in inventory or inventory[label]['object_type']!='rigid':
            raise ValueError('All three actual task bowls must be measured')


def validate_pose(pose):
    if not isinstance(pose,(list,tuple))or len(pose)!=2:raise ValueError('Missing measured current object pose')
    for value,shape in zip(pose,[(3,),(4,)]):
        a=np.asarray(value,dtype=np.float64)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Require finite current object position3 and rotation4')
    if not np.any(np.asarray(pose[1],dtype=np.float64)):raise ValueError('Invalid zero current object quaternion')


def validate_snapshot(value):
    if(not isinstance(value,dict)or value.get('schema')!=SCHEMA
            or set(value)!={'schema','object_inventory','object_poses','measured_joint_gripper_state','physics_clock'}):
        raise ValueError('Complete dynamic current-layout freeze snapshot required')
    validate_inventory(value['object_inventory']);poses=value['object_poses']
    if not isinstance(poses,dict)or set(poses)!=set(value['object_inventory']):raise ValueError('Not every current object pose was measured')
    for pose in poses.values():validate_pose(pose)
    state=value['measured_joint_gripper_state']
    if isinstance(state,dict):
        keys=('left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state')
        arrays=[np.asarray(state.get(key),dtype=np.float64)for key in keys]
        if [a.shape for a in arrays]!=[(6,),(1,),(6,),(1,)]:raise ValueError('Complete measured robot6+1+6+1 required')
        state=np.concatenate(arrays)
    else:state=np.asarray(state,dtype=np.float64)
    if state.shape!=(14,)or not np.isfinite(state).all():raise ValueError('Finite measured robot joint14 required')
    # JSON serialization also rejects nonfinite or unsupported extra state values.
    canonical(value)
    return value


def snapshot(env,obs):
    manager=env.reward_manager.func_parser.layout_manager
    clock=read_physics_clock(env);inventory=current_inventory(manager);poses={}
    for label,row in inventory.items():
        pose=plain(manager.get_instance_pose(inst_name=row['instance_name'],env_idx=0))
        validate_pose(pose);poses[label]=pose
    if current_inventory(manager)!=inventory:raise RuntimeError('Current object inventory changed while measuring freeze snapshot')
    if canonical(clock)!=canonical(read_physics_clock(env)):raise RuntimeError('Simulation advanced while measuring freeze snapshot')
    return validate_snapshot({'schema':SCHEMA,'object_inventory':inventory,'object_poses':poses,
            'measured_joint_gripper_state':plain(obs['state']),'physics_clock':clock})


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def verify_static(before,after):
    validate_snapshot(before);validate_snapshot(after)
    if canonical(before)!=canonical(after):raise RuntimeError('Simulation moved during DeepSeek/policy wait')
    return {'passed':True,'before_sha256':hashlib.sha256(canonical(before)).hexdigest(),
            'after_sha256':hashlib.sha256(canonical(after)).hexdigest()}
