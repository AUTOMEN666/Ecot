"""Bind the dataset instruction and actual SDK task without importing IsaacSim."""
import ast
import hashlib
import json
from pathlib import Path

TASK = 'classify_objects'
INSTRUCTION = 'Sort the objects by category into the three baskets.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_task(manifest_path, sdk_task_path, checkpoint_metadata=None):
    manifest_path, sdk_task_path = Path(manifest_path), Path(sdk_task_path)
    data = json.loads(manifest_path.read_text())
    if data.get('task_name') != TASK or data.get('complete') is not True:
        raise ValueError('Require prepared classify_objects data')
    episodes = data.get('episodes', [])
    if len(episodes) != 100 or sorted(e['episode_index'] for e in episodes) != list(range(200, 300)):
        raise ValueError('Require exactly target100 trajectories 200..299')
    if (data.get('splits') != {'train': list(range(200, 300)), 'val': []}
            or any(e.get('split') != 'train' or e.get('task_index') != 2 for e in episodes)):
        raise ValueError('All target100 trajectories must train; no validation split')
    if any(e.get('task_name') != TASK or e.get('instruction') != INSTRUCTION for e in episodes):
        raise ValueError('Dataset task/instruction differs from classify_objects contract')
    contract = data.get('data_contract', {})
    expected = {'fps': 25, 'action_dim': 14, 'horizon': 50,
                'alignment': 'action_t_equals_state_t_plus_1',
                'cameras': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('Training camera/action/current-frame semantics changed')
    for episode in episodes:
        if (episode.get('next_state_label_shift') != 1
                or episode.get('final_action_unpaired_excluded') is not True
                or episode.get('valid_action_count') != episode.get('frame_count', 0)-1):
            raise ValueError('Next-state frame pairing differs from training contract')
        if any(episode.get('images_224', {}).get(camera, {}).get('shape') != [episode['frame_count'], 224, 224, 3]
               for camera in expected['cameras']):
            raise ValueError('All three current 224 RGB camera streams required')
    if checkpoint_metadata is not None and checkpoint_metadata.get('identity', {}).get('dataset_manifest_sha256') != sha(manifest_path):
        raise ValueError('Checkpoint was not trained against this exact dataset manifest')
    tree = ast.parse(sdk_task_path.read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'gen_instruction']
    if len(methods) != 1:
        raise ValueError('SDK instruction method changed; inspect before evaluation')
    assignments = [n for n in methods[0].body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'templates' for t in n.targets)]
    returns = [n for n in methods[0].body if isinstance(n, ast.Return)]
    if (len(assignments) != 1 or ast.literal_eval(assignments[0].value) != [INSTRUCTION]
            or len(returns) != 1 or not isinstance(returns[0].value, ast.Name)
            or returns[0].value.id != 'templates'):
        raise ValueError('SDK task prompt differs from dataset instruction')
    return {'task': TASK, 'instruction': INSTRUCTION,
            'dataset_manifest': str(manifest_path), 'dataset_manifest_sha256': sha(manifest_path),
            'sdk_task_file': str(sdk_task_path), 'sdk_task_sha256': sha(sdk_task_path),
            'train_episodes': 100, 'validation_episodes': 0,
            'target_data_normalization': True, 'strict_no_target_data': False,
            'base_gradient_updates': 0 if checkpoint_metadata is None else None,
            'training_matched_layouts': False}
