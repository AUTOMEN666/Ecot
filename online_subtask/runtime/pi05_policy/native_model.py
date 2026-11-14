"""Native OpenPI inference with exact checkpoint and RoboDojo observation semantics.

The OpenPI output transforms undo quantile scaling and add the SAME current
joint state to each predicted arm target. No extra delta conversion is done here.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from task_contract import TASK, INSTRUCTION

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_obs(obs):
    """Input RGB is decoded by the official WS server; retain SDK gripper state."""
    state = obs['state']
    if isinstance(state, dict):
        keys = ('left_arm_joint_state', 'left_ee_joint_state',
                'right_arm_joint_state', 'right_ee_joint_state')
        arrays = [np.asarray(state[k], dtype=np.float32) for k in keys]
        if [a.shape for a in arrays] != [(6,), (1,), (6,), (1,)]:
            raise ValueError('Require official RoboDojo 6+1+6+1 state shape')
        state = np.concatenate(arrays)
    else:
        state = np.asarray(state, dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Require finite official SDK joint14 state')
    instruction = obs.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Real nonempty task instruction required')
    vision = obs.get('images', obs.get('vision', {}))
    aliases = {'cam_high': ('cam_high', 'cam_head', 'head_camera', 'top_camera'),
               'cam_left_wrist': ('cam_left_wrist', 'left_camera'),
               'cam_right_wrist': ('cam_right_wrist', 'right_camera')}
    images = {}
    for camera in CAMERAS:
        key = next((k for k in aliases[camera] if k in vision), None)
        if key is None:
            raise KeyError('Missing real camera ' + camera)
        image = vision[key]
        if isinstance(image, dict):
            image = image['color'] if 'color' in image else image['rgb']
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError('Camera must be 3D array')
        if image.shape[-1] == 3:
            pass
        elif image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError('Camera must have exactly3 RGB channels')
        if image.shape != (480, 640, 3):
            raise ValueError('Require original 640x480 RGB for each current camera')
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Floating RGB must be in0..1')
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError('RGB must be uint8 or normalized float')
        # Native AlohaInputs always rearranges c h w -> h w c. Its public input
        # contract is CHW, even though the simulator and our decode work in HWC.
        images[camera] = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return {'state': state, 'images': images, 'prompt': instruction}


def to_commands(predictions):
    raw = np.asarray(predictions, dtype=np.float32)
    if raw.shape != (50, 14) or not np.isfinite(raw).all():
        raise ValueError('Native pi05 output must be finite [50,14] after transforms')
    # Preserve all original predictions in audit. Only SDK gripper commands are
    # bounded to its documented range; training and recorded raw outputs are not.
    actions = raw[:8].copy()
    actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0, 1)
    commands = [{'left_arm_joint_state': a[:6], 'left_ee_joint_state': a[6:7],
                 'right_arm_joint_state': a[7:13], 'right_ee_joint_state': a[13:14]}
                for a in actions]
    return commands, actions


class Model:
    def __init__(self, spec, policy=None):
        if policy is None:
            raise ValueError('Use model_server.create_model; online Subtask injection is mandatory')
        self.spec = spec
        self.audit_path = Path(spec['audit_path'])
        self.metadata = dict(spec.get('binding', {}))
        self._jax = None
        self.policy = policy
        self.episode_id = None
        self.current = None
        self.frame_index = -1
        self.predicted_at = -1

    def audit(self, row):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def reset(self):
        self.episode_id = uuid4().hex
        self.current = None
        self.frame_index = self.predicted_at = -1
        if self._jax is not None:
            self.policy._rng = self._jax.random.key(int(self.spec.get('noise_seed', 42)))
        self.audit({'event': 'reset', 'episode_id': self.episode_id, 'binding': self.metadata})

    def update_obs(self, obs):
        t = obs.get('frame_index')
        if not self.episode_id or type(t) is not int or t < 0 or t % 8:
            raise ValueError('Require reset and actual replan control frame0,8,...')
        if self.frame_index >= t:
            raise ValueError('Each plan must use a new current observation')
        current = encode_obs(obs)
        if self.spec.get('instruction') and current['prompt'] != self.spec['instruction']:
            raise ValueError('Actual SDK instruction differs from training instruction')
        self.current = current
        self.frame_index = t
        self.clock = obs.get('physics_clock')

    def get_action(self):
        raise RuntimeError('Online Subtask runtime required; use model_server.create_model')

    def record_terminal_observation(self, payload):
        count = payload['executed_commands']
        if self.predicted_at < 0 or not self.predicted_at < count <= self.predicted_at + 8:
            raise ValueError('Terminal action count inconsistent with most recent plan')
        self.audit({'event': 'terminal', 'episode_id': self.episode_id, **payload})
        return {'recorded': True}


if __name__ == '__main__':
    raise SystemExit('Use runtime/model_server.py; task-only serving is disabled')
