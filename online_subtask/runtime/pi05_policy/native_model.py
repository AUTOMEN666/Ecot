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
from support.task_contract import TASK, INSTRUCTION
from support.weights_contract import validate_weights

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
        self.spec = spec
        self.audit_path = Path(spec['audit_path'])
        self.metadata = dict(spec.get('binding', {}))
        if policy is None:
            if spec.get('task') != TASK or spec.get('instruction') != INSTRUCTION:
                raise ValueError('Explicit classify_objects task and instruction binding required')
            binding = spec['weights_binding']
            validate_weights(binding)
            checkpoint = Path(binding['checkpoint']).resolve(strict=True)
            norm_path = Path(binding['norm_stats_path']).resolve(strict=True)
            import jax
            from openpi.models.pi0_config import Pi0Config
            from openpi.policies.policy_config import create_trained_policy
            from openpi.shared.normalize import load
            from openpi.training.config import TrainConfig
            from strict_prompt import strict_aloha_config
            identity = binding['checkpoint_metadata']['identity']
            if identity.get('train_ecot') is not True or identity.get('eval_ecot') is not False:
                raise ValueError('Require the independent full-parameter ECoT train-only checkpoint')
            max_token_len = identity['max_token_len']
            cfg = TrainConfig(name='pi05_classify_objects_ssr_fullparam_eval',
                              model=Pi0Config(pi05=True, action_horizon=50, action_dim=32, max_token_len=max_token_len),
                              data=strict_aloha_config(repo_id=None, adapt_to_pi=False,
                                                         use_delta_joint_actions=True))
            policy = create_trained_policy(cfg, checkpoint, norm_stats=load(norm_path.parent),
                                           sample_kwargs={'num_steps': 10})
            # Bind sampling seed across checkpoints; reset it for every real case.
            self._jax = jax
            self.metadata.update(binding)
            self.metadata.update(predict_steps=50, execute_steps=8, flow_steps=10,
                                 image_resolution=[224,224], raw_camera_resolution=[640,480], adapt_to_pi=False,
                                 ecot=False, train_ecot=True, eval_ecot=False, max_token_len=max_token_len,
                                 history=False, state_dim=14,
                                 task=spec['task'], instruction=spec['instruction'])
        else:
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
        if self.current is None or self.predicted_at == self.frame_index:
            raise ValueError('Fresh observation required before sampling')
        before = time.perf_counter()
        raw = np.asarray(self.policy.infer(self.current)['actions'], dtype=np.float32)
        commands, executed = to_commands(raw)
        self.predicted_at = self.frame_index
        self.audit({'event': 'inference', 'episode_id': self.episode_id,
                    'executed_action_step': self.frame_index, 'binding': self.metadata,
                    'raw_current_state': self.current['state'].tolist(),
                    'state_source': 'official_RoboDojo_observation_joint_and_gripper',
                    'prompt': self.current['prompt'],
                    'camera_shapes': {k: list(v.shape) for k, v in self.current['images'].items()},
                    'camera_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest()
                                      for k, v in self.current['images'].items()},
                    'raw_predicted_absolute_joint14': raw.tolist(),
                    'executed_first8_absolute_joint14': executed.tolist(),
                    'gripper_outside_01_count': int(((raw[:, [6, 13]] < 0) | (raw[:, [6, 13]] > 1)).sum()),
                    'inference_seconds': time.perf_counter() - before,
                    'physics_clock': self.clock})
        return commands

    def record_terminal_observation(self, payload):
        count = payload['executed_commands']
        if self.predicted_at < 0 or not self.predicted_at < count <= self.predicted_at + 8:
            raise ValueError('Terminal action count inconsistent with most recent plan')
        self.audit({'event': 'terminal', 'episode_id': self.episode_id, **payload})
        return {'recorded': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--ready-output', type=Path, required=True)
    args = parser.parse_args()
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    model = Model(json.loads(args.config.read_text()))
    server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=args.port,
                         ws_ping_interval_s=20, ws_ping_timeout_s=120))
    logging.basicConfig(level=logging.INFO)
    async def run():
        await server.start()
        args.ready_output.write_text(json.dumps({'pid': os.getpid(), 'binding': model.metadata}) + '\n')
        await server.serve_forever()
    asyncio.run(run())


if __name__ == '__main__':
    main()
