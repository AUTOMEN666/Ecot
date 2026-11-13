"""Record real main-camera RGB at every executed 25 Hz action boundary.

Only render/readback/encoding are performed here. No get_obs, take_action,
physics step, policy request, interpolation, or duplicated frame is introduced.
The SDK's three existing videos remain in their original result directory.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from .physics_clock import read_physics_clock, verify_clock_delta


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class MainCameraAction25Hz:
    """One initial frame and one fresh capture per actual control command.

    writer_factory/render_sync injection supports CPU tests; deployment uses
    the existing SDK VideoStreamWriter and requires its actual render fence.
    """
    def __init__(self, env, writer_factory=None, render_sync=None):
        self.env = env
        self.writer_factory = writer_factory
        self.render_sync = render_sync
        self.writer = self.stream = self.folder = None
        self.initial = self.previous = None
        self.count = 0
        self.closed = False
        self.binding = None

    def start(self, observation, control_step=0):
        if self.initial is not None or self.closed or control_step != 0:
            raise ValueError('Recorder must start once at actual command boundary zero')
        env = self.env
        if env.num_envs != 1:
            raise ValueError('Require the existing one-environment-per-lane protocol')
        names = list(env.obs_manager.camera_manager.camera_names[0])
        matches = [name for name in names if name in ('cam_high', 'cam_head', 'head_camera', 'top_camera')]
        if len(matches) != 1:
            raise ValueError('Require one unambiguous main camera')
        self.camera = matches[0]
        self.camera_id = names.index(self.camera)
        if self.writer_factory is None and self.render_sync is None:
            from env.camera_manager.capture import render_sync
            self.render_sync = render_sync
        if self.render_sync is not None and not self.render_sync.zero_delay_active():
            raise ValueError('Actual zero-delay camera render fence settings required')
        self.initial = self.previous = read_physics_clock(env)
        sdk_save_dir = Path(env.save_dir).resolve()
        self.identity = {
            'schema': 'pi05_main_camera_action25hz_v1',
            'scene_seed': int(env.eval_seed), 'layout_id': int(env.current_env_seed_map[0]),
            'original_SDK_save_dir': str(sdk_save_dir),
            'camera': self.camera, 'SDK_camera_id': self.camera_id,
            'control_frequency_hz': 25, 'encoded_fps': 25,
            'includes_initial_boundary': True, 'interpolation_used': False,
            'synthetic_or_repeated_padding_frames': False,
            'actual_render_fence_instrumented': self.render_sync is not None,
            'raw_RGB_shape': [480, 640, 3],
        }
        self.folder = sdk_save_dir.parent / '_action25hz' / sdk_save_dir.name / (
            f'scene{self.identity["scene_seed"]}_layout{self.identity["layout_id"]}_{uuid4().hex}')
        self.folder.mkdir(parents=True, exist_ok=False)
        self.stream = (self.folder / 'frames.jsonl').open('x')
        self._append(observation['vision'][self.camera]['color'], 0, self.initial)
        verify_clock_delta(self.initial, read_physics_clock(env), 0, 'recorder_initial_encode')

    def after_action(self, control_step):
        if self.initial is None or self.closed or control_step != self.count:
            raise ValueError('Every executed command requires one ordered capture')
        before = read_physics_clock(self.env)
        verify_clock_delta(self.previous, before, 10, 'recorder_actual_control_interval')
        manager = self.env.obs_manager
        before_delivery = self.render_sync.dispatcher_time() if self.render_sync else None
        manager.render_for_capture()
        after_delivery = self.render_sync.dispatcher_time() if self.render_sync else None
        render_proof = None
        if self.render_sync:
            if (not self.render_sync.zero_delay_active() or before_delivery is None or after_delivery is None
                    or not math.isfinite(before_delivery) or not math.isfinite(after_delivery)
                    or after_delivery == before_delivery):
                raise ValueError('No actual new SDG camera delivery at this action boundary')
            render_proof = {'zero_delay_active': True, 'dispatcher_before': before_delivery,
                            'dispatcher_after': after_delivery,
                            'clock_scope': 'SDG reference time; not compared with physical seconds'}
        # The SDK returns selected cameras as a compact list, not by cam_id.
        selected = manager.capture_manager.step(env_ids=[0], cam_ids=[self.camera_id])
        self._append(selected[0]['rgb'][0]['data'][..., :3], control_step, before, render_proof)
        after = read_physics_clock(self.env)
        verify_clock_delta(before, after, 0, 'recorder_render_readback_encode')
        self.previous = after

    def _append(self, color, control_step, clock, render_proof=None):
        color = np.ascontiguousarray(color)
        if color.dtype != np.uint8 or color.shape != (480, 640, 3):
            raise ValueError('Require original SDK 640x480 HWC RGB uint8; no resizing or channel conversion')
        if control_step != self.count:
            raise ValueError('Missing or repeated action-boundary frame')
        if self.writer is None:
            factory = self.writer_factory
            if factory is None:
                from utils.save_file import VideoStreamWriter
                factory = VideoStreamWriter
            self.writer = factory(str(self.folder / 'main_rgb.mp4'), *color.shape, fps=25.0, is_rgb=True)
        self.writer.append(color)
        row = {'frame_index': self.count, 'executed_action_step': control_step,
               'simulation_time_seconds': clock['simulation_time_seconds'],
               'relative_simulation_time_seconds': clock['simulation_time_seconds'] - self.initial['simulation_time_seconds'],
               'context_physics_step_index': clock['context_physics_step_index'],
               'SDK_physics_step_counter': clock['SDK_physics_step_counter'],
               'rgb_sha256': hashlib.sha256(color.tobytes()).hexdigest(), 'render_sync': render_proof}
        self.stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.stream.flush()
        self.count += 1

    def close(self, completed):
        if self.closed:
            return self.binding
        self.closed = True
        if self.folder is None:
            return None
        error = None
        final = None
        try:
            if self.writer is not None:
                self.writer.close(announce=False)
            final = read_physics_clock(self.env)
            if completed:
                verify_clock_delta(self.initial, final, (self.count - 1) * 10, 'recorder_whole_episode')
                if self.writer is None or getattr(self.writer, 'n_frames', self.count) != self.count:
                    raise ValueError('Encoder frame count differs from capture ledger')
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if self.stream is not None:
                self.stream.close()
            if final is None:
                final = read_physics_clock(self.env)
            actual_physics_steps = final['SDK_physics_step_counter'] - self.initial['SDK_physics_step_counter']
            summary = {**self.identity,
                       'status': 'complete' if completed and error is None else 'incomplete',
                       'frame_count': self.count, 'captured_action_count': max(0, self.count - 1),
                       'actual_action_count': actual_physics_steps // 10,
                       'simulation_span_seconds': max(0, self.count - 1) / 25,
                       'encoded_duration_seconds': self.count / 25,
                       'initial_clock': self.initial, 'final_clock': final, 'error': error,
                       'video_path': str(self.folder / 'main_rgb.mp4'),
                       'frames_path': str(self.folder / 'frames.jsonl'),
                       'frames_sha256': sha_file(self.folder / 'frames.jsonl'),
                       'video_decode_verified': False,
                       'scope': 'real action-boundary captures and encoder finalization; SDK success is separate'}
            video = self.folder / 'main_rgb.mp4'
            if video.exists():
                summary['video_sha256'] = sha_file(video)
            path = self.folder / 'summary.json'
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(summary, indent=2) + '\n')
            tmp.replace(path)
            self.binding = {**summary, 'summary_path': str(path), 'summary_sha256': sha_file(path)}
        return self.binding
