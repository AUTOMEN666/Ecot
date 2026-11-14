"""Stack Bowls EMA10000 online Subtask policy, exact original training formatter."""
from __future__ import annotations
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from export_binding import read, validate_binding
from task_contract import (TASK, TRAIN_TASK, INSTRUCTION, CHECKPOINT_STEP, VARIANT,
                           PROTOCOL_CONFIG_ID, OUTPUT_NAMESPACE, compose_prompt)


def initialize_paths(parent):
    parent = Path(parent)
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(parent / 'src')]


def validate_protocol_spec(spec):
    from multiview_protocol import PROTOCOLS, CATALOG_SHA
    expected = {
        'task': TASK, 'training_task': TRAIN_TASK, 'variant': VARIANT,
        'condition': 'online_subtask', 'effective_eval_ecot': True,
        'source_training_eval_ecot': False, 'step': CHECKPOINT_STEP,
        'noise_seed': 42, 'instruction': INSTRUCTION, 'provider': 'deepseek',
        'protocol_mode': 'expanded_catalog_id', 'catalog_sha256': CATALOG_SHA,
        'generator_protocol': PROTOCOLS['expanded_catalog_id'],
        'config_id': PROTOCOL_CONFIG_ID, 'protocol_config_id': PROTOCOL_CONFIG_ID,
        'max_tokens': 65536, 'past_history_cameras': ['cam_high'],
        'maximum_total_images': 96, 'maximum_past_timepoints': 93,
        'selector_timeout_seconds': 600, 'selector_retry_budget_seconds': 1850,
        'policy_rpc_timeout_seconds': 2100, 'cadence_execution_frames': 8,
        'executed_subtask_history_enabled': True,
        'executed_history_protocol': 'ecot.actual_executed_subtask_history.v1',
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Exact Stack Bowls Random/EMA10000 online protocol required')
    if OUTPUT_NAMESPACE not in Path(spec['audit_path']).parts:
        raise ValueError('New isolated Stack Bowls output namespace required')
    return {'config_id': PROTOCOL_CONFIG_ID, 'maximum_past_timepoints': 93,
            'maximum_total_images': 96}


def create_model(spec, policy=None, jax_module=None, provider=None):
    initialize_paths(spec['parent_root'])
    from pi05_policy.native_model import Model as NativeModel, to_commands, encode_obs
    from multiview_protocol import PROTOCOLS, CAMERAS, SubtaskAbstention
    validate_protocol_spec(spec)
    binding = read(spec['export_binding'])
    validate_binding(binding, full_hashes=False)
    if (binding['step'] != CHECKPOINT_STEP or binding['variant'] != VARIANT
            or binding['parent_root'] != str(Path(spec['parent_root']).resolve())
            or binding['training_identity'].get('eval_ecot') is not False
            or binding['training_identity'].get('train_ecot') is not True):
        raise ValueError('Genuine Stack Bowls training identity differs; do not rewrite it')
    online, variant, protocol_mode = True, VARIANT, 'expanded_catalog_id'
    condition = 'online_subtask'
    if provider is None:
        from multiview_provider import MultiviewHistoryProvider
        provider = MultiviewHistoryProvider(Path(spec['audit_path']).parent / 'subtask_requests',
                                            selector_kind='deepseek',
                                            timeout_seconds=spec['selector_timeout_seconds'])
    if policy is None:
        import jax
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared.normalize import load
        from openpi.training.config import TrainConfig
        from strict_prompt import strict_aloha_config
        cfg = TrainConfig(name='stack_bowls100_subtask_ema10000_random_online',
                          model=Pi0Config(pi05=True, action_horizon=50, action_dim=32,
                                         max_token_len=binding['training_identity']['max_token_len']),
                          data=strict_aloha_config(repo_id=None, adapt_to_pi=False,
                                                   use_delta_joint_actions=True))
        policy = create_trained_policy(cfg, Path(binding['export_path']),
                                       norm_stats=load(Path(binding['norm_stats_path']).parent),
                                       sample_kwargs={'num_steps': 10})
        jax_module = jax
    if jax_module is None:
        raise ValueError('Actual JAX RNG implementation required')
    metadata = {
        'schema': 'ecot.stack_bowls_random.online_subtask.runtime.v1',
        'protocol_config_id': PROTOCOL_CONFIG_ID, 'task': TASK, 'training_task': TRAIN_TASK,
        'instruction': INSTRUCTION, 'variant': VARIANT,
        'source_export_binding_sha256': binding['binding_sha256'],
        'source_export_path': binding['export_path'], 'step': CHECKPOINT_STEP,
        'inference_params': 'EMA', 'source_training_eval_ecot': False,
        'planned_eval_ecot': True, 'effective_eval_ecot': True,
        'parameter_item_tree_sha256': binding['parameter_item_tree_sha256'],
        'runtime_condition': condition, 'provider_kind': 'deepseek_mainhist93_tok65536',
        'generator_protocol': PROTOCOLS[protocol_mode], 'catalog_sha256': spec['catalog_sha256'],
        'context_fields': ['subtask'], 'context_injected': True,
        'generator_current3RGB': True, 'generator_measured_state14': True,
        'generator_actual_executed_subtask_history': True, 'generator_reference': False,
        'generator_full_training_video_reference': False,
        'policy_history': False, 'policy_image_history': False,
        'policy_uses_only_current3RGB_state14_and_composed_prompt': True,
        'predict_steps': 50, 'execute_steps': 8, 'flow_steps': 10, 'noise_seed': 42,
        'state_dim': 14, 'raw_camera_resolution': [640, 480], 'cadence_execution_frames': 8,
        'SDK_action_limit': 800, 'max_tokens': 65536,
    }

    class RuntimeModel(NativeModel):
        def __init__(self):
            # Reuse the immutable native observation/action implementation with
            # an already-verified policy; never falsify a legacy weight identity.
            super().__init__({'audit_path':spec['audit_path'],'binding':metadata,
                              'noise_seed':42,'instruction':INSTRUCTION},policy=policy)
            self._jax=jax_module;self.latest_injection=None;self.latest_commands=None
        def reset(self):
            super().reset();self.latest_injection=None;self.latest_commands=None
            if online:provider.reset(self.episode_id)
        def record_history_observation(self,obs):
            # This path never calls update_obs/get_action/infer or changes policy RNG.
            frame=obs.get('frame_index')
            if self.episode_id is None:raise ValueError('History RPC requires an actual model reset')
            before_frame,before_predicted,before_current=self.frame_index,self.predicted_at,self.current
            key_data=self._jax.random.key_data if hasattr(self._jax.random,'key_data')else(lambda value:value)
            rng_before=np.asarray(key_data(self.policy._rng)).copy()
            measured=encode_obs(obs)
            if measured['prompt']!=INSTRUCTION:raise ValueError('Original Instruction required in history RPC')
            from subtask_client import png_current_rgb
            pngs={camera:png_current_rgb(measured['images'][camera])for camera in CAMERAS}
            record=provider.capture_history(pngs,INSTRUCTION,state14=measured['state'].tolist(),episode_id=self.episode_id,frame=frame)
            rng_after=np.asarray(key_data(self.policy._rng))
            if (not np.array_equal(rng_before,rng_after)or self.frame_index!=before_frame or self.predicted_at!=before_predicted or self.current is not before_current):
                raise RuntimeError('History RPC altered action-policy state or RNG')
            proof={'event':'actual_history_capture','episode_id':self.episode_id,'history_frame':frame,'physics_clock':obs.get('physics_clock'),
                'history_source_index':record['history_source_index'],'current_views':record['current_views'],
                'raw_measured_state14':measured['state'].tolist(),'policy_called':False,'RNG_advanced':False,
                'RNG_before_sha256':hashlib.sha256(rng_before.tobytes()).hexdigest(),'RNG_after_sha256':hashlib.sha256(rng_after.tobytes()).hexdigest(),
                'policy_frame_index_unchanged':True,'action_decoder_called':False}
            self.audit(proof)
            return {'history_frame':frame,'episode_id':self.episode_id,'policy_called':False,'RNG_advanced':False,
                'capture_count':record['history_source_index']+1,'current_views':record['current_views']}

        def acknowledge_executed_subtask(self,obs):
            if not online or self.latest_injection is None or self.latest_commands is None:
                raise ValueError('Cannot commit a generation without actual policy injection/commands')
            before_frame,before_predicted,before_current=self.frame_index,self.predicted_at,self.current
            key_data=self._jax.random.key_data if hasattr(self._jax.random,'key_data')else(lambda value:value)
            rng_before=np.asarray(key_data(self.policy._rng)).copy()
            receipt=provider.acknowledge_executed_subtask(obs,expected_injection=self.latest_injection,
                expected_actions=self.latest_commands,expected_clock=self.clock)
            rng_after=np.asarray(key_data(self.policy._rng))
            if (not np.array_equal(rng_before,rng_after)or self.frame_index!=before_frame
                    or self.predicted_at!=before_predicted or self.current is not before_current):
                raise RuntimeError('Executed-history ACK altered policy state/RNG')
            self.audit({**receipt,'event':'actual_executed_subtask_ACK','physics_clock_at_injection':self.clock,
                'RNG_before_sha256':hashlib.sha256(rng_before.tobytes()).hexdigest(),
                'RNG_after_sha256':hashlib.sha256(rng_after.tobytes()).hexdigest(),'policy_frame_index_unchanged':True})
            return receipt

        def current_injection(self):
            if not online or self.latest_injection is None or self.latest_injection['execution_frame']!=self.frame_index:
                raise ValueError('No current actual Subtask injection is available')
            return deepcopy(self.latest_injection)
        def get_action(self):
            if self.current is None or self.predicted_at==self.frame_index:raise ValueError('Fresh actual SDK observation required')
            if self.current['prompt']!=INSTRUCTION:raise ValueError('SDK must supply exactly the full original Instruction')
            started=time.monotonic();context=None;lineage=None;prompt=INSTRUCTION
            if online:
                from subtask_client import png_current_rgb,validate_context,canonical
                pngs={camera:png_current_rgb(self.current['images'][camera])for camera in CAMERAS}
                try:
                    context,lineage=provider.generate(pngs,INSTRUCTION,state14=self.current['state'].tolist(),
                        episode_id=self.episode_id,frame=self.frame_index)
                except SubtaskAbstention:
                    self.audit({'event':'generator_abstention','episode_id':self.episode_id,
                        'executed_action_step':self.frame_index,'physics_clock':self.clock,
                        'protocol_mode':protocol_mode,'actual_injected':False,'policy_called':False,'fallback_used':False})
                    raise
                validate_context(context)
                prompt=compose_prompt(INSTRUCTION,context['subtask'],True)
                if prompt!=INSTRUCTION+' Current subtask: '+context['subtask']:
                    raise ValueError('Actual training/inference Subtask composition differs')
            actual={**self.current,'prompt':prompt}
            _,sample_rng=self._jax.random.split(self.policy._rng)
            noise=np.asarray(self._jax.random.normal(sample_rng,(1,50,32)),dtype=np.float32)
            if noise.shape!=(1,50,32)or not np.isfinite(noise).all():raise ValueError('Invalid actual sampling noise')
            noise_path=Path(spec['audit_path']).parent/'arrays'/f'{self.episode_id}_{self.frame_index:06d}_noise.npy'
            noise_path.parent.mkdir(parents=True,exist_ok=True)
            with noise_path.open('xb')as f:np.save(f,noise,allow_pickle=False)
            raw=np.asarray(self.policy.infer(actual,noise=noise)['actions'],dtype=np.float32)
            commands,executed=to_commands(raw);self.predicted_at=self.frame_index
            if online:
                from pi05_policy.freeze import plain
                self.latest_commands=plain(commands)
                self.latest_injection={'episode_id':self.episode_id,'execution_frame':self.frame_index,
                    'context':context,'context_json':canonical(context).decode(),'prompt':prompt,
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'lineage':lineage,
                    'actual_injected':True,'checkpoint_step':binding['step'],'training_composition_byte_equal':True}
            self.audit({'event':'inference','episode_id':self.episode_id,'executed_action_step':self.frame_index,
                'binding':self.metadata,'raw_current_state':self.current['state'].tolist(),
                'state_source':'official_RoboDojo_observation_joint_and_gripper',
                'raw_instruction':INSTRUCTION,'prompt':prompt,'context':context,'lineage':lineage,
                'context_source_frame':self.frame_index if online else None,
                'context_input_episode_id':self.episode_id if online else None,'goal_protocol_metadata':None,
                'generation_seconds':lineage['generation_total_seconds']if online else 0.0,
                'camera_shapes':{k:list(v.shape)for k,v in self.current['images'].items()},
                'camera_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in self.current['images'].items()},
                'noise_shape':list(noise.shape),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                'noise_path':str(noise_path),'raw_predicted_absolute_joint14':raw.tolist(),
                'executed_first8_absolute_joint14':executed.tolist(),
                'inference_seconds':time.monotonic()-started,'physics_clock':self.clock})
            return commands
    return RuntimeModel()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--ready-output',type=Path,required=True)
    args=p.parse_args();spec=read(args.config);initialize_paths(spec['parent_root']);model=create_model(spec)
    from client_server.ws.model_server import PolicyServer,PolicyServerConfig
    server=PolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=args.port,ws_ping_interval_s=20,ws_ping_timeout_s=2100))
    async def run():
        await server.start();args.ready_output.write_text(json.dumps({'pid':__import__('os').getpid(),
            'binding':model.metadata,'provider':spec.get('provider'),'effective_eval_ecot':spec['effective_eval_ecot']},allow_nan=False)+'\n')
        await server.serve_forever()
    asyncio.run(run())
if __name__=='__main__':main()
