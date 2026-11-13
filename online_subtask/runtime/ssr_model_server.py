"""Generate admitted public SSR before policy inference; commit only real ACKs."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

from multiview_protocol import CAMERAS,INSTRUCTION,canonical
from rpc_lifecycle import exclusive_rpc,lifecycle_lock
from ssr_protocol import CONFIG_ID,CONTEXT_FIELDS,REFERENCE_SHA,validate_context,validate_spec
from ssr_provider import SSRSteadyProvider

def create_model(spec,policy=None,jax_module=None,provider=None):
    from grounded_ssr_protocol import CONFIG_ID as GROUNDED_ID
    from strict_grounded_ssr_protocol import CONFIG_ID as STRICT_ID
    from enum_grounded_ssr_protocol import CONFIG_ID as ENUM_ID
    from phase_grounded_ssr_protocol import CONFIG_ID as PHASE_ID
    from release_grounded_ssr_protocol import CONFIG_ID as RELEASE_ID
    from temporal_grounded_ssr_protocol import CONFIG_ID as TEMPORAL_ID
    temporal=spec.get('config_id')==TEMPORAL_ID
    releasing=spec.get('config_id')==RELEASE_ID
    phased=spec.get('config_id')==PHASE_ID
    enumerated=spec.get('config_id')in(ENUM_ID,PHASE_ID,RELEASE_ID,TEMPORAL_ID)
    strict=spec.get('config_id')==STRICT_ID
    grounded=spec.get('config_id')in(GROUNDED_ID,STRICT_ID,ENUM_ID,PHASE_ID,RELEASE_ID,TEMPORAL_ID)
    if temporal:
        from temporal_grounded_ssr_protocol import validate_spec as validate_temporal_spec
        validate_temporal_spec(spec)
    elif releasing:
        from release_grounded_ssr_protocol import validate_spec as validate_release_spec
        validate_release_spec(spec)
    elif phased:
        from phase_grounded_ssr_protocol import validate_spec as validate_phase_spec
        validate_phase_spec(spec)
    elif enumerated:
        from enum_grounded_ssr_protocol import validate_spec as validate_enum_spec
        validate_enum_spec(spec)
    elif strict:
        from strict_grounded_ssr_protocol import validate_spec as validate_strict_spec
        validate_strict_spec(spec)
    elif grounded:
        from grounded_ssr_protocol import validate_spec as validate_grounded_spec
        validate_grounded_spec(spec)
    else:validate_spec(spec)
    protocol_id=spec['config_id']
    parent=Path(spec['parent_root'])
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(parent/'src'),str(parent/'evaluation')]
    from ssr_binding import actual_training_prompt,load_weights
    binding=load_weights(spec)
    from pi05_policy.native_model import Model as NativeModel,encode_obs,to_commands
    from ssr_protocol import load_reference
    # Fail before loading/sampling a policy if the real external formatter differs.
    actual_training_prompt(INSTRUCTION,load_reference()['examples'][0]['context'])
    if provider is None:
        if temporal:
            from temporal_grounded_ssr_provider import TemporalGroundedSSRProvider
            provider_class=TemporalGroundedSSRProvider
        elif releasing:
            from release_grounded_ssr_provider import ReleaseGroundedSSRProvider
            provider_class=ReleaseGroundedSSRProvider
        elif phased:
            from phase_grounded_ssr_provider import PhaseGroundedSSRProvider
            provider_class=PhaseGroundedSSRProvider
        elif enumerated:
            from enum_grounded_ssr_provider import EnumGroundedSSRProvider
            provider_class=EnumGroundedSSRProvider
        elif strict:
            from strict_grounded_ssr_provider import StrictGroundedSSRProvider
            provider_class=StrictGroundedSSRProvider
        elif grounded:
            from grounded_ssr_provider import GroundedSSRProvider
            provider_class=GroundedSSRProvider
        else:provider_class=SSRSteadyProvider
        provider=provider_class(Path(spec['audit_path']).parent/'ssr_requests',selector_kind='deepseek',
            timeout_seconds=spec['selector_timeout_seconds'],checkpoint_step=binding['step'])
    if not isinstance(provider,SSRSteadyProvider)or provider.executed_history.checkpoint_step!=binding['step']:
        raise ValueError('Actual SSR provider/checkpoint step binding required')
    if getattr(provider,'protocol_config_id',None)!=protocol_id:
        raise ValueError('Actual generation provider differs from the declared SSR revision')

    class RuntimeModel(NativeModel):
        def __init__(self):
            self._rpc_lock=lifecycle_lock()
            super().__init__({'task':'classify_objects','instruction':INSTRUCTION,'audit_path':spec['audit_path'],
                              'noise_seed':42,'weights_binding':binding},policy=policy)
            if jax_module is not None:self._jax=jax_module
            if self._jax is None:raise ValueError('Actual JAX RNG implementation required')
            self.latest_injection=None;self.latest_commands=None
            self.metadata={'schema':'ecot.classify_objects.steady_SSR_runtime.v1',
                           'source_weights_binding':deepcopy(binding)}
            self.metadata.update(protocol_config_id=protocol_id,steady_subtask_enabled=True,ssr_enabled=True,
                context_fields=CONTEXT_FIELDS,task='classify_objects',step=binding['step'],inference_params='EMA',
                source_weights_identity_sha256=binding['weights_identity_sha256'],
                source_checkpoint=binding['checkpoint'],source_training_eval_ecot=False,effective_eval_ecot=True,
                planned_eval_ecot=True,eval_ecot=True,train_ecot=True,ecot=True,context_injected=True,max_token_len=512,
                source_training_dropout_probability=0.0,context_reference_sha256=REFERENCE_SHA,
                generator_reference=True,generator_reference_kind='exact_training_text_examples',generator_full_training_video_reference=False,
                generator_current3RGB=True,history_max_total_images=96,maximum_past_timepoints=93,
                past_history_cameras=['cam_high'],max_tokens=65536,policy_image_history=False,
                policy_uses_only_current3RGB_state14_and_composed_prompt=True,
                predict_steps=50,execute_steps=8,flow_steps=10,noise_seed=42,action_hz=25,action_limit=1100,
                semantic_quality_proven=False,checkpoint_training_identity_rewritten=False)

        def rng_snapshot(self):
            key_data=getattr(self._jax.random,'key_data',lambda x:x)
            return np.asarray(key_data(self.policy._rng)).copy()

        @exclusive_rpc
        def reset(self):
            super().reset();self.latest_injection=None;self.latest_commands=None;provider.reset(self.episode_id)

        @exclusive_rpc
        def update_obs(self,obs):return super().update_obs(deepcopy(obs))

        @exclusive_rpc
        def record_history_observation(self,obs):
            if self.episode_id is None:raise ValueError('History capture requires actual reset')
            before=(self.frame_index,self.predicted_at,self.current);rng=self.rng_snapshot()
            current=encode_obs(deepcopy(obs))
            from subtask_client import png_current_rgb
            result=provider.capture_history({c:png_current_rgb(current['images'][c])for c in CAMERAS},
                current['prompt'],state14=current['state'].tolist(),episode_id=self.episode_id,frame=obs['frame_index'])
            if (not np.array_equal(rng,self.rng_snapshot())or self.frame_index!=before[0]
                    or self.predicted_at!=before[1]or self.current is not before[2]):
                raise RuntimeError('SSR history capture altered policy frame/RNG')
            self.audit({'event':'actual_history_capture','episode_id':self.episode_id,'history_frame':obs['frame_index'],
                        'physics_clock':obs.get('physics_clock'),'policy_called':False,'RNG_advanced':False})
            return {'history_frame':obs['frame_index'],'episode_id':self.episode_id,'policy_called':False,
                    'RNG_advanced':False,'capture_count':result['history_source_index']+1,
                    'current_views':result['current_views']}

        @exclusive_rpc
        def get_action(self):
            if self.current is None or self.predicted_at==self.frame_index:raise ValueError('Fresh current observation required')
            if self.current['prompt']!=INSTRUCTION:raise ValueError('Exact original task instruction required')
            from subtask_client import png_current_rgb
            started=time.monotonic();rng=self.rng_snapshot()
            context,lineage=provider.generate({c:png_current_rgb(self.current['images'][c])for c in CAMERAS},
                INSTRUCTION,state14=self.current['state'].tolist(),episode_id=self.episode_id,frame=self.frame_index)
            if not np.array_equal(rng,self.rng_snapshot()):raise RuntimeError('Generator advanced policy RNG')
            validate_context(context)
            provider.validate_injection(context,lineage,episode_id=self.episode_id,frame=self.frame_index)
            prompt=actual_training_prompt(INSTRUCTION,context)
            actual=deepcopy(self.current);actual['prompt']=prompt
            _,sample_rng=self._jax.random.split(self.policy._rng)
            noise=np.asarray(self._jax.random.normal(sample_rng,(1,50,32)),dtype=np.float32)
            if noise.shape!=(1,50,32)or not np.isfinite(noise).all():raise ValueError('Native noise42 input invalid')
            raw=np.asarray(self.policy.infer(actual,noise=noise)['actions'],dtype=np.float32)
            commands,executed=to_commands(raw);self.predicted_at=self.frame_index
            from pi05_policy.freeze import plain
            self.latest_commands=plain(commands)
            self.latest_injection={'episode_id':self.episode_id,'execution_frame':self.frame_index,
                'context':deepcopy(context),'context_json':canonical(context),'prompt':prompt,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'lineage':deepcopy(lineage),
                'actual_injected':True,'checkpoint_step':binding['step'],'training_composition_byte_equal':True,
                'context_protocol':protocol_id}
            self.audit({'event':'inference','episode_id':self.episode_id,'executed_action_step':self.frame_index,
                'binding':self.metadata,'raw_current_state':self.current['state'].tolist(),'raw_instruction':INSTRUCTION,
                'prompt':prompt,'context':context,'lineage':lineage,'context_source_frame':self.frame_index,
                'generation_seconds':lineage['generation_total_seconds'],'inference_seconds':time.monotonic()-started,
                'camera_sha256':{c:hashlib.sha256(v.tobytes()).hexdigest()for c,v in self.current['images'].items()},
                'noise_shape':list(noise.shape),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                'raw_predicted_absolute_joint14':raw.tolist(),'executed_first8_absolute_joint14':executed.tolist(),
                'physics_clock':self.clock,'all_three_fields_generated_before_policy':True})
            return commands

        @exclusive_rpc
        def current_injection(self):
            if self.latest_injection is None or self.latest_injection['execution_frame']!=self.frame_index:
                raise ValueError('No current actual SSR injection')
            return deepcopy(self.latest_injection)

        @exclusive_rpc
        def acknowledge_executed_subtask(self,obs):
            ack=obs
            if self.latest_injection is None or self.latest_commands is None:raise ValueError('No actual SSR injection/actions to acknowledge')
            before=(self.frame_index,self.predicted_at,self.current);rng=self.rng_snapshot()
            receipt=provider.acknowledge_executed_subtask(ack,expected_injection=self.latest_injection,
                expected_actions=self.latest_commands,expected_clock=self.clock)
            if (not np.array_equal(rng,self.rng_snapshot())or self.frame_index!=before[0]
                    or self.predicted_at!=before[1]or self.current is not before[2]):
                raise RuntimeError('SSR ACK altered policy frame/RNG')
            self.audit(dict(receipt,event='actual_executed_subtask_ACK'))
            return receipt

        @exclusive_rpc
        def record_terminal_observation(self,obs):return super().record_terminal_observation(obs)

    return RuntimeModel()
