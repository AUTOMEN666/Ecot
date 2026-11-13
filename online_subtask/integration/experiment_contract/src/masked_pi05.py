"""OpenPI pi05 flow objective with explicit *temporal* validity.

The model, random distributions, flow convention and 32-channel averaging are
unchanged. Missing trajectory suffixes are excluded both as attention keys and
from loss. No model parameter or checkpoint name changes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


def temporal_mean(per_time, action_valid, sample_valid):
    """Mean over real control targets; batch padding never changes weighting."""
    valid = action_valid.astype(jnp.bool_) & sample_valid[:, None].astype(jnp.bool_)
    count = jnp.sum(valid)
    return jnp.sum(jnp.where(valid, per_time, 0.0)) / jnp.maximum(count, 1), count


def masked_compute_loss(model, rng, observation, actions, action_valid, *, train=False):
    if not model.pi05:
        raise ValueError("This isolated implementation only accepts pi05=True")
    if actions.shape[-2] != model.action_horizon or action_valid.shape != actions.shape[:-1]:
        raise ValueError("Temporal action validity shape mismatch")
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=train)
    valid = action_valid.astype(jnp.bool_)
    # Replace unavailable labels before interpolation. Invalid tokens are not
    # visible to valid tokens, so neither padding labels nor their noise leaks.
    actions = jnp.where(valid[..., None], actions, 0.0)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    suffix_mask = suffix_mask & valid
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    velocity = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    per_time = jnp.mean(jnp.square(velocity - u_t), axis=-1)
    return jnp.where(valid, per_time, 0.0)
