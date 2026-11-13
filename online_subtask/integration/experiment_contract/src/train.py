"""Isolated full-parameter OpenPI pi05 training on classify_objects.

Run with the project's pinned OpenPI source on PYTHONPATH. One process owns an
eight-device JAX FSDP mesh. Checkpoint numbers count COMPLETED optimizer updates.
No wandb/cloud logger, no implicit resume, no overwriting an existing run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import optimizer
from openpi.training import sharding
from openpi.training import weight_loaders

from cover_dataset import CoverBlocksDataset
from strict_prompt import strict_model_transforms
from checkpoint_recovery import select_resume_checkpoint, quarantine_uncommitted
from masked_pi05 import masked_compute_loss, temporal_mean
from fullparam_contract import (
    PARAMETER_POLICY, EXPERIMENT_PROTOCOL, validate_training_spec,
    audit_model_parameters, parameter_inventory, gradient_group_metrics,
    snapshot_update_proof, compare_update_proof,
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_train_config(spec):
    validate_training_spec(spec)
    model = pi0_config.Pi0Config(pi05=True, action_horizon=50, action_dim=32,
                                max_token_len=int(spec["max_token_len"]))
    return config_lib.TrainConfig(
        name=EXPERIMENT_PROTOCOL,
        exp_name="seed42",
        model=model,
        weight_loader=weight_loaders.CheckpointWeightLoader(spec["base_params"]),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=int(spec.get("warmup_steps", 200)),
            peak_lr=float(spec.get("peak_lr", 2.5e-5)),
            decay_steps=int(spec["num_train_steps"]),
            decay_lr=float(spec.get("decay_lr", 2.5e-6)),
        ),
        optimizer=optimizer.AdamW(),
        freeze_filter=nnx.Nothing,
        ema_decay=float(spec.get("ema_decay", 0.99)),
        seed=int(spec.get("seed", 42)),
        batch_size=int(spec.get("batch_size", 32)),
        num_train_steps=int(spec["num_train_steps"]),
        fsdp_devices=int(spec.get("fsdp_devices", 8)),
        wandb_enabled=False,
    )


class BatchSource:
    """Pinned transforms shared with the official ARX-X5 policy adapter."""
    def __init__(self, dataset, norm_dir, model_config, batch_size, seed, workers=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = math.ceil(len(dataset) / batch_size)
        self.norm_stats = normalize.load(Path(norm_dir))
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.input_transforms = [
            aloha_policy.AlohaInputs(adapt_to_pi=False),
            transforms.DeltaActions(transforms.make_bool_mask(6, -1, 6, -1)),
            transforms.Normalize(self.norm_stats, use_quantiles=True),
            *strict_model_transforms(model_config).inputs,
        ]
        self._order_cache = {}
        self._condition_audits = {}

    def data_config(self):
        return config_lib.DataConfig(norm_stats=self.norm_stats, asset_id="arx_x5_sim")

    def order(self, epoch, shuffle=True):
        key = (epoch, shuffle)
        if key not in self._order_cache:
            order = np.arange(len(self.dataset), dtype=np.int64)
            if shuffle:
                np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).shuffle(order)
            self._order_cache[key] = order
        return self._order_cache[key]

    def transform_one(self, index, epoch=None, *, return_condition=False):
        raw = self.dataset.get_raw(int(index), epoch=epoch)
        valid = np.asarray(raw["action_valid"], dtype=bool)
        if valid.shape != (50,) or not valid.any() or np.any(np.diff(valid.astype(int)) > 0):
            raise ValueError(f"Invalid/non-prefix tail validity at index {index}")
        data = {
            "images": {k: np.array(v, copy=True) for k, v in raw["images"].items()},
            "state": np.array(raw["state"], dtype=np.float32, copy=True),
            "actions": np.array(raw["actions"], dtype=np.float32, copy=True),
            "prompt": str(raw["prompt"]),
        }
        for ti, transform in enumerate(self.input_transforms):
            data = transform(data)
            if ti == 0 and any(np.asarray(v).shape != (224, 224, 3) for v in data["image"].values()):
                raise ValueError("AlohaInputs requires dataset CHW images and must produce HWC 224x224x3")
        if data["actions"].shape != (50, 32) or data["state"].shape != (32,):
            raise ValueError("Expected native OpenPI 32 padded channels")
        if not np.isfinite(data["actions"][valid]).all() or not np.isfinite(data["state"]).all():
            raise ValueError("Nonfinite normalized sample")
        if return_condition:
            condition = {"index": int(index), "epoch": epoch,
                "kept": raw["ecot_condition_kept"], "prompt": raw["prompt"],
                "model_tokens_sha256": hashlib.sha256(np.asarray(data["tokenized_prompt"]).tobytes()).hexdigest()}
            return data, valid, condition
        return data, valid

    def condition_audit(self, epoch, batch_index):
        return dict(self._condition_audits[(epoch, batch_index)])

    def batch(self, epoch, batch_index, shuffle=True):
        order = self.order(epoch, shuffle)
        indices = order[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        real_count = len(indices)
        if not real_count:
            raise IndexError(batch_index)
        # The last batch is padded only for static XLA shapes. These copies do
        # not contribute to the objective or sample/target counts.
        if real_count < self.batch_size:
            indices = np.concatenate([indices, np.repeat(indices[-1], self.batch_size - real_count)])
        samples = list(self.pool.map(functools.partial(self.transform_one, epoch=epoch, return_condition=True), indices))
        actual = [sample[2] for sample in samples[:real_count]]
        kept = sum(row["kept"] for row in actual)
        self._condition_audits[(epoch, batch_index)] = {
            "ecot_kept_samples": int(kept), "ecot_dropped_samples": int(real_count - kept),
            "ecot_padding_excluded": int(self.batch_size - real_count),
            "ecot_condition_epoch": int(epoch), "ecot_condition_batch_index": int(batch_index),
            "ecot_applied_inputs_sha256": canonical_sha(actual),
            "ecot_epoch_mask_sha256": self.dataset.mask_sha256(epoch)}
        data = jax.tree.map(lambda *x: np.stack(x), *(s[0] for s in samples))
        observation = model_lib.Observation.from_dict(data)
        return (observation, np.asarray(data["actions"]), np.stack([s[1] for s in samples]),
                np.arange(self.batch_size) < real_count)

    def close(self):
        self.pool.shutdown()


def train_step(config, rng, state, batch, *, fullparam_proof=False):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions, action_valid, sample_valid = batch
    train_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(model):
        per_time = masked_compute_loss(model, train_rng, observation, actions, action_valid, train=True)
        return temporal_mean(per_time, action_valid, sample_valid)[0]

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(model)
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(new_state, ema_params=jax.tree.map(
            lambda a, b: state.ema_decay * a + (1 - state.ema_decay) * b, state.ema_params, new_params))
    _, valid_targets = temporal_mean(jnp.zeros_like(action_valid, dtype=jnp.float32), action_valid, sample_valid)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "update_norm": optax.global_norm(updates),
        "valid_targets": valid_targets,
        "valid_samples": sample_valid.sum(),
    }
    if fullparam_proof:
        info.update(gradient_group_metrics(grads))
    return new_state, info


def validate_step(rng, state, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions, valid, sample_valid = batch
    return temporal_mean(masked_compute_loss(model, rng, observation, actions, valid, train=False), valid, sample_valid)


def state_fingerprint(state):
    """Hash every array, including Adam moments and EMA, one leaf at a time."""
    result = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        a = np.asarray(jax.device_get(leaf))
        result.append({"path": jax.tree_util.keystr(path), "shape": list(a.shape), "dtype": str(a.dtype),
                       "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    return {"tree_sha256": canonical_sha(result), "leaves": result}


def load_official_train(vendor_root):
    path = Path(vendor_root) / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("_pi05_native_train_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

