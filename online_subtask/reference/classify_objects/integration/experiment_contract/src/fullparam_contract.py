"""Pinned full-parameter pi0.5 selection and measured update evidence.

Shape audits are prerequisites, not evidence of optimization. GPU receipts also
require finite/nonzero gradients and actual raw/EMA changes in every group.
Only the explicit second-expert names below identify the action expert; vision
contains unrelated LayerNorm_1/Dense_1 names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiment_settings import PARAMETER_POLICY, PROTOCOL as EXPERIMENT_PROTOCOL, ECOT_VARIANT, EVAL_ECOT, DROPOUT_PROBABILITY
EXPERT_PATHS = frozenset({
    "final_norm_1/Dense_0/bias", "final_norm_1/Dense_0/kernel",
    "layers/attn/attn_vec_einsum_1/w", "layers/attn/kv_einsum_1/w",
    "layers/attn/q_einsum_1/w", "layers/mlp_1/gating_einsum", "layers/mlp_1/linear",
    "layers/pre_attention_norm_1/Dense_0/bias", "layers/pre_attention_norm_1/Dense_0/kernel",
    "layers/pre_ffw_norm_1/Dense_0/bias", "layers/pre_ffw_norm_1/Dense_0/kernel",
})
LANGUAGE_PATHS = frozenset({
    "embedder/input_embedding", "final_norm/scale", "layers/attn/attn_vec_einsum/w",
    "layers/attn/kv_einsum/w", "layers/attn/q_einsum/w", "layers/mlp/gating_einsum",
    "layers/mlp/linear", "layers/pre_attention_norm/scale", "layers/pre_ffw_norm/scale",
})
HEADS = frozenset({"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"})
TRAINABLE_GROUPS = frozenset({"vision", "language", "action_expert", *HEADS})
EXPECTED_LEAVES = {"vision": 23, "language": 9, "action_expert": 11, **{h: 2 for h in HEADS}}
EXPECTED_PARAMETERS = {"vision": 414803696, "language": 2508531712, "action_expert": 427932672,
                       "action_in_proj": 33792, "action_out_proj": 32800,
                       "time_mlp_in": 1049600, "time_mlp_out": 1049600}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def path_string(path):
    return path if isinstance(path, str) else "/".join(map(str, path))


def classify_parameter_path(path):
    path = path_string(path)
    if path.startswith("PaliGemma/img/"):
        return "vision"
    if path.startswith("PaliGemma/llm/"):
        relative = path[len("PaliGemma/llm/"):]
        if relative in EXPERT_PATHS:
            return "action_expert"
        if relative in LANGUAGE_PATHS:
            return "language"
    parts = path.split("/")
    if len(parts) == 2 and parts[0] in HEADS and parts[1] in {"kernel", "bias"}:
        return parts[0]
    raise ValueError(f"Unclassified parameter in pinned pi05 model: {path}")


def validate_training_spec(spec):
    required = {"frozen_vlm": False, "ecot_variant": ECOT_VARIANT, "train_ecot": True,
                "eval_ecot": EVAL_ECOT, "experiment_protocol": EXPERIMENT_PROTOCOL,
                "ecot_dropout_probability": DROPOUT_PROBABILITY, "max_token_len": 512}
    for key, value in required.items():
        if key not in spec or spec[key] != value or (
                isinstance(value, bool) and type(spec[key]) is not bool):
            raise ValueError(f"Full-parameter input-only compact-no-bbox contract requires {key}={value!r}")
    if Path(spec.get("base_params", "")).parts[-3:] != ("models", "pi05_base", "params"):
        raise ValueError("Full-parameter experiment must initialize from official models/pi05_base/params")
    if spec.get("freeze_filter") or spec.get("lora", False):
        raise ValueError("No freeze filter or LoRA is permitted in full-parameter training")
    if spec.get("text_ce_weight", 0.0) != 0.0:
        raise ValueError("Reasoning is input conditioning only; no text CE objective is permitted")


def validate_checkpoint_identity(identity, spec):
    validate_training_spec(spec)
    expected = {"frozen_vlm": False, "ecot_variant": ECOT_VARIANT, "train_ecot": True,
                "eval_ecot": EVAL_ECOT, "max_token_len": 512, "full_parameter_finetuning": True,
                "experiment_protocol": EXPERIMENT_PROTOCOL, "parameter_policy": PARAMETER_POLICY}
    if any(identity.get(key) != value or (isinstance(value, bool) and type(identity.get(key)) is not bool)
           for key, value in expected.items()):
        raise ValueError("Checkpoint full-parameter/group/prompt contract differs from this experiment")
    inventory_sha = identity.get("parameter_inventory_sha256", "")
    if len(inventory_sha) != 64 or any(c not in "0123456789abcdef" for c in inventory_sha):
        raise ValueError("Checkpoint lacks audited native full-parameter inventory identity")
    return expected


def inventory_from_rows(rows):
    """Pure CPU validation of complete, actually selected NNX parameter rows."""
    groups, paths = {}, set()
    for row in rows:
        path = row["path"]
        if path in paths or row["group"] != classify_parameter_path(path):
            raise ValueError(f"Duplicate/misclassified native parameter: {path}")
        paths.add(path)
        if row["trainable"] is not True:
            raise ValueError(f"Full-parameter optimizer/gradient selection omitted {path}")
        count = math.prod(row["shape"])
        if count != row["parameters"]:
            raise ValueError(f"Parameter shape/count mismatch: {path}")
        group = groups.setdefault(row["group"], {"leaves": 0, "parameters": 0, "trainable": True})
        group["leaves"] += 1
        group["parameters"] += count
    if {g: v["leaves"] for g, v in groups.items()} != EXPECTED_LEAVES:
        raise ValueError("Native parameter leaves differ from pinned pi05 model")
    if {g: v["parameters"] for g, v in groups.items()} != EXPECTED_PARAMETERS:
        raise ValueError("Native parameter counts differ from pinned pi05 model")
    rows = sorted(rows, key=lambda row: row["path"])
    return {"schema": "pi05_fullparam_inventory_v1", "passed": True,
            "parameter_policy": PARAMETER_POLICY, "groups": groups, "leaves": rows,
            "parameter_leaves": len(rows), "trainable_leaves": len(rows),
            "trainable_parameters": sum(r["parameters"] for r in rows), "frozen_parameters": 0,
            "inventory_sha256": canonical_sha(rows),
            "scope": "NNX parameter selection and shape/dtype audit only; actual updates require GPU proof"}


def parameter_inventory(params, trainable_filter):
    import flax.nnx as nnx
    params = params.filter(nnx.Param)
    selected = set(params.filter(trainable_filter).flat_state())
    rows = []
    for path, variable in sorted(params.flat_state().items()):
        value = variable.value
        rows.append({"path": path_string(path), "shape": list(value.shape), "dtype": str(value.dtype),
                     "parameters": math.prod(value.shape), "group": classify_parameter_path(path),
                     "trainable": path in selected})
    if len(selected) != len(rows):
        raise ValueError("Full-parameter selection differs from complete native parameter tree")
    return inventory_from_rows(rows)


def audit_model_parameters(config):
    import flax.nnx as nnx
    import jax
    params = nnx.eval_shape(lambda: nnx.state(config.model.create(jax.random.key(config.seed)), nnx.Param))
    return parameter_inventory(params, config.trainable_filter)


def gradient_group_metrics(grads):
    """JIT-compatible scalars; called only by explicit update-proof smoke runs."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    groups = {group: [] for group in TRAINABLE_GROUPS}
    result = {}
    for path, variable in sorted(grads.filter(nnx.Param).flat_state().items()):
        group = classify_parameter_path(path)
        value = variable.value
        # Check the full leaf, including entries outside the norm's reduction.
        finite = jnp.all(jnp.isfinite(value))
        nonzero = jnp.any(value != 0)
        norm2 = jnp.sum(jnp.square(value.astype(jnp.float32)))
        groups[group].append((finite, nonzero, norm2))
    if {g: len(values) for g, values in groups.items()} != EXPECTED_LEAVES:
        raise ValueError("Gradient tree does not cover all native pi05 parameter leaves")
    for group, values in groups.items():
        result[f"fullparam_grad/{group}/finite"] = jnp.all(jnp.stack([v[0] for v in values]))
        result[f"fullparam_grad/{group}/nonzero_leaves"] = jnp.sum(jnp.stack([v[1] for v in values]))
        result[f"fullparam_grad/{group}/norm"] = jnp.sqrt(jnp.sum(jnp.stack([v[2] for v in values])))
    return result


def snapshot_update_proof(state):
    import flax.nnx as nnx
    import jax
    import numpy as np
    if state.ema_params is None:
        raise ValueError("Full-parameter experiment requires native EMA")
    trees = {}
    for name, params in (("raw", state.params), ("ema", state.ema_params)):
        rows = []
        for path, variable in sorted(params.filter(nnx.Param).flat_state().items()):
            array = np.asarray(jax.device_get(variable.value))
            rows.append({"path": path_string(path), "shape": list(array.shape), "dtype": str(array.dtype),
                         "group": classify_parameter_path(path), "finite": bool(np.isfinite(array).all()),
                         "sha256": hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()})
        trees[name] = rows
    return {"schema": "pi05_fullparam_update_snapshot_v1", "completed_step": int(state.step),
            "trees": trees, "snapshot_sha256": canonical_sha(trees)}


def compare_update_proof(before, after, gradient_metrics):
    for snapshot in (before, after):
        if snapshot["snapshot_sha256"] != canonical_sha(snapshot["trees"]):
            raise ValueError("Full-parameter snapshot digest changed")
    first, last = before["completed_step"], after["completed_step"]
    if (first, last) != (0, 2):
        raise ValueError("Full-parameter update smoke must measure the first two fresh-base optimizer updates")
    if [m.get("step") for m in gradient_metrics] != [first + 1, last]:
        raise ValueError("Gradient evidence must cover both measured updates in order")
    for metric in gradient_metrics:
        for group in TRAINABLE_GROUPS:
            prefix = f"fullparam_grad/{group}/"
            if metric.get(prefix + "finite") != 1 or not (
                    math.isfinite(metric.get(prefix + "norm", float("nan"))) and metric[prefix + "norm"] > 0
                    and 0 < metric.get(prefix + "nonzero_leaves", 0) <= EXPECTED_LEAVES[group]):
                raise ValueError(f"Missing finite/nonzero gradient evidence for {group} update {metric['step']}")
    result = {"schema": "pi05_fullparam_update_receipt_v1", "parameter_policy": PARAMETER_POLICY,
              "passed": True, "accepted": True, "all_groups_finite_nonzero_gradients": True,
              "completed_updates_checked": 2, "before_step": first, "after_step": last,
              "before_sha256": before["snapshot_sha256"], "after_sha256": after["snapshot_sha256"],
              "gradient_metrics": gradient_metrics, "trees": {}}
    for name in ("raw", "ema"):
        old = {r["path"]: r for r in before["trees"][name]}
        new = {r["path"]: r for r in after["trees"][name]}
        if len(old) != len(before["trees"][name]) or len(new) != len(after["trees"][name]) or old.keys() != new.keys():
            raise ValueError("Duplicate/changed parameter paths in full-parameter snapshots")
        counts, changed, unchanged = {}, {}, []
        for path, row in old.items():
            other = new[path]
            group = classify_parameter_path(path)
            counts[group] = counts.get(group, 0) + 1
            if row["group"] != group or any(row[k] != other[k] for k in ("shape", "dtype", "group")):
                raise ValueError(f"Parameter metadata changed: {path}")
            if row.get("finite") is not True or other.get("finite") is not True:
                raise ValueError(f"Nonfinite {name} model weights: {path}")
            if row["sha256"] != other["sha256"]:
                changed.setdefault(group, []).append(path)
            else:
                unchanged.append(path)
        if counts != EXPECTED_LEAVES or set(changed) != TRAINABLE_GROUPS:
            raise ValueError(f"Missing complete native tree or actual {name} updates: {sorted(TRAINABLE_GROUPS-set(changed))}")
        result["trees"][name] = {"changed_groups": sorted(changed), "changed_paths_by_group": changed,
                                  "unchanged_paths": unchanged, "all_parameters_finite": True}
    return result


def main():
    parser = argparse.ArgumentParser(description="CPU-only eval_shape full-parameter selection audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import sys
    spec = json.loads(Path(args.config).read_text())
    validate_training_spec(spec)
    vendor = Path(spec["vendor_root"])
    sys.path[:0] = [str(vendor / "src"), str(vendor / "packages/openpi-client/src"), str(vendor.parent / "lerobot")]
    import jax
    from train import build_train_config
    report = audit_model_parameters(build_train_config(spec))
    report.update(gpu_used=False, model_weights_allocated=False, devices=[str(d) for d in jax.devices()],
                  frozen_vlm=False, ecot_variant=ECOT_VARIANT, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "leaves"}), flush=True)


if __name__ == "__main__":
    main()
