"""Pure-CPU identity, admission and schedule for the fresh independently conditioned experiment.

This module imports neither JAX nor CUDA. Existing receipts cannot admit this
experiment: config, source and actual data/token audits bind its own inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from experiment_settings import (PROTOCOL, VARIANT, EVAL_STEPS, ECOT_VARIANT,
    RETAINED_FIELDS, EVAL_ECOT, DROPOUT_PROBABILITY, DROPOUT_MODE, PROMPT_VERSION,
    EVAL_EPISODES, EVAL_SCENE_SEEDS, EVAL_LAYOUT_IDS)
VARIANTS = (VARIANT,)
TRAIN_SAMPLES = 69562
STEPS_PER_EPOCH = math.ceil(TRAIN_SAMPLES / 32)
NATIVE_FILES = (
    "scripts/train.py", "src/openpi/models/pi0.py", "src/openpi/models/pi0_config.py",
    "src/openpi/transforms.py", "src/openpi/training/optimizer.py",
    "src/openpi/training/checkpoints.py", "src/openpi/policies/aloha_policy.py",
)
LOCAL_FILES = ("train_ablation.py", "ablation_training_contract.py", "subtask_data.py",
               "preflight_subtask.py", "train.py", "strict_prompt.py", "cover_dataset.py",
               "masked_pi05.py", "fullparam_contract.py", "checkpoint_recovery.py")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing SHA256: {name}")


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_spec(spec):
    required = {
        "experiment_protocol": PROTOCOL, "task_name": "classify_objects",
        "batch_size": 32, "fsdp_devices": 8, "seed": 42, "warmup_steps": 200,
        "peak_lr": 2.5e-5, "decay_lr": 2.5e-6, "ema_decay": 0.99,
        "num_train_steps": 43480, "training_end_step": 20000, "max_token_len": 512,
        "frozen_vlm": False, "train_ecot": True, "eval_ecot": EVAL_ECOT,
        "validation_enabled": False,
        "validation_reason": "all_100_demonstrations_used_for_training_no_heldout_split",
        "ecot_variant": ECOT_VARIANT, "text_ce_weight": 0.0,
        "eval_steps": list(EVAL_STEPS), "eval_episodes": EVAL_EPISODES, "eval_scene_seeds": list(EVAL_SCENE_SEEDS), "eval_layout_ids": list(EVAL_LAYOUT_IDS),
        "ecot_retained_fields": list(RETAINED_FIELDS),
        "ecot_dropout_probability": DROPOUT_PROBABILITY, "dropout_mode": DROPOUT_MODE, "dropout_seed": 42,
        "subtask_boundary_dropout": False, "prompt_version": PROMPT_VERSION,
    }
    for key, value in required.items():
        if key not in spec or spec[key] != value or (isinstance(value, bool) and type(spec[key]) is not bool):
            raise ValueError(f"Ablation contract requires {key}={value!r}")
    if spec.get("ablation_variant") not in VARIANTS:
        raise ValueError("Unknown ablation variant")
    if spec.get("freeze_filter") or spec.get("lora", False):
        raise ValueError("The independent experiment requires full-parameter optimization")
    for key, value in (("subtask_action_mask", False), ("action_horizon", 50),
                       ("ecot_dropout_probability", DROPOUT_PROBABILITY), ("eval_seed", 1),
                       ("noise_seed", 42), ("execute", 8), ("flow_steps", 10)):
        if key in spec and (spec[key] != value or (isinstance(value, bool) and type(spec[key]) is not bool)):
            raise ValueError(f"Optional explicit {key} contradicts the confirmed setting")
    if Path(spec.get("base_params", "")).parts[-3:] != ("models", "pi05_base", "params"):
        raise ValueError("Fresh initialization must use the bound official pi05_base params")
    for key in ("vendor_root", "base_params", "dataset_manifest",
                "norm_dir", "output_dir", "smoke_output_dir", "data_audit",
                "token_audit", "preflight_cpu", "jax_cache_dir", "ecot_context_manifest"):
        if not isinstance(spec.get(key), str) or not Path(spec[key]).is_absolute():
            raise ValueError(f"Explicit absolute {key} required")
    for key in ("boundary_manifest", "context_cache"):
        if spec.get(key): raise ValueError("Direct cache overrides and boundary manifests are forbidden: " + key)
    if Path(spec["output_dir"]).resolve() == Path(spec["smoke_output_dir"]).resolve():
        raise ValueError("Formal and smoke output namespaces must be independent")
    return spec


def source_hashes(project_root, spec):
    root = Path(project_root)
    # Seal actual admission, data, model, transforms and reused support sources.
    # Keep the native dependency closure rather than a hand-maintained shortlist.
    own = set()
    for directory in ("src",):
        own.update(p for p in (root / directory).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".sh"))
    result = {str(p.relative_to(root)): file_sha(p) for p in sorted(own)}
    vendor = Path(spec["vendor_root"])
    native = set((vendor / "src/openpi").rglob("*.py"))
    native.add(vendor / "scripts/train.py")
    result.update({"vendor/" + str(p.relative_to(vendor)): file_sha(p)
                   for p in sorted(native)})
    return result


def audit_paths(spec):
    from subtask_data import source_context_cache_path
    return {"data_audit": Path(spec["data_audit"]), "token_audit": Path(spec["token_audit"]),
            "dataset_manifest": Path(spec["dataset_manifest"]),
            "norm_stats": Path(spec["norm_dir"]) / "norm_stats.json",
            "ecot_context_manifest": Path(spec["ecot_context_manifest"]),
            "ecot_context_cache": source_context_cache_path(spec["ecot_context_manifest"])}


def validate_preflight(project_root, spec):
    validate_spec(spec); receipt = read_json(spec["preflight_cpu"])
    if (receipt.get("schema") != PROTOCOL + ".preflight.v1"
            or receipt.get("passed") is not True or receipt.get("cpu_only") is not True
            or receipt.get("ablation_variant") != VARIANT
            or receipt.get("config_sha256") != canonical_sha(spec)
            or receipt.get("train_samples") != TRAIN_SAMPLES
            or receipt.get("steps_per_epoch") != STEPS_PER_EPOCH
            or receipt.get("train_ecot") is not True or receipt.get("eval_ecot") is not EVAL_ECOT):
        raise ValueError("Own condition-specific CPU receipt/config/count mismatch")
    require_sha(receipt.get("base_params_tree_sha256"), "official base tree")
    if receipt.get("source_sha256") != source_hashes(project_root, spec):
        raise ValueError("Current baseline/native source changed since own CPU acceptance")
    expected = {key: file_sha(path) for key, path in audit_paths(spec).items()}
    if receipt.get("audit_sha256") != expected:
        raise ValueError("Current numeric data/normalizer/actual audit bytes changed")
    data = read_json(spec["data_audit"]); token = read_json(spec["token_audit"])
    for audit, schema in ((data, PROTOCOL + ".data.v1"),
                          (token, PROTOCOL + ".token.v1")):
        if (audit.get("schema") != schema or audit.get("passed") is not True
                or audit.get("ablation_variant") != VARIANT or audit.get("training_starts") != TRAIN_SAMPLES
                or audit.get("dataset_manifest_sha256") != expected["dataset_manifest"]
                or audit.get("retained_projection_fields") != list(RETAINED_FIELDS)
                or audit.get("retained_branch_checked_starts") != TRAIN_SAMPLES
                or audit.get("dropped_branch_checked_starts") != TRAIN_SAMPLES):
            raise ValueError("Actual condition-specific data/token traversal incomplete or wrong input")
    if (token.get("all_starts_checked") is not True or token.get("max_token_len") != 512
            or token.get("normalized_state14_checked") is not True or token.get("no_truncation") is not True):
        raise ValueError("Every native original-instruction/state token sequence must pass unchanged")
    if (type(token.get("max_observed_tokens")) is not int or not 0 < token["max_observed_tokens"] <= 512
            or token.get("native_tokenizer_model_sha256") != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"):
        raise ValueError("Actual native tokenizer/model/capacity proof is missing")
    require_sha(token.get("actual_native_tokens_sha256"), "native token traversal")
    from subtask_data import MASK_SCHEME, MASK_DOMAIN
    require_sha(data.get("subtask_projection_sha256"), "actual original Subtask projection")
    if (data.get("subtask_projection_sha256") != token.get("subtask_projection_sha256")
            or token.get("all_starts_both_branches_checked") is not True
            or data.get("dropout_mask_scheme") != MASK_SCHEME
            or data.get("dropout_mask_domain") != list(MASK_DOMAIN)):
        raise ValueError("Actual projection/both-branch/dropout algorithm audit changed")
    from subtask_data import condition_mask
    masks = data.get("actual_epoch_masks", [])
    if [row.get("epoch") for row in masks] != [0,1,2]:
        raise ValueError("Actual epoch0/1/2 condition masks required")
    for row in masks:
        actual = condition_mask(TRAIN_SAMPLES,row['epoch'])
        expected = {"epoch":row['epoch'],"kept":int(actual.sum()),"dropped":int((~actual).sum()),
                    "sha256":hashlib.sha256(actual.tobytes()).hexdigest()}
        if row != expected:raise ValueError("Actual condition mask differs from the frozen independent RNG algorithm")
    if DROPOUT_PROBABILITY == 0.0:
        if any(row['kept'] != TRAIN_SAMPLES for row in masks):raise ValueError("Subtask-only training must retain every condition")
    elif len({row['sha256']for row in masks})!=3 or any(not 0<row['kept']<TRAIN_SAMPLES for row in masks):
        raise ValueError("Independent Bernoulli masks must vary by epoch and include both branches")
    batches = data.get("actual_batch_path_checks", [])
    if (len(batches) < 4 or any(row.get("ecot_kept_samples", -1) + row.get("ecot_dropped_samples", -1)
            != row.get("real_samples") for row in batches)
            or data.get("cold_resume_condition_replay_exact") is not True):
        raise ValueError("Actual explicit-epoch threaded batch/dropout and coldcursor replay proof required")
    from subtask_data import batch_condition_counts
    for row in batches:
        expected = batch_condition_counts(row['ecot_condition_epoch'],row['ecot_condition_batch_index'])
        if any(row.get(key)!=value for key,value in expected.items()):
            raise ValueError("Actual batch condition counts differ from independent sampler-mask reconstruction")
        require_sha(row.get('ecot_applied_inputs_sha256'),'actual transformed batch condition inputs')
    if data.get('conditioning_projection_sha256')!=data['subtask_projection_sha256'] or token.get('conditioning_projection_sha256')!=data['subtask_projection_sha256']:
        raise ValueError("Condition projection aliases differ")
    payloads = data.get("payload_bindings", [])
    if len(payloads) != 400 or data.get("all_real_action_windows_verified") is not True:
        raise ValueError("All100 numeric and300 image payloads plus full50 targets must be audited")
    manifest = read_json(spec["dataset_manifest"])
    from subtask_data import validate_manifest
    validate_manifest(manifest)
    actual_manifest_files = []
    for episode in manifest["episodes"]:
        actual_manifest_files.append((episode["source_arrays_cache"], episode["source_arrays_cache_sha256"]))
        actual_manifest_files.extend((item["path"], item["sha256"]) for item in episode["images_224"].values())
    if sorted((row["path"], row["sha256"]) for row in payloads) != sorted(actual_manifest_files):
        raise ValueError("Hashed payload inventory differs from all400 actual manifest inputs")
    for row in payloads:
        stat = Path(row["path"]).stat()
        if any(getattr(stat, key) != row[key] for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("Previously hashed numeric/image payload changed: " + row["path"])
        require_sha(row.get("sha256"), "actual numeric/image payload")
    return receipt


def sampler_cursor(completed_step, *, steps_per_epoch=STEPS_PER_EPOCH):
    if type(completed_step) is not int or completed_step < 0 or steps_per_epoch <= 0:
        raise ValueError("Nonnegative completed optimizer update required")
    return completed_step // steps_per_epoch, completed_step % steps_per_epoch


def validate_phase(start, stop, *, smoke=False, restore_only=False):
    if type(start) is not int or type(stop) is not int or start < 0:
        raise ValueError("Integer cumulative optimizer steps required")
    if restore_only:
        if start <= 0 or stop != start:
            raise ValueError("Restore-only must preserve an existing completed step")
        return
    if smoke:
        if (start, stop) != (0, 2):
            raise ValueError("Smoke measures exactly two fresh-base updates")
    elif (start, stop) not in ((0, 5000), *zip(EVAL_STEPS, EVAL_STEPS[1:])):
        raise ValueError("Fresh0->5000 then exact authorized full-state milestones through20000")


def verify_smoke(project_root, spec, preflight):
    """Validate actual update, exact cold-restore and source/config bindings."""
    root = Path(project_root)
    run = Path(spec.get("smoke_output_dir", str(Path(spec["output_dir"]).parent / "smoke")))
    update = read_json(run / "fullparam_update_receipt.json")
    restore = read_json(run / "restore_check.json")
    identity = read_json(run / "run_identity.json")
    if (update.get("passed") is not True or update.get("after_step") != 2
            or restore.get("passed") is not True or restore.get("completed_step") != 2
            or restore.get("fullparam_update_proof_verified") is not True
            or update.get("identity") != identity or restore.get("identity") != identity
            or identity.get("config_sha256") != canonical_sha(spec)
            or identity.get("cpu_acceptance_sha256") != file_sha(spec["preflight_cpu"])
            or identity.get("official_base_params_tree_sha256") != preflight["base_params_tree_sha256"]
            or identity.get("source_sha256") != source_hashes(root, spec)
            or identity.get("ablation_variant") != spec["ablation_variant"]
            or restore.get("fullparam_update_receipt_sha256") != file_sha(run / "fullparam_update_receipt.json")):
        raise ValueError("Own fresh two-update/cold-restore smoke is not complete or identity-bound")
    # Recompute proof from its actual arrays' snapshots, not a passed boolean.
    from fullparam_contract import compare_update_proof
    computed = compare_update_proof(read_json(run / "fullparam_before.json"),
                                    read_json(run / "fullparam_after.json"), update["gradient_metrics"])
    if any(update.get(k) != value for k, value in computed.items()):
        raise ValueError("Measured full-parameter smoke proof content differs")
    fingerprints = read_json(run / "checkpoints/2/state_fingerprint.json")
    if restore.get("tree_sha256") != fingerprints.get("tree_sha256"):
        raise ValueError("Cold-restored state fingerprint differs")
    return {"schema": PROTOCOL + ".smoke_acceptance.v1", "passed": True,
            "ablation_variant": spec["ablation_variant"], "config_sha256": canonical_sha(spec),
            "preflight_sha256": file_sha(spec["preflight_cpu"]),
            "update_receipt_sha256": file_sha(run / "fullparam_update_receipt.json"),
            "restore_sha256": file_sha(run / "restore_check.json"), "completed_step": 2,
            "source_sha256": source_hashes(root, spec)}


def reconcile_metrics(path, committed_step):
    """Preserve uncommitted history before replay; never double-count updates."""
    path = Path(path)
    if not path.exists():
        if committed_step:
            raise ValueError("A committed run is missing its metrics history")
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [row.get("step") for row in rows]
    if steps != list(range(1, len(rows) + 1)) or len(rows) < committed_step:
        raise ValueError("Metrics do not prove a contiguous optimizer history")
    if len(rows) == committed_step:
        return None
    archive = path.with_name("metrics_uncommitted_" + str(time.time_ns()) + ".jsonl")
    archive.write_bytes(path.read_bytes())
    temp = path.with_suffix(".reconciled.tmp")
    temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:committed_step]))
    temp.replace(path)
    return str(archive)
