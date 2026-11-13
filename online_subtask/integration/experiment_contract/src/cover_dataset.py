"""Read-only, current-frame RoboDojo data for the isolated pi0.5 experiment.

Public samples deliberately remain in absolute joint coordinates.  The trainer
must apply AlohaInputs(adapt_to_pi=False), DeltaActions and Normalize once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])
PREPARED_SCHEMA = "pi05_robodojo_current3_prepared_v1"


def validate_data_contract(manifest: dict) -> dict:
    """Fail closed on the explicit task, whitelist and all-training split contract.

    The whitelist is supplied by the independently audited source inventory;
    it is never inferred from whichever files happen to be present.
    """
    contract = manifest.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError("Missing explicit data_contract")
    task = contract.get("task_name")
    if not isinstance(task, str) or not task or manifest.get("task_name", task) != task:
        raise ValueError("Missing or inconsistent task_name")
    ids = contract.get("episode_indices")
    if (not isinstance(ids, list) or len(ids) != 100
            or any(type(e) is not int or e < 0 for e in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Expected a sorted, unique, explicit 100-episode whitelist")
    required = {
        "split_seed": 42,
        "split_algorithm": "all_episodes_train",
        "val_episode_count": 0,
        "fps": 25,
        "action_dim": 14,
        "horizon": 50,
        "alignment": "action_t_equals_state_t_plus_1",
        "cameras": list(CAMERAS),
        "source_image_shape": [480, 640, 3],
    }
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError((key, "unsupported or missing data contract", contract.get(key), value))
    shuffled = ids.copy()
    random.Random(contract["split_seed"]).shuffle(shuffled)
    val_ids = set(shuffled[:contract["val_episode_count"]])
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or sorted(e["episode_index"] for e in episodes) != ids:
        raise ValueError("Episodes do not exactly match the explicit whitelist")
    for ep in episodes:
        eid = ep["episode_index"]
        if ep.get("task_name") != task or ep.get("fps") != contract["fps"]:
            raise ValueError((eid, "task or FPS mismatch"))
        if ep.get("split") != ("val" if eid in val_ids else "train"):
            raise ValueError((eid, "train-all split changed"))
        if not isinstance(ep.get("instruction"), str) or not ep["instruction"].strip():
            raise ValueError((eid, "missing task instruction"))
        n = ep.get("frame_count")
        if type(n) is not int or n < 2 or ep.get("valid_action_count") != n - 1:
            raise ValueError((eid, "invalid frame or action count"))
    return contract


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def training_validation_enabled(spec: dict, manifest: dict) -> bool:
    """Require the explicit train-all policy; never synthesize a validation loss."""
    contract = validate_data_contract(manifest)
    if (spec.get("validation_enabled") is not False
            or spec.get("validation_reason") != "all_100_demonstrations_used_for_training_no_heldout_split"
            or contract["val_episode_count"] != 0
            or any(ep["split"] != "train" for ep in manifest["episodes"])):
        raise ValueError("This experiment requires explicit train100/val0 with validation disabled")
    return False


def absolute_chunk(state: np.ndarray, action: np.ndarray, t: int, horizon: int = 50):
    """action[t] already targets state[t+1]; never shift this source again."""
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError("Expected equal [L,14] state and action arrays")
    n = len(state) - 1
    if not 0 <= t < n:
        raise IndexError((t, n))
    valid_len = min(horizon, n - t)
    target = np.zeros((horizon, 14), dtype=np.float32)
    target[:valid_len] = action[t : t + valid_len]
    mask = np.arange(horizon) < valid_len
    return target, mask


def delta_chunk(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The same current state anchors every arm target, grippers stay absolute."""
    result = np.array(action, dtype=np.float32, copy=True)
    result[..., ARM_MASK] -= np.asarray(state, np.float32)[..., ARM_MASK]
    return result


class RoboDojoDataset:
    def __init__(self, prepared_manifest_path: str | Path, split: str = "train", horizon: int = 50):
        self.manifest_path = Path(prepared_manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != PREPARED_SCHEMA:
            raise ValueError("Unexpected or unprepared manifest")
        if self.manifest.get("complete") is not True:
            raise ValueError("Dataset preparation has not completed")
        self.data_contract = validate_data_contract(self.manifest)
        if horizon != self.manifest["horizon"]:
            raise ValueError("Requested horizon differs from bound statistics")
        if split not in ("train", "val"):
            raise ValueError(split)
        self.split, self.horizon = split, horizon
        self.norm_stats_path = self.manifest["norm_stats_path"]
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        self._cum = np.cumsum([0] + [e["valid_action_count"] for e in self.episodes])
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._images: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self):
        return int(self._cum[-1])

    def locate(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ei = int(np.searchsorted(self._cum, index, side="right") - 1)
        return ei, int(index - self._cum[ei])

    def epoch_indices(self, seed: int, epoch: int):
        return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(self))

    def get_raw(self, index: int):
        ei, t = self.locate(index)
        ep = self.episodes[ei]
        if ei not in self._arrays:
            with np.load(ep["source_arrays_cache"], allow_pickle=False) as a:
                self._arrays[ei] = (a["state"].astype(np.float32), a["action"].astype(np.float32))
        state, action = self._arrays[ei]
        targets, mask = absolute_chunk(state, action, t, self.horizon)
        images = {}
        for camera in CAMERAS:
            key = (ei, camera)
            if key not in self._images:
                self._images[key] = np.load(ep["images_224"][camera]["path"], mmap_mode="r", allow_pickle=False)
            # Official AlohaInputs unconditionally rearranges CHW -> HWC.
            images[camera] = np.array(self._images[key][t].transpose(2, 0, 1), copy=True)
        return {
            "images": images,
            "state": np.array(state[t], copy=True),
            "actions": targets,
            "action_valid": mask,
            "prompt": ep["instruction"],
            "episode_index": ep["episode_index"],
            "frame_index": t,
        }

    __getitem__ = get_raw


# Kept only for the existing trainer import. The task and episode whitelist are
# supplied by the manifest, not by this historical class name.
CoverBlocksDataset = RoboDojoDataset

