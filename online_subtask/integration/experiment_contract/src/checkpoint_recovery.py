"""Application-commit selection over Orbax's already committed checkpoints.

An Orbax commit alone does not promise that our sampler/RNG metadata was
committed. Preserve and quarantine incomplete application commits before
retrying their step numbers; never restore a different experiment identity.
This module deliberately has no model, JAX, or CUDA dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


def select_resume_checkpoint(checkpoint_root, native_steps, identity):
    root = Path(checkpoint_root)
    skipped = []
    for step in sorted({int(s) for s in native_steps}, reverse=True):
        path = root / str(step) / "resume_metadata.json"
        if not path.is_file():
            skipped.append({"step": step, "reason": "missing_application_commit"})
            continue
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append({"step": step, "reason": "unreadable_application_commit", "error": str(exc)})
            continue
        if not isinstance(meta, dict) or "identity" not in meta:
            skipped.append({"step": step, "reason": "incomplete_application_commit"})
            continue
        if meta["identity"] != identity:
            # Do not silently roll past a foreign run and hide an identity mix.
            raise RuntimeError(f"Checkpoint identity mismatch at completed step {step}; refusing restore")
        required = {"completed_step", "epoch", "next_batch_in_epoch"}
        if not required.issubset(meta):
            skipped.append({"step": step, "reason": "incomplete_application_commit"})
            continue
        epoch_size = int(identity["steps_per_epoch"])
        if (meta["completed_step"] != step or meta["epoch"] != step // epoch_size
                or meta["next_batch_in_epoch"] != step % epoch_size):
            raise RuntimeError(f"Checkpoint completed-step/sampler cursor mismatch at {step}")
        return {"selected_step": step, "metadata": meta, "skipped": skipped}
    return {"selected_step": None, "metadata": None, "skipped": skipped}


def quarantine_uncommitted(checkpoint_root, skipped, quarantine_root):
    """Reversibly move incomplete checkpoints out of Orbax's step namespace."""
    root = Path(checkpoint_root)
    quarantine = Path(quarantine_root)
    moves = []
    for item in skipped:
        source = root / str(int(item["step"]))
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"Expected an owned checkpoint directory, refusing move: {source}")
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / f"step_{int(item['step'])}_{time.time_ns()}"
        source.rename(destination)
        moves.append({**item, "original_path": str(source), "preserved_path": str(destination)})
    return moves
