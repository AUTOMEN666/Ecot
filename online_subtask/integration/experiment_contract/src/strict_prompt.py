"""Native PaliGemma encoding with explicit fail-closed token capacity.

Only the length failure policy differs from the pinned tokenizer. Native Task / 
State / Action serialization and state discretization are preserved exactly.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np

from experiment_settings import PROMPT_VERSION, ECOT_VARIANT, RETAINED_FIELDS

def native_unpadded_tokens(native, prompt, state=None):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:
        state = np.asarray(state)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Tokenization must see finite normalized joint14 before padding")
        discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nAction: "
        tokens = native._tokenizer.encode(text, add_bos=True)
    else:
        tokens = native._tokenizer.encode(cleaned, add_bos=True) + native._tokenizer.encode("\n")
    return np.asarray(tokens)


class StrictPaligemmaTokenizer:
    def __init__(self, max_len, native=None):
        if type(max_len) is not int or max_len <= 0:
            raise ValueError("Explicit positive max_token_len required")
        if native is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            native = PaligemmaTokenizer(max_len)
        model_sha = hashlib.sha256(native._tokenizer.serialized_model_proto()).hexdigest()
        if model_sha != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6":
            raise ValueError("SentencePiece model differs from audited pinned native tokenizer")
        self.native, self.max_len = native, max_len

    def tokenize(self, prompt, state=None):
        expected = native_unpadded_tokens(self.native, prompt, state)
        if len(expected) > self.max_len:
            raise ValueError(f"Prompt/state token overflow: {len(expected)} > {self.max_len}; refusing truncation")
        tokens, mask = self.native.tokenize(prompt, state)
        if (len(tokens) != self.max_len or int(np.asarray(mask).sum()) != len(expected)
                or not np.array_equal(np.asarray(tokens)[np.asarray(mask, bool)], expected)):
            raise RuntimeError("Pinned native tokenization changed or truncated state/Action suffix")
        return tokens, mask


def strict_model_transforms(model_config):
    from openpi import transforms
    from openpi.training.config import ModelTransformFactory
    group = ModelTransformFactory()(model_config)
    changed = 0
    inputs = []
    for transform in group.inputs:
        if isinstance(transform, transforms.TokenizePrompt):
            transform = dataclasses.replace(transform,
                tokenizer=StrictPaligemmaTokenizer(model_config.max_token_len))
            changed += 1
        inputs.append(transform)
    if changed != 1:
        raise RuntimeError("Expected exactly one native TokenizePrompt")
    return dataclasses.replace(group, inputs=inputs)


def strict_aloha_config(**kwargs):
    from openpi.training.config import LeRobotAlohaDataConfig
    @dataclasses.dataclass(frozen=True)
    class StrictAlohaDataConfig(LeRobotAlohaDataConfig):
        def create(self, assets_dirs, model_config):
            config = super().create(assets_dirs, model_config)
            return dataclasses.replace(config, model_transforms=strict_model_transforms(model_config))
    return StrictAlohaDataConfig(**kwargs)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

