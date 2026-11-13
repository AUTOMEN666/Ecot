"""Isolated causal RGB+measured-State14 ECoT diagnostic request builder.

No formal evaluator integration. Import/build/parse are offline and stdlib-only.
The optional HTTP client performs network I/O ONLY when explicitly invoked.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import urllib.error
import urllib.request
import uuid
import zlib
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
MODEL = "deepseek-v4-1-flash-261017"
ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
FPS = 25
REFERENCE_TEXT_SHA256 = "54b0581ce3055476a22267b935bb9d0a8ca307b151d067377050dd13e29c2a6e"
REFERENCE_VIDEO_SHA256 = "5b4a6ffae72f78b30ed4ae2766ec0be22c098ef5817d41a568adc1e8aba8f676"
REFERENCE_KEYS = {"text_payload", "text_sha256", "text_utf8_bytes", "video_path",
                  "video_sha256", "video_bytes", "source_fps", "source_frame_count",
                  "reference_only", "trajectory_id"}
STATE_SCHEMA = {
    "order": ["left_arm_joint_0", "left_arm_joint_1", "left_arm_joint_2",
              "left_arm_joint_3", "left_arm_joint_4", "left_arm_joint_5",
              "left_gripper_measurement", "right_arm_joint_0", "right_arm_joint_1",
              "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4",
              "right_arm_joint_5", "right_gripper_measurement"],
    "interpretation": "Native SDK measured values; units, Cartesian axes and gripper sign unknown."
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer, not bool")
    return value


def _state(value: Any) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 14:
        raise ValueError("State must contain exactly 14 measured values")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("State14 must contain finite real values, excluding bool")
    return tuple(value)


def decode_png_chw(png: bytes) -> bytes:
    """Verify collector RGB8/filter0-or-Up2 PNG, return original CHW bytes.

    Deliberately accepts only the collector's lossless 640x480 encoding. Metadata,
    all filters except original0 and lossless Up2, alternate encodings and trailing
    bytes fail rather than get normalized. CRC/shape/stream bounds stay strict.
    Independent reviewer uses Pillow to cross-check the decoded original pixels.
    """
    if type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not an original RGB PNG")
    offset, tags, compressed = 8, [], bytearray()
    while offset < len(png):
        if offset + 12 > len(png):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack(">I", png[offset:offset + 4])[0]
        if length > 2_000_000 or offset + length + 12 > len(png):
            raise ValueError("Invalid PNG chunk length")
        tag = png[offset + 4:offset + 8]
        data = png[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length:offset + 12 + length])[0]
        if zlib.crc32(tag + data) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        offset += length + 12
        if tag not in (b"IHDR", b"IDAT", b"IEND"):
            raise ValueError("Only original collector PNG chunks allowed")
        if tag == b"IHDR":
            if tags or data != struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0):
                raise ValueError("Expected original 640x480 RGB8 PNG")
        elif tag == b"IDAT":
            if not tags or tags[-1] not in (b"IHDR", b"IDAT"):
                raise ValueError("Unexpected IDAT order")
            compressed.extend(data)
        else:
            if length or not tags or tags[-1] != b"IDAT" or offset != len(png):
                raise ValueError("Bad PNG end/trailing data")
        tags.append(tag)
    if not tags or tags[0] != b"IHDR" or tags[-1] != b"IEND":
        raise ValueError("Incomplete PNG")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(compressed), 480 * 1921 + 1)
    if (len(raw) != 480 * 1921 or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError("Unexpected decompressed PNG size/stream")
    rgb = bytearray(480 * 1920)
    for y in range(480):
        filter_kind=raw[y * 1921]
        if filter_kind not in (0,2):
            raise ValueError("Collector PNG must use original filter0 or lossless Up2")
        row=raw[y * 1921 + 1:(y + 1) * 1921]
        start=y*1920
        if filter_kind==2 and y:
            previous=rgb[start-1920:start]
            row=bytes((value+previous[i])&255 for i,value in enumerate(row))
        rgb[start:start+1920]=row
    return bytes(rgb[0::3] + rgb[1::3] + rgb[2::3])


@dataclass(frozen=True)
class CausalFrame:
    episode_id: str
    frame: int
    time_seconds: float
    state14: tuple
    image_png: bytes
    image_sha256: str
    camera_chw_sha256: str
    state_sha256: str


@dataclass(frozen=True)
class CausalWindow:
    episode_id: str
    current_frame: int
    frames: tuple[CausalFrame, ...]
    fps: int = FPS
    # Local provenance only. This mapping is NEVER serialized into model input.
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


def make_frame(episode_id: str, frame: int, state14: Any, image_png: bytes,
               *, time_seconds: float | None = None,
               image_sha256: str | None = None, camera_chw_sha256: str | None = None,
               state_sha256: str | None = None) -> CausalFrame:
    """Capture matched observation bytes; supplied original hashes must match.

    Matching a state to its exposure is the collector's responsibility. This
    function checks the pair's integrity, not simulator sensor synchronization.
    """
    frame = _int(frame, "frame")
    state = _state(state14)
    chw = decode_png_chw(image_png)
    image_hash, chw_hash, state_hash = sha(image_png), sha(chw), sha(canonical(state).encode())
    for given, actual, name in ((image_sha256, image_hash, "image"),
                                (camera_chw_sha256, chw_hash, "camera CHW"),
                                (state_sha256, state_hash, "state")):
        if given is not None and given != actual:
            raise ValueError(name + " SHA mismatch")
    result = CausalFrame(episode_id, frame, frame / FPS if time_seconds is None else time_seconds,
                         state, image_png, image_hash, chw_hash, state_hash)
    _validate_frame(result)
    return result


def _validate_frame(value: CausalFrame) -> None:
    if not isinstance(value, CausalFrame) or type(value.episode_id) is not str or not value.episode_id:
        raise ValueError("Each frame needs an explicit episode identity")
    _int(value.frame, "frame")
    if (type(value.time_seconds) not in (int, float) or not math.isfinite(value.time_seconds)
            or abs(value.time_seconds - value.frame / FPS) > 1e-9):
        raise ValueError("Frame timestamp must equal frame/25")
    state = _state(value.state14)
    if sha(canonical(state).encode()) != value.state_sha256:
        raise ValueError("State integrity mismatch")
    if sha(value.image_png) != value.image_sha256:
        raise ValueError("Image integrity mismatch")
    if sha(decode_png_chw(value.image_png)) != value.camera_chw_sha256:
        raise ValueError("Original RGB CHW integrity mismatch")


def validate_history(history: CausalWindow) -> None:
    if not isinstance(history, CausalWindow) or type(history.episode_id) is not str or not history.episode_id:
        raise ValueError("Expected CausalWindow with episode identity")
    current = _int(history.current_frame, "current_frame")
    if type(history.fps) is not int or history.fps != FPS:
        raise ValueError("This diagnostic is defined at native25Hz")
    if not isinstance(history.frames, tuple) or not 1 <= len(history.frames) <= 16:
        raise ValueError("Expected 1..16 chronological observations")
    previous = -1
    for value in history.frames:
        _validate_frame(value)
        if value.episode_id != history.episode_id:
            raise ValueError("Cross-episode frame is forbidden")
        if value.frame <= previous or value.frame > current:
            raise ValueError("Duplicate, unordered or future observation")
        previous = value.frame
    if history.frames[-1].frame != current:
        raise ValueError("Current frame must occur exactly once and last")


def load_window(query: Mapping[str, Any], window_root: str | Path, *, fps: int = FPS) -> CausalWindow:
    """Read Root-collected query with strict observation allowlist.

    Other query fields are local provenance only. Image paths must be original
    query/frame_N.png under this root, with no symlinks or traversal.
    """
    root = Path(window_root).resolve(strict=True)
    query_id = query["id"]
    if type(query_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]+", query_id):
        raise ValueError("Unsafe query identity")
    instruction = query["instruction"]
    _instruction(instruction)
    frames = []
    for row in query["frames"]:
        n = _int(row["frame"], "frame")
        relative = row["image"]
        if type(relative) is not str or PurePosixPath(relative).parts != (query_id, f"frame_{n}.png"):
            raise ValueError("Image must belong to this query, without traversal")
        path = root
        for component in PurePosixPath(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("Image symlinks forbidden")
        if not path.is_file() or root not in path.resolve(strict=True).parents:
            raise ValueError("Image is not a file inside window root")
        if row["raw_instruction"] != instruction:
            raise ValueError("Frame instruction differs from query")
        frames.append(make_frame(row["episode_id"], n, row["measured_state14"], path.read_bytes(),
                                 time_seconds=row["relative_simulation_seconds"],
                                 image_sha256=row["image_sha256"],
                                 camera_chw_sha256=row["original_camera_chw_sha256"],
                                 state_sha256=row.get("state_sha256")))
    result = CausalWindow(query["episode_id"], query["current_frame"], tuple(frames), fps,
                          {"query_id": query_id, "instruction": instruction,
                           "source_provenance_only": {k: v for k, v in query.items() if k != "frames"}})
    validate_history(result)
    return result


def _instruction(value: Any) -> str:
    if type(value) is not str or not value.strip() or len(value) > 10000:
        raise ValueError("A nonempty total-task Instruction is required")
    return value


def _reference_parts(reference: Mapping[str, Any]) -> tuple[list, dict]:
    # Pin the one independently prepared TRAINING trajectory: arbitrary test
    # images/annotations cannot be smuggled in under a reference role.
    if not isinstance(reference, Mapping) or set(reference) != REFERENCE_KEYS:
        raise ValueError("Unknown training-reference fields")
    text = reference["text_payload"]
    if type(text) is not str or sha(text.encode()) != REFERENCE_TEXT_SHA256:
        raise ValueError("Training reference text differs from its fixed seal")
    if (reference["text_sha256"] != REFERENCE_TEXT_SHA256
            or reference["text_utf8_bytes"] != len(text.encode())
            or reference["reference_only"] is not True
            or reference["trajectory_id"] != "training_episode200"
            or reference["source_fps"] != 25 or reference["source_frame_count"] != 747):
        raise ValueError("Wrong training-reference identity/timing")
    path = Path(reference["video_path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("Reference video must be original regular file")
    video = path.read_bytes()
    if (sha(video) != REFERENCE_VIDEO_SHA256 or reference["video_sha256"] != REFERENCE_VIDEO_SHA256
            or reference["video_bytes"] != len(video) or len(video) != 8748531):
        raise ValueError("Training reference video differs from its fixed seal")
    parts = [{"type": "text", "text": text},
             {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video).decode()}}]
    proof = {"trajectory_id": "training_episode200", "reference_only": True,
             "text_sha256": REFERENCE_TEXT_SHA256, "text_utf8_bytes": len(text.encode()),
             "video_sha256": REFERENCE_VIDEO_SHA256, "video_bytes": len(video),
             "source_fps": 25, "source_frame_count": 747,
             "model_video_sampling_or_full_frame_coverage_verified": False}
    return parts, proof


def _prepare(history: CausalWindow, instruction: str, mode: str,
             reference: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    validate_history(history)
    _instruction(instruction)
    if "instruction" in history.metadata and history.metadata["instruction"] != instruction:
        raise ValueError("Instruction differs from recorded current episode")
    if mode not in ("subtask", "ssr"):
        raise ValueError("Mode must be subtask or ssr")
    prompt_path = HERE / "prompts" / (mode + ".md")
    prompt = prompt_path.read_text(encoding="utf-8")
    messages = [{"role": "system", "content": prompt}]
    reference_proof = None
    if reference is not None:
        parts, reference_proof = _reference_parts(reference)
        messages.append({"role": "user", "content": parts})
    content = [{"type": "text", "text": canonical({
        "source_role": "current_test_causal_observations", "instruction": instruction,
        "query_frame": history.current_frame, "fps": history.fps, "state_schema": STATE_SCHEMA})}]
    frame_proofs = []
    for observation in history.frames:
        content.append({"type": "text", "text": canonical({
            "frame": observation.frame, "time_seconds": observation.time_seconds,
            "is_current": observation.frame == history.current_frame,
            "state14": observation.state14})})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(observation.image_png).decode()}})
        frame_proofs.append({"frame": observation.frame, "time_seconds": observation.time_seconds,
                             "state_sha256": observation.state_sha256,
                             "image_sha256": observation.image_sha256,
                             "original_camera_chw_sha256": observation.camera_chw_sha256})
    messages.append({"role": "user", "content": content})
    body = {"model": MODEL, "temperature": 0, "thinking": {"type": "disabled"},
            "max_tokens": 512 if mode == "subtask" else 1200,
            "response_format": {"type": "json_object"}, "messages": messages}
    proof = {"schema": "ecot.causal_history_request.v1", "mode": mode,
             "model": MODEL, "prompt_sha256": sha(prompt.encode()),
             "client_source_sha256": sha(Path(__file__).read_bytes()),
             "instruction_sha256": sha(instruction.encode()), "current_frame": history.current_frame,
             "fps": history.fps, "observations": frame_proofs,
             "frame_list_sha256": sha(canonical(frame_proofs).encode()),
             "reference": reference_proof, "request_sha256": sha(canonical(body).encode()),
             "request_bytes": len(canonical(body).encode()),
             "episode_id_local_only": history.episode_id, "local_metadata_only": dict(history.metadata),
             "scope": "Offline saved causal window or explicit live capture. No policy/SDK integration; no semantic quality guarantee."}
    return body, proof


def build_request(history: CausalWindow, instruction: str, mode: str,
                  reference: Mapping[str, Any] | None = None) -> dict:
    """Build explicit-allowlist POST body; no network or environment access."""
    return _prepare(history, instruction, mode, reference)[0]


def request_proof(history: CausalWindow, instruction: str, mode: str,
                  reference: Mapping[str, Any] | None = None) -> dict:
    """Local metadata only, NEVER include this proof in API messages."""
    return _prepare(history, instruction, mode, reference)[1]


def _unique_pairs(pairs: list) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate output JSON key")
        out[key] = value
    return out


def parse_output(raw: str, mode: str) -> dict:
    """Strict original schema, no projections, markdown recovery or fallbacks."""
    if mode not in ("subtask", "ssr") or type(raw) is not str:
        raise ValueError("Invalid mode/content")
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    obj = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    expected = {"subtask"} if mode == "subtask" else {"subtask", "action", "reasoning"}
    if type(obj) is not dict or set(obj) != expected:
        raise ValueError("Output fields do not match requested mode")
    def text(value, limit=2400):
        if type(value) is not str or not value.strip() or value != value.strip() or len(value) > limit:
            raise ValueError("Output must contain bounded, nonempty trimmed strings")
        return value
    subtask = text(obj["subtask"], 1200)
    if mode == "subtask":
        return {"subtask": subtask}
    if type(obj["action"]) is not dict or set(obj["action"]) != {"summary"}:
        raise ValueError("Action must contain only summary")
    return {"subtask": subtask, "action": {"summary": text(obj["action"]["summary"])},
            "reasoning": text(obj["reasoning"])}


def parse_response(response: Mapping[str, Any], mode: str) -> dict:
    if not isinstance(response, Mapping):
        raise ValueError("Response must be an object")
    if response.get("model") != MODEL:
        raise ValueError("Response model identity differs")
    choices = response.get("choices")
    if (type(choices) is not list or len(choices) != 1 or not isinstance(choices[0], Mapping)
            or choices[0].get("finish_reason") != "stop"):
        raise ValueError("Expected one complete, stopped response")
    message = choices[0].get("message", {})
    if not isinstance(message, Mapping):
        raise ValueError("Response message must be an object")
    if message.get("tool_calls") or message.get("refusal"):
        raise ValueError("Tool call/refusal is not a valid ECoT output")
    return parse_output(message.get("content"), mode)


class CausalRingBuffer:
    """Bounded live-capture holder; explicit reset, no automatic episode reuse.

    This class does not step/read a simulator and has no generated-label cache.
    Its caller must atomically capture each PNG and its matched measured State14
    while paused, and must reset at every actual episode boundary.
    """
    def __init__(self, capacity: int = 64):
        if type(capacity) is not int or not 16 <= capacity <= 256:
            raise ValueError("Ring capacity must be 16..256 frames")
        self._frames = deque(maxlen=capacity)
        self._episode = None
        self._anchor = None

    def reset(self, episode_id: str) -> None:
        if type(episode_id) is not str or not episode_id:
            raise ValueError("Reset requires explicit episode identity")
        self._frames.clear()
        self._anchor = None
        self._episode = episode_id

    def push(self, observation: CausalFrame) -> None:
        _validate_frame(observation)
        if self._episode is None or observation.episode_id != self._episode:
            raise ValueError("Explicit episode reset required before capture")
        if self._anchor is None:
            if observation.frame != 0:
                raise ValueError("Live capture must start at episode frame0")
            self._anchor = observation
        elif observation.frame <= self._frames[-1].frame:
            raise ValueError("Live observations must strictly advance")
        self._frames.append(observation)

    def window(self) -> CausalWindow:
        if not self._frames or self._anchor is None:
            raise ValueError("No captured observation in this episode")
        current = self._frames[-1].frame
        available = {v.frame: v for v in (self._anchor, *self._frames)}
        selected = {0: self._anchor, current: self._frames[-1]}
        for offset in (96, 48, 16):
            candidates = [n for n in available if n <= max(0, current - offset)]
            n = max(candidates)
            selected[n] = available[n]
        result = CausalWindow(self._episode, current, tuple(selected[n] for n in sorted(selected)),
                              metadata={"capture_kind": "live_causal_ring",
                                        "selection": "frame0 plus nearest retained <= current-96/-48/-16/current"})
        validate_history(result)
        return result


class HistoryClient:
    """Explicitly invoked HTTP diagnostic, immutable body and bounded retries.

    No model, task-only, null, stale-label, or alternative-mode fallback. The
    credential is read only in generate() and never saved in request/proof files.
    Root owns actual invocation and any subsequent live integration.
    """
    def __init__(self, output_dir: str | Path, *, timeout: int = 60, attempts: int = 3,
                 api_key_env: str = "ECOT_AUTHORIZED_API_KEY"):
        if type(timeout) is not int or not 1 <= timeout <= 60:
            raise ValueError("Timeout must be 1..60 seconds")
        if type(attempts) is not int or not 1 <= attempts <= 3:
            raise ValueError("Attempts must be 1..3")
        if type(api_key_env) is not str or not api_key_env:
            raise ValueError("Credential environment name required")
        self.output_dir = Path(output_dir)
        self.timeout, self.attempts, self.api_key_env = timeout, attempts, api_key_env

    def _post(self, body: bytes, key: str) -> bytes:
        request = urllib.request.Request(ENDPOINT, body, {
            "Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(8_388_609)
        if len(raw) > 8_388_608:
            raise ValueError("Response exceeds 8MiB diagnostic bound")
        return raw

    def generate(self, history: CausalWindow, instruction: str, mode: str,
                 reference: Mapping[str, Any] | None = None) -> dict:
        request, proof = _prepare(history, instruction, mode, reference)
        body = canonical(request).encode()
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError("Authorized API credential is absent")
        # uuid directory is exclusive: no previous request or result overwritten.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run = self.output_dir / uuid.uuid4().hex
        run.mkdir(exist_ok=False)
        (run / "request.json").write_bytes(body)
        proof.update({"endpoint": ENDPOINT, "timeout_seconds": self.timeout,
                      "max_identical_body_attempts": self.attempts})
        (run / "proof.json").write_text(canonical(proof), encoding="utf-8")
        failures = []
        for attempt in range(1, self.attempts + 1):
            try:
                raw = self._post(body, key)
                (run / f"response_attempt_{attempt}.json").write_bytes(raw)
                response = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
                result = parse_response(response, mode)
                (run / "result.json").write_text(canonical({
                    "result": result, "successful_attempt": attempt,
                    "request_sha256": sha(body), "response_sha256": sha(raw),
                    "previous_attempt_failures": failures}), encoding="utf-8")
                return result
            except Exception as exc:
                failure = {"attempt": attempt, "type": type(exc).__name__,
                           "request_sha256": sha(body)}
                if isinstance(exc, urllib.error.HTTPError):
                    failure["http_status"] = exc.code
                    # Persist server body separately; never persist credentials/headers.
                    (run / f"http_error_attempt_{attempt}.bin").write_bytes(exc.read(8_388_608))
                failures.append(failure)
                (run / "attempt_failures.json").write_text(canonical(failures), encoding="utf-8")
        (run / "failed.json").write_text(canonical({
            "status": "failed", "request_sha256": sha(body), "failures": failures,
            "fallback_used": False}), encoding="utf-8")
        raise RuntimeError("History ECoT diagnostic failed; raw evidence: " + str(run))
