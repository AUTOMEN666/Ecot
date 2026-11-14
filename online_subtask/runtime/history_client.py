"""Causal lossless image/State14 validation; no reference, API or model loading."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math,struct,zlib
from typing import Any
FPS=25

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
