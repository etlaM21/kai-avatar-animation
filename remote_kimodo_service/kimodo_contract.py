"""The NPZ that crosses the network: what the Spark service returns for one Kimodo clip.

numpy only, no package-relative imports, so the SAME file is imported by the service
on the Spark (kimodo_service.py) and by the Windows client. Change it on one side and
the other side changes with it.

The service sends raw Kimodo data and knows nothing about Manny (CLAUDE.md, procedural
invariant 5). One clip, T frames, Kimodo's 77 somaskel77 joints in Kimodo's own
`bone_order_names` order (= the BVH joints minus the `Root` wrapper):

    global_rot_mats   (T, 77, 3, 3) float32  global joint rotations, Kimodo's frame
                                             (RH, Y up, +Z forward), standard T-pose
                                             convention (identity = T-pose)
    root_positions    (T, 3)        float32  metres
    posed_joints      (T, 77, 3)    float32  metres; global joint positions. Lets the
                                             client check, by FK, that the rotations mean
                                             what it assumes (convention, transpose, root)
    foot_contacts     (T, N)        float32  as Kimodo returns them (bool -> 0/1; N = 6 from
                                             kimodo 1.0.0, not the 4 its docs state)
    fps               ()            float32
    bone_order_names  (77,)         unicode
    meta_json         ()            unicode  prompt, seed, num_frames, steps, model, timings

Loaded with allow_pickle=False: nothing in the file can execute code on load.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field

import numpy as np

CONTRACT_VERSION = 1
NUM_JOINTS = 77
DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.1"
KIMODO_FPS = 30
MAX_SECONDS = 10.0           # Kimodo's limit per prompt
DEFAULT_SECONDS = 9.0
# Malte, 2026-10-09: 100 steps is unnecessary, 33 is the default (it is what he had been
# typing as `/s 33`). Measured on the GB10 for 9 s of motion: 100 steps 5.8 s, 50 steps
# 2.9 s, 25 steps 1.5 s. Shared with the Spark: its service default follows after a
# `git pull` there.
DEFAULT_STEPS = 33


class ContractError(ValueError):
    """The bytes are not a clip this contract describes."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    seed: int
    num_frames: int
    steps: int = DEFAULT_STEPS
    model: str = DEFAULT_MODEL

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        # Whitespace never changes what the text encoder sees in a way we care about,
        # and it would otherwise split one clip into several cache entries.
        return " ".join(prompt.split())

    @classmethod
    def build(cls, prompt: str, seed: int, seconds: float = DEFAULT_SECONDS,
              steps: int = DEFAULT_STEPS, model: str = DEFAULT_MODEL) -> "GenerationRequest":
        prompt = cls.normalize_prompt(prompt)
        if not prompt:
            raise ValueError("empty prompt")
        if not 0.0 < seconds <= MAX_SECONDS:
            raise ValueError(f"seconds must be in (0, {MAX_SECONDS:g}], got {seconds:g}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        return cls(prompt=prompt, seed=int(seed), num_frames=max(1, round(seconds * KIMODO_FPS)),
                   steps=int(steps), model=model)

    def cache_key(self) -> str:
        """Same on both sides: the client's disk cache and the service's result cache."""
        canon = json.dumps({"v": CONTRACT_VERSION, "model": self.model, "prompt": self.prompt,
                            "seed": self.seed, "num_frames": self.num_frames, "steps": self.steps},
                           sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {"prompt": self.prompt, "seed": self.seed, "num_frames": self.num_frames,
                "steps": self.steps, "model": self.model}


@dataclass
class KimodoClip:
    global_rot_mats: np.ndarray
    root_positions: np.ndarray
    posed_joints: np.ndarray
    foot_contacts: np.ndarray
    fps: float
    bone_order_names: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.global_rot_mats.shape[0])


def pack(clip: KimodoClip) -> bytes:
    meta = dict(clip.meta)
    meta["contract_version"] = CONTRACT_VERSION
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        global_rot_mats=np.asarray(clip.global_rot_mats, dtype=np.float32),
        root_positions=np.asarray(clip.root_positions, dtype=np.float32),
        posed_joints=np.asarray(clip.posed_joints, dtype=np.float32),
        foot_contacts=np.asarray(clip.foot_contacts, dtype=np.float32),
        fps=np.float32(clip.fps),
        bone_order_names=np.array(clip.bone_order_names, dtype=str),
        meta_json=np.array(json.dumps(meta)),
    )
    return buf.getvalue()


def _shape(name: str, arr: np.ndarray, expected: tuple) -> None:
    if arr.ndim != len(expected) or any(e is not None and a != e for a, e in zip(arr.shape, expected)):
        raise ContractError(f"{name} has shape {arr.shape}, expected {expected} (None = any)")


def unpack(data: bytes) -> KimodoClip:
    """Structural checks only (keys, shapes, finite values). Whether the rotations mean
    what the retarget assumes is checked by FK on the Windows side (kimodo_adapter)."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as e:  # zipfile.BadZipFile, ValueError, EOFError, ...
        raise ContractError(f"not a readable NPZ ({type(e).__name__}: {e})") from None

    required = ("global_rot_mats", "root_positions", "posed_joints", "foot_contacts",
                "fps", "bone_order_names", "meta_json")
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ContractError(f"missing keys: {', '.join(missing)}")

    rots = arrays["global_rot_mats"]
    _shape("global_rot_mats", rots, (None, NUM_JOINTS, 3, 3))
    T = rots.shape[0]
    if T < 1:
        raise ContractError("clip has no frames")
    _shape("root_positions", arrays["root_positions"], (T, 3))
    _shape("posed_joints", arrays["posed_joints"], (T, NUM_JOINTS, 3))
    _shape("foot_contacts", arrays["foot_contacts"], (T, None))
    names = [str(n) for n in arrays["bone_order_names"]]
    if len(names) != NUM_JOINTS:
        raise ContractError(f"{len(names)} bone names, expected {NUM_JOINTS}")
    for k in ("global_rot_mats", "root_positions", "posed_joints"):
        if not np.all(np.isfinite(arrays[k])):
            raise ContractError(f"{k} contains NaN or inf")
    fps = float(arrays["fps"])
    if not fps > 0:
        raise ContractError(f"fps is {fps}")
    try:
        meta = json.loads(str(arrays["meta_json"]))
    except json.JSONDecodeError as e:
        raise ContractError(f"meta_json is not JSON: {e}") from None
    if meta.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(f"contract version {meta.get('contract_version')}, expected {CONTRACT_VERSION}")

    return KimodoClip(global_rot_mats=rots.astype(np.float64), root_positions=arrays["root_positions"].astype(np.float64),
                      posed_joints=arrays["posed_joints"].astype(np.float64),
                      foot_contacts=arrays["foot_contacts"], fps=fps, bone_order_names=names, meta=meta)
