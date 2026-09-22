"""
Landmark recorder - dumps the raw (pre-solve, pre-smoothing) MediaPipe output
of a live session to a .npz file, and loads it back as landmark objects the
solver accepts unchanged.

Why this exists: every solver change has to be checked against ground truth
(see tests/solver_checks.py), and that needs real captures that can be replayed
without a camera. Recording the RAW landmarks rather than solved bones means a
single recording stays useful across solver rewrites.

File layout (all arrays share the leading frame axis N; missing data is NaN,
never zeros, so a dropped detection can't masquerade as a pose at the origin):

    t_ms          (N,)          conductor timestamp
    pose_valid    (N,)          bool
    pose_world    (N, 33, 5)    x, y, z, visibility, presence  (metric, hip-centred)
    pose_image    (N, 33, 5)    same fields, image-normalised
    lhand_valid   (N,)          bool
    lhand_world   (N, 21, 3)    metric, hand-centred
    lhand_image   (N, 21, 3)    image-normalised
    rhand_*                     same as lhand_*
    face_valid    (N,)          bool
    face_image    (N, 478, 3)   Holistic face mesh, image-normalised (no world variant exists)
    head_ypr_deg  (N, 3)        yaw, pitch, roll from HeadPoseCapture (NaN if that pass missed)
    frame_size    (2,)          capture width, height - needed to un-normalise image x vs y
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_POSE = 33
N_HAND = 21
N_FACE = 478


@dataclass
class Landmark:
    """Duck-types MediaPipe's NormalizedLandmark/Landmark: the solver only
    reads .x .y .z (and, from later steps on, .visibility .presence)."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


def _fill(lms: Any, n: int, fields: int) -> np.ndarray:
    out = np.full((n, fields), np.nan, dtype=np.float32)
    if not lms:
        return out
    for i, lm in enumerate(lms[:n]):
        row = [lm.x, lm.y, lm.z]
        if fields == 5:
            # Tasks API leaves these as None on some landmark types.
            row += [lm.visibility if lm.visibility is not None else np.nan,
                    lm.presence if lm.presence is not None else np.nan]
        out[i] = row
    return out


class LandmarkRecorder:
    def __init__(self, path: str | Path, frame_size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.frame_size = frame_size
        self._rows: dict[str, list] = {}

    def _push(self, key: str, value: Any) -> None:
        self._rows.setdefault(key, []).append(value)

    def add(self, t_ms: int, pose_frame: Any, face_frame: Any, hands_frame: Any,
            head_pose_frame: Any) -> None:
        self._push("t_ms", t_ms)
        self._push("pose_valid", bool(pose_frame.valid))
        self._push("pose_world", _fill(pose_frame.world_landmarks if pose_frame.valid else None, N_POSE, 5))
        self._push("pose_image", _fill(pose_frame.landmarks if pose_frame.valid else None, N_POSE, 5))
        for prefix, hand in (("lhand", hands_frame.left), ("rhand", hands_frame.right)):
            self._push(f"{prefix}_valid", bool(hand.valid))
            self._push(f"{prefix}_world", _fill(hand.world_landmarks if hand.valid else None, N_HAND, 3))
            self._push(f"{prefix}_image", _fill(hand.landmarks if hand.valid else None, N_HAND, 3))
        self._push("face_valid", bool(face_frame.valid))
        self._push("face_image", _fill(face_frame.landmarks if face_frame.valid else None, N_FACE, 3))
        if head_pose_frame.valid:
            ypr = (head_pose_frame.yaw_deg, head_pose_frame.pitch_deg, head_pose_frame.roll_deg)
        else:
            ypr = (np.nan, np.nan, np.nan)
        self._push("head_ypr_deg", np.array(ypr, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._rows.get("t_ms", []))

    def save(self) -> None:
        if not len(self):
            print(f"[record] no frames captured - {self.path} not written")
            return
        arrays = {k: np.asarray(v) for k, v in self._rows.items()}
        arrays["frame_size"] = np.asarray(self.frame_size, dtype=np.int32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **arrays)
        print(f"[record] {len(self)} frames -> {self.path}")


# ---- loading ------------------------------------------------------------------

def to_landmarks(arr: np.ndarray) -> list[Landmark]:
    """One frame's (n, 3|5) array -> landmark objects, or [] if the frame is
    missing (all-NaN). Returning [] matches what the live capture hands the
    solver when a detection drops, so the solver's own fallbacks kick in."""
    if np.isnan(arr[:, :3]).all():
        return []
    if arr.shape[1] == 5:
        return [Landmark(*map(float, row)) for row in arr]
    return [Landmark(float(r[0]), float(r[1]), float(r[2])) for r in arr]


def load_recording(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
