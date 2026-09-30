"""Minimal BVH reader: hierarchy, per-frame local rotations and translations, and
forward kinematics. No retargeting knowledge lives here.

Conventions, as written by Kimodo's `kimodo.exports.bvh.motion_to_bvh` and checked
against the files in this repo:
  - Rotation channels are Euler angles in degrees, composed in the order the
    channels are listed (Kimodo writes `Zrotation Yrotation Xrotation`, i.e.
    R = Rz * Ry * Rx acting on column vectors, child-to-parent).
  - A joint with position channels uses them INSTEAD of its OFFSET. Measured on
    the Kimodo clips: Hips has OFFSET (0, 100, 0) and its Yposition channel reads
    ~99.8 on a standing frame, so the channel is the absolute local translation,
    not an addition to the offset.
  - Units are whatever the file uses (Kimodo: centimetres, Y up, +Z forward).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class BvhJoint:
    name: str
    parent: int                      # index into BvhClip.joints, -1 for the root
    offset: np.ndarray               # (3,) rest translation from the parent
    channels: list[str] = field(default_factory=list)
    end_site: np.ndarray | None = None


@dataclass
class BvhClip:
    joints: list[BvhJoint]
    frame_time: float
    # (T, J) rotations, local to the parent joint.
    local_rot: list[R]
    # (T, J, 3) local translations: channel values where a joint has position
    # channels, its OFFSET otherwise.
    local_pos: np.ndarray
    path: Path | None = None

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def num_frames(self) -> int:
        return self.local_pos.shape[0]

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def parents(self) -> list[int]:
        return [j.parent for j in self.joints]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def offsets(self) -> np.ndarray:
        return np.stack([j.offset for j in self.joints])

    def forward_kinematics(self) -> tuple[list[R], np.ndarray]:
        """Global rotations (one Rotation of length T per joint) and global
        positions (T, J, 3). Joints are stored parents-first, as BVH requires."""
        n = len(self.joints)
        glob: list[R] = [None] * n  # type: ignore[list-item]
        pos = np.zeros_like(self.local_pos)
        for i, joint in enumerate(self.joints):
            p = joint.parent
            if p == -1:
                glob[i] = self.local_rot[i]
                pos[:, i] = self.local_pos[:, i]
            else:
                glob[i] = glob[p] * self.local_rot[i]
                pos[:, i] = pos[:, p] + glob[p].apply(self.local_pos[:, i])
        return glob, pos


def _tokens(text: str):
    for tok in text.split():
        yield tok


def read_bvh(path: str | Path) -> BvhClip:
    path = Path(path)
    tokens = _tokens(path.read_text(encoding="utf-8"))
    joints: list[BvhJoint] = []
    stack: list[int] = []
    pending_end = False

    tok = next(tokens)
    if tok != "HIERARCHY":
        raise ValueError(f"{path}: expected HIERARCHY, got {tok!r}")

    for tok in tokens:
        if tok in ("ROOT", "JOINT"):
            name = next(tokens)
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(name=name, parent=parent, offset=np.zeros(3)))
            stack.append(len(joints) - 1)
        elif tok == "End":
            next(tokens)  # "Site"
            pending_end = True
        elif tok == "{":
            continue
        elif tok == "}":
            if pending_end:
                pending_end = False
            else:
                stack.pop()
        elif tok == "OFFSET":
            vec = np.array([float(next(tokens)) for _ in range(3)])
            if pending_end:
                joints[stack[-1]].end_site = vec
            else:
                joints[stack[-1]].offset = vec
        elif tok == "CHANNELS":
            count = int(next(tokens))
            joints[stack[-1]].channels = [next(tokens) for _ in range(count)]
        elif tok == "MOTION":
            break
        else:
            raise ValueError(f"{path}: unexpected token {tok!r} in HIERARCHY")

    if next(tokens) != "Frames:":
        raise ValueError(f"{path}: expected 'Frames:'")
    num_frames = int(next(tokens))
    if (next(tokens), next(tokens)) != ("Frame", "Time:"):
        raise ValueError(f"{path}: expected 'Frame Time:'")
    frame_time = float(next(tokens))

    num_channels = sum(len(j.channels) for j in joints)
    values = np.fromiter((float(t) for t in tokens), dtype=np.float64)
    if values.size != num_frames * num_channels:
        raise ValueError(f"{path}: {values.size} motion values, expected "
                         f"{num_frames} frames x {num_channels} channels")
    values = values.reshape(num_frames, num_channels)

    local_rot: list[R] = []
    local_pos = np.zeros((num_frames, len(joints), 3))
    col = 0
    for i, joint in enumerate(joints):
        local_pos[:, i] = joint.offset
        rot_axes = ""
        rot_cols: list[int] = []
        for ch in joint.channels:
            axis = ch[0].upper()
            if ch.endswith("position"):
                local_pos[:, i, "XYZ".index(axis)] = values[:, col]
            elif ch.endswith("rotation"):
                rot_axes += axis
                rot_cols.append(col)
            else:
                raise ValueError(f"{path}: unknown channel {ch!r} on {joint.name}")
            col += 1
        if rot_axes:
            # Upper-case = intrinsic: R = R_first * R_second * R_third, the BVH order.
            local_rot.append(R.from_euler(rot_axes, values[:, rot_cols], degrees=True))
        else:
            local_rot.append(R.identity(num_frames))

    return BvhClip(joints=joints, frame_time=frame_time, local_rot=local_rot,
                   local_pos=local_pos, path=path)
