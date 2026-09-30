"""Source-skeleton motion -> the 60 Manny BoneTransforms that `mirroring` streams.

The one piece of real math in the procedural lane (procedural-animation.md sec. 4.1).
Same technique as pose_solver: global orientations in Unreal component space against
Manny's MEASURED rest pose (pose_solver's own FK of FULL_CHAIN / FINGER_CHAIN), then
local = parent_global^-1 * global over Manny's full chain, including the unstreamed
spine_03, spine_05 and neck_02.

Per Manny bone b driven by source joint j:
    G_ue[j](t)  = C * G_src[j](t) * C             change of basis (C = C^-1 = C^T)
    A[b]        = swing(manny_dir[b] -> C*src_rest_dir[j]) * manny_rest_global[b]
    G_manny[b]  = G_ue[j](t) * G_ue_rest[j]^-1 * A[b]
A is computed once from the two rest poses. It takes Manny's bind orientation into
the source's rest pose (A-pose -> T-pose for SOMA) with minimal swing, so bone roll
at rest is whatever the minimal swing leaves - tests/procedural_checks.py measures
that against independent anatomy rather than assuming it.

Bone directions come out exact by construction: G_manny[b] carries manny_dir[b] to
C * G_src[j] * src_rest_dir[j], the source bone's own direction in component space.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import MIRRORING_DIR  # noqa: F401  (puts mirroring on sys.path)
from .bvh_reader import BvhClip
from .source_skeletons import SourceSkeleton

from pose_solver import (  # read-only imports - pose_solver is never edited from here
    BIND_POSITIONS, BONE_NAMES, FINGER_CHAIN, FINGER_BONE_NAMES, FULL_CHAIN,
    BoneTransform, PoseSolver, normalize, swing_between,
)

# The 60 bones of one pose packet, in wire order: 22 body, 19 left hand, 19 right
# hand (conductor.py sends raw_bones + left_bones + right_bones).
STREAMED_BONES: list[str] = BONE_NAMES + FINGER_BONE_NAMES

# What each Manny bone points along in its rest pose, for the alignment A. A child
# bone's rest position where the bone has one; otherwise the bone's own local axis,
# which is +X for the spine/head/left side and -X for the right side (measured on
# the rest FK: head +X is straight up, ball_l +X and ball_r -X both point forward,
# index_03_l +X and index_03_r -X both point down the finger - the same rule
# pose_solver uses for hand_* and the fingertip bones).
# hand_* aims at middle_01 (the middle knuckle), matching SOMA's Hand -> Middle2.
_MANNY_AIM_AT: dict[str, str] = {
    "pelvis": "spine_01", "spine_01": "spine_02", "spine_02": "spine_03",
    "spine_04": "spine_05", "neck_01": "neck_02",
    "clavicle_l": "upperarm_l", "upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l",
    "hand_l": "middle_01_l",
    "clavicle_r": "upperarm_r", "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r",
    "hand_r": "middle_01_r",
    "thigh_l": "calf_l", "calf_l": "foot_l", "foot_l": "ball_l",
    "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r",
}
for _name, _parent, _rot, _off in FINGER_CHAIN:
    if _parent not in ("hand_l", "hand_r"):
        _MANNY_AIM_AT[_parent] = _name


# Bones whose FULL rest orientation corresponds between the rigs, so the alignment
# is the identity map of rest frames rather than a swing. The head: both rest poses
# look straight ahead. Swinging Manny's head axis (straight up) onto SOMA's
# Head -> HeadEnd, which leans back 6.5 deg in SOMA's skull geometry, tilted every
# retargeted head up by that much - measured on three clips as a constant +6.5-7 deg
# offset between Manny's head pitch and the source's head->eyes elevation.
FULL_FRAME_BONES = {"head"}


def _manny_axis(name: str) -> np.ndarray:
    return np.array([-1.0, 0.0, 0.0]) if name.endswith("_r") else np.array([1.0, 0.0, 0.0])


def _change_basis(C: np.ndarray, rot: R) -> R:
    mats = rot.as_matrix()
    return R.from_matrix(C @ mats @ C.T)


@dataclass
class RetargetedMotion:
    fps: float
    # (T, 60, 4) parent-local quaternions (x, y, z, w) in STREAMED_BONES order.
    local_quats: np.ndarray
    pelvis_pos: np.ndarray            # (T, 3) component space, cm
    head_rotation: R                  # (T,) as PoseSolver.head_rotation
    manny_global: dict[str, R]        # every FULL_CHAIN + finger bone, (T,) each

    @property
    def num_frames(self) -> int:
        return self.local_quats.shape[0]

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def bone_transforms(self, frame: int) -> list[BoneTransform]:
        return _to_bone_transforms(self.local_quats[frame], self.pelvis_pos[frame])

    def sample(self, time_s: float) -> tuple[list[BoneTransform], R]:
        """Pose at an arbitrary time, slerped between source frames (so a 30 fps
        clip plays smoothly at 60 Hz). Clamped to the clip. Returns the bones and
        the head rotation for the face channel."""
        f = min(max(time_s * self.fps, 0.0), self.num_frames - 1.0)
        i = int(np.floor(f))
        j = min(i + 1, self.num_frames - 1)
        a = f - i
        ri, rj = R.from_quat(self.local_quats[i]), R.from_quat(self.local_quats[j])
        rot = ri * R.from_rotvec((ri.inv() * rj).as_rotvec() * a)
        pos = self.pelvis_pos[i] * (1.0 - a) + self.pelvis_pos[j] * a
        hi, hj = self.head_rotation[i], self.head_rotation[j]
        head = hi * R.from_rotvec((hi.inv() * hj).as_rotvec() * a)
        return _to_bone_transforms(rot.as_quat(), pos), head


_FINGER_OFFSET = {name: off for name, _p, _r, off in FINGER_CHAIN}


def _to_bone_transforms(quats: np.ndarray, pelvis_pos: np.ndarray) -> list[BoneTransform]:
    out: list[BoneTransform] = []
    for k, name in enumerate(STREAMED_BONES):
        qx, qy, qz, qw = (float(v) for v in quats[k])
        if k < len(BONE_NAMES):
            pos = dict(BIND_POSITIONS[k])
            if k == 0:
                pos = {"x": float(pelvis_pos[0]), "y": float(pelvis_pos[1]), "z": float(pelvis_pos[2])}
        else:
            off = _FINGER_OFFSET[name]
            pos = {"x": float(off[0]), "y": float(off[1]), "z": float(off[2])}
        out.append(BoneTransform(name=name, rotation={"x": qx, "y": qy, "z": qz, "w": qw}, position=pos))
    return out


class Retargeter:
    def __init__(self, source: SourceSkeleton) -> None:
        self.source = source
        C = source.to_ue
        self.C = C

        # Manny's rest pose, from pose_solver's own FK of the verified rig tables.
        solver = PoseSolver()
        self.manny_order: list[str] = [c[0] for c in FULL_CHAIN] + FINGER_BONE_NAMES
        self.manny_parent: dict[str, str | None] = {}
        self.manny_bind_local: dict[str, R] = {}
        self.manny_rest_global: dict[str, R] = {}
        self.manny_rest_pos: dict[str, np.ndarray] = {}
        for i, (name, parent, _rot, _off) in enumerate(FULL_CHAIN):
            self.manny_parent[name] = None if parent == -1 else FULL_CHAIN[parent][0]
            self.manny_bind_local[name] = solver.full_bind[i]
            self.manny_rest_global[name] = solver.rest_global[i]
            self.manny_rest_pos[name] = solver.rest_pos[i]
        for name, parent, _rot, _off in FINGER_CHAIN:
            self.manny_parent[name] = parent
            self.manny_bind_local[name] = solver.finger_bind_rot[name]
            self.manny_rest_global[name] = solver.finger_rest_global[name]
            self.manny_rest_pos[name] = solver.finger_rest_pos[name]

        # Alignment, once: G_manny[b] = G_ue[j] * pre[b], pre[b] = G_ue_rest[j]^-1 * A[b].
        src_pos = source.rest_positions()
        self.pre: dict[str, R] = {}
        self.driver: dict[str, int] = {}
        for bone, (joint, aim) in source.to_manny.items():
            j, k = source.index(joint), source.index(aim)
            src_dir = normalize(C @ (src_pos[k] - src_pos[j]))
            if bone in _MANNY_AIM_AT:
                manny_dir = normalize(self.manny_rest_pos[_MANNY_AIM_AT[bone]] - self.manny_rest_pos[bone])
            else:
                manny_dir = normalize(self.manny_rest_global[bone].apply(_manny_axis(bone)))
            if bone in FULL_FRAME_BONES:
                align = self.manny_rest_global[bone]
            else:
                align = swing_between(manny_dir, src_dir) * self.manny_rest_global[bone]
            ue_rest = _change_basis(C, source.rest_global[j])
            self.pre[bone] = ue_rest.inv() * align
            self.driver[bone] = j

        root = source.index(source.root_joint)
        self._root = root
        # Pelvis translation: the source root, in cm, through C, scaled by pelvis
        # height so stride length matches Manny's legs, anchored so the source rest
        # lands exactly on Manny's rest pelvis.
        src_root_rest_ue = C @ (src_pos[root] * source.units_to_cm)
        manny_pelvis = self.manny_rest_pos["pelvis"]
        self.root_scale = float(manny_pelvis[2] / src_root_rest_ue[2])
        self.root_anchor = manny_pelvis - src_root_rest_ue * self.root_scale

    # -- entry points ---------------------------------------------------------
    def retarget_clip(self, clip: BvhClip) -> RetargetedMotion:
        glob, pos = clip.forward_kinematics()
        return self.retarget_globals(glob, pos[:, self._root], clip.fps)

    def retarget_globals(self, src_global: list[R], src_root_pos: np.ndarray,
                         fps: float) -> RetargetedMotion:
        """src_global: per source joint, a Rotation of length T in the source's own
        frame (e.g. Kimodo's global_rot_mats). src_root_pos: (T, 3), source units."""
        C = self.C
        T = len(src_global[0])
        g: dict[str, R] = {}
        for name in self.manny_order:
            if name in self.pre:
                g[name] = _change_basis(C, src_global[self.driver[name]]) * self.pre[name]
            else:
                # Unstreamed (spine_03, spine_05, neck_02) or no source joint (e.g.
                # fingers on a source without them): held at bind relative to the
                # parent, which is exactly what Unreal does with it.
                parent = self.manny_parent[name]
                g[name] = g[parent] * self.manny_bind_local[name]

        quats = np.zeros((T, len(STREAMED_BONES), 4))
        for k, name in enumerate(STREAMED_BONES):
            parent = self.manny_parent[name]
            local = g[name] if parent is None else g[parent].inv() * g[name]
            quats[:, k] = local.as_quat()

        pelvis = (src_root_pos * self.source.units_to_cm) @ C.T * self.root_scale + self.root_anchor
        head = g["head"] * self.manny_rest_global["head"].inv()
        return RetargetedMotion(fps=fps, local_quats=quats, pelvis_pos=pelvis,
                                head_rotation=head, manny_global=g)
