"""Source skeleton tables: joint names, parents, rest pose, coordinate frame, and
which source joint drives which Manny bone.

Only the SOURCE side lives here. What Manny's bones point along is Manny knowledge
and lives in retarget.py next to pose_solver's tables.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import MODULE_ROOT
from .bvh_reader import BvhClip, BvhJoint, read_bvh

SOMA77_TPOSE_BVH = MODULE_ROOT / "kimodo" / "soma_skeleton" / "somaskel77_standard_tpose.bvh"

# Source (right-handed, Y up, +Z forward, +X = character's left) -> Manny component
# space (+X = character's left, +Y forward, +Z up): (x, y, z) -> (x, z, y).
# A reflection (det -1), and its own inverse. Rotations change basis as C R C.
# Checked on the data: the SOMA T-pose has LeftHand at +X and the eyes at +Z;
# Manny's rest has thigh_l at +X and ball_l ahead of foot_l in +Y.
Y_UP_Z_FWD_TO_UE = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])


@dataclass
class SourceSkeleton:
    name: str
    joint_names: list[str]
    parents: list[int]
    rest_offsets: np.ndarray          # (J, 3) in source units, rest pose
    # Global rotation of each joint in the rest pose the offsets describe. Identity
    # for Kimodo's standard T-pose convention, where identity locals ARE the rest.
    rest_global: list[R]
    to_ue: np.ndarray                 # 3x3 change of basis into Manny component space
    units_to_cm: float
    root_joint: str                   # the joint whose translation is the body's
    # Manny bone -> (source joint whose global rotation drives it, source joint it
    # aims at in the rest pose). The aim is only used once, for the rest alignment.
    to_manny: dict[str, tuple[str, str]]

    def index(self, name: str) -> int:
        return self.joint_names.index(name)

    def rest_positions(self) -> np.ndarray:
        pos = np.zeros_like(self.rest_offsets)
        for i, p in enumerate(self.parents):
            pos[i] = self.rest_offsets[i] if p == -1 else (
                pos[p] + self.rest_global[p].apply(self.rest_offsets[i]))
        return pos


# SOMA77 -> Manny. Naming trap (procedural-animation.md sec. 1): in SOMA, LeftLeg is
# the THIGH and LeftShin the calf.
#
# Spine: SOMA has Spine1, Spine2, Chest between Hips and Neck1; Manny streams
# spine_01, spine_02, spine_04. spine_03, spine_05 and neck_02 are NOT given a
# source joint (the plan suggested Spine2 -> spine_03, Neck2 -> neck_02): they are
# never streamed, so Unreal holds them at bind relative to their parent, and the
# retarget has to model them the same way or every child local is computed against
# a parent Unreal doesn't have. SOMA's Neck2 bend therefore lands on neck_01/head;
# the head's global orientation is still exact.
#
# Fingers: SOMA's <Finger>1 is the metacarpal (Index1 -> Index2 is 6.4 cm, the next
# segments 3.7 / 2.3 / 2.8), so Index1..4 -> index_metacarpal, index_01..03. Thumb1
# is the thumb metacarpal, the same bone as Manny's thumb_01.
_SOMA77_BODY: dict[str, tuple[str, str]] = {
    "pelvis":     ("Hips", "Spine1"),
    "spine_01":   ("Spine1", "Spine2"),
    "spine_02":   ("Spine2", "Chest"),
    "spine_04":   ("Chest", "Neck1"),
    "neck_01":    ("Neck1", "Neck2"),
    "head":       ("Head", "HeadEnd"),
    "clavicle_l": ("LeftShoulder", "LeftArm"),
    "upperarm_l": ("LeftArm", "LeftForeArm"),
    "lowerarm_l": ("LeftForeArm", "LeftHand"),
    "hand_l":     ("LeftHand", "LeftHandMiddle2"),
    "clavicle_r": ("RightShoulder", "RightArm"),
    "upperarm_r": ("RightArm", "RightForeArm"),
    "lowerarm_r": ("RightForeArm", "RightHand"),
    "hand_r":     ("RightHand", "RightHandMiddle2"),
    "thigh_l":    ("LeftLeg", "LeftShin"),
    "calf_l":     ("LeftShin", "LeftFoot"),
    "foot_l":     ("LeftFoot", "LeftToeBase"),
    "ball_l":     ("LeftToeBase", "LeftToeEnd"),
    "thigh_r":    ("RightLeg", "RightShin"),
    "calf_r":     ("RightShin", "RightFoot"),
    "foot_r":     ("RightFoot", "RightToeBase"),
    "ball_r":     ("RightToeBase", "RightToeEnd"),
}


def _soma77_fingers() -> dict[str, tuple[str, str]]:
    table: dict[str, tuple[str, str]] = {}
    for side, prefix in (("_l", "LeftHand"), ("_r", "RightHand")):
        table[f"thumb_01{side}"] = (f"{prefix}Thumb1", f"{prefix}Thumb2")
        table[f"thumb_02{side}"] = (f"{prefix}Thumb2", f"{prefix}Thumb3")
        table[f"thumb_03{side}"] = (f"{prefix}Thumb3", f"{prefix}ThumbEnd")
        for finger, soma in (("index", "Index"), ("middle", "Middle"),
                             ("ring", "Ring"), ("pinky", "Pinky")):
            table[f"{finger}_metacarpal{side}"] = (f"{prefix}{soma}1", f"{prefix}{soma}2")
            table[f"{finger}_01{side}"] = (f"{prefix}{soma}2", f"{prefix}{soma}3")
            table[f"{finger}_02{side}"] = (f"{prefix}{soma}3", f"{prefix}{soma}4")
            table[f"{finger}_03{side}"] = (f"{prefix}{soma}4", f"{prefix}{soma}End")
    return table


SOMA77_TO_MANNY: dict[str, tuple[str, str]] = {**_SOMA77_BODY, **_soma77_fingers()}


def soma77_from_bvh(clip: BvhClip) -> SourceSkeleton:
    return SourceSkeleton(
        name="SOMA77",
        joint_names=clip.names,
        parents=clip.parents,
        rest_offsets=clip.offsets(),
        rest_global=[R.identity() for _ in clip.joints],
        to_ue=Y_UP_Z_FWD_TO_UE,
        units_to_cm=1.0,  # Kimodo's BVH export is in centimetres (Hips at y=100)
        root_joint="Hips",
        to_manny=SOMA77_TO_MANNY,
    )


def load_soma77(path: str | Path = SOMA77_TPOSE_BVH) -> SourceSkeleton:
    """SOMA77 in Kimodo's standard T-pose convention, the convention the Kimodo API's
    local_rot_mats / global_rot_mats are in."""
    return soma77_from_bvh(read_bvh(path))


# Kimodo's per-joint rest rotations for somaskel77 (kimodo/assets/skeletons/somaskel77/
# standard_t_pose_global_offsets_rots.p at nv-tlabs/kimodo main, 2026-09-29), read from
# the torch archive's raw float32 buffer without unpickling. (77, 3, 3), indexed by
# BVH joint index - 1 (the table has no entry for the BVH's Root wrapper).
# Kimodo's transforms.change_tpose: G_native = G_std * O_j, so G_std = G_native * O_j^T.
# Verified against this repo's own BVHs, independently of Kimodo's code: O_parent
# maps every native offset onto the standard one, all 76 joints to 0.00000 cm (the
# transposed reading misses by up to 62 cm).
SOMA77_NATIVE_REST_ROTS = Path(__file__).resolve().parent / "data" / "somaskel77_standard_t_pose_global_offsets_rots.npy"


def _native_rest_rots(num_joints: int) -> list[R]:
    table = np.load(SOMA77_NATIVE_REST_ROTS).astype(np.float64)
    if table.shape != (num_joints - 1, 3, 3):
        raise ValueError(f"native rest table is {table.shape}, expected ({num_joints - 1}, 3, 3)")
    return [R.identity()] + [R.from_matrix(m) for m in table]


def to_standard_convention(clip: BvhClip, reference: SourceSkeleton,
                           tol_cm: float = 1e-3) -> tuple[BvhClip, str]:
    """The clip in Kimodo's standard T-pose convention, plus which convention it came
    in ('standard' or 'native').

    Kimodo's save_motion_bvh(standard_tpose=False) - what every service in this repo
    except kimodo_service_handoff_v2.py used - writes the same motion against per-joint
    rest frames (bones along local +X). Those are converted with Kimodo's own O_j
    table; joint positions are unchanged by construction, which is checked here on
    load (and asserted over every clip by procedural_checks).
    """
    if clip.names != reference.joint_names:
        raise ValueError(f"{clip.path}: joint list differs from {reference.name}")
    off = clip.offsets()
    if np.max(np.abs(off - reference.rest_offsets)) <= tol_cm:
        return clip, "standard"
    same_lengths = np.allclose(np.linalg.norm(off, axis=1),
                               np.linalg.norm(reference.rest_offsets, axis=1), atol=0.05)
    if not same_lengths:
        raise ValueError(f"{clip.path}: SOMA77 joint names but unrecognised rest offsets")

    rest = _native_rest_rots(len(clip.joints))
    native_global, _ = clip.forward_kinematics()
    std_global = [g * o.inv() for g, o in zip(native_global, rest)]
    local_rot = [g if p == -1 else std_global[p].inv() * g
                 for g, p in zip(std_global, clip.parents)]
    local_pos = clip.local_pos.copy()
    for i, joint in enumerate(clip.joints):
        if not any(ch.endswith("position") for ch in joint.channels):
            local_pos[:, i] = reference.rest_offsets[i]
    joints = [BvhJoint(name=j.name, parent=j.parent, offset=reference.rest_offsets[i].copy(),
                       channels=list(j.channels), end_site=j.end_site)
              for i, j in enumerate(clip.joints)]
    converted = BvhClip(joints=joints, frame_time=clip.frame_time, local_rot=local_rot,
                        local_pos=local_pos, path=clip.path)
    # Matching bone lengths alone don't prove the file is a native Kimodo export. The
    # conversion must leave every joint where it was, or the clip is refused.
    _, native_pos = clip.forward_kinematics()
    _, std_pos = converted.forward_kinematics()
    moved = float(np.abs(native_pos - std_pos).max())
    if moved > tol_cm:
        raise ValueError(f"{clip.path}: not a native SOMA77 export, conversion moves joints by {moved:.3f} cm")
    return converted, "native"
