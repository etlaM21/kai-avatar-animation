"""A Kimodo NPZ clip -> the inputs Retargeter.retarget_globals() takes. Windows only.

Three differences between the NPZ and the BVH skeleton the Retargeter is built from
(source_skeletons.load_soma77, read from Kimodo's own T-pose BVH):
  - the BVH has a `Root` wrapper joint at index 0; Kimodo's 77 joints are the BVH
    joints after it, same order. Root gets an identity rotation.
  - the NPZ is in metres, the BVH skeleton in centimetres. The Retargeter scales the
    root by the rest pelvis height, which cancels a unit factor applied to BOTH the
    rest offsets and the frames, but not one applied to only one of them - so the
    frames are converted to cm here, to match the skeleton.
  - the root is the Hips joint's own position (posed_joints[:, Hips]), the same
    quantity the BVH path uses (BvhClip FK at the Hips joint).

Nothing is assumed about what global_rot_mats means: forward kinematics over the
standard T-pose offsets with those rotations must land on Kimodo's own posed_joints,
or the clip is refused. Kimodo's native per-joint rest convention (what
save_motion_bvh(standard_tpose=False) writes) is recognised and converted, exactly
as source_skeletons.to_standard_convention does for BVH files.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from procedural_animation.retarget import Retargeter, RetargetedMotion
from procedural_animation.source_skeletons import SOMA77_NATIVE_REST_ROTS, SourceSkeleton

from .kimodo_contract import ContractError, KimodoClip

M_TO_CM = 100.0
# float32 metres over a ~10-joint chain round to ~1e-4 cm; a wrong convention or a
# transposed rotation misses by tens of cm. Measured on real clips once the Spark
# answers - recorded in CLAUDE.md.
TOL_FK_CM = 0.5


@dataclass
class AdaptedClip:
    src_global: list[R]          # one (T,) Rotation per BVH joint, Root included
    root_cm: np.ndarray          # (T, 3) Hips position, cm, Kimodo's frame
    fps: float
    convention: str              # "standard" or "native"
    fk_error_cm: float           # worst FK vs posed_joints over the clip
    root_vs_hips_cm: float       # worst |root_positions - posed Hips|, a report


def _fk(rots: np.ndarray, hips_cm: np.ndarray, offsets_cm: np.ndarray, parents: list[int]) -> np.ndarray:
    """Positions (T, 77, 3) from global rotations (T, 77, 3, 3). parents/offsets are
    in the 77-joint order; parent -1 is the Hips."""
    pos = np.zeros(rots.shape[:2] + (3,))
    for j, p in enumerate(parents):
        pos[:, j] = hips_cm if p == -1 else pos[:, p] + np.einsum("tij,j->ti", rots[:, p], offsets_cm[j])
    return pos


def adapt(clip: KimodoClip, skeleton: SourceSkeleton, tol_cm: float = TOL_FK_CM) -> AdaptedClip:
    if skeleton.joint_names[0] != "Root" or clip.bone_order_names != skeleton.joint_names[1:]:
        raise ContractError("bone_order_names differ from the SOMA77 BVH joints after Root")
    parents = [p - 1 for p in skeleton.parents[1:]]          # Root (0) -> -1
    offsets = skeleton.rest_offsets[1:]
    posed_cm = clip.posed_joints * M_TO_CM
    hips_cm = posed_cm[:, 0]
    rots = clip.global_rot_mats

    def err(r: np.ndarray) -> float:
        return float(np.linalg.norm(_fk(r, hips_cm, offsets, parents) - posed_cm, axis=-1).max())

    convention, fk_err = "standard", err(rots)
    if fk_err > tol_cm:
        # Kimodo's native convention: G_native = G_std * O_j, so G_std = G_native * O_j^T.
        table = np.load(SOMA77_NATIVE_REST_ROTS).astype(np.float64)     # (77, 3, 3)
        converted = np.einsum("tjab,jcb->tjac", rots, table)
        native_err = err(converted)
        if native_err > tol_cm:
            raise ContractError(f"rotations don't reproduce posed_joints: off by {fk_err:.2f} cm as standard, "
                                f"{native_err:.2f} cm as native convention")
        convention, fk_err, rots = "native", native_err, converted

    T = clip.num_frames
    src_global = [R.identity(T)] + [R.from_matrix(rots[:, j]) for j in range(rots.shape[1])]
    root_vs_hips = float(np.linalg.norm(clip.root_positions * M_TO_CM - hips_cm, axis=-1).max())
    return AdaptedClip(src_global=src_global, root_cm=hips_cm, fps=clip.fps, convention=convention,
                       fk_error_cm=fk_err, root_vs_hips_cm=root_vs_hips)


def retarget(clip: KimodoClip, retargeter: Retargeter) -> tuple[RetargetedMotion, AdaptedClip]:
    adapted = adapt(clip, retargeter.source)
    # rest_offsets are cm and units_to_cm is 1 for the BVH-derived skeleton, so the
    # cm root goes in unchanged.
    motion = retargeter.retarget_globals(adapted.src_global, adapted.root_cm, adapted.fps)
    return motion, adapted
