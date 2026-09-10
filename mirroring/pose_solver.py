"""
Pose Solver Module - MediaPipe 33 Landmarks to Unreal Engine 22-Bone Skeleton.

Transforms raw 3D point positions from MediaPipe Pose into a hierarchical
skeletal pose with local bone rotations (quaternions) and root translations.

Coordinate Space Conventions:
    MediaPipe World Landmarks:
        +X = Right (Screen Right / Subject's Left)
        +Y = Down (-Y is Up)
        +Z = Forward (Away from camera / Depth)
        Units: Meters, centered at the mid-hip point.

    Unreal Engine (UE5):
        +X = Forward
        +Y = Right
        +Z = Up
        Units: Centimeters (1 m = 100 cm).

    Conversion Matrix (MediaPipe -> UE):
        UE_X = -MP_Z * 100.0  (Depth -> Forward)
        UE_Y =  MP_X * 100.0  (Right -> Right)
        UE_Z = -MP_Y * 100.0  (Down -> Up)

Requires:
    pip install numpy scipy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from mediapipe_pose_capture import PoseLandmark


# ---------------------------------------------------------------------------
# Bone Hierarchy Definitions (Matching MediaPipeLiveLinkSource.cpp)
# ---------------------------------------------------------------------------

BONE_NAMES: list[str] = [
    "pelvis",       # 0
    "spine_01",     # 1
    "spine_02",     # 2
    "spine_04",     # 3
    "neck_01",      # 4
    "head",         # 5
    "clavicle_l",   # 6
    "upperarm_l",   # 7
    "lowerarm_l",   # 8
    "hand_l",       # 9
    "clavicle_r",   # 10
    "upperarm_r",   # 11
    "lowerarm_r",   # 12
    "hand_r",       # 13
    "thigh_l",      # 14
    "calf_l",       # 15
    "foot_l",       # 16
    "ball_l",       # 17
    "thigh_r",      # 18
    "calf_r",       # 19
    "foot_r",       # 20
    "ball_r",       # 21
]

BONE_PARENTS: list[int] = [
    -1,  # pelvis (root)
    0,   # spine_01   -> pelvis
    1,   # spine_02   -> spine_01
    2,   # spine_04   -> spine_02
    3,   # neck_01    -> spine_04
    4,   # head       -> neck_01
    3,   # clavicle_l -> spine_04
    6,   # upperarm_l -> clavicle_l
    7,   # lowerarm_l -> upperarm_l
    8,   # hand_l     -> lowerarm_l
    3,   # clavicle_r -> spine_04
    10,  # upperarm_r -> clavicle_r
    11,  # lowerarm_r -> upperarm_r
    12,  # hand_r     -> lowerarm_r
    0,   # thigh_l    -> pelvis
    14,  # calf_l     -> thigh_l
    15,  # foot_l     -> calf_l
    16,  # ball_l     -> foot_l
    0,   # thigh_r    -> pelvis
    18,  # calf_r     -> thigh_r
    19,  # foot_r     -> calf_r
    20,  # ball_r     -> foot_r
]


@dataclass
class BoneTransform:
    """Represents a single bone transform in local space relative to its parent."""
    name: str
    rotation: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})


# ---------------------------------------------------------------------------
# Vector Math Utilities
# ---------------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    """Safely normalizes a 3D vector, returning zero vector if norm is near 0."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm


def rotation_from_orthonormal_basis(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> R:
    """Creates a scipy Rotation object from 3 orthogonal basis vectors (columns)."""
    matrix = np.column_stack((x_axis, y_axis, z_axis))
    return R.from_matrix(matrix)


def build_look_at_rotation(aim_axis_dir: np.ndarray, hint_up_dir: np.ndarray, aim_axis: str = "x", up_axis: str = "z") -> R:
    """
    Constructs a strictly right-handed 3D rotation matrix aligning a specified bone 
    primary axis with aim_axis_dir, and a secondary axis with hint_up_dir.
    """
    primary = normalize(aim_axis_dir)
    hint = normalize(hint_up_dir)

    # Calculate a vector orthogonal to both
    third = normalize(np.cross(primary, hint))
    # Recalculate secondary axis to guarantee strict 90-degree orthogonality
    secondary = normalize(np.cross(third, primary))

    # Build final coordinate frames enforcing right-hand rule (Determinant +1)
    if aim_axis == "x" and up_axis == "z":
        # primary = X, secondary = Z. Derive Y = Z x X
        y_axis = np.cross(secondary, primary)
        return rotation_from_orthonormal_basis(primary, y_axis, secondary)
        
    elif aim_axis == "y" and up_axis == "x":
        # primary = Y, secondary = X. Derive Z = X x Y
        z_axis = np.cross(secondary, primary)
        return rotation_from_orthonormal_basis(secondary, primary, z_axis)
        
    elif aim_axis == "-z" and up_axis == "x":
        # primary = -Z, secondary = X. Derive Y = Z x X
        y_axis = np.cross(-primary, secondary)
        return rotation_from_orthonormal_basis(secondary, y_axis, -primary)

    # Fallback default: align X with primary, Z with secondary
    y_axis = np.cross(secondary, primary)
    return rotation_from_orthonormal_basis(primary, y_axis, secondary)


# ---------------------------------------------------------------------------
# Pose Solver Core
# ---------------------------------------------------------------------------

class PoseSolver:
    """
    Converts 33 MediaPipe landmark coordinates into 22 UE skeletal bone transforms.
    """

    def __init__(self, pelvis_default_height_cm: float = 95.0) -> None:
        self.pelvis_default_height_cm = pelvis_default_height_cm

    def _convert_landmarks_to_ue_space(self, raw_landmarks: list[Any]) -> np.ndarray:
        """
        Converts a list of 33 Landmark objects into a (33, 3) NumPy array in UE space (cm).
        """
        points = np.zeros((len(raw_landmarks), 3), dtype=np.float64)
        for i, lm in enumerate(raw_landmarks):
            # MediaPipe world landmarks are metric
            points[i, 0] = -lm.z * 100.0  # Depth -> Forward (+X)
            points[i, 1] =  lm.x * 100.0  # Right -> Right (+Y)
            points[i, 2] = -lm.y * 100.0  # Down -> Up (+Z)
        return points

    def solve(self, raw_world_landmarks: list[Any]) -> list[BoneTransform]:
        """
        Main solver entry point:
        1. Converts points to UE coordinate space.
        2. Computes global segment rotations.
        3. Converts global rotations to parent-local space.
        4. Returns local bone transforms ready for Live Link.
        """
        if len(raw_world_landmarks) < len(PoseLandmark):
            return [BoneTransform(name=name) for name in BONE_NAMES]

        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)

        # ---- Key Joint Centers ----
        l_hip = pts[PoseLandmark.LEFT_HIP]
        r_hip = pts[PoseLandmark.RIGHT_HIP]
        hip_center = (l_hip + r_hip) * 0.5

        l_shoulder = pts[PoseLandmark.LEFT_SHOULDER]
        r_shoulder = pts[PoseLandmark.RIGHT_SHOULDER]
        shoulder_center = (l_shoulder + r_shoulder) * 0.5

        l_elbow = pts[PoseLandmark.LEFT_ELBOW]
        r_elbow = pts[PoseLandmark.RIGHT_ELBOW]
        l_wrist = pts[PoseLandmark.LEFT_WRIST]
        r_wrist = pts[PoseLandmark.RIGHT_WRIST]
        l_index = pts[PoseLandmark.LEFT_INDEX]
        r_index = pts[PoseLandmark.RIGHT_INDEX]

        l_knee = pts[PoseLandmark.LEFT_KNEE]
        r_knee = pts[PoseLandmark.RIGHT_KNEE]
        l_ankle = pts[PoseLandmark.LEFT_ANKLE]
        r_ankle = pts[PoseLandmark.RIGHT_ANKLE]
        l_foot = pts[PoseLandmark.LEFT_FOOT_INDEX]
        r_foot = pts[PoseLandmark.RIGHT_FOOT_INDEX]

        nose = pts[PoseLandmark.NOSE]

        # Container for absolute/global rotation of each of the 22 bones
        global_rotations: list[R] = [R.identity() for _ in range(len(BONE_NAMES))]
        bone_positions: list[np.ndarray] = [np.zeros(3) for _ in range(len(BONE_NAMES))]

        # -------------------------------------------------------------------
        # 1. Torso / Pelvis (Root)
        # -------------------------------------------------------------------
        # Spine direction = from hip center up to shoulder center
        spine_up = normalize(shoulder_center - hip_center)
        # Lateral direction = from left hip to right hip
        hip_lateral = normalize(r_hip - l_hip)
        # Torso forward = Cross(Lateral, Up)
        torso_forward = normalize(np.cross(hip_lateral, spine_up))
        # Recompute clean orthogonal right vector
        torso_right = normalize(np.cross(spine_up, torso_forward))

        pelvis_rot = rotation_from_orthonormal_basis(torso_forward, torso_right, spine_up)
        global_rotations[0] = pelvis_rot
        # Pelvis location in UE centimeters (relative to root floor)
        bone_positions[0] = hip_center + np.array([0.0, 0.0, self.pelvis_default_height_cm])

        # Spine chain interpolated upwards
        global_rotations[1] = pelvis_rot  # spine_01
        global_rotations[2] = pelvis_rot  # spine_02
        global_rotations[3] = pelvis_rot  # spine_04

        # -------------------------------------------------------------------
        # 2. Neck & Head
        # -------------------------------------------------------------------
        neck_up = normalize(nose - shoulder_center)
        neck_rot = build_look_at_rotation(neck_up, torso_forward, aim_axis="x", up_axis="z")
        global_rotations[4] = neck_rot  # neck_01
        global_rotations[5] = neck_rot  # head

        # -------------------------------------------------------------------
        # 3. Left Arm Chain
        # -------------------------------------------------------------------
        # Clavicle: shoulder center to left shoulder
        clav_l_dir = normalize(l_shoulder - shoulder_center)
        global_rotations[6] = build_look_at_rotation(clav_l_dir, spine_up, aim_axis="y", up_axis="x")

        # Upperarm: left shoulder to left elbow
        upperarm_l_dir = normalize(l_elbow - l_shoulder)
        elbow_l_bend_normal = normalize(np.cross(upperarm_l_dir, normalize(l_wrist - l_elbow)))
        global_rotations[7] = build_look_at_rotation(upperarm_l_dir, elbow_l_bend_normal, aim_axis="y", up_axis="x")

        # Lowerarm: left elbow to left wrist
        lowerarm_l_dir = normalize(l_wrist - l_elbow)
        global_rotations[8] = build_look_at_rotation(lowerarm_l_dir, elbow_l_bend_normal, aim_axis="y", up_axis="x")

        # Hand: left wrist to left index
        hand_l_dir = normalize(l_index - l_wrist)
        global_rotations[9] = build_look_at_rotation(hand_l_dir, elbow_l_bend_normal, aim_axis="y", up_axis="x")

        # -------------------------------------------------------------------
        # 4. Right Arm Chain
        # -------------------------------------------------------------------
        # Clavicle: shoulder center to right shoulder
        clav_r_dir = normalize(r_shoulder - shoulder_center)
        global_rotations[10] = build_look_at_rotation(clav_r_dir, spine_up, aim_axis="y", up_axis="x")

        # Upperarm: right shoulder to right elbow
        upperarm_r_dir = normalize(r_elbow - r_shoulder)
        elbow_r_bend_normal = normalize(np.cross(upperarm_r_dir, normalize(r_wrist - r_elbow)))
        global_rotations[11] = build_look_at_rotation(upperarm_r_dir, elbow_r_bend_normal, aim_axis="y", up_axis="x")

        # Lowerarm: right elbow to right wrist
        lowerarm_r_dir = normalize(r_wrist - r_elbow)
        global_rotations[12] = build_look_at_rotation(lowerarm_r_dir, elbow_r_bend_normal, aim_axis="y", up_axis="x")

        # Hand: right wrist to right index
        hand_r_dir = normalize(r_index - r_wrist)
        global_rotations[13] = build_look_at_rotation(hand_r_dir, elbow_r_bend_normal, aim_axis="y", up_axis="x")

        # -------------------------------------------------------------------
        # 5. Left Leg Chain
        # -------------------------------------------------------------------
        # Thigh: left hip to left knee
        thigh_l_dir = normalize(l_knee - l_hip)
        knee_l_bend_normal = normalize(np.cross(thigh_l_dir, normalize(l_ankle - l_knee)))
        global_rotations[14] = build_look_at_rotation(thigh_l_dir, torso_forward, aim_axis="-z", up_axis="x")

        # Calf: left knee to left ankle
        calf_l_dir = normalize(l_ankle - l_knee)
        global_rotations[15] = build_look_at_rotation(calf_l_dir, torso_forward, aim_axis="-z", up_axis="x")

        # Foot: left ankle to left foot tip
        foot_l_dir = normalize(l_foot - l_ankle)
        global_rotations[16] = build_look_at_rotation(foot_l_dir, spine_up, aim_axis="x", up_axis="z")
        global_rotations[17] = global_rotations[16]  # ball_l

        # -------------------------------------------------------------------
        # 6. Right Leg Chain
        # -------------------------------------------------------------------
        # Thigh: right hip to right knee
        thigh_r_dir = normalize(r_knee - r_hip)
        knee_r_bend_normal = normalize(np.cross(thigh_r_dir, normalize(r_ankle - r_knee)))
        global_rotations[18] = build_look_at_rotation(thigh_r_dir, torso_forward, aim_axis="-z", up_axis="x")

        # Calf: right knee to right ankle
        calf_r_dir = normalize(r_ankle - r_knee)
        global_rotations[19] = build_look_at_rotation(calf_r_dir, torso_forward, aim_axis="-z", up_axis="x")

        # Foot: right ankle to right foot tip
        foot_r_dir = normalize(r_foot - r_ankle)
        global_rotations[20] = build_look_at_rotation(foot_r_dir, spine_up, aim_axis="x", up_axis="z")
        global_rotations[21] = global_rotations[20]  # ball_r

        # -------------------------------------------------------------------
        # 7. Convert Global Rotations -> Local Rotations
        # -------------------------------------------------------------------
        # Formula: R_local[child] = (R_global[parent])^-1 * R_global[child]
        result_transforms: list[BoneTransform] = []

        for i, name in enumerate(BONE_NAMES):
            parent_idx = BONE_PARENTS[i]
            if parent_idx == -1:
                # Root bone: local rotation is identical to global rotation
                local_rot = global_rotations[i]
            else:
                parent_global_rot = global_rotations[parent_idx]
                local_rot = parent_global_rot.inv() * global_rotations[i]

            # Convert to [x, y, z, w] quaternion dict
            qx, qy, qz, qw = local_rot.as_quat()
            rot_dict = {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}

            # Only root carries world translation; child bones use rig bone lengths
            pos_dict = (
                {"x": float(bone_positions[i][0]), "y": float(bone_positions[i][1]), "z": float(bone_positions[i][2])}
                if parent_idx == -1
                else {"x": 0.0, "y": 0.0, "z": 0.0}
            )

            result_transforms.append(BoneTransform(name=name, rotation=rot_dict, position=pos_dict))

        return result_transforms