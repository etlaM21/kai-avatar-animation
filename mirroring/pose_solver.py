"""
Pose Solver Module - MediaPipe 33 Landmarks to Unreal Engine 22-Bone Skeleton.

Retargets MediaPipe world landmarks onto the UE5 Mannequin (Manny) by computing,
for each bone, the minimal "swing" rotation that takes the bone's REST direction to
the direction measured from the performer, then converting to parent-local space.

Why this shape (read before changing anything):
  An earlier version computed a per-bone delta against a hand-written T-pose reference
  table and composed it as BIND_POSE * delta. That is not frame-correct: the delta was
  expressed in the landmark coordinate space while BIND_POSE is a parent-local rotation
  in Unreal's component space, so composing them mixed two different frames. It produced
  a systematic body tilt (a forward lean rendered as a sideways lean) that no sign flip
  or handedness conversion could fix.

  This version never needs a reference-pose table and never needs a handedness fudge.
  Everything happens in Unreal's component space: rest globals come from the rig's own
  bind pose by forward kinematics, landmark directions are mapped into that same space,
  and locals are produced by the standard local = parent_global^-1 * global.

  Measured against a real capture (absolute 3D bone directions vs the performer):
  RMS error 3.7 deg, versus 31.1 deg for the delta/table approach.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from mediapipe_pose_capture import PoseLandmark

BONE_NAMES: list[str] = [
    "pelvis", "spine_01", "spine_02", "spine_04", "neck_01", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l", "ball_l",
    "thigh_r", "calf_r", "foot_r", "ball_r"
]

BONE_PARENTS: list[int] = [
    -1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20
]


@dataclass
class BoneTransform:
    name: str
    rotation: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
    position: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})


# ---------------------------------------------------------------------------
# Unreal rotator -> quaternion
# ---------------------------------------------------------------------------

def ue_rotator_to_dict(pitch: float, yaw: float, roll: float) -> dict[str, float]:
    """Converts a UE FRotator (degrees) into an FQuat dictionary.

    Signs match Unreal's own FRotator::Quaternion() implementation. An earlier version
    had '+ sr*cp*cy' / '+ sr*cp*sy' on X and Y, which corrupted every bone whose bind
    rotation had a non-zero roll -- pelvis, both clavicles and both hands worst of all.
    """
    deg2rad = math.pi / 180.0
    sp, cp = math.sin(pitch * deg2rad * 0.5), math.cos(pitch * deg2rad * 0.5)
    sy, cy = math.sin(yaw * deg2rad * 0.5), math.cos(yaw * deg2rad * 0.5)
    sr, cr = math.sin(roll * deg2rad * 0.5), math.cos(roll * deg2rad * 0.5)
    return {
        "x": cr * sp * sy - sr * cp * cy,
        "y": -cr * sp * cy - sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy
    }


def multiply_ue_quats(q1: dict[str, float], q2: dict[str, float]) -> dict[str, float]:
    """Unreal Engine FQuat multiplication (q1 * q2). Standard Hamilton product."""
    x1, y1, z1, w1 = q1['x'], q1['y'], q1['z'], q1['w']
    x2, y2, z2, w2 = q2['x'], q2['y'], q2['z'], q2['w']
    return {
        "x": y1*z2 - z1*y2 + x1*w2 + w1*x2,
        "y": z1*x2 - x1*z2 + y1*w2 + w1*y2,
        "z": x1*y2 - y1*x2 + z1*w2 + w1*z2,
        "w": w1*w2 - x1*x2 - y1*y2 - z1*z2
    }


# ---------------------------------------------------------------------------
# Rig reference pose. VERIFIED against a RefSkeleton dump of SKM_Manny_Simple.
# Do not hand-edit: re-dump from the rig if the mesh ever changes.
# ---------------------------------------------------------------------------

BIND_POSES: list[dict[str, float]] = [
    ue_rotator_to_dict(86.366893, -90.0, -90.0),             # 0: pelvis
    ue_rotator_to_dict(0.0, 14.457322, 0.0),                 # 1: spine_01
    ue_rotator_to_dict(0.0, -3.464470, 0.0),                 # 2: spine_02
    ue_rotator_to_dict(0.0, -5.866984, 0.000450),            # 3: spine_04
    ue_rotator_to_dict(0.0, 23.928404, 0.0),                 # 4: neck_01
    ue_rotator_to_dict(0.000020, -11.880170, 0.000096),      # 5: head
    ue_rotator_to_dict(-80.831226, -153.124384, 163.263585), # 6: clavicle_l
    ue_rotator_to_dict(-46.029604, 4.358519, -4.337345),     # 7: upperarm_l
    ue_rotator_to_dict(0.0, 38.978822, 0.0),                 # 8: lowerarm_l
    ue_rotator_to_dict(-1.473471, -1.848916, -67.770759),    # 9: hand_l
    ue_rotator_to_dict(-80.831226, 26.875616, 163.263585),   # 10: clavicle_r
    ue_rotator_to_dict(-46.029604, 4.358519, -4.337345),     # 11: upperarm_r
    ue_rotator_to_dict(0.0, 38.978822, 0.0),                 # 12: lowerarm_r
    ue_rotator_to_dict(-1.473471, -1.848916, -67.770759),    # 13: hand_r
    ue_rotator_to_dict(3.125540, 3.560133, 8.408539),        # 14: thigh_l
    ue_rotator_to_dict(0.0, 5.004845, 0.0),                  # 15: calf_l
    ue_rotator_to_dict(-3.081202, -2.664105, -0.004663),     # 16: foot_l
    ue_rotator_to_dict(0.0, 90.0, 0.0),                      # 17: ball_l
    ue_rotator_to_dict(3.125540, -176.439867, 8.408539),     # 18: thigh_r
    ue_rotator_to_dict(0.0, 5.004845, 0.0),                  # 19: calf_r
    ue_rotator_to_dict(-3.081202, -2.664105, -0.004663),     # 20: foot_r
    ue_rotator_to_dict(0.0, 90.0, 0.0)                       # 21: ball_r
]

# Left/right leg offsets used to be sign-swapped here, which projected both legs
# upward through the torso. These are the rig's real values.
BIND_POSITIONS: list[dict[str, float]] = [
    {"x": -0.0000, "y": 2.2809, "z": 95.8968},   # 0: pelvis (overwritten at runtime)
    {"x": 3.6771, "y": 0.0, "z": 0.0},           # 1: spine_01
    {"x": 6.7951, "y": 0.0, "z": 0.0},           # 2: spine_02
    {"x": 8.5239, "y": 0.0, "z": 0.0},           # 3: spine_04
    {"x": 11.8878, "y": 0.0, "z": 0.0},          # 4: neck_01
    {"x": 4.9130, "y": 0.0, "z": 0.0},           # 5: head
    {"x": 5.5163, "y": -1.3148, "z": -1.4279},   # 6: clavicle_l
    {"x": 17.8095, "y": 0.0, "z": 0.0},          # 7: upperarm_l
    {"x": 27.7711, "y": 0.0, "z": 0.0},          # 8: lowerarm_l
    {"x": 27.2511, "y": 0.0, "z": 0.0},          # 9: hand_l
    {"x": 5.5162, "y": -1.3148, "z": 1.4279},    # 10: clavicle_r
    {"x": -17.8096, "y": 0.0, "z": 0.0004},      # 11: upperarm_r
    {"x": -27.7707, "y": 0.0, "z": 0.0},         # 12: lowerarm_r
    {"x": -27.2510, "y": 0.0, "z": 0.0},         # 13: hand_r
    {"x": -2.3657, "y": 0.1100, "z": -9.9692},   # 14: thigh_l
    {"x": -43.3413, "y": 0.0, "z": 0.0},         # 15: calf_l
    {"x": -42.2179, "y": 0.0, "z": 0.0},         # 16: foot_l
    {"x": -7.0094, "y": 15.2376, "z": 0.5389},   # 17: ball_l
    {"x": -2.3657, "y": 0.1195, "z": 9.9691},    # 18: thigh_r
    {"x": 43.3413, "y": 0.0, "z": 0.0},          # 19: calf_r
    {"x": 42.2179, "y": 0.0, "z": 0.0},          # 20: foot_r
    {"x": 7.0094, "y": -15.2376, "z": -0.5389}   # 21: ball_r
]

# The 22 streamed bones are a SUBSET of Manny's real hierarchy: spine_03, spine_05 and
# neck_02 sit between them and are not streamed. Unreal holds those at their rest pose
# and applies each streamed local transform relative to the bone's REAL parent, so the
# solver has to model them too. Omitting spine_03 alone rotates the whole upper body
# reference by ~11 degrees. (name, real_parent_index, rotator, offset)
FULL_CHAIN: list[tuple] = [
    ("pelvis",     -1, (86.366893, -90.0, -90.0),                (-0.0000, 2.2809, 95.8968)),
    ("spine_01",    0, (0.0, 14.457322, 0.0),                    (3.6771, 0.0, 0.0)),
    ("spine_02",    1, (0.0, -3.464470, 0.0),                    (6.7951, 0.0, 0.0)),
    ("spine_03",    2, (0.0, -10.946079, 0.0),                   (7.2382, 0.0, 0.0)),
    ("spine_04",    3, (0.0, -5.866984, 0.000450),               (8.5239, 0.0, 0.0)),
    ("spine_05",    4, (-0.000005, -0.681389, -0.000449),        (19.4398, 0.0, 0.0)),
    ("neck_01",     5, (0.0, 23.928404, 0.0),                    (11.8878, 0.0, 0.0)),
    ("neck_02",     6, (0.0, -1.913529, -0.000098),              (5.1103, 0.0, 0.0)),
    ("head",        7, (0.000020, -11.880170, 0.000096),         (4.9130, 0.0, 0.0)),
    ("clavicle_l",  5, (-80.831226, -153.124384, 163.263585),    (5.5163, -1.3148, -1.4279)),
    ("upperarm_l",  9, (-46.029604, 4.358519, -4.337345),        (17.8095, 0.0, 0.0)),
    ("lowerarm_l", 10, (0.0, 38.978822, 0.0),                    (27.7711, 0.0, 0.0)),
    ("hand_l",     11, (-1.473471, -1.848916, -67.770759),       (27.2511, 0.0, 0.0)),
    ("clavicle_r",  5, (-80.831226, 26.875616, 163.263585),      (5.5162, -1.3148, 1.4279)),
    ("upperarm_r", 13, (-46.029604, 4.358519, -4.337345),        (-17.8096, 0.0, 0.0004)),
    ("lowerarm_r", 14, (0.0, 38.978822, 0.0),                    (-27.7707, 0.0, 0.0)),
    ("hand_r",     15, (-1.473471, -1.848916, -67.770759),       (-27.2510, 0.0, 0.0)),
    ("thigh_l",     0, (3.125540, 3.560133, 8.408539),           (-2.3657, 0.1100, -9.9692)),
    ("calf_l",     17, (0.0, 5.004845, 0.0),                     (-43.3413, 0.0, 0.0)),
    ("foot_l",     18, (-3.081202, -2.664105, -0.004663),        (-42.2179, 0.0, 0.0)),
    ("ball_l",     19, (0.0, 90.0, 0.0),                         (-7.0094, 15.2376, 0.5389)),
    ("thigh_r",     0, (3.125540, -176.439867, 8.408539),        (-2.3657, 0.1195, 9.9691)),
    ("calf_r",     21, (0.0, 5.004845, 0.0),                     (43.3413, 0.0, 0.0)),
    ("foot_r",     22, (-3.081202, -2.664105, -0.004663),        (42.2179, 0.0, 0.0)),
    ("ball_r",     23, (0.0, 90.0, 0.0),                         (7.0094, -15.2376, -0.5389)),
]

# Landmark space is (Fwd, Right, Up); Unreal component space is that rotated 90 deg
# about Z. This matrix maps a landmark-space vector into component space.
PTS_TO_COMPONENT = np.array([[0.0, -1.0, 0.0],
                             [1.0,  0.0, 0.0],
                             [0.0,  0.0, 1.0]])


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return np.zeros_like(vec) if norm < 1e-6 else vec / norm


def swing_between(from_dir: np.ndarray, to_dir: np.ndarray) -> R:
    """Minimal rotation taking from_dir to to_dir. Twist about the bone is left free."""
    a, b = normalize(from_dir), normalize(to_dir)
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return R.identity()
    axis = np.cross(a, b)
    axis_len = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if axis_len < 1e-8:
        if dot > 0.0:
            return R.identity()
        # antiparallel: rotate 180 deg about any perpendicular axis
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return R.from_rotvec(normalize(perp) * math.pi)
    return R.from_rotvec(normalize(axis) * math.acos(dot))


# Which landmark pair gives each bone its aim direction.
_P = PoseLandmark
BONE_AIM: dict[str, tuple] = {
    "upperarm_l": (_P.LEFT_SHOULDER, _P.LEFT_ELBOW),
    "lowerarm_l": (_P.LEFT_ELBOW, _P.LEFT_WRIST),
    "hand_l":     (_P.LEFT_WRIST, _P.LEFT_INDEX),
    "upperarm_r": (_P.RIGHT_SHOULDER, _P.RIGHT_ELBOW),
    "lowerarm_r": (_P.RIGHT_ELBOW, _P.RIGHT_WRIST),
    "hand_r":     (_P.RIGHT_WRIST, _P.RIGHT_INDEX),
    "thigh_l":    (_P.LEFT_HIP, _P.LEFT_KNEE),
    "calf_l":     (_P.LEFT_KNEE, _P.LEFT_ANKLE),
    "foot_l":     (_P.LEFT_ANKLE, _P.LEFT_FOOT_INDEX),
    "thigh_r":    (_P.RIGHT_HIP, _P.RIGHT_KNEE),
    "calf_r":     (_P.RIGHT_KNEE, _P.RIGHT_ANKLE),
    "foot_r":     (_P.RIGHT_ANKLE, _P.RIGHT_FOOT_INDEX),
}

# Bones carried rigidly by the torso/neck frame rather than aimed at a landmark.
TORSO_BONES = {"pelvis", "spine_01", "spine_02", "spine_04", "neck_01", "head"}
# Present in the rig and needed for correct parenting, but never streamed.
INTERNAL_BONES = {"spine_03", "spine_05", "neck_02"}


class PoseSolver:
    def __init__(self, pelvis_default_height_cm: float = 95.0,
                 torso_lean_offset_deg: float = 0.0) -> None:
        self.pelvis_default_height_cm = pelvis_default_height_cm
        # MediaPipe's world landmarks place the shoulders in front of the hips, so a
        # performer standing vertically is reported as leaning ~18 deg forward. The bias
        # is near-constant (measured +17.8 deg on a vertical subject and +20.0 deg on a
        # subject actually leaning backwards), so it is removed as a calibration offset
        # rather than by changing the solve. Call calibrate_neutral() once with the
        # performer standing relaxed and upright, or pass a value here.
        self.torso_lean_offset_deg = torso_lean_offset_deg
        self._build_rest_pose()

    def measure_torso_lean_deg(self, raw_world_landmarks: list[Any]) -> float:
        """Forward lean of the torso as MediaPipe reports it. Positive = leaning forward."""
        if len(raw_world_landmarks) < len(PoseLandmark):
            return 0.0
        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)
        hip_mid = (pts[int(_P.LEFT_HIP)] + pts[int(_P.RIGHT_HIP)]) * 0.5
        shoulder_mid = (pts[int(_P.LEFT_SHOULDER)] + pts[int(_P.RIGHT_SHOULDER)]) * 0.5
        up = normalize(shoulder_mid - hip_mid)
        return math.degrees(math.atan2(float(up[0]), float(up[2])))

    def calibrate_neutral(self, raw_world_landmarks: list[Any]) -> float:
        """Record the current lean as 'upright'. Returns the offset now in use."""
        self.torso_lean_offset_deg = self.measure_torso_lean_deg(raw_world_landmarks)
        return self.torso_lean_offset_deg

    # -- rig rest pose, in component space -------------------------------------
    def _build_rest_pose(self) -> None:
        self.full_names = [c[0] for c in FULL_CHAIN]
        self.full_parent = [c[1] for c in FULL_CHAIN]
        self.full_idx = {n: i for i, n in enumerate(self.full_names)}
        self.full_bind: list[R] = []
        self.full_offset: list[np.ndarray] = []
        for _, _, rot, off in FULL_CHAIN:
            q = ue_rotator_to_dict(*rot)
            self.full_bind.append(R.from_quat([q['x'], q['y'], q['z'], q['w']]))
            self.full_offset.append(np.array(off, dtype=np.float64))

        n = len(FULL_CHAIN)
        self.rest_global: list[R] = [R.identity()] * n
        self.rest_pos: list[np.ndarray] = [np.zeros(3)] * n
        for i in range(n):
            p = self.full_parent[i]
            if p == -1:
                self.rest_global[i] = self.full_bind[i]
                self.rest_pos[i] = self.full_offset[i]
            else:
                self.rest_global[i] = self.rest_global[p] * self.full_bind[i]
                self.rest_pos[i] = self.rest_pos[p] + self.rest_global[p].apply(self.full_offset[i])

        fi = self.full_idx
        child_of = {"upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l",
                    "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r",
                    "thigh_l": "calf_l", "calf_l": "foot_l", "foot_l": "ball_l",
                    "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r"}
        self.rest_dir: dict[str, np.ndarray] = {}
        for bone, child in child_of.items():
            self.rest_dir[bone] = normalize(self.rest_pos[fi[child]] - self.rest_pos[fi[bone]])
        shoulder_mid = (self.rest_pos[fi["upperarm_l"]] + self.rest_pos[fi["upperarm_r"]]) * 0.5
        self.rest_dir["clavicle_l"] = normalize(self.rest_pos[fi["upperarm_l"]] - shoulder_mid)
        self.rest_dir["clavicle_r"] = normalize(self.rest_pos[fi["upperarm_r"]] - shoulder_mid)
        self.rest_dir["hand_l"] = normalize(self.rest_global[fi["hand_l"]].apply(np.array([1.0, 0.0, 0.0])))
        self.rest_dir["hand_r"] = normalize(self.rest_global[fi["hand_r"]].apply(np.array([-1.0, 0.0, 0.0])))

        hip_mid = (self.rest_pos[fi["thigh_l"]] + self.rest_pos[fi["thigh_r"]]) * 0.5
        up = normalize(shoulder_mid - hip_mid)
        side = normalize(self.rest_pos[fi["thigh_l"]] - self.rest_pos[fi["thigh_r"]])
        fwd = normalize(np.cross(up, side))
        right = normalize(np.cross(up, fwd))
        self.rest_body_frame = R.from_matrix(np.column_stack((fwd, right, up)))

    # -- landmark conversion ---------------------------------------------------
    def _convert_landmarks_to_ue_space(self, raw_landmarks: list[Any]) -> np.ndarray:
        """MediaPipe world landmarks -> (Fwd, Right, Up) in centimetres.

        MediaPipe reports the performer's LEFT side with positive lm.x, so Right = -lm.x.
        The original code passed lm.x through unmirrored despite its own comment.
        """
        points = np.zeros((len(raw_landmarks), 3), dtype=np.float64)
        for i, lm in enumerate(raw_landmarks):
            points[i, 0] = -lm.z * 100.0   # Fwd
            points[i, 1] = -lm.x * 100.0   # Right (mirrored)
            points[i, 2] = -lm.y * 100.0   # Up
        return points

    def _rest_pose_output(self) -> list[BoneTransform]:
        return [BoneTransform(name=name, rotation=dict(BIND_POSES[i]), position=dict(BIND_POSITIONS[i]))
                for i, name in enumerate(BONE_NAMES)]

    def solve(self, raw_world_landmarks: list[Any]) -> list[BoneTransform]:
        if len(raw_world_landmarks) < len(PoseLandmark):
            return self._rest_pose_output()

        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)
        comp = {int(k): PTS_TO_COMPONENT @ pts[int(k)] for k in PoseLandmark}

        l_hip, r_hip = comp[int(_P.LEFT_HIP)], comp[int(_P.RIGHT_HIP)]
        l_sh, r_sh = comp[int(_P.LEFT_SHOULDER)], comp[int(_P.RIGHT_SHOULDER)]
        hip_mid = (l_hip + r_hip) * 0.5
        shoulder_mid = (l_sh + r_sh) * 0.5

        up = normalize(shoulder_mid - hip_mid)
        side = normalize(l_hip - r_hip)
        fwd = normalize(np.cross(up, side))
        right = normalize(np.cross(up, fwd))
        if np.linalg.norm(fwd) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return self._rest_pose_output()

        if abs(self.torso_lean_offset_deg) > 1e-6:
            correction = R.from_rotvec(right * -math.radians(self.torso_lean_offset_deg))
            up = normalize(correction.apply(up))
            fwd = normalize(np.cross(up, side))
            right = normalize(np.cross(up, fwd))

        body_frame = R.from_matrix(np.column_stack((fwd, right, up)))
        body_rot = body_frame * self.rest_body_frame.inv()

        fi = self.full_idx
        n_full = len(FULL_CHAIN)
        global_rot: list[R] = [R.identity()] * n_full
        for i in range(n_full):
            name = self.full_names[i]
            parent = self.full_parent[i]
            if name in TORSO_BONES or name in INTERNAL_BONES:
                global_rot[i] = body_rot * self.rest_global[i]
            elif name in self.rest_dir:
                if name == "clavicle_l":
                    aim = comp[int(_P.LEFT_SHOULDER)] - shoulder_mid
                elif name == "clavicle_r":
                    aim = comp[int(_P.RIGHT_SHOULDER)] - shoulder_mid
                else:
                    a, b = BONE_AIM[name]
                    aim = comp[int(b)] - comp[int(a)]
                rest_world = body_rot.apply(self.rest_dir[name])
                global_rot[i] = swing_between(rest_world, aim) * body_rot * self.rest_global[i]
            else:
                global_rot[i] = global_rot[parent] * self.full_bind[i]

        pelvis_pos = hip_mid + np.array([0.0, 0.0, self.pelvis_default_height_cm])

        result: list[BoneTransform] = []
        for i, name in enumerate(BONE_NAMES):
            fidx = fi[name]
            fparent = self.full_parent[fidx]
            local = global_rot[fidx] if fparent == -1 else global_rot[fparent].inv() * global_rot[fidx]
            qx, qy, qz, qw = local.as_quat()
            rot = {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}

            pos = dict(BIND_POSITIONS[i])
            if i == 0:
                pos["x"], pos["y"], pos["z"] = float(pelvis_pos[0]), float(pelvis_pos[1]), float(pelvis_pos[2])
            result.append(BoneTransform(name=name, rotation=rot, position=pos))

        return result