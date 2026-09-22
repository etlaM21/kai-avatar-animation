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
import time
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe_pose_capture import PoseLandmark

HandLandmark = hand_landmarker.HandLandmark

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
class CalibrationState:
    """What a running timed calibration is doing right now, for the debug overlay
    and the GUI. phase is 'idle', 'countdown', 'sampling' or 'done'."""
    phase: str
    seconds_left: float
    samples: int
    samples_wanted: int
    lean_offset_deg: float


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


def _landmark_confidence(landmarks: list[Any], indices) -> float:
    """Lowest confidence among the landmarks a bone is aimed by. visibility ("is it
    occluded") and presence ("is it in frame at all") both matter, and MediaPipe drops
    them independently - the legs in the recordings fall to 0.63 visibility but 0.2
    presence. Missing or None fields count as fully confident, so synthetic landmarks
    (and any source that doesn't populate them) are never gated."""
    worst = 1.0
    for i in indices:
        lm = landmarks[int(i)]
        for value in (getattr(lm, "visibility", None), getattr(lm, "presence", None)):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            worst = min(worst, float(value))
    return worst


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

# Bones carried rigidly by the torso frame rather than aimed at a landmark.
TORSO_BONES = {"pelvis", "spine_01", "spine_02", "spine_04"}
# Present in the rig and needed for correct parenting, but never streamed.
# neck_02 is deliberately NOT here: it sits between neck_01 and head, and Unreal
# holds it at bind RELATIVE TO neck_01 - so once neck_01 rotates independently of
# the torso, neck_02 has to follow neck_01 by FK (the generic branch), not the torso.
INTERNAL_BONES = {"spine_03", "spine_05"}
# Share of the head's rotation (relative to the torso) given to neck_01; head gets
# the full orientation on top. Anatomically the cervical spine and the skull joint
# split the motion; 0.4 is a visual choice, not a measured value.
NECK_SHARE = 0.4

# Holistic face mesh indices for the head basis. Chosen for being bone-backed, so
# expressions don't move them: cheek contour extremes for the side axis (454 is the
# performer's LEFT - it lies on the image's right), eye outer corners to stabilise
# it, forehead top (10) and chin (152) for the up axis. The chin moves a little with
# jaw opening; eye->mouth would move with every smile.
# Share of the wrist's twist handed back to the forearm. MediaPipe can't observe
# pronation from joint positions, but the palm plane can - without this the whole
# roll lands on the wrist joint and the forearm stays unrotated. 0.5 is a visual
# choice; 0.0 disables the redistribution entirely.
FOREARM_TWIST_SHARE = 0.5

# Occlusion gating. MediaPipe always returns all 33 landmarks: when a limb leaves
# frame it INVENTS a plausible one rather than reporting nothing, and only
# visibility/presence say so. Measured on the recordings: torso and arms sit at
# ~1.0 throughout, while the legs fall to 0.42-0.63 visibility (presence ~0.2) for
# 10-17% of frames - so these thresholds gate real dropouts, not normal tracking.
# Hysteresis (hold below LOW, resume above HIGH) stops a limb flickering in and out
# of hold while it hovers at the threshold.
CONFIDENCE_HOLD_BELOW = 0.5
CONFIDENCE_RESUME_ABOVE = 0.65
# Frames to ease between following and holding, ~0.25 s at 30 fps. Entering the hold
# is blended too: the held pose IS the live pose at that moment, so it costs nothing,
# and it means a brief dip can't snap anything.
HOLD_BLEND_FRAMES = 8

FACE_LEFT, FACE_RIGHT = 454, 234
FACE_EYE_LEFT, FACE_EYE_RIGHT = 263, 33
FACE_TOP, FACE_CHIN = 10, 152


# ---------------------------------------------------------------------------
# Finger rig, extending the chain above past hand_l/hand_r. Same technique
# (minimal swing from rest direction to measured direction, parent-local
# conversion), same rig-verification discipline: VERIFIED against a
# RefSkeleton dump of SKM_Manny_Simple, do not hand-edit. Each hand has 19
# real bones - a metacarpal + 3 phalanges for index/middle/ring/pinky, and
# just 3 phalanges (no metacarpal) for the thumb - and MediaPipe's 21-point
# hand topology maps onto that directly, one landmark-pair per bone.
#
# Unlike the body chain, there's no unstreamed-bone gap to model here:
# hand_l/hand_r parent every finger bone directly, no equivalent of
# spine_03/05/neck_02 in between.
#
# (name, parent, rotator (pitch, yaw, roll), offset (x, y, z)). parent is
# either "hand_l"/"hand_r" (already in BONE_NAMES/FULL_CHAIN above) or
# another finger bone earlier in this same list.
# ---------------------------------------------------------------------------

FINGER_CHAIN_L: list[tuple] = [
    ("thumb_01_l", "hand_l", (-39.904178, -20.508676, 73.564464), (1.9924, -1.3566, -2.5815)),
    ("thumb_02_l", "thumb_01_l", (1.932290, -23.246006, 3.530628), (4.3780, 0.0, 0.0)),
    ("thumb_03_l", "thumb_02_l", (0.0, -10.000000, 0.0), (3.0860, 0.0, 0.0)),

    ("index_metacarpal_l", "hand_l", (-7.325502, 0.606162, 3.287746), (3.4445, 0.3847, -2.3793)),
    ("index_01_l", "index_metacarpal_l", (0.0, -23.373000, 0.0), (5.8771, -0.0432, 0.2409)),
    ("index_02_l", "index_01_l", (0.0, -14.892568, 0.0), (4.0800, 0.0, 0.0)),
    ("index_03_l", "index_02_l", (0.0, -12.516401, 0.0), (2.5950, 0.0, 0.0)),

    ("middle_metacarpal_l", "hand_l", (0.130751, 2.318392, -4.272500), (3.3758, 0.7536, -0.1829)),
    ("middle_01_l", "middle_metacarpal_l", (0.0, -31.572682, 0.0), (6.0982, 0.0, 0.0)),
    ("middle_02_l", "middle_01_l", (0.0, -20.769210, 0.0), (5.1690, 0.0, 0.0)),
    ("middle_03_l", "middle_02_l", (0.0, -10.000000, 0.0), (2.4740, 0.0, 0.0)),

    ("ring_metacarpal_l", "hand_l", (11.809319, 1.594563, -13.299835), (3.3743, 0.5425, 1.0918)),
    ("ring_01_l", "ring_metacarpal_l", (0.116938, -29.414482, 6.395844), (5.6455, 0.0416, -0.0207)),
    ("ring_02_l", "ring_01_l", (0.0, -18.964000, 0.0), (4.9770, 0.0, 0.0)),
    ("ring_03_l", "ring_02_l", (0.0, -9.168000, 0.0), (2.2650, 0.0, 0.0)),

    ("pinky_metacarpal_l", "hand_l", (19.527703, -11.850627, -27.769049), (3.3144, 0.3059, 2.3911)),
    ("pinky_01_l", "pinky_metacarpal_l", (-0.605043, -14.833681, 10.491640), (4.9576, 0.1431, -0.1988)),
    ("pinky_02_l", "pinky_01_l", (0.0, -21.286999, 0.0), (3.8160, 0.0, 0.0)),
    ("pinky_03_l", "pinky_02_l", (0.0, -4.917000, 0.0), (2.0400, 0.0, 0.0)),
]

FINGER_CHAIN_R: list[tuple] = [
    ("thumb_01_r", "hand_r", (-39.904178, -20.508676, 73.564464), (-1.9928, 1.3567, 2.5813)),
    ("thumb_02_r", "thumb_01_r", (1.932290, -23.246006, 3.530628), (-4.3778, 0.0, 0.0)),
    ("thumb_03_r", "thumb_02_r", (0.0, -10.000000, 0.0), (-3.0860, 0.0, 0.0)),

    ("index_metacarpal_r", "hand_r", (-7.325502, 0.606162, 3.287746), (-3.4445, -0.3852, 2.3793)),
    ("index_01_r", "index_metacarpal_r", (0.0, -23.373000, 0.0), (-5.8772, 0.0434, -0.2410)),
    ("index_02_r", "index_01_r", (0.0, -14.892568, 0.0), (-4.0799, 0.0, 0.0)),
    ("index_03_r", "index_02_r", (0.0, -12.516401, 0.0), (-2.5951, 0.0, 0.0)),

    ("middle_metacarpal_r", "hand_r", (0.130751, 2.318392, -4.272500), (-3.3758, -0.7540, 0.1828)),
    ("middle_01_r", "middle_metacarpal_r", (0.0, -31.572682, 0.0), (-6.0984, 0.0001, 0.0)),
    ("middle_02_r", "middle_01_r", (0.0, -20.769210, 0.0), (-5.1690, 0.0001, 0.0)),
    ("middle_03_r", "middle_02_r", (0.0, -10.000000, 0.0), (-2.4740, 0.0, 0.0)),

    ("ring_metacarpal_r", "hand_r", (11.809319, 1.594563, -13.299835), (-3.3742, -0.5430, -1.0918)),
    ("ring_01_r", "ring_metacarpal_r", (0.116938, -29.414482, 6.395844), (-5.6457, -0.0414, 0.0207)),
    ("ring_02_r", "ring_01_r", (0.0, -18.964000, 0.0), (-4.9771, 0.0, 0.0)),
    ("ring_03_r", "ring_02_r", (0.0, -9.168000, 0.0), (-2.2650, -0.0001, 0.0)),

    ("pinky_metacarpal_r", "hand_r", (19.527703, -11.850627, -27.769049), (-3.3147, -0.3059, -2.3913)),
    ("pinky_01_r", "pinky_metacarpal_r", (-0.605043, -14.833681, 10.491640), (-4.9573, -0.1433, 0.1989)),
    ("pinky_02_r", "pinky_01_r", (0.0, -21.286999, 0.0), (-3.8160, 0.0, 0.0)),
    ("pinky_03_r", "pinky_02_r", (0.0, -4.917000, 0.0), (-2.0400, 0.0, 0.0)),
]

FINGER_CHAIN: list[tuple] = FINGER_CHAIN_L + FINGER_CHAIN_R
FINGER_BONE_NAMES: list[str] = [c[0] for c in FINGER_CHAIN]

# Which pair of MediaPipe hand landmarks aims each bone. A finger's metacarpal
# (or, for the thumb, which has none in Manny, thumb_01 taking that role) aims
# WRIST -> its own MCP-equivalent; each phalanx then aims through the next
# joint out - exactly mirroring BONE_AIM's one-pair-per-bone shape above.
#
# Known, accepted approximation: each metacarpal's REST direction runs from
# its own rest position (offset a little from the wrist for where that
# specific finger meets the palm) to its child's, but its aim is measured
# WRIST -> MCP - MediaPipe has no landmark at that per-finger palm offset,
# only one shared WRIST point. Confirmed (via this module's finger tests)
# to cost a few degrees on metacarpals alone; every phalanx joint - the ones
# that actually drive curl and splay - solves to 0.000 deg in the bind-pose
# round-trip test. Not fixable without estimating the missing offset from
# something other than measurement, i.e. guessing again.
#
# The thumb used to be mapped one joint further in (WRIST->CMC, CMC->MCP,
# MCP->IP), which sheared the whole thumb: Manny's thumb_01 IS the thumb
# metacarpal, so it runs CMC->MCP, not WRIST->CMC. Measured at 18-32 deg RMS
# error per thumb bone on a real capture - even with the performer's hands at
# rest - against the anatomical mapping in tests/solver_checks.py.
_HL = HandLandmark
FINGER_AIM_L: dict[str, tuple] = {
    "thumb_01_l": (_HL.THUMB_CMC, _HL.THUMB_MCP),
    "thumb_02_l": (_HL.THUMB_MCP, _HL.THUMB_IP),
    "thumb_03_l": (_HL.THUMB_IP, _HL.THUMB_TIP),

    "index_metacarpal_l": (_HL.WRIST, _HL.INDEX_FINGER_MCP),
    "index_01_l": (_HL.INDEX_FINGER_MCP, _HL.INDEX_FINGER_PIP),
    "index_02_l": (_HL.INDEX_FINGER_PIP, _HL.INDEX_FINGER_DIP),
    "index_03_l": (_HL.INDEX_FINGER_DIP, _HL.INDEX_FINGER_TIP),

    "middle_metacarpal_l": (_HL.WRIST, _HL.MIDDLE_FINGER_MCP),
    "middle_01_l": (_HL.MIDDLE_FINGER_MCP, _HL.MIDDLE_FINGER_PIP),
    "middle_02_l": (_HL.MIDDLE_FINGER_PIP, _HL.MIDDLE_FINGER_DIP),
    "middle_03_l": (_HL.MIDDLE_FINGER_DIP, _HL.MIDDLE_FINGER_TIP),

    "ring_metacarpal_l": (_HL.WRIST, _HL.RING_FINGER_MCP),
    "ring_01_l": (_HL.RING_FINGER_MCP, _HL.RING_FINGER_PIP),
    "ring_02_l": (_HL.RING_FINGER_PIP, _HL.RING_FINGER_DIP),
    "ring_03_l": (_HL.RING_FINGER_DIP, _HL.RING_FINGER_TIP),

    "pinky_metacarpal_l": (_HL.WRIST, _HL.PINKY_MCP),
    "pinky_01_l": (_HL.PINKY_MCP, _HL.PINKY_PIP),
    "pinky_02_l": (_HL.PINKY_PIP, _HL.PINKY_DIP),
    "pinky_03_l": (_HL.PINKY_DIP, _HL.PINKY_TIP),
}
# Same landmark indices apply against right_hand_world_landmarks - only the
# bone names differ, so build FINGER_AIM_R by suffix rather than retyping it.
FINGER_AIM_R: dict[str, tuple] = {name[:-1] + "r": pair for name, pair in FINGER_AIM_L.items()}
FINGER_AIM: dict[str, tuple] = {**FINGER_AIM_L, **FINGER_AIM_R}


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
        # Head orientation relative to the torso, as a rotation in the torso frame.
        # Held across face-detection dropouts (a moment of lost face tracking
        # shouldn't snap the head back onto the chest); identity until a face is seen.
        self._head_rel = R.identity()
        # Head pose recorded as "looking straight ahead" by calibrate_neutral(). The
        # face-mesh basis isn't exactly the rig's head axes (forehead->chin leans back
        # a few degrees, people carry their heads differently). Identity = mesh as-is.
        self.head_neutral = R.identity()
        self._last_face_rot: R | None = None
        # Per-hand rotation from the rig's rest hand to the measured one, set by
        # _solve_body_globals() and read by solve_hands() so both agree on the wrist.
        self._hand_delta: dict[str, R | None] = {"_l": None, "_r": None}
        # Running timed calibration, or None. See begin_calibration().
        self._cal: dict[str, Any] | None = None
        # Occlusion gating state, per aimed bone: last trusted pose (relative to the
        # torso, so a held limb still travels with the body), how far into the hold
        # the blend is, and whether the bone is currently gated (for the hysteresis).
        self._hold_pose: dict[str, R] = {}
        self._hold_weight: dict[str, float] = {}
        self._hold_gated: dict[str, bool] = {}
        self._build_rest_pose()
        self._build_finger_rest_pose()
        self._build_hand_frames()

    def measure_torso_lean_deg(self, raw_world_landmarks: list[Any]) -> float:
        """Forward lean of the performer's torso RELATIVE TO THE RIG'S OWN rest torso.
        Positive = leaning further forward than Manny stands. Zero means the torso
        already matches the rig, which is exactly when body_rot comes out as identity.

        Measured against vertical instead (as this did originally), a calibration on a
        genuinely upright performer still cancels Manny's own 5.8 deg backward torso
        lean, leaving the character 5.8 deg off its bind pose at the performer's
        neutral. Caught by the synthetic calibration check: calibrating on the rig's
        own rest pose has to be a no-op, and wasn't."""
        if len(raw_world_landmarks) < len(PoseLandmark):
            return 0.0
        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)
        hip_mid = (pts[int(_P.LEFT_HIP)] + pts[int(_P.RIGHT_HIP)]) * 0.5
        shoulder_mid = (pts[int(_P.LEFT_SHOULDER)] + pts[int(_P.RIGHT_SHOULDER)]) * 0.5
        up = normalize(shoulder_mid - hip_mid)
        return math.degrees(math.atan2(float(up[0]), float(up[2]))) - self.rest_lean_deg

    # -- timed calibration -------------------------------------------------------
    def begin_calibration(self, countdown_s: float = 3.0, samples: int = 30,
                          now_s: float | None = None) -> None:
        """Start a countdown, then average the neutral pose over `samples` frames.

        Averaging matters: the lean measurement is noisy frame to frame, and a single
        snapshot bakes in whatever jitter existed at that instant. The countdown is
        what lets the performer get back into position and stand still first."""
        self._cal = {"started": now_s if now_s is not None else time.monotonic(),
                     "countdown_s": countdown_s, "want": samples,
                     "leans": [], "poses": [], "faces": [], "done_at": None}

    def cancel_calibration(self) -> None:
        self._cal = None

    def update_calibration(self, raw_world_landmarks: list[Any],
                           face_landmarks: list[Any] | None = None,
                           image_size: tuple[int, int] | None = None,
                           now_s: float | None = None) -> CalibrationState:
        """Drive one frame of a running calibration. Safe to call every frame; a
        no-op returning phase='idle' when none is running."""
        cal = self._cal
        if cal is None:
            return CalibrationState("idle", 0.0, 0, 0, self.torso_lean_offset_deg)
        now = now_s if now_s is not None else time.monotonic()

        left = cal["countdown_s"] - (now - cal["started"])
        if left > 0.0:
            return CalibrationState("countdown", left, 0, cal["want"], self.torso_lean_offset_deg)

        if cal["done_at"] is not None:
            if now - cal["done_at"] > 2.0:  # leave the result on screen briefly
                self._cal = None
            return CalibrationState("done", 0.0, len(cal["leans"]), cal["want"], self.torso_lean_offset_deg)

        if len(raw_world_landmarks) >= len(PoseLandmark):
            cal["leans"].append(self.measure_torso_lean_deg(raw_world_landmarks))
            cal["poses"].append(raw_world_landmarks)
            face_rot = (self._face_mesh_rotation(face_landmarks, image_size)
                        if face_landmarks and image_size else None)
            cal["faces"].append(face_rot)

        if len(cal["leans"]) < cal["want"]:
            return CalibrationState("sampling", 0.0, len(cal["leans"]), cal["want"], self.torso_lean_offset_deg)

        self.torso_lean_offset_deg = float(np.mean(cal["leans"]))
        # Head neutral is averaged AFTER the lean offset lands, since the head pose is
        # recorded relative to the (now corrected) torso frame.
        rels = [body.inv() * face
                for pose, face in zip(cal["poses"], cal["faces"]) if face is not None
                and (body := self._body_rotation(pose)) is not None]
        if rels:
            self.head_neutral = R.concatenate(rels).mean()
            self._head_rel = R.identity()
        cal["done_at"] = now
        return CalibrationState("done", 0.0, len(cal["leans"]), cal["want"], self.torso_lean_offset_deg)

    def calibrate_neutral(self, raw_world_landmarks: list[Any]) -> float:
        """Record the current lean as 'upright' and, if a face has been seen, the
        current head pose as 'looking straight ahead'. Returns the lean offset now in
        use. The head part uses the most recent face mesh passed to solve(), so the
        signature - and every existing caller - stays as it was."""
        self.torso_lean_offset_deg = self.measure_torso_lean_deg(raw_world_landmarks)
        if self._last_face_rot is not None:
            body_rot = self._body_rotation(raw_world_landmarks)  # with the new lean offset
            if body_rot is not None:
                self.head_neutral = body_rot.inv() * self._last_face_rot
                self._head_rel = R.identity()
        return self.torso_lean_offset_deg

    # -- head --------------------------------------------------------------------
    def _face_mesh_rotation(self, face_landmarks: list[Any], image_size: tuple[int, int]) -> R | None:
        """Head orientation in component space from the Holistic face mesh: the
        rotation taking the rig's rest head axes (side=+X, fwd=+Y, up=+Z) onto the
        face's. None if the mesh is missing or degenerate.

        The mesh only exists image-normalised (x of width, y of height, z roughly in
        x's units). Scaling x and z by width and y by height makes it metrically
        consistent - measured on a real capture, z*1.0 keeps the face most rigid
        across head turns (2.4% shape variation vs 3.5% at z*1.25, 4.2% at z*0.75).
        It is then in the same camera-aligned axes as the pose world landmarks, so it
        goes through the same conversion. Orthographic, which is fine for an object
        the size of a head at performance distance.

        Why the face mesh and not the pose model's own ear/eye/nose points (which are
        metric already): on the same capture the pose points registered ~8 deg of a
        ~37 deg head turn and 18 deg of pitch range vs the mesh's 72 - smooth, but
        they barely move.
        """
        if not face_landmarks or len(face_landmarks) <= max(FACE_LEFT, FACE_RIGHT, FACE_CHIN):
            return None
        w, h = image_size

        def pt(i: int) -> np.ndarray:
            lm = face_landmarks[i]
            # (Fwd, Right, Up) exactly as _convert_landmarks_to_ue_space, then to component.
            return PTS_TO_COMPONENT @ np.array([-lm.z * w, -lm.x * w, -lm.y * h])

        side = normalize((pt(FACE_LEFT) - pt(FACE_RIGHT)) + (pt(FACE_EYE_LEFT) - pt(FACE_EYE_RIGHT)))
        up_hint = pt(FACE_TOP) - pt(FACE_CHIN)
        up = normalize(up_hint - np.dot(up_hint, side) * side)
        if np.linalg.norm(side) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return None
        fwd = np.cross(up, side)  # Z x X = Y
        return R.from_matrix(np.column_stack((side, fwd, up)))

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
        # Manny's own hip->shoulder line is 5.8 deg off vertical; torso lean is measured
        # relative to this, not to vertical. See measure_torso_lean_deg().
        self.rest_lean_deg = math.degrees(math.atan2(float(up[1]), float(up[2])))
        side = normalize(self.rest_pos[fi["thigh_l"]] - self.rest_pos[fi["thigh_r"]])
        fwd = normalize(np.cross(up, side))
        right = normalize(np.cross(up, fwd))
        self.rest_body_frame = R.from_matrix(np.column_stack((fwd, right, up)))

    # -- finger rig rest pose, extending the FK chain past hand_l/hand_r ------
    def _build_finger_rest_pose(self) -> None:
        fi = self.full_idx
        self.finger_rest_global: dict[str, R] = {
            "hand_l": self.rest_global[fi["hand_l"]],
            "hand_r": self.rest_global[fi["hand_r"]],
        }
        self.finger_rest_pos: dict[str, np.ndarray] = {
            "hand_l": self.rest_pos[fi["hand_l"]],
            "hand_r": self.rest_pos[fi["hand_r"]],
        }
        self.finger_bind_rot: dict[str, R] = {}
        self.finger_bind_rot_dict: dict[str, dict] = {}
        self.finger_bind_offset: dict[str, np.ndarray] = {}
        self.finger_parent_of: dict[str, str] = {}
        finger_child_of: dict[str, str] = {}

        for name, parent, rot, off in FINGER_CHAIN:
            q = ue_rotator_to_dict(*rot)
            self.finger_bind_rot_dict[name] = q
            self.finger_bind_rot[name] = R.from_quat([q["x"], q["y"], q["z"], q["w"]])
            self.finger_bind_offset[name] = np.array(off, dtype=np.float64)
            self.finger_parent_of[name] = parent
            self.finger_rest_global[name] = self.finger_rest_global[parent] * self.finger_bind_rot[name]
            self.finger_rest_pos[name] = (
                self.finger_rest_pos[parent] + self.finger_rest_global[parent].apply(self.finger_bind_offset[name])
            )
            if parent not in ("hand_l", "hand_r"):
                finger_child_of[parent] = name

        # Rest direction each bone aims along: toward its own child bone when
        # it has one, otherwise (the last phalanx of each finger) its local
        # aim axis rotated into rest-global space - the same fallback used
        # above for hand_l/hand_r's own rest_dir, since a fingertip bone has
        # no further bone to point a direction at.
        self.finger_rest_dir: dict[str, np.ndarray] = {}
        for name, _, _, _ in FINGER_CHAIN:
            if name in finger_child_of:
                child = finger_child_of[name]
                self.finger_rest_dir[name] = normalize(self.finger_rest_pos[child] - self.finger_rest_pos[name])
            else:
                axis = np.array([1.0, 0.0, 0.0]) if name.endswith("_l") else np.array([-1.0, 0.0, 0.0])
                self.finger_rest_dir[name] = normalize(self.finger_rest_global[name].apply(axis))

    # -- hand orientation from the palm plane ---------------------------------
    @staticmethod
    def _palm_frame(wrist: np.ndarray, index_mcp: np.ndarray, middle_mcp: np.ndarray,
                    pinky_mcp: np.ndarray) -> R | None:
        """Orthonormal frame of a hand: along the fingers, across the palm, and the
        palm normal. Built identically from rig rest positions and from measured
        landmarks, so any constant geometric offset between the two cancels out.

        This is what makes the hand's ROLL observable. A swing-only aim leaves
        rotation about the bone axis free, so the fingers inherited the torso's roll
        and looked twisted; three non-collinear palm points pin it down.
        """
        fwd = normalize(middle_mcp - wrist)
        normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        normal = normalize(normal - np.dot(normal, fwd) * fwd)
        if np.linalg.norm(fwd) < 1e-6 or np.linalg.norm(normal) < 1e-6:
            return None
        return R.from_matrix(np.column_stack((fwd, normal, np.cross(fwd, normal))))

    def _build_hand_frames(self) -> None:
        self.hand_rest_frame: dict[str, R] = {}
        for side in ("_l", "_r"):
            frame = self._palm_frame(self.finger_rest_pos[f"hand{side}"],
                                     self.finger_rest_pos[f"index_01{side}"],
                                     self.finger_rest_pos[f"middle_01{side}"],
                                     self.finger_rest_pos[f"pinky_01{side}"])
            assert frame is not None, f"degenerate rest palm frame for hand{side}"
            self.hand_rest_frame[side] = frame

    def _hand_rotation(self, raw_hand_world_landmarks: list[Any], side: str) -> R | None:
        """World rotation taking the rig's rest hand onto the measured one. None if
        that hand isn't tracked this frame."""
        if not raw_hand_world_landmarks or len(raw_hand_world_landmarks) < 21:
            return None
        pts = self._convert_landmarks_to_ue_space(raw_hand_world_landmarks)
        comp = {i: PTS_TO_COMPONENT @ pts[i] for i in range(len(pts))}
        frame = self._palm_frame(comp[int(_HL.WRIST)], comp[int(_HL.INDEX_FINGER_MCP)],
                                 comp[int(_HL.MIDDLE_FINGER_MCP)], comp[int(_HL.PINKY_MCP)])
        return None if frame is None else frame * self.hand_rest_frame[side].inv()

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

    def _body_rotation(self, raw_world_landmarks: list[Any]) -> R | None:
        if len(raw_world_landmarks) < len(PoseLandmark):
            return None
        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)
        return self._body_rotation_from_comp({int(k): PTS_TO_COMPONENT @ pts[int(k)] for k in PoseLandmark})

    def _body_rotation_from_comp(self, comp: dict[int, np.ndarray]) -> R | None:
        """Whole-body orientation (measured torso frame vs the rig's rest torso
        frame), lean calibration applied. None if degenerate."""
        l_hip, r_hip = comp[int(_P.LEFT_HIP)], comp[int(_P.RIGHT_HIP)]
        l_sh, r_sh = comp[int(_P.LEFT_SHOULDER)], comp[int(_P.RIGHT_SHOULDER)]
        hip_mid = (l_hip + r_hip) * 0.5
        shoulder_mid = (l_sh + r_sh) * 0.5

        up = normalize(shoulder_mid - hip_mid)
        side = normalize(l_hip - r_hip)
        fwd = normalize(np.cross(up, side))
        right = normalize(np.cross(up, fwd))
        if np.linalg.norm(fwd) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return None

        if abs(self.torso_lean_offset_deg) > 1e-6:
            correction = R.from_rotvec(right * -math.radians(self.torso_lean_offset_deg))
            up = normalize(correction.apply(up))
            fwd = normalize(np.cross(up, side))
            right = normalize(np.cross(up, fwd))

        body_frame = R.from_matrix(np.column_stack((fwd, right, up)))
        return body_frame * self.rest_body_frame.inv()

    def _apply_hold(self, name: str, live: R, body_rot: R, confidence: float) -> R:
        """Follow the measurement while it can be trusted; hold the last trusted pose
        while it can't. Held relative to the torso, so an occluded limb still turns
        and travels with the body instead of freezing in world space."""
        was_gated = self._hold_gated.get(name, False)
        gated = confidence < (CONFIDENCE_RESUME_ABOVE if was_gated else CONFIDENCE_HOLD_BELOW)
        self._hold_gated[name] = gated

        # Into the hold immediately, out of it gradually. The held pose is the one the
        # character is already in, so snapping to it can't pop - whereas blending IN
        # means showing a frame or two of the invented pose first (measured: a 40 deg
        # bogus knee bend still reached 35 deg before the hold caught it). Coming back
        # does need the ramp: the live pose by then is somewhere else entirely.
        weight = 1.0 if gated else max(0.0, self._hold_weight.get(name, 0.0) - 1.0 / HOLD_BLEND_FRAMES)
        self._hold_weight[name] = weight

        if weight <= 0.0:
            self._hold_pose[name] = body_rot.inv() * live  # trusted: this is the pose to hold
            return live
        held = self._hold_pose.get(name)
        if held is None:
            return live
        target = body_rot * held
        return live * R.from_rotvec((live.inv() * target).as_rotvec() * weight)

    @staticmethod
    def _twist_about(rot: R, axis: np.ndarray) -> R:
        """The part of `rot` that spins about `axis` (swing-twist decomposition)."""
        q = rot.as_quat()  # x, y, z, w
        proj = np.dot(q[:3], axis) * axis
        twist = np.array([proj[0], proj[1], proj[2], q[3]])
        norm = np.linalg.norm(twist)
        return R.identity() if norm < 1e-8 else R.from_quat(twist / norm)

    def _solve_body_globals(
        self,
        raw_world_landmarks: list[Any],
        face_landmarks: list[Any] | None = None,
        image_size: tuple[int, int] | None = None,
        left_hand_world_landmarks: list[Any] | None = None,
        right_hand_world_landmarks: list[Any] | None = None,
    ) -> tuple[list[R], R, np.ndarray] | None:
        """Shared by solve() and solve_hands(): computes every FULL_CHAIN
        bone's global rotation, plus the whole-body orientation (body_rot)
        and hip midpoint, from raw pose world landmarks. Returns None on the
        same triggers solve() used to fall back to rest pose on (too few
        landmarks, or a degenerate fwd/up) - each caller applies its own
        fallback in that case. Pulled out of solve() so solve_hands() can
        read hand_l/hand_r's global rotation from the exact same computation
        solve() itself uses - not an independent approximation of it.

        face_landmarks / image_size (Holistic face mesh, image-normalised, and the
        capture size in pixels) drive neck_01/head. Without them the head keeps its
        last torso-relative pose - welded to the chest until a face is first seen."""
        if len(raw_world_landmarks) < len(PoseLandmark):
            return None

        pts = self._convert_landmarks_to_ue_space(raw_world_landmarks)
        comp = {int(k): PTS_TO_COMPONENT @ pts[int(k)] for k in PoseLandmark}
        body_rot = self._body_rotation_from_comp(comp)
        if body_rot is None:
            return None
        hip_mid = (comp[int(_P.LEFT_HIP)] + comp[int(_P.RIGHT_HIP)]) * 0.5
        shoulder_mid = (comp[int(_P.LEFT_SHOULDER)] + comp[int(_P.RIGHT_SHOULDER)]) * 0.5

        if face_landmarks is not None and image_size is not None:
            face_rot = self._face_mesh_rotation(face_landmarks, image_size)
            if face_rot is not None:
                self._last_face_rot = face_rot
                self._head_rel = body_rot.inv() * face_rot * self.head_neutral.inv()
        neck_rel = R.from_rotvec(self._head_rel.as_rotvec() * NECK_SHARE)

        n_full = len(FULL_CHAIN)
        global_rot: list[R] = [R.identity()] * n_full
        for i in range(n_full):
            name = self.full_names[i]
            parent = self.full_parent[i]
            if name in TORSO_BONES or name in INTERNAL_BONES:
                global_rot[i] = body_rot * self.rest_global[i]
            elif name == "neck_01":
                global_rot[i] = body_rot * neck_rel * self.rest_global[i]
            elif name == "head":
                # The full head orientation regardless of the split: NECK_SHARE only
                # moves the pivot, not where the face ends up pointing.
                global_rot[i] = body_rot * self._head_rel * self.rest_global[i]
            elif name in self.rest_dir:
                if name == "clavicle_l":
                    aim = comp[int(_P.LEFT_SHOULDER)] - shoulder_mid
                    conf = _landmark_confidence(raw_world_landmarks, (_P.LEFT_SHOULDER,))
                elif name == "clavicle_r":
                    aim = comp[int(_P.RIGHT_SHOULDER)] - shoulder_mid
                    conf = _landmark_confidence(raw_world_landmarks, (_P.RIGHT_SHOULDER,))
                else:
                    a, b = BONE_AIM[name]
                    aim = comp[int(b)] - comp[int(a)]
                    conf = _landmark_confidence(raw_world_landmarks, (a, b))
                rest_world = body_rot.apply(self.rest_dir[name])
                live = swing_between(rest_world, aim) * body_rot * self.rest_global[i]
                global_rot[i] = self._apply_hold(name, live, body_rot, conf)
            else:
                global_rot[i] = global_rot[parent] * self.full_bind[i]

        # Hands, once the arm chain above is solved: a full palm basis replaces the
        # swing aimed at the pose model's INDEX point, which is both roll-free and
        # ~17 deg off the hand bone's own axis. Falls back to that aim per hand when
        # the 21-point hand isn't tracked.
        self._hand_delta = {
            "_l": self._hand_rotation(left_hand_world_landmarks or [], "_l"),
            "_r": self._hand_rotation(right_hand_world_landmarks or [], "_r"),
        }
        for side in ("_l", "_r"):
            delta = self._hand_delta[side]
            if delta is None:
                continue
            hand_i = self.full_idx[f"hand{side}"]
            lower_i = self.full_idx[f"lowerarm{side}"]
            global_rot[hand_i] = delta * self.rest_global[hand_i]
            # MediaPipe can't see forearm pronation (joint positions only), but the
            # palm now can: pass a share of the wrist's twist back up the forearm so
            # the roll comes from the arm rather than snapping at the wrist. The
            # hand's own global orientation is unchanged by this - only the pivot.
            #
            # Measured against where the forearm WOULD carry the hand at bind, not
            # against the forearm itself: hand_l's bind local already contains a
            # -67.8 deg roll, and treating that as twist tipped the whole arm by 34 deg.
            if FOREARM_TWIST_SHARE > 0.0:
                carried = global_rot[lower_i] * self.full_bind[hand_i]
                residual = global_rot[hand_i] * carried.inv()
                axis = normalize(global_rot[lower_i].apply(
                    np.array([1.0, 0.0, 0.0]) if side == "_l" else np.array([-1.0, 0.0, 0.0])))
                twist = self._twist_about(residual, axis)
                global_rot[lower_i] = R.from_rotvec(twist.as_rotvec() * FOREARM_TWIST_SHARE) * global_rot[lower_i]

        return global_rot, body_rot, hip_mid

    def solve(
        self,
        raw_world_landmarks: list[Any],
        face_landmarks: list[Any] | None = None,
        image_size: tuple[int, int] | None = None,
        left_hand_world_landmarks: list[Any] | None = None,
        right_hand_world_landmarks: list[Any] | None = None,
    ) -> list[BoneTransform]:
        solved = self._solve_body_globals(raw_world_landmarks, face_landmarks, image_size,
                                          left_hand_world_landmarks, right_hand_world_landmarks)
        if solved is None:
            return self._rest_pose_output()
        global_rot, _body_rot, hip_mid = solved

        fi = self.full_idx
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

    # -- fingers ----------------------------------------------------------
    def _rest_finger_bone(self, name: str) -> BoneTransform:
        return BoneTransform(
            name=name,
            rotation=dict(self.finger_bind_rot_dict[name]),
            position={"x": float(self.finger_bind_offset[name][0]),
                      "y": float(self.finger_bind_offset[name][1]),
                      "z": float(self.finger_bind_offset[name][2])},
        )

    def _solve_one_hand(
        self,
        side_suffix: str,
        raw_hand_world_landmarks: list[Any],
        hand_delta: R | None,
        root_global: R | None,
    ) -> list[BoneTransform]:
        """hand_delta is the rotation from the rig's rest hand to the measured one
        (see _hand_rotation) - the frame every finger's rest direction is carried
        into. It used to be body_rot, the TORSO's rotation, which meant the fingers
        inherited the chest's roll instead of the wrist's: turn the wrist over and
        the fingers kept pointing the old way, which is what read as 'clawed'."""
        names = [n for n in FINGER_BONE_NAMES if n.endswith(side_suffix)]
        if hand_delta is None or root_global is None or len(raw_hand_world_landmarks) < 21:
            return [self._rest_finger_bone(name) for name in names]

        pts = self._convert_landmarks_to_ue_space(raw_hand_world_landmarks)
        comp = {i: PTS_TO_COMPONENT @ pts[i] for i in range(len(raw_hand_world_landmarks))}

        global_rot: dict[str, R] = {("hand_l" if side_suffix == "_l" else "hand_r"): root_global}
        for name in names:
            a, b = FINGER_AIM[name]
            aim = comp[int(b)] - comp[int(a)]
            rest_world = hand_delta.apply(self.finger_rest_dir[name])
            global_rot[name] = swing_between(rest_world, aim) * hand_delta * self.finger_rest_global[name]

        result: list[BoneTransform] = []
        for name in names:
            parent = self.finger_parent_of[name]
            local = global_rot[parent].inv() * global_rot[name]
            qx, qy, qz, qw = local.as_quat()
            result.append(BoneTransform(
                name=name,
                rotation={"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
                position={"x": float(self.finger_bind_offset[name][0]),
                          "y": float(self.finger_bind_offset[name][1]),
                          "z": float(self.finger_bind_offset[name][2])},
            ))
        return result

    def solve_hands(
        self,
        pose_world_landmarks: list[Any],
        left_hand_world_landmarks: list[Any],
        right_hand_world_landmarks: list[Any],
    ) -> tuple[list[BoneTransform], list[BoneTransform]]:
        """Returns (left_hand_bones, right_hand_bones), each 19 long in
        FINGER_CHAIN order. Either half falls back to bind pose independently
        if that hand (or the body) isn't tracked this frame - never raises,
        never returns a short list. hand_l/hand_r's own live orientation
        comes straight out of _solve_body_globals(), the same computation
        solve() uses for the streamed body bones - so finger parent-local
        rotations always compose against the value actually being sent to
        Unreal for that bone this frame - including the palm-plane orientation,
        since the same hand landmarks are passed through to it here."""
        solved = self._solve_body_globals(pose_world_landmarks, None, None,
                                          left_hand_world_landmarks, right_hand_world_landmarks)
        if solved is None:
            hand_delta = {"_l": None, "_r": None}
            hand_l_global = hand_r_global = None
        else:
            global_rot, _body_rot, _hip_mid = solved
            fi = self.full_idx
            hand_delta = self._hand_delta
            hand_l_global = global_rot[fi["hand_l"]]
            hand_r_global = global_rot[fi["hand_r"]]

        left = self._solve_one_hand("_l", left_hand_world_landmarks, hand_delta["_l"], hand_l_global)
        right = self._solve_one_hand("_r", right_hand_world_landmarks, hand_delta["_r"], hand_r_global)
        return left, right