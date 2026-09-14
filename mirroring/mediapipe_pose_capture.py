"""
Pose data model - shared by mediapipe_holistic_capture.py, pose_solver.py and
mediapipe_pose_osc_protocol.py.

Detection itself now happens inside HolisticLandmarker (see
mediapipe_holistic_capture.py) rather than a standalone PoseLandmarker, but
PoseLandmark and PoseFrame are kept in this module - unchanged in shape and
meaning - so pose_solver.py's `from mediapipe_pose_capture import
PoseLandmark` and every other existing import site keep working without
edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class PoseLandmark(IntEnum):
    """Mirrors MediaPipe's fixed 33-point BlazePose topology, in the
    exact order MediaPipe itself returns landmarks (index 0 = NOSE,
    ... index 32 = RIGHT_FOOT_INDEX). Also used by
    mediapipe_pose_osc_protocol.py and conductor.py to build
    human-readable channel names instead of bare indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


NUM_POSE_LANDMARKS = len(PoseLandmark)


@dataclass
class PoseFrame:
    """One frame's worth of pose tracking data, ready to hand to Conductor."""

    valid: bool
    timestamp_ms: int
    # Both lists, when valid, have exactly NUM_POSE_LANDMARKS entries, each
    # an object with .x .y .z .visibility .presence - the same shape
    # MediaPipe returns.
    landmarks: list = field(default_factory=list)        # image-normalized
    world_landmarks: list = field(default_factory=list)  # metric, hip-centered
