"""
MediaPipe Pose capture module - the "MediaPipe Pose" detector.

Pure ML inference wrapper, mirroring mediapipe_face_capture.py exactly:
given an already-captured frame (mp.Image) and a timestamp, runs
PoseLandmarker (Tasks API, LIVE_STREAM mode) and returns a PoseFrame.
No camera ownership here either - see mediapipe_face_capture.py's
module docstring for why that responsibility now lives in Conductor.

Requires:
    pip install mediapipe opencv-python

Model:
    Download pose_landmarker_full.task from the MediaPipe Pose
    Landmarker model index and place it next to this script, or pass
    --pose-model to conductor.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)


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
    # MediaPipe returns, just already unwrapped from "list of one pose"
    # (result.pose_landmarks[0]) since num_poses=1 below.
    landmarks: list = field(default_factory=list)        # image-normalized
    world_landmarks: list = field(default_factory=list)  # metric, hip-centered


class MediaPipePoseCapture:
    """Wraps PoseLandmarker. Owns no camera - call process() once per frame."""

    def __init__(self, model_path: str = "pose_landmarker_full.task") -> None:
        self._lock = threading.Lock()
        self._latest_result: PoseLandmarkerResult | None = None

        # All option names here are snake_case - the Python Tasks API is
        # snake_case throughout, no exceptions. (MediaPipe's docs mix in
        # JS/Android camelCase examples on the same pages, which is an
        # easy trap - cross-check against a working Python call if in doubt.)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with self._lock:
            self._latest_result = result

    def process(self, mp_image: mp.Image, timestamp_ms: int) -> PoseFrame:
        """Same async-dispatch-then-read-latest pattern as
        MediaPipeFaceCapture.process() - see its docstring for the
        LIVE_STREAM lag caveat, which applies identically here."""
        self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._lock:
            result = self._latest_result

        if not result or not result.pose_landmarks:
            return PoseFrame(valid=False, timestamp_ms=timestamp_ms)

        return PoseFrame(
            valid=True,
            timestamp_ms=timestamp_ms,
            landmarks=result.pose_landmarks[0],
            # note: pose_world_landmarks, not worldLandmarks - snake_case,
            # same trap as the options above.
            world_landmarks=result.pose_world_landmarks[0],
        )

    def close(self) -> None:
        self.landmarker.close()