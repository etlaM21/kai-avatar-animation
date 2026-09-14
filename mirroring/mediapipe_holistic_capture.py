"""
MediaPipe Holistic capture module - runs HolisticLandmarker (Tasks API,
LIVE_STREAM mode) once per frame and returns pose, face and hand data in a
single pass, instead of running separate Face/PoseLandmarker detectors that
each redo their own internal detection step.

Why one detector instead of two:
    - Fewer inference passes per frame: Holistic detects the pose once and
      derives the face and hand regions of interest from it, rather than
      each detector independently re-running its own detection stage.
    - Face tracking now comes from that pose-derived ROI crop, so it keeps
      working at full-body distance where a standalone FaceLandmarker's own
      face detector would lose the (relatively tiny) face.
    - Hand landmarks (21 points per hand, plus world landmarks) come along
      for free - not wired into anything yet, kept for future finger
      support.

What Holistic does NOT give you: HolisticLandmarkerResult has no
facial_transformation_matrixes field, so head yaw/pitch/roll has no source
here. See head_pose_capture.py for why that's a separate, much cheaper
FaceLandmarker pass rather than something reimplemented on top of
face_landmarks.

Does NOT own a camera - Conductor owns the single shared camera and hands
the same frame here and to HeadPoseCapture.

Requires:
    pip install mediapipe opencv-python numpy

Model:
    Download the HolisticLandmarker model bundle from:
    https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
    save it as holistic_landmarker.task next to this script, or pass
    --holistic-model to conductor.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    HolisticLandmarkerResult,
    RunningMode,
)

from mediapipe_pose_capture import PoseFrame


@dataclass
class FaceFrame:
    """One frame's worth of face tracking data, ready to hand to Conductor.

    Same shape as the old mediapipe_face_capture.FaceFrame. head_yaw/pitch/
    roll_deg are NOT populated here (Holistic has no transformation-matrix
    output) - they default to 0.0 and Conductor overwrites them from a
    HeadPoseFrame (head_pose_capture.py) before this frame is used.
    """

    valid: bool  # True only if a face was actually found this frame
    timestamp_ms: int
    # ARKit blendshape name -> score. 52 entries when valid, empty otherwise.
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmarks: list = field(default_factory=list)
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0


@dataclass
class HandFrame:
    """One hand's worth of tracking data. Not consumed anywhere yet - kept
    for future finger support."""

    valid: bool
    landmarks: list = field(default_factory=list)        # image-normalized
    world_landmarks: list = field(default_factory=list)  # metric


@dataclass
class HandsFrame:
    timestamp_ms: int
    left: HandFrame = field(default_factory=lambda: HandFrame(valid=False))
    right: HandFrame = field(default_factory=lambda: HandFrame(valid=False))


class MediaPipeHolisticCapture:
    """Wraps HolisticLandmarker. Owns no camera - call process() once per
    frame."""

    def __init__(self, model_path: str = "holistic_landmarker.task") -> None:
        # Same lock rationale as head_pose_capture.HeadPoseCapture:
        # detect_async() fires _on_result() on a MediaPipe worker thread,
        # process() reads it from Conductor's main-loop thread.
        self._lock = threading.Lock()
        self._latest_result: HolisticLandmarkerResult | None = None

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            output_face_blendshapes=True,
            output_segmentation_mask=False,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            result_callback=self._on_result,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: HolisticLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with self._lock:
            self._latest_result = result

    def process(
        self, mp_image: mp.Image, timestamp_ms: int
    ) -> tuple[PoseFrame, FaceFrame, HandsFrame]:
        """Dispatches one frame for (async) detection and returns whatever
        the most recently *completed* detection produced - see
        head_pose_capture.HeadPoseCapture.process()'s docstring for the
        LIVE_STREAM lag caveat, which applies identically here.

        Unlike FaceLandmarkerResult/PoseLandmarkerResult, every list on
        HolisticLandmarkerResult is already flat (one holistic detection
        per image, not one-list-per-detected-instance), so there's no `[0]`
        unwrap here the way the old capture modules needed.
        """
        self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._lock:
            result = self._latest_result

        if result is None:
            return (
                PoseFrame(valid=False, timestamp_ms=timestamp_ms),
                FaceFrame(valid=False, timestamp_ms=timestamp_ms),
                HandsFrame(timestamp_ms=timestamp_ms),
            )

        if result.pose_landmarks:
            pose_frame = PoseFrame(
                valid=True,
                timestamp_ms=timestamp_ms,
                landmarks=result.pose_landmarks,
                world_landmarks=result.pose_world_landmarks,
            )
        else:
            pose_frame = PoseFrame(valid=False, timestamp_ms=timestamp_ms)

        if result.face_blendshapes:
            blendshapes = {c.category_name: c.score for c in result.face_blendshapes}
            face_frame = FaceFrame(
                valid=True,
                timestamp_ms=timestamp_ms,
                blendshapes=blendshapes,
                landmarks=result.face_landmarks,
            )
        else:
            face_frame = FaceFrame(valid=False, timestamp_ms=timestamp_ms)

        hands_frame = HandsFrame(
            timestamp_ms=timestamp_ms,
            left=HandFrame(
                valid=bool(result.left_hand_landmarks),
                landmarks=result.left_hand_landmarks,
                world_landmarks=result.left_hand_world_landmarks,
            ),
            right=HandFrame(
                valid=bool(result.right_hand_landmarks),
                landmarks=result.right_hand_landmarks,
                world_landmarks=result.right_hand_world_landmarks,
            ),
        )

        return pose_frame, face_frame, hands_frame

    def close(self) -> None:
        self.landmarker.close()
