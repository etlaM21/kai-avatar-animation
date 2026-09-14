"""
Head pose capture module - a slim FaceLandmarker used for exactly one thing:
head yaw/pitch/roll via output_facial_transformation_matrixes.

This module used to be mediapipe_face_capture.py and did the full job (face
blendshapes, face mesh landmarks, head rotation). That responsibility has
moved to HolisticLandmarker (see mediapipe_holistic_capture.py), which runs
once per frame instead of two separate detectors and gives face tracking
from its internal pose-derived ROI crop - better at full-body distance than
a standalone FaceLandmarker.

HolisticLandmarkerResult has no equivalent of
facial_transformation_matrixes, though - head rotation isn't exposed there
at all. Rather than reimplement it via solvePnP against a canonical face
model (which would need its own from-scratch verification against the
Engine - see pose_solver.py's module docstring for how fiddly that
verification was for bone rotations), this module keeps running a second,
much cheaper FaceLandmarker pass alongside Holistic, with blendshapes and
full landmark output turned off since Holistic already provides those.

Requires:
    pip install mediapipe opencv-python numpy

Model:
    Reuses face_landmarker.task, already present for the pre-Holistic
    pipeline - download link in this repo's history / MediaPipe's model
    index if it's missing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)


@dataclass
class HeadPoseFrame:
    """One frame's worth of head rotation, ready to be stitched into a
    FaceFrame by Conductor."""

    valid: bool  # True only if this detector found a face this frame
    timestamp_ms: int
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


def _rotation_matrix_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decomposes the 3x3 rotation part of MediaPipe's 4x4 transform into
    yaw/pitch/roll (degrees), via OpenCV's projection-matrix decomposition.

    This is the approach already tuned by hand against the Engine (note
    the pitch sign flip below) - kept as-is rather than reverting to a
    from-scratch version, since it was already verified against a real
    turning head rather than guessed at.
    """
    # decomposeProjectionMatrix expects a 3x4 projection matrix; a plain
    # rotation matrix with a zero translation column is a valid special
    # case of one (no extra camera intrinsics involved here).
    proj_matrix = np.hstack((matrix[:3, :3], np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    # cv2 returns the three angles in the order [pitch(x), yaw(y), roll(z)].
    pitch_x, yaw_y, roll_z = euler_angles.flatten()

    # Empirically-tuned sign flip on pitch, verified against the Engine.
    return float(yaw_y), float(-pitch_x), float(roll_z)


class HeadPoseCapture:
    """Wraps a slim FaceLandmarker. Owns no camera - call process() once
    per frame, same async-dispatch-then-read-latest pattern as
    MediaPipeHolisticCapture."""

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # A lock is required because detect_async() runs the model on a
        # MediaPipe-internal worker thread; _on_result() below fires on
        # that thread, while process() reads the result from whichever
        # thread Conductor's main loop runs on.
        self._lock = threading.Lock()
        self._latest_result: FaceLandmarkerResult | None = None

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=1,
            # Blendshapes and full landmarks come from Holistic now - only
            # the transformation matrix is unique to this detector.
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._on_result,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with self._lock:
            self._latest_result = result

    def process(self, mp_image: mp.Image, timestamp_ms: int) -> HeadPoseFrame:
        """Same LIVE_STREAM lag caveat as MediaPipeHolisticCapture.process():
        the HeadPoseFrame returned here reflects the previous completed
        detection, not necessarily the frame just dispatched."""
        self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._lock:
            result = self._latest_result

        if not result or not result.facial_transformation_matrixes:
            return HeadPoseFrame(valid=False, timestamp_ms=timestamp_ms)

        matrix = np.array(result.facial_transformation_matrixes[0])
        yaw, pitch, roll = _rotation_matrix_to_euler(matrix)

        return HeadPoseFrame(
            valid=True,
            timestamp_ms=timestamp_ms,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    def close(self) -> None:
        self.landmarker.close()
