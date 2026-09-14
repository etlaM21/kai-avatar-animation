"""
MediaPipe Face capture module - the "MediaPipe Face" detector.

Pure ML inference wrapper: given an already-captured frame (as an
mp.Image) and a timestamp, runs FaceLandmarker (Tasks API, LIVE_STREAM
mode) and returns a FaceFrame. Does NOT own a webcam - Conductor owns
the single shared camera and hands the same frame to this and to
MediaPipePoseCapture, so both detectors see exactly the same image at
exactly the same moment, from one physical device. (Two processes each
independently opening cv2.VideoCapture(0) is what we're avoiding here -
most webcams only allow one reader at a time.)

Requires:
    pip install mediapipe opencv-python numpy

Model:
    Download the FaceLandmarker model bundle from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    save it as face_landmarker.task next to this script, or pass
    --face-model to conductor.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

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
class FaceFrame:
    """One frame's worth of face tracking data, ready to hand to Conductor."""

    valid: bool  # True only if a face was actually found this frame
    timestamp_ms: int
    # ARKit blendshape name -> score. 52 entries when valid, empty otherwise.
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmarks: list = field(default_factory=list)
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0


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


class MediaPipeFaceCapture:
    """Wraps FaceLandmarker. Owns no camera - call process() once per frame."""

    def __init__(self, model_path: str = "face_landmarker.task") -> None:
        # A lock is required because detect_async() runs the model on a
        # MediaPipe-internal worker thread; _on_result() below fires on
        # that thread, while process() reads the result from whichever
        # thread Conductor's main loop runs on. Without this, the two
        # threads could race on _latest_result.
        self._lock = threading.Lock()
        self._latest_result: FaceLandmarkerResult | None = None

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Called by MediaPipe on its own worker thread once inference for
        a dispatched frame finishes. Just stashes the result - process()
        below decides what to do with it."""
        with self._lock:
            self._latest_result = result

    def process(self, mp_image: mp.Image, timestamp_ms: int) -> FaceFrame:
        """Dispatches one frame for (async) detection and returns whatever
        the most recently *completed* detection produced.

        LIVE_STREAM subtlety worth remembering: because detect_async()
        doesn't block, the FaceFrame returned here reflects the previous
        completed detection, not necessarily the frame just dispatched -
        there's a small, unavoidable lag of roughly one processing cycle.
        timestamp_ms below is the capture time you passed in, not
        necessarily the exact moment the returned detection finished.
        """
        self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._lock:
            result = self._latest_result

        if not result or not result.face_blendshapes:
            return FaceFrame(valid=False, timestamp_ms=timestamp_ms)

        blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        landmarks = result.face_landmarks[0] if result.face_landmarks else [],

        yaw = pitch = roll = 0.0
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0])
            yaw, pitch, roll = _rotation_matrix_to_euler(matrix)

        return FaceFrame(
            valid=True,
            timestamp_ms=timestamp_ms,
            blendshapes=blendshapes,
            landmarks=landmarks,
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            head_roll_deg=roll,
        )

    def close(self) -> None:
        self.landmarker.close()