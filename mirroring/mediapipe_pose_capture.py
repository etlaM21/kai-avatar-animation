"""
MediaPipe Pose capture module (LIVE_STREAM mode).

Uses MediaPipe Tasks' asynchronous LIVE_STREAM running mode and provides
an optional OpenCV debug overlay for webcam feed and pose landmarks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
    drawing_utils,
    drawing_styles,
    PoseLandmarksConnections
)


@dataclass
class PoseFrame:
    """Tracking data and optional visual frame for debugging."""
    valid: bool
    timestamp_ms: int
    landmarks: list = field(default_factory=list)
    worldLandmarks: list = field(default_factory=list)

class MediaPipePoseCapture:
    """Captures webcam video using MediaPipe LIVE_STREAM asynchronous mode."""

    def __init__(
        self,
        model_path: str = "pose_landmarker_full.task",
        camera_index: int = 0,
        show_debug: bool = False,
    ) -> None:
        self.show_debug = show_debug
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self._lock = threading.Lock()
        self._latest_result: PoseLandmarkerResult | None = None
        self._start_time = time.perf_counter()

        # Configure Tasks API for Live Stream mode
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
        """Asynchronous callback executed when inference completes."""
        with self._lock:
            self._latest_result = result

    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def read(self) -> PoseFrame:
        """Grabs a frame, triggers async inference, and optionally draws debug overlay."""
        ts = self._timestamp_ms()
        success, frame = self.cap.read()
        if not success:
            return PoseFrame(valid=False, timestamp_ms=ts)

        # Convert OpenCV BGR image to MediaPipe Image and dispatch inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, ts)

        # Retrieve the latest async result safely
        with self._lock:
            result = self._latest_result

        valid = False
        pose_landmarks = []
        world_landmarks = []

        if result and result.pose_landmarks:
            valid = True
            pose_landmarks = result.pose_landmarks
            world_landmarks = result.pose_world_landmarks
            # Draw visual landmarks on the debug window
            if self.show_debug:
                h, w, _ = frame.shape
                annotated_image = frame
                pos_landmarks = result.pose_landmarks[0]
                pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
                pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

                # The new Tasks API drawing_utils accepts the raw landmark list natively
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=pos_landmarks,
                    connections=PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=pose_landmark_style,
                    connection_drawing_spec=pose_connection_style
                )

        if self.show_debug:
            status = "TRACKING" if valid else "SEARCHING"
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.putText(
                frame, f"Status: {status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.imshow("MediaPipe LiveLink Debug", frame)
            cv2.waitKey(1)

        return PoseFrame(
            valid=valid,
            timestamp_ms=ts,
            landmarks=pose_landmarks,
            worldLandmarks=world_landmarks
        )

    def close(self) -> None:
        self.cap.release()
        if self.show_debug:
            cv2.destroyAllWindows()
        self.landmarker.close()