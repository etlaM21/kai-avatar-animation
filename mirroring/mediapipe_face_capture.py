"""
MediaPipe Face capture module (LIVE_STREAM mode).

Uses MediaPipe Tasks' asynchronous LIVE_STREAM running mode and provides
an optional OpenCV debug overlay for webcam feed and facial landmarks.
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
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
    drawing_utils,
    drawing_styles,
    FaceLandmarksConnections
)


@dataclass
class FaceFrame:
    """Tracking data and optional visual frame for debugging."""
    valid: bool
    timestamp_ms: int
    blendshapes: dict[str, float] = field(default_factory=dict)
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0


def _rotation_matrix_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decomposes the 4x4 transform into correctly mapped yaw, pitch, and roll (degrees)."""
    # Append a zero-translation column to make it a valid 3x4 projection matrix
    proj_matrix = np.hstack((matrix[:3, :3], np.zeros((3, 1))))
    
    # Use OpenCV to robustly extract Euler angles from the camera-space matrix
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    # cv2 returns axes in the order: Pitch (X), Yaw (Y), Roll (Z)
    pitch_x, yaw_y, roll_z = euler_angles.flatten()
    
    return float(yaw_y), float(-pitch_x), float(roll_z)


class MediaPipeFaceCapture:
    """Captures webcam video using MediaPipe LIVE_STREAM asynchronous mode."""

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        camera_index: int = 0,
        show_debug: bool = False,
    ) -> None:
        self.show_debug = show_debug
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self._lock = threading.Lock()
        self._latest_result: FaceLandmarkerResult | None = None
        self._start_time = time.perf_counter()

        # Configure Tasks API for Live Stream mode
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
        """Asynchronous callback executed when inference completes."""
        with self._lock:
            self._latest_result = result

    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def read(self) -> FaceFrame:
        """Grabs a frame, triggers async inference, and optionally draws debug overlay."""
        ts = self._timestamp_ms()
        success, frame = self.cap.read()
        if not success:
            return FaceFrame(valid=False, timestamp_ms=ts)

        # Convert OpenCV BGR image to MediaPipe Image and dispatch inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, ts)

        # Retrieve the latest async result safely
        with self._lock:
            result = self._latest_result

        valid = False
        blendshapes: dict[str, float] = {}
        yaw = pitch = roll = 0.0

        if result and result.face_landmarks and result.face_blendshapes:
            valid = True
            blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}

            if result.facial_transformation_matrixes:
                matrix = np.array(result.facial_transformation_matrixes[0])
                yaw, pitch, roll = _rotation_matrix_to_euler(matrix)

            # Draw visual landmarks on the debug window
            if self.show_debug:
                h, w, _ = frame.shape
                annotated_image = frame
                face_landmarks = result.face_landmarks[0]

                # Draw the face landmarks using Google's example: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

        if self.show_debug:
            status = "TRACKING" if valid else "SEARCHING"
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.putText(
                frame, f"Status: {status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.imshow("MediaPipe LiveLink Debug", frame)
            cv2.waitKey(1)

        return FaceFrame(
            valid=valid,
            timestamp_ms=ts,
            blendshapes=blendshapes,
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            head_roll_deg=roll,
        )

    def close(self) -> None:
        self.cap.release()
        if self.show_debug:
            cv2.destroyAllWindows()
        self.landmarker.close()