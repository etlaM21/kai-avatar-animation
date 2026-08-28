"""
MediaPipe Pose -> OSC sender for the MediaPipe/Kimodo LiveLink pipeline.

Captures body pose from a webcam using MediaPipe's PoseLandmarker (the
Tasks API - the maintained successor to the legacy `mp.solutions.pose` /
`mp.solutions.face_mesh` used by older demos like JimWest/MeFaMo, which
Google stopped updating after the 2023 Solutions -> Tasks migration).

Streams the 33 pose *world* landmarks per frame as a single OSC message,
so this plugs into the same "OSC over UDP" transport lane that a Kimodo/
SOMA sender would use, just on a different OSC address - the custom
LiveLink Source plugin (ILiveLinkSource) tells the two subjects apart by
address, not by port.

Wire format:
    address: /mediapipe/pose
    args:    [timestamp_ms, num_landmarks,
              x0, y0, z0, visibility0, presence0,
              x1, y1, z1, visibility1, presence1,
              ... (33 landmarks total)]
    -> 2 + 33 * 5 = 167 args per frame

World landmarks (not the image-normalized `pose_landmarks`) are used
deliberately: they're metric (meters) and hip-centered, which is the
right space to drive a 3D skeleton rather than a 2D image overlay.

Requires:
    pip install mediapipe opencv-python python-osc

Model:
    Download `pose_landmarker_full.task` from the MediaPipe Pose
    Landmarker model index (Google AI Edge / MediaPipe Solutions site)
    and place it next to this script, or pass --model with a path.
"""

from __future__ import annotations

import argparse
import time
from enum import IntEnum
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from pythonosc.udp_client import SimpleUDPClient


class PoseLandmark(IntEnum):
    """Mirrors MediaPipe's fixed 33-point BlazePose topology.

    Useful downstream (e.g. in the LiveLink plugin / a Remap Asset) for
    turning named joints into bone-local rotations instead of juggling
    raw indices.
    """

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


NUM_LANDMARKS = len(PoseLandmark)


class MediaPipePoseSender:
    """Captures webcam pose and streams world-space landmarks over OSC."""

    def __init__(
        self,
        model_path: str = "pose_landmarker_full.task",
        camera_index: int = 0,
        ip: str = "127.0.0.1",
        port: int = 9001,
        osc_address: str = "/mediapipe/pose",
        show_debug: bool = False,
    ) -> None:
        self.osc_address = osc_address
        self.show_debug = show_debug
        self.osc_client = SimpleUDPClient(ip, port)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self._start_time = time.perf_counter()

    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def _encode_frame(self, world_landmarks) -> list[float]:
        """Flattens the 33 world landmarks into a single OSC arg list."""
        args: list[float] = [float(self._timestamp_ms()), float(NUM_LANDMARKS)]
        for lm in world_landmarks:
            args.extend([lm.x, lm.y, lm.z, lm.visibility, lm.presence])
        return args

    def _draw_debug(self, image, result) -> None:
        if not result.pose_landmarks:
            return
        h, w, _ = image.shape
        for lm in result.pose_landmarks[0]:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

    def run(self) -> None:
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms())

                if result.pose_world_landmarks:
                    args = self._encode_frame(result.pose_world_landmarks[0])
                    self.osc_client.send_message(self.osc_address, args)

                if self.show_debug:
                    self._draw_debug(frame, result)
                    cv2.imshow("MediaPipe Pose -> OSC", cv2.flip(frame, 1))
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            self.cap.release()
            self.landmarker.close()
            if self.show_debug:
                cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pose_landmarker_full.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ip", default="127.0.0.1", help="LiveLink Source plugin host")
    parser.add_argument("--port", type=int, default=9001, help="OSC listen port on the plugin")
    parser.add_argument("--debug", action="store_true", help="show a webcam preview with landmarks")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Download pose_landmarker_full.task "
            "from the MediaPipe Pose Landmarker model index (Google AI Edge site) "
            "and place it next to this script, or pass --model."
        )

    sender = MediaPipePoseSender(
        model_path=args.model,
        camera_index=args.camera,
        ip=args.ip,
        port=args.port,
        show_debug=args.debug,
    )
    sender.run()


if __name__ == "__main__":
    main()
