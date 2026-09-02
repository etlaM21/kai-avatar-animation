"""
Conductor - orchestrates the MediaPipe Face capture pipeline and drives
Unreal Engine's stock "Live Link Face" plugin over UDP.

Responsibilities (per the agreed architecture):
    1. Starts MediaPipeFaceCapture and pulls its data each loop iteration.
    2. Transforms the data: EMA smoothing per channel, and holds the last
       valid frame when face detection drops out rather than sending
       zeros/garbage.
    3. Encodes via LiveLinkFaceEncoder and sends over UDP to Unreal's
       Live Link Face listener (default port 11111).

Requires:
    pip install mediapipe opencv-python numpy
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import cv2

from live_link_face_protocol import CHANNEL_ORDER, LiveLinkFaceEncoder
from mediapipe_face_capture import FaceFrame, MediaPipeFaceCapture

LIVE_LINK_FACE_PORT = 11111  # Unreal's stock Live Link Face plugin default

# Eye bone rotation isn't derived from anything yet in this first pass -
# MediaPipe's blendshapes cover eye *look* direction as shape keys
# (eyeLookInLeft etc.), but not the separate eye bone yaw/pitch/roll
# Live Link Face also carries. Left at 0.0 deliberately; iris landmarks
# would be the way to compute this later if it's needed.
_UNRESOLVED_EYE_CHANNELS = (
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
)


class Smoother:
    """Simple per-channel exponential moving average filter."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def apply(self, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            prev = self._state.get(name, value)
            self._state[name] = self.alpha * value + (1.0 - self.alpha) * prev
        return dict(self._state)


class Conductor:
    def __init__(
        self,
        capture: MediaPipeFaceCapture,
        encoder: LiveLinkFaceEncoder,
        target_ip: str = "127.0.0.1",
        target_port: int = LIVE_LINK_FACE_PORT,
        smoothing_alpha: float = 0.5,
    ) -> None:
        self.capture = capture
        self.encoder = encoder
        self.smoother = Smoother(alpha=smoothing_alpha)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (target_ip, target_port)

        # Neutral default until the first valid detection arrives.
        self._last_valid_values: dict[str, float] = {name: 0.0 for name in CHANNEL_ORDER}

    def _frame_to_channel_values(self, frame: FaceFrame) -> dict[str, float]:
        values = dict(frame.blendshapes)
        # First-pass scale: degrees -> roughly [-1, 1], matching the range
        # Live Link Face expects for head rotation. Sign/scale will likely
        # need tuning once you can see it moving in the Engine.
        values["headYaw"] = frame.head_yaw_deg / 90.0
        values["headPitch"] = frame.head_pitch_deg / 90.0
        values["headRoll"] = frame.head_roll_deg / 90.0
        for name in _UNRESOLVED_EYE_CHANNELS:
            values[name] = 0.0
        return values

    def run(self) -> None:
        try:
            while True:
                frame = self.capture.read()

                if frame.valid:
                    raw_values = self._frame_to_channel_values(frame)
                    self._last_valid_values.update(raw_values)
                else:
                    # Hold the last valid data instead of sending zeros.
                    raw_values = self._last_valid_values

                smoothed = self.smoother.apply(raw_values)
                packet = self.encoder.encode(smoothed)
                self.sock.sendto(packet, self.target)
        except KeyboardInterrupt:
            pass
        finally:
            self.capture.close()
            self.sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="face_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ip", default="127.0.0.1", help="Target Unreal Engine host")
    parser.add_argument("--port", type=int, default=LIVE_LINK_FACE_PORT)
    parser.add_argument(
        "--smoothing", type=float, default=0.5,
        help="EMA alpha, 0-1. Lower = smoother but laggier."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Display camera feed and landmark overlay")
    parser.add_argument(
        "--name", default="PythonConductor_Face",
        help="Subject name as it will appear in UE5's Live Link panel"
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Download "
            "face_landmarker.task from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task and place it next "
            "to this script, or pass --model."
        )

    capture = MediaPipeFaceCapture(
        model_path=args.model,
        camera_index=args.camera,
        show_debug=args.debug,
    )
    encoder = LiveLinkFaceEncoder(name=args.name)
    conductor = Conductor(
        capture, encoder,
        target_ip=args.ip, target_port=args.port,
        smoothing_alpha=args.smoothing,
    )
    conductor.run()


if __name__ == "__main__":
    main()