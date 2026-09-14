"""
Conductor - owns the single shared webcam, runs both MediaPipe detectors
against every frame, transforms their output, and drives both Unreal
Engine endpoints.
 
    Face -> Epic's stock "Live Link Face" UDP listener (port 11111),
            via live_link_face_protocol.LiveLinkFaceEncoder. This side
            is fully working end to end.
    Pose -> a custom OSC endpoint (default port 9001), via
            mediapipe_pose_osc_protocol.PoseOSCEncoder. IMPORTANT: as of
            this version, nothing in Unreal understands this data yet -
            there is no stock listener the way there is for face. See
            the "NEXT STEPS" comment at the bottom of this file.
 
Why one shared camera feeding two detectors, instead of two separate
scripts each with their own: most webcams only allow one reader at a
time. Two processes each independently calling cv2.VideoCapture(0)
tend to fight over the device (one fails to open, or gets a frozen
feed). Conductor captures exactly once per loop and hands the same
frame to both MediaPipeFaceCapture.process() and
MediaPipePoseCapture.process(), which removes that conflict at its root
instead of routing around it (e.g. with a virtual camera app).
 
Responsibilities, per loop iteration:
    1. Capture ONE frame from the one shared camera.
    2. Run both detectors against it.
    3. Transform each channel independently:
         - EMA smoothing per channel (see the Smoother class).
         - Hold the last valid values when detection drops out, rather
           than sending zeros/garbage that would make the character
           twitch or snap to a neutral pose.
         - For pose specifically: also send an explicit "present" flag
           (1.0 live / 0.0 held-stale) alongside the data, so the
           receiving side can tell "no one is in frame right now" apart
           from "the process died" - something the face channel can't
           do, since Epic's Live Link Face wire format is a fixed
           61-float struct with no room for extra metadata.
    4. Encode and send each channel to its own target via its own
       protocol module. Conductor is the only thing that owns sockets -
       neither protocol module sends anything itself.
 
Requires:
    pip install mediapipe opencv-python numpy python-osc
"""
from __future__ import annotations
 
import argparse
import socket
import time
from pathlib import Path

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient
import numpy as np
 
from live_link_face_protocol import CHANNEL_ORDER as FACE_CHANNEL_ORDER
from live_link_face_protocol import LiveLinkFaceEncoder
from mediapipe_face_capture import FaceFrame, MediaPipeFaceCapture
from mediapipe_pose_capture import MediaPipePoseCapture, PoseFrame, PoseLandmark
from mediapipe_pose_osc_protocol import (
    POSE_CHANNEL_ORDER,
    PoseOSCEncoder,
    pose_landmark_channel_prefix,
)
from pose_solver import BoneTransform, PoseSolver
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder
 
LIVE_LINK_FACE_PORT = 11111  # Unreal's stock Live Link Face plugin default
POSE_OSC_PORT = 9001         # arbitrary - must match whatever the custom LiveLink Source ends up listening on
 
# Eye bone rotation still isn't derived from anything - MediaPipe gives
# eye-look blendshapes (eyeLookInLeft etc.), not the separate eye bone
# yaw/pitch/roll Live Link Face's wire format also carries. Left at 0.0
# deliberately, same as the original single-channel Conductor.
_UNRESOLVED_EYE_CHANNELS = (
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
)

class Smoother:
    """Simple per-channel exponential moving average filter.
 
    Generic over any dict[str, float] - used as one independent
    instance for the face channel with its own internal state.
    """
 
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha  # 0-1: lower = smoother but laggier
        self._state: dict[str, float] = {}
 
    def apply(self, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            # First time a channel is seen, start at its own value
            # rather than 0 - avoids a fake fade-in from zero at startup.
            prev = self._state.get(name, value)
            self._state[name] = self.alpha * value + (1.0 - self.alpha) * prev
        return dict(self._state)
    
class PoseSmoother:
    """
    Applies Exponential Moving Average (EMA) to 3D position vectors and 
    Normalized Linear Interpolation (NLERP) to 4D rotation quaternions.
    Used to smooth the pose channel with its own internal state.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha  # 0 to 1: lower = smoother but laggier
        # Stores the previous frame's fully smoothed bone transforms
        self._state: dict[str, BoneTransform] = {}

    def apply(self, current_bones: list[BoneTransform]) -> list[BoneTransform]:
        smoothed_bones = []
        
        for curr in current_bones:
            # First time seeing this bone, start at its own value
            if curr.name not in self._state:
                self._state[curr.name] = curr
                smoothed_bones.append(curr)
                continue
            
            prev = self._state[curr.name]
            
            # --- 1. Position Smoothing (Standard EMA) ---
            # P_new = alpha * P_curr + (1 - alpha) * P_prev
            px = self.alpha * curr.position["x"] + (1.0 - self.alpha) * prev.position["x"]
            py = self.alpha * curr.position["y"] + (1.0 - self.alpha) * prev.position["y"]
            pz = self.alpha * curr.position["z"] + (1.0 - self.alpha) * prev.position["z"]
            smoothed_pos = {"x": px, "y": py, "z": pz}
            
            # --- 2. Rotation Smoothing (NLERP) ---
            q_curr = np.array([curr.rotation["x"], curr.rotation["y"], curr.rotation["z"], curr.rotation["w"]])
            q_prev = np.array([prev.rotation["x"], prev.rotation["y"], prev.rotation["z"], prev.rotation["w"]])
            
            # Quaternions "double-cover" 3D rotations: q and -q represent the exact same pose.
            # If the dot product is negative, the quaternions are pointing in opposite 4D hemispheres.
            # We must flip one to ensure we blend across the shortest path, avoiding a 360-degree spin.
            if np.dot(q_curr, q_prev) < 0.0:
                q_curr = -q_curr
                
            # Step A: Standard linear blend
            q_blended = self.alpha * q_curr + (1.0 - self.alpha) * q_prev
            
            # Step B: Normalize back to a unit quaternion (length = 1)
            norm = np.linalg.norm(q_blended)
            if norm > 1e-6:
                q_blended = q_blended / norm
            else:
                # Math fallback: if normalization fails due to bad data, hold the previous frame
                q_blended = q_prev
                
            smoothed_rot = {
                "x": float(q_blended[0]), 
                "y": float(q_blended[1]), 
                "z": float(q_blended[2]), 
                "w": float(q_blended[3])
            }
            
            # --- 3. Package and Store ---
            smoothed_bone = BoneTransform(name=curr.name, rotation=smoothed_rot, position=smoothed_pos)
            self._state[curr.name] = smoothed_bone
            smoothed_bones.append(smoothed_bone)
            
        return smoothed_bones

class Conductor:
    def __init__(
        self,
        face_capture: MediaPipeFaceCapture,
        pose_capture: MediaPipePoseCapture,
        camera_index: int = 0,
        face_ip: str = "127.0.0.1",
        face_port: int = LIVE_LINK_FACE_PORT,
        pose_ip: str = "127.0.0.1",
        pose_port: int = POSE_OSC_PORT,
        face_smoothing_alpha: float = 0.5,
        pose_smoothing_alpha: float = 0.5,
        show_debug: bool = False,
    ) -> None:
        self.face_capture = face_capture
        self.pose_capture = pose_capture
        self.show_debug = show_debug
        self._start_time = time.perf_counter()
 
        # --- the one shared camera, opened exactly once ---
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
 
        # --- face channel: transform state + network target ---
        self.face_smoother = Smoother(alpha=face_smoothing_alpha)
        self.face_encoder = LiveLinkFaceEncoder()
        self.face_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.face_target = (face_ip, face_port)
        # Neutral default until the first valid face detection arrives.
        self._last_valid_face_values: dict[str, float] = {
            name: 0.0 for name in FACE_CHANNEL_ORDER
        }
 
        # --- pose channel: transform state + network target ---
        # self.pose_smoother = Smoother(alpha=pose_smoothing_alpha)
        # self.pose_encoder = PoseOSCEncoder()
        # self.pose_osc_client = SimpleUDPClient(pose_ip, pose_port)
        # self._last_valid_pose_values: dict[str, float] = {
        #    name: 0.0 for name in POSE_CHANNEL_ORDER
        #}
        
        self.pose_smoother = PoseSmoother(alpha=pose_smoothing_alpha)

        # --- pose channel: transform state + network target ---
        self.pose_solver = PoseSolver()
        self.pose_encoder = LiveLinkPoseOSCEncoder(ip=pose_ip, port=pose_port)
        self._last_valid_bone_transforms: list[BoneTransform] = []
 
    # ---- timing ---------------------------------------------------------
 
    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)
 
    # ---- per-channel transforms ------------------------------------------
 
    def _face_frame_to_channel_values(self, frame: FaceFrame) -> dict[str, float]:
        """Turns a FaceFrame into the flat channel dict both the smoother
        and the encoder work with."""
        values = dict(frame.blendshapes)
        # First-pass scale: degrees -> roughly [-1, 1], the range Live
        # Link Face expects for head rotation. Already empirically tuned
        # in mediapipe_face_capture.py's rotation decomposition itself.
        values["headYaw"] = frame.head_yaw_deg / 90.0
        values["headPitch"] = frame.head_pitch_deg / 90.0
        values["headRoll"] = frame.head_roll_deg / 90.0
        for name in _UNRESOLVED_EYE_CHANNELS:
            values[name] = 0.0
        return values
 
    def _pose_frame_to_channel_values(self, frame: PoseFrame) -> dict[str, float]:
        """Turns a PoseFrame's world landmarks into the same kind of flat
        channel dict, keyed with POSE_CHANNEL_ORDER's names - so the same
        generic Smoother class above can be reused unmodified for pose,
        exactly as it is for face."""
        values: dict[str, float] = {}
        for landmark_enum, point in zip(PoseLandmark, frame.world_landmarks):
            prefix = pose_landmark_channel_prefix(landmark_enum)
            values[f"{prefix}_x"] = point.x
            values[f"{prefix}_y"] = point.y
            values[f"{prefix}_z"] = point.z
            values[f"{prefix}_visibility"] = point.visibility
            values[f"{prefix}_presence"] = point.presence
        return values
 
    # ---- per-channel send ------------------------------------------------
 
    def _handle_face(self, frame: FaceFrame) -> None:
        if frame.valid:
            raw_values = self._face_frame_to_channel_values(frame)
            self._last_valid_face_values.update(raw_values)
        else:
            # Hold the last valid data instead of sending zeros/neutral -
            # a dropped detection for a frame or two shouldn't make the
            # face visibly snap to a blank expression.
            raw_values = self._last_valid_face_values
 
        smoothed = self.face_smoother.apply(raw_values)
        packet = self.face_encoder.encode(smoothed)
        self.face_socket.sendto(packet, self.face_target)
 
    def _handle_pose(self, frame: PoseFrame, timestamp_ms: int) -> None:
        if frame.valid and len(frame.world_landmarks) > 0:
            # Log world pose landmarks
            '''
            for lm_enum in PoseLandmark:
                lm = frame.world_landmarks[int(lm_enum)]
                print(lm_enum.name, lm.x, lm.y, lm.z)
            '''
            # Get raw solved bones from MediaPipe
            raw_bones = self.pose_solver.solve(frame.world_landmarks)
            self._last_valid_bone_transforms = raw_bones

            # Apply quaternion-safe smoothing
            smoothed_bones = self.pose_smoother.apply(raw_bones)
            self.pose_encoder.send(smoothed_bones, present=True)
        else:
            # Hold the last valid rig pose if tracking drops out
            transforms_to_send = (
                self._last_valid_bone_transforms
                if self._last_valid_bone_transforms
                else self.pose_solver.solve([]) # Fallback to identity rest pose
            )
            self.pose_encoder.send(transforms_to_send, present=False)
        '''
        present = frame.valid  # the flag Live Link Face's protocol has no room for -> validates that a pose is currently present
        if frame.valid:
            raw_values = self._pose_frame_to_channel_values(frame)
            self._last_valid_pose_values.update(raw_values)
        else:
            raw_values = self._last_valid_pose_values
 
        smoothed = self.pose_smoother.apply(raw_values)
        args = self.pose_encoder.build_args(smoothed, present=present, timestamp_ms=timestamp_ms)
        self.pose_osc_client.send_message(self.pose_encoder.address, args)
        '''
 
    # ---- debug overlay -----------------------------------------------------
 
    def _draw_debug(self, frame, face_frame: FaceFrame, pose_frame: PoseFrame) -> None:
        """Lightweight status overlay - a single combined window showing
        whether each channel is currently tracking, not a full landmark
        mesh/skeleton drawing. Kept deliberately simple here to keep this
        merge focused; the full mesh overlay (mp.solutions.drawing_utils,
        with the landmark_pb2 conversion covered in earlier review) can
        be added back per-channel if wanted."""
        face_status = "FACE: TRACKING" if face_frame.valid else "FACE: SEARCHING"
        pose_status = "POSE: TRACKING" if pose_frame.valid else "POSE: SEARCHING"
        face_color = (0, 255, 0) if face_frame.valid else (0, 0, 255)
        pose_color = (0, 255, 0) if pose_frame.valid else (0, 0, 255)
        cv2.putText(frame, face_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
        cv2.putText(frame, pose_status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)
        cv2.imshow("Conductor Debug", frame)
        cv2.waitKey(1)
 
    # ---- main loop ---------------------------------------------------------
 
    def run(self) -> None:
        try:
            while True:
                ts = self._timestamp_ms()
                success, raw_frame = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
 
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
 
                # Built once per detector rather than shared as a single
                # object, to sidestep any question of whether two Tasks
                # API landmarkers reading the *same* mp.Image concurrently
                # is safe - this costs one extra cheap wrap, not a real
                # performance concern next to the model inference itself.
                face_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
                face_frame = self.face_capture.process(face_mp_image, ts)
                pose_frame = self.pose_capture.process(pose_mp_image, ts)
 
                self._handle_face(face_frame)
                self._handle_pose(pose_frame, ts)
 
                if self.show_debug:
                    self._draw_debug(raw_frame, face_frame, pose_frame)
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            self.face_capture.close()
            self.pose_capture.close()
            self.face_socket.close()
            if self.show_debug:
                cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-model", default="face_landmarker.task")
    parser.add_argument("--pose-model", default="pose_landmarker_full.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--face-ip", default="127.0.0.1")
    parser.add_argument("--face-port", type=int, default=LIVE_LINK_FACE_PORT)
    parser.add_argument("--pose-ip", default="127.0.0.1")
    parser.add_argument("--pose-port", type=int, default=POSE_OSC_PORT)
    parser.add_argument("--face-smoothing", type=float, default=0.5,
                         help="EMA alpha for face, 0-1. Lower = smoother but laggier.")
    parser.add_argument("--pose-smoothing", type=float, default=0.5,
                         help="EMA alpha for pose, 0-1. Lower = smoother but laggier.")
    parser.add_argument("--debug", action="store_true",
                         help="show a combined webcam + tracking-status debug window")
    args = parser.parse_args()
 
    for label, path in (("Face", args.face_model), ("Pose", args.pose_model)):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} model not found at {path}. See mediapipe_face_capture.py's "
                "and mediapipe_pose_capture.py's module docstrings for download links."
            )
 
    face_capture = MediaPipeFaceCapture(model_path=args.face_model)
    pose_capture = MediaPipePoseCapture(model_path=args.pose_model)
 
    conductor = Conductor(
        face_capture, pose_capture,
        camera_index=args.camera,
        face_ip=args.face_ip, face_port=args.face_port,
        pose_ip=args.pose_ip, pose_port=args.pose_port,
        face_smoothing_alpha=args.face_smoothing,
        pose_smoothing_alpha=args.pose_smoothing,
        show_debug=args.debug,
    )
    conductor.run()
 
 
if __name__ == "__main__":
    main()
 
# ---- NEXT STEPS ----------------------------------------------------------
#
# Face: done. Add a Live Link Face source in UE5's Live Link panel
# (listens on 11111 by default) and it should just work.
#
# Pose: two separate pieces of work remain, and they're independent of
# each other - either order is fine:
#
#   1. The custom ILiveLinkSource C++ plugin. Nothing in Unreal is
#      listening on port 9001 yet - these OSC packets currently go
#      nowhere. Before writing any C++, sanity-check the wire format
#      first with a plain OSC monitor (a five-line python-osc dummy
#      server, or a tool like Protokol) and confirm you see 168 args
#      per message, present flipping 0/1 as you step in and out of
#      frame, and values changing as you move - cheap to verify, and
#      isolates "is my data right" from "is my C++ right" as separate
#      questions.
#
#   2. Landmark positions -> actual bone rotations. This is the "why
#      don't the landmarks form a rig" question - see the chat answer
#      for the full explanation. Short version: MediaPipe only gives you
#      33 point *positions* in space, never rotations, and a rig needs
#      rotations. This conversion doesn't exist yet anywhere in this
#      pipeline. Worth doing in Python (numpy/scipy have solid rotation
#      math, easier to iterate on than C++) before sending over OSC,
#      producing bone-style rotation+position data closer to what
#      DollarsMoCap's body channel already looked like - which would
#      also make the eventual C++ plugin simpler, since it wouldn't
#      need to know anything about landmark math at all.