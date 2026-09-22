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
 
Why one shared camera feeding both detectors, instead of two separate
scripts each with their own: most webcams only allow one reader at a
time. Two processes each independently calling cv2.VideoCapture(0)
tend to fight over the device (one fails to open, or gets a frozen
feed). Conductor captures exactly once per loop and hands the same
frame to both MediaPipeHolisticCapture.process() (pose, face, hands
in one pass) and HeadPoseCapture.process() (head yaw/pitch/roll only,
via a second, much cheaper FaceLandmarker - HolisticLandmarkerResult
has no transformation-matrix output), which removes that conflict at
its root instead of routing around it (e.g. with a virtual camera app).
 
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
from head_pose_capture import HeadPoseCapture
from mediapipe_holistic_capture import FaceFrame, HandsFrame, MediaPipeHolisticCapture
from mediapipe_pose_capture import PoseFrame, PoseLandmark
from mediapipe_pose_osc_protocol import (
    POSE_CHANNEL_ORDER,
    PoseOSCEncoder,
    pose_landmark_channel_prefix,
)
from pose_solver import BoneTransform, PoseSolver
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder
from landmark_recorder import LandmarkRecorder

# Tasks-API drawing helpers. The legacy mp.solutions namespace (and
# mediapipe.framework.formats.landmark_pb2 with it) no longer exists in current
# mediapipe builds - the Tasks drawing_utils takes the raw landmark list
# natively, so no protobuf conversion is needed.
from mediapipe.tasks.python.vision import (
    drawing_utils,
    drawing_styles,
    PoseLandmarksConnections,
    FaceLandmarksConnections,
    HandLandmarksConnections,
)
 
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

# Candidate capture modes, highest first. Cameras silently fall back to their
# nearest supported mode instead of failing, so the only way to know what you
# actually got is to set it and read it back - which is what the loop does.
_PREFERRED_MODES = ((3840, 2160), (2560, 1440), (1920, 1080), (1280, 720))


def open_camera(index: int, width: int | None = None, height: int | None = None):
    """Opens the shared webcam at the highest mode it will actually accept.

    MJPG matters more than it looks: many webcams will agree to 1080p+ over the
    default YUY2 but then deliver it at 5-10 fps, which reads as "the tracking is
    laggy" rather than "the camera is starved".
    """
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():  # some backends dislike the explicit flag
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    modes = [(width, height)] if width and height else list(_PREFERRED_MODES)
    for w, h in modes:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h):
            break
    cap.set(cv2.CAP_PROP_FPS, 60)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {index}: {got_w}x{got_h} @ {got_fps:.0f} fps (MJPG)")
    return cap


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
    DEBUG_WINDOW = "Conductor Debug"

    def __init__(
        self,
        holistic_capture: MediaPipeHolisticCapture,
        head_pose_capture: HeadPoseCapture,
        camera_index: int = 0,
        camera_width: int | None = None,
        camera_height: int | None = None,
        torso_lean_offset_deg: float = 0.0,
        preview_scale: float = 0.5,
        face_ip: str = "127.0.0.1",
        face_port: int = LIVE_LINK_FACE_PORT,
        pose_ip: str = "127.0.0.1",
        pose_port: int = POSE_OSC_PORT,
        face_smoothing_alpha: float = 0.5,
        pose_smoothing_alpha: float = 0.5,
        show_debug: bool = False,
        record_path: str | None = None,
    ) -> None:
        self.holistic_capture = holistic_capture
        self.head_pose_capture = head_pose_capture
        self.show_debug = show_debug
        self._warned_no_pose_lms = False
        self._warned_no_face_lms = False
        self._start_time = time.perf_counter()
 
        # --- the one shared camera, opened exactly once ---
        self.cap = open_camera(camera_index, camera_width, camera_height)

        # Optional raw-landmark dump for offline solver checks (see
        # landmark_recorder.py / tests/solver_checks.py). None = no recording,
        # zero overhead.
        # Capture size in pixels: the face mesh is image-normalised, and the head
        # solve needs it to put x, y and z back in the same units.
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.recorder: LandmarkRecorder | None = None
        if record_path:
            self.recorder = LandmarkRecorder(record_path, self.frame_size)

        # The capture frame is deliberately large (MediaPipe crops its ROI from it,
        # so resolution helps detection at distance), but a 4K debug window is
        # unusable. WINDOW_NORMAL makes it user-resizable; the initial size is just
        # a fraction of the capture so it fits on screen. This scales the WINDOW
        # only - the full-resolution frame still goes to the detectors.
        if self.show_debug:
            cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv2.namedWindow(self.DEBUG_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.DEBUG_WINDOW,
                             max(320, int(cap_w * preview_scale)),
                             max(180, int(cap_h * preview_scale)))
 
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
        self.pose_solver = PoseSolver(torso_lean_offset_deg=torso_lean_offset_deg)
        self.pose_encoder = LiveLinkPoseOSCEncoder(ip=pose_ip, port=pose_port)
        self._last_valid_bone_transforms: list[BoneTransform] = []
        # Kept so the 'c' key can calibrate the torso lean from the live frame.
        self._last_valid_world_landmarks: list = []
        # Held per-hand (not per-frame) so one hand losing tracking doesn't
        # snap ITS fingers to bind pose while the other hand keeps moving -
        # same "hold last valid" rationale as _last_valid_bone_transforms.
        # Empty until the first valid detection, same as that list.
        self._last_valid_left_hand_bones: list[BoneTransform] = []
        self._last_valid_right_hand_bones: list[BoneTransform] = []
 
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
        # in head_pose_capture.py's rotation decomposition itself.
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
 
    def _handle_pose(self, frame: PoseFrame, timestamp_ms: int, hands_frame: HandsFrame,
                     face_frame: FaceFrame | None = None) -> None:
        if frame.valid and len(frame.world_landmarks) > 0:
            # Get raw solved bones from MediaPipe
            # The face mesh drives neck/head; without a face this frame the solver
            # holds the last head pose relative to the torso.
            face_lms = face_frame.landmarks if face_frame is not None and face_frame.valid else None
            # The hands go in here too, not just into solve_hands(): hand_l/hand_r are
            # BODY bones, and their orientation now comes from the palm plane. Passing
            # the same landmarks to both keeps the wrist and the fingers consistent.
            raw_bones = self.pose_solver.solve(
                frame.world_landmarks, face_lms, self.frame_size,
                hands_frame.left.world_landmarks if hands_frame.left.valid else None,
                hands_frame.right.world_landmarks if hands_frame.right.valid else None,
            )
            self._last_valid_bone_transforms = raw_bones
            self._last_valid_world_landmarks = frame.world_landmarks
            present = True
        else:
            # Hold the last valid rig pose if tracking drops out
            raw_bones = (
                self._last_valid_bone_transforms
                if self._last_valid_bone_transforms
                else self.pose_solver.solve([])  # Fallback to identity rest pose
            )
            present = False

        # Fingers ride along in the same message. solve_hands() falls back to
        # bind pose per-hand on its own when a hand isn't tracked; hold the
        # last valid finger pose instead, same "don't snap to neutral"
        # rationale as the body and face channels, but per-hand so one hand
        # losing tracking doesn't affect the other.
        left_bones, right_bones = self.pose_solver.solve_hands(
            frame.world_landmarks, hands_frame.left.world_landmarks, hands_frame.right.world_landmarks
        )
        if hands_frame.left.valid:
            self._last_valid_left_hand_bones = left_bones
        elif self._last_valid_left_hand_bones:
            left_bones = self._last_valid_left_hand_bones
        if hands_frame.right.valid:
            self._last_valid_right_hand_bones = right_bones
        elif self._last_valid_right_hand_bones:
            right_bones = self._last_valid_right_hand_bones

        # Apply quaternion-safe smoothing to body + both hands together -
        # PoseSmoother keys purely by BoneTransform.name, so this needed no
        # changes to support more bones.
        smoothed_bones = self.pose_smoother.apply(raw_bones + left_bones + right_bones)
        self.pose_encoder.send(smoothed_bones, present=present)
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
 
    @staticmethod
    def _image_landmarks(frame) -> list | None:
        """Finds the IMAGE-SPACE landmarks on a Face/PoseFrame.

        Deliberately not frame.world_landmarks: those are metric and hip-centred,
        so drawing them over the picture would put a skeleton in the corner.

        Handles two different shapes defensively: FaceLandmarker/PoseLandmarker
        return one landmark list PER DETECTED PERSON/FACE (a list of lists) -
        unwrap the first entry in that case. HolisticLandmarker's lists are
        already flat (single detection per image), so use them as-is. Field
        naming is up to the capture modules, so try the plausible names and
        give up quietly rather than killing the loop.
        """
        for attr in ("landmarks", "image_landmarks", "normalized_landmarks",
                     "pose_landmarks", "face_landmarks"):
            value = getattr(frame, attr, None)
            if not value:
                continue
            first = value[0]
            # a list-of-lists means it is still per-person; unwrap it
            if isinstance(first, (list, tuple)):
                return list(first) if first else None
            return list(value)
        return None

    def _draw_debug(self, frame, face_frame: FaceFrame, pose_frame: PoseFrame,
                     hands_frame: HandsFrame | None = None) -> int:
        """Combined webcam view: tracking status plus the full landmark overlay
        for both channels. Returns the key pressed, so the caller can act on it."""
        pose_lms = self._image_landmarks(pose_frame) if pose_frame.valid else None
        if pose_lms:
            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=pose_lms,
                connections=PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
            )
        elif pose_frame.valid and not self._warned_no_pose_lms:
            print("[debug] PoseFrame exposes no image-space landmarks - skeleton overlay "
                  "disabled. Expose the Tasks API's result.pose_landmarks on PoseFrame "
                  "(alongside world_landmarks) to enable it.")
            self._warned_no_pose_lms = True

        face_lms = self._image_landmarks(face_frame) if face_frame.valid else None
        if face_lms:
            drawing_utils.draw_landmarks(
                image=frame, landmark_list=face_lms,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
            drawing_utils.draw_landmarks(
                image=frame, landmark_list=face_lms,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        elif face_frame.valid and not self._warned_no_face_lms:
            print("[debug] FaceFrame exposes no image-space landmarks - mesh overlay "
                  "disabled. Expose the Tasks API's result.face_landmarks on FaceFrame.")
            self._warned_no_face_lms = True

        if hands_frame:
            for hand in (hands_frame.left, hands_frame.right):
                if hand.valid and hand.landmarks:
                    drawing_utils.draw_landmarks(
                        image=frame, landmark_list=hand.landmarks,
                        connections=HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
                    )

        # Text drawn in pixels on a 4K frame would be unreadable once the window is
        # scaled down, so scale it with the frame height.
        s = max(0.7, frame.shape[0] / 1080.0)
        t = max(2, int(round(2 * s)))
        face_status = "FACE: TRACKING" if face_frame.valid else "FACE: SEARCHING"
        pose_status = "POSE: TRACKING" if pose_frame.valid else "POSE: SEARCHING"
        face_color = (0, 255, 0) if face_frame.valid else (0, 0, 255)
        pose_color = (0, 255, 0) if pose_frame.valid else (0, 0, 255)
        cv2.putText(frame, face_status, (int(20*s), int(34*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, face_color, t)
        cv2.putText(frame, pose_status, (int(20*s), int(68*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, pose_color, t)

        # MediaPipe reports a standing performer as leaning ~18 deg forward; this
        # shows the offset currently cancelling that, and how to set it.
        lean_now = (self.pose_solver.measure_torso_lean_deg(self._last_valid_world_landmarks)
                    if self._last_valid_world_landmarks else 0.0)
        cv2.putText(frame,
                    f"lean raw {lean_now:+5.1f}  offset {self.pose_solver.torso_lean_offset_deg:+5.1f}"
                    f"  [c] calibrate upright",
                    (int(20*s), int(100*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (255, 200, 0), t)

        cv2.imshow(self.DEBUG_WINDOW, frame)
        return cv2.waitKey(1) & 0xFF
 
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
                holistic_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                head_pose_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                pose_frame, face_frame, hands_frame = self.holistic_capture.process(holistic_mp_image, ts)
                head_pose_frame = self.head_pose_capture.process(head_pose_mp_image, ts)
                # HolisticLandmarkerResult has no transformation-matrix
                # equivalent, so head rotation is stitched in from the
                # separate slim FaceLandmarker pass. Leave face_frame's
                # 0.0 defaults in place if that pass missed this frame -
                # _handle_face's hold-last-valid logic then carries the
                # last known head pose forward, same as it already does
                # for every other face channel.
                if head_pose_frame.valid:
                    face_frame.head_yaw_deg = head_pose_frame.yaw_deg
                    face_frame.head_pitch_deg = head_pose_frame.pitch_deg
                    face_frame.head_roll_deg = head_pose_frame.roll_deg
                if self.recorder is not None:
                    self.recorder.add(ts, pose_frame, face_frame, hands_frame, head_pose_frame)
                self._handle_face(face_frame)
                self._handle_pose(pose_frame, ts, hands_frame, face_frame)

                if self.show_debug:
                    key = self._draw_debug(raw_frame, face_frame, pose_frame, hands_frame)
                    if key == ord("c") and self._last_valid_world_landmarks:
                        offset = self.pose_solver.calibrate_neutral(self._last_valid_world_landmarks)
                        print(f"Torso lean calibrated: offset {offset:+.1f} deg")
                    elif key == 27:  # Esc
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if self.recorder is not None:
                self.recorder.save()
            self.cap.release()
            self.holistic_capture.close()
            self.head_pose_capture.close()
            self.face_socket.close()
            if self.show_debug:
                cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holistic-model", default="holistic_landmarker.task")
    parser.add_argument("--head-pose-model", default="face_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=None,
                         help="force a capture width; default picks the highest mode the camera accepts")
    parser.add_argument("--height", type=int, default=None,
                         help="force a capture height; use together with --width")
    parser.add_argument("--preview-scale", type=float, default=0.5,
                         help="initial debug window size as a fraction of the capture "
                              "resolution. The window is resizable regardless.")
    parser.add_argument("--torso-lean-offset", type=float, default=0.0,
                         help="degrees of forward lean to cancel out. MediaPipe reports a "
                              "vertical performer as leaning ~18 deg forward; press 'c' in the "
                              "debug window while standing upright to measure it, then pass it here.")
    parser.add_argument("--face-ip", default="127.0.0.1")
    parser.add_argument("--face-port", type=int, default=LIVE_LINK_FACE_PORT)
    parser.add_argument("--pose-ip", default="127.0.0.1")
    parser.add_argument("--pose-port", type=int, default=POSE_OSC_PORT)
    parser.add_argument("--face-smoothing", type=float, default=0.5,
                         help="EMA alpha for face, 0-1. Lower = smoother but laggier.")
    parser.add_argument("--pose-smoothing", type=float, default=0.5,
                         help="EMA alpha for pose, 0-1. Lower = smoother but laggier.")
    parser.add_argument("--record", default=None, metavar="FILE.npz",
                         help="dump raw landmarks of this session to FILE.npz on exit, "
                              "for offline solver checks (tests/solver_checks.py)")
    parser.add_argument("--debug", action="store_true",
                         help="show a combined webcam + tracking-status debug window")
    args = parser.parse_args()
 
    for label, path in (("Holistic", args.holistic_model), ("Head pose", args.head_pose_model)):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} model not found at {path}. See mediapipe_holistic_capture.py's "
                "and head_pose_capture.py's module docstrings for download links."
            )

    holistic_capture = MediaPipeHolisticCapture(model_path=args.holistic_model)
    head_pose_capture = HeadPoseCapture(model_path=args.head_pose_model)

    conductor = Conductor(
        holistic_capture, head_pose_capture,
        camera_index=args.camera,
        camera_width=args.width, camera_height=args.height,
        torso_lean_offset_deg=args.torso_lean_offset,
        preview_scale=args.preview_scale,
        face_ip=args.face_ip, face_port=args.face_port,
        pose_ip=args.pose_ip, pose_port=args.pose_port,
        face_smoothing_alpha=args.face_smoothing,
        pose_smoothing_alpha=args.pose_smoothing,
        show_debug=args.debug,
        record_path=args.record,
    )
    conductor.run()
 
 
if __name__ == "__main__":
    main()