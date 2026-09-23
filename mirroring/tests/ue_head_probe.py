"""
Unreal head-rotation probe - streams KNOWN inputs to a running editor so the
MetaHuman's response can be read off, instead of reasoned about.

    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py                    # every stage in turn
    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py --stage q2-torso-yaw
    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py --value 1.0         # bigger curve input

Stop conductor.py first: both would be sending to the same ports.

Why this exists: the head reaches a MetaHuman by two routes - the Body stream's
head bone (OSC 9001) and Live Link Face's headYaw/Pitch/Roll curves (UDP 11111),
which something downstream of ABP_Face's layered blend (RigLogic, or the Head
Movement IK Control Rig) turns into head rotation. Which frame that curve path
works in, its units and its signs decide how pose_solver's head basis is converted
for the face channel. See CLAUDE.md, "Head rotation for the face channel".

Body inputs are built by PoseSolver itself from synthetic landmarks, so the Body
stream here is exactly what conductor.py would send for that pose. Each stage
prints the solver's own FK of it - that is what the Body stream asks for; the
viewport shows what the MetaHuman does with it.

Angles: yaw is about Unreal's up axis, positive = towards the CHARACTER's own
right (component space: X = character's left, Y = forward, Z = up).
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from solver_checks import SYNTH_FRAME, comp_to_landmark, fk_body, landmarks_to_comp, synth_face, synth_pose

from conductor import LIVE_LINK_FACE_PORT, POSE_OSC_PORT
from live_link_face_protocol import LiveLinkFaceEncoder
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder
from mediapipe_pose_capture import PoseLandmark as P
from pose_solver import BoneTransform, PoseSolver


@dataclass
class Stage:
    name: str
    torso_yaw_deg: float           # whole body turned, head with it
    head_yaw_deg: float            # body-stream head, relative to the torso
    curves: dict[str, float] = field(default_factory=dict)  # sign-scaled by --value
    look_for: str = ""


def stages() -> list[Stage]:
    out = [
        Stage("rest", 0.0, 0.0, {},
              "Reference. Character square on, head straight. Note where 'straight' is."),
        Stage("q1-body-head", 0.0, 30.0, {},
              "Q1. Body stream turns the head 30 deg to the character's right; face head "
              "curves are 0. Head turned -> the Body path reaches the head. Head straight "
              "-> the curve path overwrites it and is the only way in."),
        Stage("q2-torso", 45.0, 0.0, {},
              "Q2 baseline. Whole body turned 45 deg right, head with it; curves 0. Note "
              "where the head points - q2-torso-yaw is read against this."),
        Stage("q2-torso-yaw", 45.0, 0.0, {"headYaw": 1.0},
              "Q2. Same body, plus headYaw = +value. Head at 45 + x -> the curve rotation "
              "is relative to the neck chain (option B). Head at x alone, ignoring the "
              "45 -> it is absolute (option A)."),
    ]
    for axis in ("Yaw", "Pitch", "Roll"):
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            out.append(Stage(
                f"{axis.lower()}{tag}", 0.0, 0.0, {f"head{axis}": sign},
                f"Q3. Body at rest, head{axis} = {tag}value only. Read the angle (~29 deg "
                f"at 0.5 -> radians, 45 deg -> deg/90) and its direction. Also watch "
                f"neck_01/neck_02 (console: 'show bones'): if they move, the curve path "
                f"already splits into the neck the way NECK_SHARE does."))
    return out


def _turn_about_hips(landmarks: list, yaw_deg: float) -> list:
    """Rotate a whole landmark set about the vertical through the hip midpoint -
    a performer turning on the spot."""
    if abs(yaw_deg) < 1e-9:
        return landmarks
    pts = landmarks_to_comp(landmarks)
    hip_mid = (pts[int(P.LEFT_HIP)] + pts[int(P.RIGHT_HIP)]) * 0.5
    turn = R.from_euler("z", yaw_deg, degrees=True)
    return [comp_to_landmark(hip_mid + turn.apply(p - hip_mid)) for p in pts]


def _yaw_of(rot: R) -> float:
    """Heading of a rotation's forward (+Y) axis, degrees towards the character's right."""
    fwd = rot.apply(np.array([0.0, 1.0, 0.0]))
    return math.degrees(math.atan2(-fwd[0], fwd[1]))


def build_body(stage: Stage) -> tuple[list[BoneTransform], str]:
    """The Body stream for a stage, plus a one-line FK readout of what it asks for."""
    solver = PoseSolver()
    pose = _turn_about_hips(synth_pose(solver), stage.torso_yaw_deg)
    face = synth_face(solver, R.from_euler("z", stage.torso_yaw_deg + stage.head_yaw_deg, degrees=True))
    body = solver.solve(pose, face, SYNTH_FRAME)
    left, right = solver.solve_hands(pose, [], [])

    g_rot, _ = fk_body(solver, body)
    fi = solver.full_idx
    delta = {n: g_rot[n] * solver.rest_global[fi[n]].inv() for n in ("spine_04", "neck_01", "neck_02", "head")}
    readout = "  ".join(f"{n} {_yaw_of(d):+6.1f}" for n, d in delta.items())
    return body + left + right, f"body stream asks for (yaw off bind, deg): {readout}"


def main() -> int:
    names = [s.name for s in stages()]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all"] + names)
    ap.add_argument("--value", type=float, default=0.5, help="magnitude of the face head curve input")
    ap.add_argument("--seconds", type=float, default=10.0, help="per stage when cycling; forever with --stage")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--rate", type=float, default=60.0)
    args = ap.parse_args()

    run = stages() if args.stage == "all" else [s for s in stages() if s.name == args.stage]
    pose_encoder = LiveLinkPoseOSCEncoder(ip=args.ip, port=POSE_OSC_PORT)
    # Same subject name as conductor.py's encoder, so the MetaHuman's existing Live
    # Link Face binding picks this up with nothing changed in the editor.
    face_encoder = LiveLinkFaceEncoder()
    face_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    face_target = (args.ip, LIVE_LINK_FACE_PORT)
    period = 1.0 / args.rate

    try:
        for stage in run:
            bones, readout = build_body(stage)
            curves = {k: v * args.value for k, v in stage.curves.items()}
            print(f"\n=== {stage.name} ===")
            print(f"  face curves: {curves or 'all 0'}")
            print(f"  {readout}")
            print(f"  look for: {stage.look_for}", flush=True)

            until = time.monotonic() + (args.seconds if args.stage == "all" else math.inf)
            while time.monotonic() < until:
                pose_encoder.send(bones, present=True)
                # Re-encoded each send only for the timecode; the values are constant.
                face_socket.sendto(face_encoder.encode(curves), face_target)
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        face_socket.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
