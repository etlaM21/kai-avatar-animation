"""
Unreal head-rotation probe - streams KNOWN inputs to a running editor so the
MetaHuman's response can be read off, instead of reasoned about.

    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py                    # every stage in turn
    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py --stage q2-torso-yaw
    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py --value 1.0         # bigger curve input
    .\\venv\\Scripts\\python.exe tests\\ue_head_probe.py --measure           # read bones back, numerically

Stop conductor.py first: both would be sending to the same ports. --measure needs
Project Settings > Python > Enable Remote Execution in the editor.

Measured 2026-09-23, UE 5.8, BP_NewMetaHumanCharacter:
  - The Body stream's neck_02/head never reach the MetaHuman's head; neck_01 does.
  - The curve path is ABSOLUTE in component space: a 45 deg torso turn leaves the
    head exactly where the curves put it. It rotates neck_02 and head together.
  - 50 deg per unit on every axis, linear to at least 75 deg. headYaw + turns the
    head to the character's LEFT, headPitch + looks up, headRoll + tips the top of
    the head to the character's right.
  - Composition: pitch * roll * yaw (yaw applied first), fitted to 0.000 deg.

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
import json
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    out.append(Stage(
        "combo", 0.0, 0.0, {"headYaw": 1.0, "headPitch": 0.6, "headRoll": -0.4},
        "All three curves at once, unequal. Single-axis stages can't show the order the "
        "curve path composes yaw/pitch/roll in; this one can, read back numerically."))
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


class Streamer(threading.Thread):
    """Sends the current stage's Body + Face packets at a fixed rate until stopped.
    A thread, so --measure can query the editor while the stream keeps flowing."""

    def __init__(self, ip: str, rate: float) -> None:
        super().__init__(daemon=True)
        self.pose_encoder = LiveLinkPoseOSCEncoder(ip=ip, port=POSE_OSC_PORT)
        # Same subject name as conductor.py's encoder, so the MetaHuman's existing Live
        # Link Face binding picks this up with nothing changed in the editor.
        self.face_encoder = LiveLinkFaceEncoder()
        self.face_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.face_target = (ip, LIVE_LINK_FACE_PORT)
        self.period = 1.0 / rate
        self.bones: list[BoneTransform] = []
        self.curves: dict[str, float] = {}
        self.running = True

    def run(self) -> None:
        while self.running:
            if self.bones:
                self.pose_encoder.send(self.bones, present=True)
                # Re-encoded each send only for the timecode; the values are constant.
                self.face_socket.sendto(self.face_encoder.encode(self.curves), self.face_target)
            time.sleep(self.period)
        self.face_socket.close()


# ---------------------------------------------------------------------------
# --measure: read the result back out of the editor instead of eyeballing it.
# Needs Project Settings > Python > Enable Remote Execution.
# ---------------------------------------------------------------------------

MEASURE_BONES = ("pelvis", "spine_05", "neck_01", "neck_02", "head")
MEASURE_COMPS = ("MannySource", "Body", "Face")
_QUERY = """
import unreal, json
out = {}
for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    comps = {c.get_name(): c for c in a.get_components_by_class(unreal.SkeletalMeshComponent)}
    if not all(n in comps for n in %r):
        continue
    q = comps['MannySource'].get_world_transform().rotation
    out['frame'] = [q.x, q.y, q.z, q.w]
    for n in %r:
        c = comps[n]
        out[n] = {}
        for b in %r:
            if c.get_bone_index(b) != -1:
                q = c.get_socket_transform(b, unreal.RelativeTransformSpace.RTS_WORLD).rotation
                out[n][b] = [q.x, q.y, q.z, q.w]
    out['actor'] = a.get_actor_label()
    break
print(json.dumps(out))
""" % (MEASURE_COMPS, MEASURE_COMPS, MEASURE_BONES)


class EditorReader:
    """World rotations of the MetaHuman's bones, via the editor's Python remote
    execution. Finds the first actor carrying MannySource, Body and Face components."""

    def __init__(self, ue_root: str) -> None:
        sys.path.insert(0, str(Path(ue_root) / "Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python"))
        import remote_execution  # shipped with the engine, not pip-installable
        self._rex = remote_execution
        self._r = remote_execution.RemoteExecution()
        self._r.start()
        deadline = time.monotonic() + 5.0
        while not self._r.remote_nodes and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self._r.remote_nodes:
            raise RuntimeError("no editor answered - is Python remote execution enabled?")
        self._r.open_command_connection(self._r.remote_nodes[0]["node_id"])

    def read(self) -> dict:
        res = self._r.run_command(_QUERY, unattended=True, exec_mode=self._rex.MODE_EXEC_FILE)
        if not res["success"]:
            raise RuntimeError(res["result"])
        out = json.loads("".join(o["output"] for o in res["output"]).strip().splitlines()[-1])
        if "frame" not in out:
            raise RuntimeError("no actor with MannySource, Body and Face components in the level")
        return out

    def settled(self, min_s: float = 1.5, timeout_s: float = 30.0) -> dict:
        """Read once nothing has moved for four consecutive polls. min_s covers the
        time for the new stage's packets to arrive: settling on the OLD pose first is
        otherwise indistinguishable from settling on the new one."""
        time.sleep(min_s)
        t0, calm, prev = time.monotonic(), 0, self.read()
        while calm < 4 and time.monotonic() - t0 < timeout_s:
            time.sleep(0.25)
            cur = self.read()
            step = max(math.degrees((R.from_quat(prev[c][b]).inv() * R.from_quat(cur[c][b])).magnitude())
                       for c in MEASURE_COMPS for b in cur[c])
            calm, prev = (calm + 1 if step < 0.01 else 0), cur
        return prev

    def close(self) -> None:
        self._r.stop()


def print_deltas(stage: dict, rest: dict) -> None:
    """Each bone's world rotation relative to the rest stage, as a rotation vector
    in Manny's component axes: x about the character's left (+ = look up), y about
    forward (+ = top of the head to the character's left), z about up (+ = turn to
    the character's right)."""
    frame = R.from_quat(rest["frame"])
    print(f"  {'':12}" + "".join(f"{b:>25}" for b in MEASURE_BONES))
    for comp in MEASURE_COMPS:
        row = f"  {comp:12}"
        for b in MEASURE_BONES:
            if b not in stage[comp]:
                row += f"{'-':>25}"
                continue
            d = R.from_quat(stage[comp][b]) * R.from_quat(rest[comp][b]).inv()
            x, y, z = np.degrees((frame.inv() * d * frame).as_rotvec())
            row += f"  x{x:+6.1f} y{y:+6.1f} z{z:+6.1f}"
        print(row)


def main() -> int:
    names = [s.name for s in stages()]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all"] + names)
    ap.add_argument("--value", type=float, default=0.5, help="magnitude of the face head curve input")
    ap.add_argument("--seconds", type=float, default=10.0, help="per stage when cycling; forever with --stage")
    ap.add_argument("--measure", action="store_true",
                    help="read each stage's bone rotations back from the editor (needs Python remote "
                         "execution) and print them against 'rest', instead of timed viewing")
    ap.add_argument("--ue-root", default=r"C:\Program Files\Epic Games\UE_5.8")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--rate", type=float, default=60.0)
    args = ap.parse_args()

    run = stages() if args.stage == "all" else [s for s in stages() if s.name == args.stage]
    if args.measure and run[0].name != "rest":
        run = [s for s in stages() if s.name == "rest"] + run  # the reference every delta is against
    reader = EditorReader(args.ue_root) if args.measure else None
    streamer = Streamer(args.ip, args.rate)
    streamer.start()

    rest: dict | None = None
    try:
        for stage in run:
            bones, readout = build_body(stage)
            curves = {k: v * args.value for k, v in stage.curves.items()}
            streamer.bones, streamer.curves = bones, curves
            print(f"\n=== {stage.name} ===")
            print(f"  face curves: {curves or 'all 0'}")
            print(f"  {readout}")
            if reader is None:
                print(f"  look for: {stage.look_for}", flush=True)
                time.sleep(args.seconds if args.stage == "all" else math.inf)
                continue
            result = reader.settled()
            rest = rest or result
            print(f"  measured in '{result['actor']}' (deg, relative to rest):")
            print_deltas(result, rest)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.running = False
        streamer.join()
        if reader is not None:
            reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
