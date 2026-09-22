"""
Offline acceptance checks for pose_solver.py - no camera needed.

    .\\venv\\Scripts\\python.exe tests\\solver_checks.py                   # checks 1+2, plus every recordings/*.npz
    .\\venv\\Scripts\\python.exe tests\\solver_checks.py some_capture.npz  # checks 1+2, plus that file

Record a capture with:  .\\venv\\Scripts\\python.exe conductor.py --debug --camera 1 --record recordings\\name.npz

1. Rest-pose identity. The rig's own bind pose, expressed as MediaPipe landmarks,
   must solve back to BIND_POSES (22 body bones) and each finger's bind rotation
   (38 finger bones).
2. Bind-pose FK is upright and symmetric (body and fingers).
3. Absolute direction error on a real capture: forward-kinematic the solver's
   OUTPUT and compare each bone's 3D direction against the performer's, in
   absolute component-space terms - never relative to the spine, which is
   invariant to body roll and would hide a tilt bug.

Ground truth here is deliberately NOT read from the solver's own tables
(BONE_AIM / FINGER_AIM). If the solver mapped a bone to the wrong landmark pair,
checking it against that same pair would pass by construction. The anatomical
correspondence below (which MediaPipe joint sits where on Manny) is defined
independently, so a mis-mapped bone shows up as error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from landmark_recorder import Landmark, load_recording, to_landmarks  # noqa: E402
from mediapipe_pose_capture import PoseLandmark as P  # noqa: E402
from pose_solver import (  # noqa: E402
    BIND_POSES, BONE_NAMES, FINGER_AIM, FINGER_CHAIN, FULL_CHAIN, PTS_TO_COMPONENT, PoseSolver,
    HandLandmark as H,
)

TOL_DEG = 1e-3

# ---------------------------------------------------------------------------
# Independent anatomical correspondence: MediaPipe joint -> Manny joint.
# ---------------------------------------------------------------------------

# Body: (bone, from-landmark, to-landmark). Direction of bone = joint(bone) -> joint(child).
BODY_TRUTH: list[tuple[str, P, P]] = [
    ("upperarm_l", P.LEFT_SHOULDER, P.LEFT_ELBOW),
    ("lowerarm_l", P.LEFT_ELBOW, P.LEFT_WRIST),
    ("upperarm_r", P.RIGHT_SHOULDER, P.RIGHT_ELBOW),
    ("lowerarm_r", P.RIGHT_ELBOW, P.RIGHT_WRIST),
    ("thigh_l", P.LEFT_HIP, P.LEFT_KNEE),
    ("calf_l", P.LEFT_KNEE, P.LEFT_ANKLE),
    ("foot_l", P.LEFT_ANKLE, P.LEFT_FOOT_INDEX),
    ("thigh_r", P.RIGHT_HIP, P.RIGHT_KNEE),
    ("calf_r", P.RIGHT_KNEE, P.RIGHT_ANKLE),
    ("foot_r", P.RIGHT_ANKLE, P.RIGHT_FOOT_INDEX),
]
BODY_CHILD = {"upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l", "upperarm_r": "lowerarm_r",
              "lowerarm_r": "hand_r", "thigh_l": "calf_l", "calf_l": "foot_l", "foot_l": "ball_l",
              "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r"}

# Hand: MediaPipe hand landmark -> the Manny bone whose HEAD (start) sits at that joint.
# Manny's thumb_01 is the thumb metacarpal, so it starts at the CMC joint.
# WRIST sits at hand_l/hand_r. Each finger's MCP is where its _01 bone starts.
# TIPs have no bone; they're handled as "end of the last phalanx".
HAND_JOINT_AT: dict[H, str] = {
    H.WRIST: "hand",
    H.THUMB_CMC: "thumb_01", H.THUMB_MCP: "thumb_02", H.THUMB_IP: "thumb_03",
    H.INDEX_FINGER_MCP: "index_01", H.INDEX_FINGER_PIP: "index_02", H.INDEX_FINGER_DIP: "index_03",
    H.MIDDLE_FINGER_MCP: "middle_01", H.MIDDLE_FINGER_PIP: "middle_02", H.MIDDLE_FINGER_DIP: "middle_03",
    H.RING_FINGER_MCP: "ring_01", H.RING_FINGER_PIP: "ring_02", H.RING_FINGER_DIP: "ring_03",
    H.PINKY_MCP: "pinky_01", H.PINKY_PIP: "pinky_02", H.PINKY_DIP: "pinky_03",
}
HAND_TIP_OF: dict[H, str] = {H.THUMB_TIP: "thumb_03", H.INDEX_FINGER_TIP: "index_03",
                             H.MIDDLE_FINGER_TIP: "middle_03", H.RING_FINGER_TIP: "ring_03",
                             H.PINKY_TIP: "pinky_03"}
# Phalanx (bone) -> (from, to) landmark pair along it. Metacarpals are omitted:
# MediaPipe has no landmark at a metacarpal's base (see pose_solver.py).
FINGER_TRUTH: dict[str, tuple[H, H]] = {
    "thumb_01": (H.THUMB_CMC, H.THUMB_MCP),
    "thumb_02": (H.THUMB_MCP, H.THUMB_IP),
    "thumb_03": (H.THUMB_IP, H.THUMB_TIP),
    "index_01": (H.INDEX_FINGER_MCP, H.INDEX_FINGER_PIP),
    "index_02": (H.INDEX_FINGER_PIP, H.INDEX_FINGER_DIP),
    "index_03": (H.INDEX_FINGER_DIP, H.INDEX_FINGER_TIP),
    "middle_01": (H.MIDDLE_FINGER_MCP, H.MIDDLE_FINGER_PIP),
    "middle_02": (H.MIDDLE_FINGER_PIP, H.MIDDLE_FINGER_DIP),
    "middle_03": (H.MIDDLE_FINGER_DIP, H.MIDDLE_FINGER_TIP),
    "ring_01": (H.RING_FINGER_MCP, H.RING_FINGER_PIP),
    "ring_02": (H.RING_FINGER_PIP, H.RING_FINGER_DIP),
    "ring_03": (H.RING_FINGER_DIP, H.RING_FINGER_TIP),
    "pinky_01": (H.PINKY_MCP, H.PINKY_PIP),
    "pinky_02": (H.PINKY_PIP, H.PINKY_DIP),
    "pinky_03": (H.PINKY_DIP, H.PINKY_TIP),
}
FINGER_TIP_LEN_CM = {"thumb_03": 2.5, "index_03": 2.2, "middle_03": 2.3, "ring_03": 2.2, "pinky_03": 2.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def quat_angle_deg(q1: dict, q2: dict) -> float:
    a = R.from_quat([q1["x"], q1["y"], q1["z"], q1["w"]])
    b = R.from_quat([q2["x"], q2["y"], q2["z"], q2["w"]])
    return math.degrees((a.inv() * b).magnitude())


def comp_to_landmark(p: np.ndarray) -> Landmark:
    """Inverse of PoseSolver._convert_landmarks_to_ue_space + PTS_TO_COMPONENT."""
    fwd, right, up = PTS_TO_COMPONENT.T @ p
    return Landmark(x=-right / 100.0, y=-up / 100.0, z=-fwd / 100.0)


def landmarks_to_comp(lms: list) -> np.ndarray:
    """Landmark list -> (n, 3) component-space cm, the solver's own convention."""
    pts = np.array([[-lm.z * 100.0, -lm.x * 100.0, -lm.y * 100.0] for lm in lms])
    return pts @ PTS_TO_COMPONENT.T


def fk_body(solver: PoseSolver, bones: list) -> tuple[dict[str, R], dict[str, np.ndarray]]:
    """Forward kinematics of the solver's 22-bone OUTPUT over the full 25-bone
    chain. Unstreamed bones (spine_03/05, neck_02) sit at bind pose, exactly as
    Unreal holds them."""
    out = {b.name: b for b in bones}
    g_rot: dict[str, R] = {}
    g_pos: dict[str, np.ndarray] = {}
    for i, (name, parent, _rot, off) in enumerate(FULL_CHAIN):
        if name in out:
            q = out[name].rotation
            local = R.from_quat([q["x"], q["y"], q["z"], q["w"]])
        else:
            local = solver.full_bind[i]
        if parent == -1:
            p = out[name].position
            g_rot[name] = local
            g_pos[name] = np.array([p["x"], p["y"], p["z"]])
        else:
            pname = FULL_CHAIN[parent][0]
            g_rot[name] = g_rot[pname] * local
            g_pos[name] = g_pos[pname] + g_rot[pname].apply(np.asarray(off, dtype=float))
    return g_rot, g_pos


def fk_hand(solver: PoseSolver, side: str, hand_rot: R, hand_pos: np.ndarray,
            finger_bones: list) -> tuple[dict[str, R], dict[str, np.ndarray]]:
    out = {b.name: b for b in finger_bones}
    g_rot = {f"hand{side}": hand_rot}
    g_pos = {f"hand{side}": hand_pos}
    for name, parent, _rot, off in FINGER_CHAIN:
        if not name.endswith(side):
            continue
        q = out[name].rotation
        g_rot[name] = g_rot[parent] * R.from_quat([q["x"], q["y"], q["z"], q["w"]])
        g_pos[name] = g_pos[parent] + g_rot[parent].apply(np.asarray(off, dtype=float))
    return g_rot, g_pos


def bone_axis(name: str) -> np.ndarray:
    # Manny's left-side bones point down +X, right-side down -X (see BIND_POSITIONS).
    return np.array([-1.0, 0.0, 0.0]) if name.endswith("_r") else np.array([1.0, 0.0, 0.0])


def rest_palm_normal_local(solver: PoseSolver, side: str) -> np.ndarray:
    """Palm normal of the rig's rest hand, expressed in hand-local space. Built with
    the same formula as the measured one, so L/R handedness cancels out."""
    w = solver.finger_rest_pos[f"hand{side}"]
    n = np.cross(solver.finger_rest_pos[f"index_01{side}"] - w, solver.finger_rest_pos[f"pinky_01{side}"] - w)
    return solver.finger_rest_global[f"hand{side}"].inv().apply(n / np.linalg.norm(n))


# ---------------------------------------------------------------------------
# Synthetic bind pose as MediaPipe landmarks
# ---------------------------------------------------------------------------

def synth_pose(solver: PoseSolver, anatomical: bool = False) -> list[Landmark]:
    """anatomical=False puts the pose INDEX/PINKY/THUMB points on the hand bone's
    own axis (what the solver's aim assumes); True puts them at the rig's real
    knuckle/thumb positions, to measure how far that assumption is off."""
    pos = {n: solver.rest_pos[i] for i, n in enumerate(solver.full_names)}
    fwd, left = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    head = pos["head"] + up * 8.0
    at: dict[P, np.ndarray] = {
        P.LEFT_SHOULDER: pos["upperarm_l"], P.RIGHT_SHOULDER: pos["upperarm_r"],
        P.LEFT_ELBOW: pos["lowerarm_l"], P.RIGHT_ELBOW: pos["lowerarm_r"],
        P.LEFT_WRIST: pos["hand_l"], P.RIGHT_WRIST: pos["hand_r"],
        P.LEFT_HIP: pos["thigh_l"], P.RIGHT_HIP: pos["thigh_r"],
        P.LEFT_KNEE: pos["calf_l"], P.RIGHT_KNEE: pos["calf_r"],
        P.LEFT_ANKLE: pos["foot_l"], P.RIGHT_ANKLE: pos["foot_r"],
        P.LEFT_FOOT_INDEX: pos["ball_l"], P.RIGHT_FOOT_INDEX: pos["ball_r"],
        P.LEFT_HEEL: pos["foot_l"] - fwd * 5.0, P.RIGHT_HEEL: pos["foot_r"] - fwd * 5.0,
        # Face points: a plausible head, looking straight ahead along +Y.
        P.NOSE: head + fwd * 10.0,
        P.LEFT_EAR: head + left * 7.0, P.RIGHT_EAR: head - left * 7.0,
        P.LEFT_EYE: head + fwd * 8.0 + left * 3.0 + up * 3.0,
        P.RIGHT_EYE: head + fwd * 8.0 - left * 3.0 + up * 3.0,
        P.LEFT_EYE_INNER: head + fwd * 8.5 + left * 1.5 + up * 3.0,
        P.RIGHT_EYE_INNER: head + fwd * 8.5 - left * 1.5 + up * 3.0,
        P.LEFT_EYE_OUTER: head + fwd * 7.5 + left * 4.5 + up * 3.0,
        P.RIGHT_EYE_OUTER: head + fwd * 7.5 - left * 4.5 + up * 3.0,
        P.MOUTH_LEFT: head + fwd * 9.0 + left * 2.5 - up * 5.0,
        P.MOUTH_RIGHT: head + fwd * 9.0 - left * 2.5 - up * 5.0,
    }
    for side in ("l", "r"):
        hp = solver.finger_rest_pos
        index, pinky, thumb = ((P.LEFT_INDEX, P.LEFT_PINKY, P.LEFT_THUMB) if side == "l"
                               else (P.RIGHT_INDEX, P.RIGHT_PINKY, P.RIGHT_THUMB))
        if anatomical:
            at[index], at[pinky], at[thumb] = hp[f"index_01_{side}"], hp[f"pinky_01_{side}"], hp[f"thumb_03_{side}"]
        else:
            axis = solver.rest_global[solver.full_idx[f"hand_{side}"]].apply(bone_axis(f"hand_{side}"))
            at[index] = at[pinky] = at[thumb] = pos[f"hand_{side}"] + axis * 8.0
    return [comp_to_landmark(at[lm]) for lm in P]


SYNTH_FRAME = (1920, 1080)


def synth_face(solver: PoseSolver, turn: R | None = None) -> list[Landmark]:
    """A 478-point face mesh (image-normalised, like Holistic's) whose basis points
    describe a head looking along the rig's rest axes, optionally rotated by `turn`
    (component space). Points the solver doesn't read sit at the head centre."""
    from pose_solver import FACE_CHIN, FACE_EYE_LEFT, FACE_EYE_RIGHT, FACE_LEFT, FACE_RIGHT, FACE_TOP
    turn = turn or R.identity()
    w, h = SYNTH_FRAME
    centre = solver.rest_pos[solver.full_idx["head"]] + np.array([0.0, 3.0, 8.0])
    local = {FACE_LEFT: (7.0, 0.0, 0.0), FACE_RIGHT: (-7.0, 0.0, 0.0),
             FACE_EYE_LEFT: (4.5, 7.0, 3.0), FACE_EYE_RIGHT: (-4.5, 7.0, 3.0),
             FACE_TOP: (0.0, 6.0, 11.0), FACE_CHIN: (0.0, 6.0, -11.0)}

    def to_image(p: np.ndarray) -> Landmark:
        lm = comp_to_landmark(p)  # metres in MediaPipe axes; any uniform scale works
        return Landmark(x=lm.x / w * 100.0, y=lm.y / h * 100.0, z=lm.z / w * 100.0)

    pts = [to_image(centre)] * 478
    for i, off in local.items():
        pts[i] = to_image(centre + turn.apply(np.array(off)))
    return pts


def synth_hand(solver: PoseSolver, side: str) -> list[Landmark]:
    """Landmarks at the rig's real joint positions (anatomical placement), NOT
    placed to satisfy the solver's own aim table."""
    hp = solver.finger_rest_pos
    at: dict[int, np.ndarray] = {}
    for lm, joint in HAND_JOINT_AT.items():
        at[int(lm)] = hp[f"{joint}{side}"]
    for lm, last in HAND_TIP_OF.items():
        bone = f"{last}{side}"
        at[int(lm)] = hp[bone] + solver.finger_rest_global[bone].apply(bone_axis(bone)) * FINGER_TIP_LEN_CM[last]
    return [comp_to_landmark(at[i]) for i in range(21)]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, msg: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        self.failures += 0 if ok else 1


def check_rest_identity(solver: PoseSolver, rep: Report) -> None:
    print("\n1a. Rest-pose identity (rig's own bind pose in, bind pose out)")
    pose = synth_pose(solver)
    hands = {s: synth_hand(solver, s) for s in ("_l", "_r")}
    body = solver.solve(pose, None, None, hands["_l"], hands["_r"])
    errs = [quat_angle_deg(b.rotation, BIND_POSES[i]) for i, b in enumerate(body)]
    bad = [f"{BONE_NAMES[i]} {e:.3f}deg" for i, e in enumerate(errs) if e > TOL_DEG]
    rep.check(not bad, f"body: {22 - len(bad)}/22 bones equal BIND_POSES" + (f"  off: {', '.join(bad)}" if bad else ""))

    left, right = solver.solve_hands(pose, hands["_l"], hands["_r"])
    # Compared as GLOBAL orientation, not parent-local: a metacarpal's known aim
    # approximation (below) lands in its child's local rotation while the child still
    # points exactly where it should, and locals would report that as the child's error.
    g_rot, g_pos = fk_body(solver, body)
    got: dict[str, float] = {}
    for side, fb in (("_l", left), ("_r", right)):
        fr, _ = fk_hand(solver, side, g_rot[f"hand{side}"], g_pos[f"hand{side}"], fb)
        for b in fb:
            got[b.name] = math.degrees((solver.finger_rest_global[b.name].inv() * fr[b.name]).magnitude())
    bad = [f"{n} {e:.3f}deg" for n, e in got.items() if "metacarpal" not in n and e > TOL_DEG]
    n_exact = sum(1 for n in got if "metacarpal" not in n)
    rep.check(not bad, f"fingers: {n_exact - len(bad)}/{n_exact} phalanges + thumb bones at rest orientation"
              + (f"  off: {', '.join(bad)}" if bad else ""))
    # Metacarpals are the one disclosed approximation: MediaPipe has no landmark at
    # a metacarpal's base, so they are aimed WRIST->MCP (see pose_solver.py).
    worst = max(e for n, e in got.items() if "metacarpal" in n)
    rep.check(worst < 15.0, f"fingers: metacarpals within 15deg of rest (max {worst:.2f}deg, "
              "known WRIST->MCP approximation)")

    # Wrist roll: rolling the measured hand about the forearm must come back out of
    # the rig, which a swing-only hand aim could not do at all.
    for deg in (45.0, -60.0):
        rolled = PoseSolver()
        axis = np.array([1.0, 0.0, 0.0])
        roll = R.from_rotvec(axis * math.radians(deg))
        centre = rolled.finger_rest_pos["hand_l"]
        rolled_lms = [comp_to_landmark(centre + roll.apply(landmarks_to_comp([lm])[0] - centre))
                      for lm in hands["_l"]]
        out = rolled.solve(pose, None, None, rolled_lms, None)
        g_rot, _ = fk_body(rolled, out)
        got = g_rot["hand_l"] * rolled.rest_global[rolled.full_idx["hand_l"]].inv()
        err = math.degrees((roll.inv() * got).magnitude())
        rep.check(err < TOL_DEG, f"hand_l roll {deg:+.0f}deg about the forearm reproduced ({err:.4f}deg)")

    face_solver = PoseSolver()
    body = face_solver.solve(pose, synth_face(face_solver), SYNTH_FRAME)
    bad = [f"{BONE_NAMES[i]} {e:.3f}deg" for i, b in enumerate(body)
           if (e := quat_angle_deg(b.rotation, BIND_POSES[i])) > TOL_DEG]
    rep.check(not bad, f"body + straight-ahead face mesh: {22 - len(bad)}/22 bones equal BIND_POSES"
              + (f"  off: {', '.join(bad)}" if bad else ""))

    # A turned head must come out as exactly that head orientation after Unreal-style
    # FK (neck_02 held at bind under neck_01), whatever the neck/head split is.
    turn = R.from_euler("zxy", [30.0, -15.0, 10.0], degrees=True)  # yaw, pitch, roll in component axes
    g_rot, _ = fk_body(face_solver, face_solver.solve(pose, synth_face(face_solver, turn), SYNTH_FRAME))
    hi = face_solver.full_idx["head"]
    err = math.degrees((turn.inv() * g_rot["head"] * face_solver.rest_global[hi].inv()).magnitude())
    rep.check(err < TOL_DEG, f"turned face mesh (yaw 30, pitch -15, roll 10) reproduced by head after FK ({err:.4f}deg)")

    print("\n1b. Rest pose at the rig's REAL joint positions (report - measures the landmark mapping)")
    body = solver.solve(synth_pose(solver, anatomical=True))
    errs = {BONE_NAMES[i]: quat_angle_deg(b.rotation, BIND_POSES[i]) for i, b in enumerate(body)}
    off = {k: v for k, v in errs.items() if v > 0.05}
    print("  body locals off bind: " + (", ".join(f"{k} {v:.2f}" for k, v in off.items()) or "none"))
    left, right = solver.solve_hands(synth_pose(solver), synth_hand(solver, "_l"), synth_hand(solver, "_r"))
    g_rot, g_pos = fk_body(solver, solver.solve(synth_pose(solver)))
    for side, fb in (("_l", left), ("_r", right)):
        fr, _ = fk_hand(solver, side, g_rot[f"hand{side}"], g_pos[f"hand{side}"], fb)
        # Global orientation error per bone - locals would smear a parent's error into its child.
        line = ", ".join(f"{n[:-2]} {math.degrees((solver.finger_rest_global[n].inv() * fr[n]).magnitude()):.1f}"
                         for n, *_ in FINGER_CHAIN if n.endswith(side))
        print(f"  finger globals off rest{side} (deg): {line}")


def check_bind_fk(solver: PoseSolver, rep: Report) -> None:
    print("\n2. Bind-pose FK upright and symmetric")
    pos = {n: solver.rest_pos[i] for i, n in enumerate(solver.full_names)}
    z = lambda n: pos[n][2]  # noqa: E731
    rep.check(z("head") > z("foot_l") and z("head") > z("foot_r"), "head above feet")
    rep.check(z("thigh_l") > z("calf_l") > z("foot_l") and z("thigh_r") > z("calf_r") > z("foot_r"),
              "thigh > calf > foot descending, both legs")
    mirror = np.array([-1.0, 1.0, 1.0])
    sym = max(np.linalg.norm(pos[f"{b}_l"] * mirror - pos[f"{b}_r"])
              for b in ("thigh", "calf", "foot", "ball", "upperarm", "lowerarm", "hand"))
    rep.check(sym < 0.05, f"body L/R mirror-symmetric (max {sym:.3f} cm)")
    rep.check(np.allclose(pos["foot_l"], (14.09, -0.99, 8.24), atol=0.01),
              f"foot_l at {np.round(pos['foot_l'], 2).tolist()} == ik_foot_l (14.09, -0.99, 8.24)")
    fsym = max(np.linalg.norm(solver.finger_rest_pos[n] * mirror - solver.finger_rest_pos[n[:-1] + "r"])
               for n, *_ in FINGER_CHAIN if n.endswith("_l"))
    rep.check(fsym < 0.05, f"fingers L/R mirror-symmetric (max {fsym:.3f} cm)")
    below = all(solver.finger_rest_pos[n][2] < z("upperarm_l") for n, *_ in FINGER_CHAIN)
    rep.check(below, "all finger joints below shoulder height (A-pose hands hang)")


def check_calibration(rep: Report) -> None:
    print("\n2b. Timed calibration (synthetic clock, no camera)")
    solver = PoseSolver()
    pose, face = synth_pose(solver), synth_face(solver)
    solver.begin_calibration(countdown_s=3.0, samples=30, now_s=0.0)
    phases: list[str] = []
    for i in range(200):  # ~6.6 s at 30 fps
        phases.append(solver.update_calibration(pose, face, SYNTH_FRAME, now_s=i / 30.0).phase)
    counts = {p: phases.count(p) for p in dict.fromkeys(phases)}
    rep.check(list(counts) == ["countdown", "sampling", "done", "idle"],
              f"phases run countdown -> sampling -> done -> idle {counts}")
    rep.check(counts["countdown"] >= 89, f"countdown lasts ~3 s ({counts['countdown']} frames at 30 fps)")
    rep.check(counts["sampling"] + 1 == 30, f"averaged over 30 frames ({counts['sampling'] + 1})")
    # The rig's own rest pose is upright and looking straight ahead, so a calibration
    # taken on it must be a no-op - anything else means the calibration injects a bias.
    rep.check(abs(solver.torso_lean_offset_deg) < 1e-6,
              f"lean offset on the rig's own rest pose is 0 ({solver.torso_lean_offset_deg:+.4f} deg)")
    rep.check(math.degrees(solver.head_neutral.magnitude()) < 1e-3,
              f"head neutral on a straight-ahead face is identity "
              f"({math.degrees(solver.head_neutral.magnitude()):.4f} deg)")
    body = solver.solve(pose, face, SYNTH_FRAME)
    bad = [BONE_NAMES[i] for i, b in enumerate(body) if quat_angle_deg(b.rotation, BIND_POSES[i]) > TOL_DEG]
    rep.check(not bad, f"rest pose still solves to bind after calibrating ({22 - len(bad)}/22)")


def _stats(errs: list[float]) -> str:
    a = np.array([e for e in errs if not math.isnan(e)])
    if not a.size:
        return "      n/a"
    return f"{math.sqrt(float(np.mean(a ** 2))):6.2f} {float(np.max(a)):7.2f} {a.size:6d}"


def check_capture(path: Path) -> dict[str, list[float]]:
    print(f"\n3. Absolute direction error - {path.name}")
    # Fresh solver per recording: the head pose is held across face dropouts, and
    # must not leak from one recording into the next.
    solver = PoseSolver()  # torso_lean_offset 0: the torso row then measures the raw solve
    rec = load_recording(path)
    frame_size = tuple(int(v) for v in rec["frame_size"])
    # recordings/trims.json cuts the walk to/from the laptop at either end.
    trims_file = path.parent / "trims.json"
    trim = json.loads(trims_file.read_text()).get(path.name) if trims_file.exists() else None
    if trim:
        keep = (rec["t_ms"] >= trim["start_ms"]) & (rec["t_ms"] <= trim["end_ms"])
        rec = {k: (v[keep] if v.ndim and len(v) == len(keep) else v) for k, v in rec.items()}
        print(f"  trimmed to {trim['start_ms']}-{trim['end_ms']} ms (recordings/trims.json)")
    errs: dict[str, list[float]] = {}
    add = lambda k, v: errs.setdefault(k, []).append(v)  # noqa: E731
    palm_local = {s: rest_palm_normal_local(solver, s) for s in ("_l", "_r")}
    fwd_local = solver.rest_global[solver.full_idx["head"]].inv().apply(np.array([0.0, 1.0, 0.0]))

    n = len(rec["t_ms"])
    for f in range(n):
        if not rec["pose_valid"][f]:
            continue
        pose = to_landmarks(rec["pose_world"][f])
        face = to_landmarks(rec["face_image"][f]) if rec["face_valid"][f] else None
        lh = to_landmarks(rec["lhand_world"][f]) if rec["lhand_valid"][f] else None
        rh = to_landmarks(rec["rhand_world"][f]) if rec["rhand_valid"][f] else None
        body = solver.solve(pose, face, frame_size, lh, rh)
        g_rot, g_pos = fk_body(solver, body)
        m = landmarks_to_comp(pose)

        for bone, a, b in BODY_TRUTH:
            add(bone, angle_deg(g_pos[BODY_CHILD[bone]] - g_pos[bone], m[b] - m[a]))
        for side, wrist, index in (("_l", P.LEFT_WRIST, P.LEFT_INDEX), ("_r", P.RIGHT_WRIST, P.RIGHT_INDEX)):
            add(f"hand{side}", angle_deg(g_rot[f"hand{side}"].apply(bone_axis(f"hand{side}")), m[index] - m[wrist]))
        sh_rig = (g_pos["upperarm_l"] + g_pos["upperarm_r"]) * 0.5
        hip_rig = (g_pos["thigh_l"] + g_pos["thigh_r"]) * 0.5
        sh_m = (m[P.LEFT_SHOULDER] + m[P.RIGHT_SHOULDER]) * 0.5
        hip_m = (m[P.LEFT_HIP] + m[P.RIGHT_HIP]) * 0.5
        add("torso", angle_deg(sh_rig - hip_rig, sh_m - hip_m))
        add("clavicle_l", angle_deg(g_pos["upperarm_l"] - sh_rig, m[P.LEFT_SHOULDER] - sh_m))
        add("clavicle_r", angle_deg(g_pos["upperarm_r"] - sh_rig, m[P.RIGHT_SHOULDER] - sh_m))

        # Head yaw only: the ear->nose vector has an unknown constant pitch offset
        # against the rig's head axis, but its heading is unambiguous.
        head_fwd = g_rot["head"].apply(fwd_local)
        face_fwd = m[P.NOSE] - (m[P.LEFT_EAR] + m[P.RIGHT_EAR]) * 0.5
        flat = np.array([1.0, 1.0, 0.0])
        add("head_yaw", angle_deg(head_fwd * flat, face_fwd * flat))
        head_delta = g_rot["head"] * solver.rest_global[solver.full_idx["head"]].inv()
        if face:
            # Plumbing: after Unreal-style FK (neck_02 at bind under neck_01) the head
            # must point exactly where the face mesh says. Same source, so this checks
            # the neck split / parenting, not the mesh's accuracy.
            mesh = solver._face_mesh_rotation(face, frame_size)
            add("head_vs_mesh", math.degrees((mesh.inv() * head_delta).magnitude()))
        # How far the head turns relative to the torso - a range, not an error.
        torso_delta = g_rot["spine_04"] * solver.rest_global[solver.full_idx["spine_04"]].inv()
        rel_fwd = (torso_delta.inv() * head_delta).apply(np.array([0.0, 1.0, 0.0]))
        add("head_rel_yaw", math.degrees(math.atan2(rel_fwd[0], rel_fwd[1])))
        add("head_rel_pitch", math.degrees(math.asin(float(np.clip(rel_fwd[2], -1.0, 1.0)))))

        hands = {"_l": lh, "_r": rh}
        left, right = solver.solve_hands(pose, lh or [], rh or [])
        for side, fb in (("_l", left), ("_r", right)):
            if hands[side] is None:
                continue
            hm = landmarks_to_comp(hands[side])
            fr, fp = fk_hand(solver, side, g_rot[f"hand{side}"], g_pos[f"hand{side}"], fb)
            for bone, (a, b) in FINGER_TRUTH.items():
                name = f"{bone}{side}"
                child = {"_01": "_02", "_02": "_03"}.get(bone[-3:])
                rig_dir = (fp[name[:-5] + child + side] - fp[name]) if child else fr[name].apply(bone_axis(name))
                add(f"finger:{bone}", angle_deg(rig_dir, hm[int(b)] - hm[int(a)]))
            n_meas = np.cross(hm[int(H.INDEX_FINGER_MCP)] - hm[int(H.WRIST)], hm[int(H.PINKY_MCP)] - hm[int(H.WRIST)])
            add(f"palm_normal{side}", angle_deg(g_rot[f"hand{side}"].apply(palm_local[side]), n_meas))
            # How far the two MODELS disagree about the same joint. The hand_l/hand_r
            # rows above are measured against the pose model's INDEX point, so they
            # can never be smaller than this - it is the floor on that comparison,
            # not solver error.
            pose_index = P.LEFT_INDEX if side == "_l" else P.RIGHT_INDEX
            pose_wrist = P.LEFT_WRIST if side == "_l" else P.RIGHT_WRIST
            add("pose_vs_hand_model", angle_deg(m[pose_index] - m[pose_wrist],
                                                hm[int(H.INDEX_FINGER_MCP)] - hm[int(H.WRIST)]))

    body_keys = [b for b, *_ in BODY_TRUTH] + ["hand_l", "hand_r", "clavicle_l", "clavicle_r", "torso"]
    print(f"  {sum(rec['pose_valid'])}/{n} frames with pose, "
          f"{sum(rec['lhand_valid'])} L-hand, {sum(rec['rhand_valid'])} R-hand")
    print(f"  {'bone':<22}{'RMS':>6} {'max':>7} {'frames':>6}")
    for k in body_keys:
        print(f"  {k:<22}{_stats(errs.get(k, []))}")
    body_all = [e for k in body_keys for e in errs.get(k, [])]
    print(f"  {'BODY overall':<22}{_stats(body_all)}")
    print(f"  {'(pose vs hand model)':<22}{_stats(errs.get('pose_vs_hand_model', []))}   "
          f"floor on the hand_l/hand_r rows")
    print(f"  {'head_yaw vs pose pts':<22}{_stats(errs.get('head_yaw', []))}   (pose ear->nose; under-reads turns)")
    print(f"  {'head_vs_mesh':<22}{_stats(errs.get('head_vs_mesh', []))}")
    for k in ("head_rel_yaw", "head_rel_pitch"):
        a = np.array(errs[k])
        print(f"  {k:<22}p5 {np.percentile(a, 5):6.1f}  p95 {np.percentile(a, 95):6.1f}  (range vs torso)")
    for k in sorted(k for k in errs if k.startswith("finger:")):
        print(f"  {k:<22}{_stats(errs[k])}")
    fing_all = [e for k in errs if k.startswith("finger:") for e in errs[k]]
    print(f"  {'FINGERS overall':<22}{_stats(fing_all)}")
    for k in ("palm_normal_l", "palm_normal_r"):
        print(f"  {k:<22}{_stats(errs.get(k, []))}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", type=Path)
    args = ap.parse_args()

    solver = PoseSolver()
    rep = Report()
    check_rest_identity(solver, rep)
    check_bind_fk(solver, rep)
    check_calibration(rep)

    recs = args.recordings or sorted((ROOT / "recordings").glob("*.npz"))
    if not recs:
        print("\n3. Absolute direction error - SKIPPED (no recordings; pass a .npz or add recordings/*.npz)")
    for path in recs:
        check_capture(path)

    print(f"\n{'ALL PASS' if not rep.failures else f'{rep.failures} FAILURE(S)'} (checks 1-2; check 3 is a report)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
