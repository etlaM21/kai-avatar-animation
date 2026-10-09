"""
Offline acceptance checks for the procedural retarget - no GPU, no Kimodo, no engine.

    .\\venv\\Scripts\\python.exe procedural_animation\\tests\\procedural_checks.py
    .\\venv\\Scripts\\python.exe procedural_animation\\tests\\procedural_checks.py clip.bvh ...

procedural-animation.md sec. 5, tests 1-3:

1. Rest pose. SOMA77's own T-pose in: every Manny bone points where the source
   bone points, the pelvis lands on Manny's rest pelvis, and the roll the swing
   alignment leaves (hip line, shoulder line, palm normals) is measured.
2. Direction truth on real clips. Every frame of every Kimodo clip: Manny's FK bone
   directions vs the source's own FK, in absolute component-space terms (never
   relative to the spine). Native-convention clips are converted with Kimodo's O_j
   table; the truth stays the ORIGINAL file's FK, and the conversion must leave
   every joint position unchanged.
3. Change-of-basis sanity on synthetic poses with known answers: walking forward
   (+Z) moves Manny along +Y facing +Y; a 90 deg turn to the character's left
   faces Manny +X (a transposed/inverted rotation faces -X); raising the source's
   left arm raises upperarm_l and leaves the right arm alone.
4. Head channel. The Live Link Face curves the conductor sends, decoded the way
   the MetaHuman applies them (float32, 50 deg per unit, pitch * roll * yaw), give
   back the head that the sent body pose puts on Manny - every frame of every clip.
5. Wire format. What LiveLinkPoseOSCEncoder actually emits for a sampled pose:
   address /mediapipe/pose, 421 floats, bones in conductor.py's order.
5b. Player (remote_kimodo_service.player), real time, packets captured: a lone clip
   loops, a queued clip takes over at the end of the current pass, every packet is
   421 floats, present=1 while playing and 0 on stop, ~60 Hz.

Remote Kimodo lane (CLAUDE.md, procedural tests 6-7), no Spark needed:
6. Client resilience against a closed port and stub servers: refused, accept-and-close
   (the SSH tunnel with no service behind it), no answer, HTTP 500, a malformed NPZ.
   Each gives a one-line error, caches nothing, and the player keeps sending. A cache
   hit never touches the network. Inline prompt options parse.
7. NPZ contract: a clip built from a BVH (either convention) retargets identically via
   the NPZ and via the BVH; exactly the documented keys/shapes/dtypes, in metres; and
   transposed rotations, cm instead of m, a different joint order, NaN, a missing key
   and truncated bytes are all refused.

Ground truth is deliberately independent of retarget.py: Manny positions come
from an Unreal-style FK written here over the streamed BoneTransforms (rotators
and offsets straight from pose_solver's rig dump, unstreamed bones held at bind),
and the source side from bvh_reader FK. Correspondence is by anatomical JOINT
POSITION (ANATOMY below), not by the retarget's own bone->joint table, so a
mis-wired rotation shows up as a direction error.
"""

from __future__ import annotations

import argparse
import io
import math
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

MODULE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE_ROOT))

from procedural_animation import MIRRORING_DIR  # noqa: E402,F401
from procedural_animation.bvh_reader import BvhClip, read_bvh  # noqa: E402
from procedural_animation.retarget import STREAMED_BONES, Retargeter, RetargetedMotion  # noqa: E402
from procedural_animation.source_skeletons import (  # noqa: E402
    SOMA77_TPOSE_BVH, load_soma77, to_standard_convention,
)
from pose_solver import BONE_NAMES, FINGER_BONE_NAMES, FINGER_CHAIN, FULL_CHAIN, ue_rotator_to_dict  # noqa: E402
from live_link_face_protocol import HEAD_DEG_PER_UNIT, head_rotation_to_curves  # noqa: E402
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder  # noqa: E402
from procedural_animation.procedural_conductor import Lane, parse_prompt  # noqa: E402
from remote_kimodo_service import kimodo_contract as contract  # noqa: E402
from remote_kimodo_service.clip_cache import ClipCache  # noqa: E402
from remote_kimodo_service.fake_kimodo_server import FakeKimodo, clip_from_bvh  # noqa: E402
from remote_kimodo_service.kimodo_adapter import retarget as retarget_npz  # noqa: E402
from remote_kimodo_service.kimodo_client import GenerationFailed, KimodoClient, SparkUnavailable  # noqa: E402
from remote_kimodo_service.kimodo_contract import ContractError, GenerationRequest  # noqa: E402
from remote_kimodo_service.player import LoopPlayer, Sender  # noqa: E402

CLIP_DIRS = [MODULE_ROOT / "assets" / "kimodo" / "clips" / "editor-gen",
             MODULE_ROOT / "assets" / "kimodo" / "clips" / "kimodo-gen"]

# Unreal component space from the source's (Y up, +Z fwd): stated independently of
# source_skeletons.Y_UP_Z_FWD_TO_UE. Test 3 checks the absolute facts it implies.
TO_COMP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Which SOMA joint sits where Manny's joint sits, anatomically. Positions, not
# bone drivers: e.g. Manny's hand_l joint is the wrist = SOMA LeftHand, and
# middle_01_l is the middle-finger knuckle = SOMA LeftHandMiddle2.
ANATOMY: dict[str, str] = {
    "neck_01": "Neck1", "head": "Head",
    "clavicle_l": "LeftShoulder", "upperarm_l": "LeftArm", "lowerarm_l": "LeftForeArm", "hand_l": "LeftHand",
    "clavicle_r": "RightShoulder", "upperarm_r": "RightArm", "lowerarm_r": "RightForeArm", "hand_r": "RightHand",
    "thigh_l": "LeftLeg", "calf_l": "LeftShin", "foot_l": "LeftFoot", "ball_l": "LeftToeBase",
    "thigh_r": "RightLeg", "calf_r": "RightShin", "foot_r": "RightFoot", "ball_r": "RightToeBase",
}
for _s, _S in (("_l", "LeftHand"), ("_r", "RightHand")):
    ANATOMY[f"thumb_01{_s}"] = f"{_S}Thumb1"
    ANATOMY[f"thumb_02{_s}"] = f"{_S}Thumb2"
    ANATOMY[f"thumb_03{_s}"] = f"{_S}Thumb3"
    for _f, _F in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Pinky")):
        ANATOMY[f"{_f}_metacarpal{_s}"] = f"{_S}{_F}1"
        ANATOMY[f"{_f}_01{_s}"] = f"{_S}{_F}2"
        ANATOMY[f"{_f}_02{_s}"] = f"{_S}{_F}3"
        ANATOMY[f"{_f}_03{_s}"] = f"{_S}{_F}4"

# Single-bone segments (Manny joint pair, one bone apart): their direction must
# match the source exactly - that is what the retarget guarantees.
LIMB_SEGMENTS = [
    ("clavicle_l", "upperarm_l"), ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
    ("clavicle_r", "upperarm_r"), ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"),
    ("thigh_l", "calf_l"), ("calf_l", "foot_l"), ("foot_l", "ball_l"),
    ("thigh_r", "calf_r"), ("calf_r", "foot_r"), ("foot_r", "ball_r"),
]
FINGER_SEGMENTS = [(p, n) for n, p, _r, _o in FINGER_CHAIN
                   if p in ANATOMY and n in ANATOMY and p not in ("hand_l", "hand_r")]

# Joint pairs that are NOT one bone, so rig geometry enters and they cannot be
# exact. Measured at rest: hand -> finger bases 3.6-16 deg (the rigid palm offsets
# differ between Manny and SOMA), hand -> middle_01 5.4 deg (crosses that offset),
# neck_01 -> head 1.4 deg (through neck_02, held at bind by Unreal). Reported.
GEOMETRY_SEGMENTS = [("neck_01", "head"), ("hand_l", "middle_01_l"), ("hand_r", "middle_01_r")] + [
    (p, n) for n, p, _r, _o in FINGER_CHAIN if p in ("hand_l", "hand_r")]

TOL_DIR_DEG = 1e-3
# The NPZ travels as float32; a real Kimodo clip measured 0.00057 deg worst.
TOL_DIR_DEG_FLOAT32 = 0.01
REAL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kimodo_real_turn_around_2s.npz"
# Head curves travel as float32 (1 unit = 50 deg), so a correct round trip lands
# within ~1e-5 deg; a wrong sign, axis or composition order misses by degrees.
TOL_HEAD_CHANNEL_DEG = 0.01
# Orientation measures (hip line, palm normals) carry a constant rig-geometry offset
# too, but that offset must not CHANGE with the motion: if roll were lost, it would.
# Measured drift from the rest value over all 28 standard clips: <= 0.04 deg.
TOL_ROLL_DRIFT_DEG = 0.5
# Half of SOMA's rest hip-centre -> Neck1 distance (54.5 cm).
TORSO_MIN_CHORD_CM = 27.0


# ---------------------------------------------------------------------------
# Independent Unreal-style FK over what is actually streamed
# ---------------------------------------------------------------------------

def _bind(rot: tuple) -> np.ndarray:
    q = ue_rotator_to_dict(*rot)
    return np.array([q["x"], q["y"], q["z"], q["w"]])


def manny_fk(motion: RetargetedMotion) -> dict[str, np.ndarray]:
    """(T, 3) component-space position of every Manny joint, the way Unreal builds
    it: streamed local rotation where the bone is in the packet, bind rotation
    otherwise (spine_03, spine_05, neck_02), rig offsets, streamed pelvis position."""
    return manny_fk_full(motion)[0]


def manny_fk_full(motion: RetargetedMotion | None) -> tuple[dict[str, np.ndarray], dict[str, R]]:
    """manny_fk plus every joint's (T,) global rotation. None = Manny's bind pose
    (one frame), built from the rig dump alone."""
    T = 1 if motion is None else motion.num_frames
    streamed = {} if motion is None else {name: k for k, name in enumerate(STREAMED_BONES)}
    glob: dict[str, R] = {}
    pos: dict[str, np.ndarray] = {}
    chain = [(n, (FULL_CHAIN[p][0] if p >= 0 else None), rot, off) for n, p, rot, off in FULL_CHAIN]
    chain += [(n, p, rot, off) for n, p, rot, off in FINGER_CHAIN]
    for name, parent, rot, off in chain:
        local = (R.from_quat(motion.local_quats[:, streamed[name]]) if name in streamed
                 else R.from_quat(np.tile(_bind(rot), (T, 1))))
        if parent is None:
            glob[name] = local
            pos[name] = (np.tile(np.asarray(off, dtype=float), (T, 1)) if motion is None
                         else motion.pelvis_pos.copy())
        else:
            glob[name] = glob[parent] * local
            pos[name] = pos[parent] + glob[parent].apply(np.asarray(off, dtype=float))
    return pos, glob


def source_fk(clip: BvhClip) -> dict[str, np.ndarray]:
    _g, p = clip.forward_kinematics()
    return {name: p[:, i] @ TO_COMP.T for i, name in enumerate(clip.names)}


def angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def palm_normal(p: dict[str, np.ndarray], wrist: str, index: str, pinky: str) -> np.ndarray:
    return np.cross(p[index] - p[wrist], p[pinky] - p[wrist])


def compare(man: dict[str, np.ndarray], src: dict[str, np.ndarray],
            motion: RetargetedMotion) -> dict[str, np.ndarray]:
    """Per-frame angular error for every truth measure."""
    err: dict[str, np.ndarray] = {}
    # Where the face channel says the head looks (Manny's rest head faces +Y) vs
    # the source's Head -> mid-eyes line. Offset by the eyes' height above the head
    # joint, but that offset must stay constant if the head orientation tracks.
    eyes = (src["LeftEye"] + src["RightEye"]) * 0.5
    err["head gaze"] = angles_deg(motion.head_rotation.apply(np.array([0.0, 1.0, 0.0])), eyes - src["Head"])
    for a, b in LIMB_SEGMENTS + FINGER_SEGMENTS + GEOMETRY_SEGMENTS:
        err[f"{a}->{b}"] = angles_deg(man[b] - man[a], src[ANATOMY[b]] - src[ANATOMY[a]])
    # Whole-torso line: hip centre -> base of neck. Independent of how the spine
    # joints are split; includes the unstreamed spine_03/spine_05 held at bind.
    m_hip = (man["thigh_l"] + man["thigh_r"]) * 0.5
    s_hip = (src["LeftLeg"] + src["RightLeg"]) * 0.5
    torso = angles_deg(man["neck_01"] - m_hip, src["Neck1"] - s_hip)
    # Ill-conditioned when the chord collapses. Two native clips (v3_lightly_r1,
    # v3_languidly_r1) fold the source to 2-9 cm hip->neck against 54.5 cm at rest -
    # generated hyperflexion - where a 1 cm difference reads as up to 112 deg.
    # Those frames are masked (NaN) and counted, not averaged in.
    chord = np.linalg.norm(src["Neck1"] - s_hip, axis=-1)
    err["torso (hips->neck)"] = np.where(chord < TORSO_MIN_CHORD_CM, np.nan, torso)
    # Roll the swing alignment leaves free, observed from positions only.
    err["hip line"] = angles_deg(man["thigh_l"] - man["thigh_r"], src["LeftLeg"] - src["RightLeg"])
    err["shoulder line"] = angles_deg(man["upperarm_l"] - man["upperarm_r"], src["LeftArm"] - src["RightArm"])
    for s, S in (("_l", "LeftHand"), ("_r", "RightHand")):
        err[f"palm normal{s}"] = angles_deg(
            palm_normal(man, f"hand{s}", f"index_01{s}", f"pinky_01{s}"),
            palm_normal(src, S, f"{S}Index2", f"{S}Pinky2"))
    return err


ROLL_MEASURES = ("torso (hips->neck)", "hip line", "shoulder line", "palm normal_l", "palm normal_r", "head gaze")
# The subset whose drift is asserted. The torso and shoulder lines are chords over
# several bones of different proportions, so they legitimately vary with the pose.
DRIFT_ASSERTED = ("hip line", "palm normal_l", "palm normal_r", "head gaze")
REPORTED = ROLL_MEASURES + tuple(f"{a}->{b}" for a, b in GEOMETRY_SEGMENTS[:3])


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, msg: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            self.failures += 1


def synthetic_clip(frames: int) -> BvhClip:
    """The SOMA T-pose repeated `frames` times, to be posed by the caller."""
    rest = read_bvh(SOMA77_TPOSE_BVH)
    J = len(rest.joints)
    return BvhClip(joints=rest.joints, frame_time=rest.frame_time,
                   local_rot=[R.identity(frames) for _ in range(J)],
                   local_pos=np.tile(rest.offsets(), (frames, 1, 1)), path=None)


# ---------------------------------------------------------------------------
# 1. Rest pose
# ---------------------------------------------------------------------------

def check_rest(rt: Retargeter, rep: Report) -> dict[str, float]:
    """Returns the rest value of every reported measure, for test 2's drift check."""
    print("\n1. Rest pose (SOMA77 T-pose in)")
    clip = synthetic_clip(1)
    motion = rt.retarget_clip(clip)
    man, src = manny_fk(motion), source_fk(clip)
    err = compare(man, src, motion)
    worst_body = max(float(err[f"{a}->{b}"][0]) for a, b in LIMB_SEGMENTS)
    worst_finger = max(float(err[f"{a}->{b}"][0]) for a, b in FINGER_SEGMENTS)
    rep.check(worst_body < TOL_DIR_DEG,
              f"{len(LIMB_SEGMENTS)} limb bones point along the source: worst {worst_body:.5f} deg")
    rep.check(worst_finger < TOL_DIR_DEG,
              f"{len(FINGER_SEGMENTS)} finger bones point along the source: worst {worst_finger:.5f} deg")
    pelvis_err = float(np.linalg.norm(motion.pelvis_pos[0] - rt.manny_rest_pos["pelvis"]))
    rep.check(pelvis_err < 1e-6, f"pelvis lands on Manny's rest pelvis: {pelvis_err:.2e} cm")
    rep.check(len(motion.bone_transforms(0)) == 60, "60 BoneTransforms per frame")
    # Both rest poses look straight ahead, so the face channel must be neutral. A
    # swing onto SOMA's Head -> HeadEnd (6.5 deg back) used to fail exactly this.
    head_off = float(np.degrees(motion.head_rotation[0].magnitude()))
    rep.check(head_off < TOL_DIR_DEG, f"face-channel head is neutral at the source's rest: {head_off:.5f} deg")
    print("  report - rig-geometry offsets at rest (Manny and SOMA proportions differ):")
    for k in REPORTED:
        print(f"    {k:<24} {float(err[k][0]):7.3f} deg")
    worst_palm = max(float(err[f"{a}->{b}"][0]) for a, b in GEOMETRY_SEGMENTS[3:])
    print(f"    {'hand -> finger bases':<24} {worst_palm:7.3f} deg worst (palm shape)")
    return {k: float(err[k][0]) for k in REPORTED}


# ---------------------------------------------------------------------------
# 2. Direction truth on real clips
# ---------------------------------------------------------------------------

def check_clips(rt: Retargeter, paths: list[Path], rest: dict[str, float], rep: Report) -> None:
    """Truth is always the ORIGINAL file's FK, so a native clip's conversion to the
    standard convention is checked by the same measures as the retarget itself."""
    print("\n2. Direction truth on real clips (and 4. head channel)")
    rest_head = manny_fk_full(None)[1]["head"][0]
    groups: dict[str, list[tuple[Path, BvhClip, BvhClip]]] = {"standard": [], "native": []}
    other: list[tuple[Path, str]] = []
    for path in paths:
        raw = read_bvh(path)
        try:
            clip, convention = to_standard_convention(raw, rt.source)
        except ValueError as e:
            other.append((path, str(e)))
            continue
        groups[convention].append((path, raw, clip))
    print(f"  {len(groups['standard'])} standard-convention clips, {len(groups['native'])} native-convention "
          f"(converted with Kimodo's O_j table), {len(other)} unrecognised")
    rep.check(not other, "every clip is recognised as one of the two SOMA77 conventions"
              + ("" if not other else f": {other[0][1]}"))

    if groups["native"]:
        moved = 0.0
        for path, raw, clip in groups["native"]:
            a, b = source_fk(raw), source_fk(clip)
            moved = max(moved, max(float(np.abs(a[n] - b[n]).max()) for n in a))
        rep.check(moved < 1e-3, f"native -> standard conversion keeps every joint of every frame in place: "
                                f"worst {moved:.2e} cm")

    for convention, items in groups.items():
        if not items:
            continue
        print(f"  -- {convention} convention --")
        worst_seg, worst_seg_where, frames = 0.0, "", 0
        worst_head, worst_head_where = 0.0, ""
        roll_all: dict[str, list[np.ndarray]] = {k: [] for k in REPORTED}
        for path, raw, clip in items:
            motion = rt.retarget_clip(clip)
            err = compare(manny_fk(motion), source_fk(raw), motion)
            frames += clip.num_frames
            e = head_channel_error(motion, rest_head)
            if e > worst_head:
                worst_head, worst_head_where = e, path.name
            for a, b in LIMB_SEGMENTS + FINGER_SEGMENTS:
                e = float(err[f"{a}->{b}"].max())
                if e > worst_seg:
                    worst_seg, worst_seg_where = e, f"{a}->{b} in {path.name}"
            for k in REPORTED:
                roll_all[k].append(err[k])
        rep.check(worst_seg < TOL_DIR_DEG,
                  f"{len(LIMB_SEGMENTS) + len(FINGER_SEGMENTS)} bones x {frames} frames ({len(items)} clips): "
                  f"worst {worst_seg:.5f} deg" + (f" ({worst_seg_where})" if worst_seg >= TOL_DIR_DEG else ""))
        rep.check(worst_head < TOL_HEAD_CHANNEL_DEG,
                  f"4. face-channel head decodes to the sent body's head: worst {worst_head:.5f} deg"
                  + (f" ({worst_head_where})" if worst_head >= TOL_HEAD_CHANNEL_DEG else ""))
        for k in DRIFT_ASSERTED:
            drift = float(np.abs(np.concatenate(roll_all[k]) - rest[k]).max())
            rep.check(drift < TOL_ROLL_DRIFT_DEG,
                      f"{k} keeps its rest offset through the motion (roll tracks): drift {drift:.3f} deg")
        print("  report - over all frames (median / 95th pct / max), rest value in brackets:")
        for k in REPORTED:
            e = np.concatenate(roll_all[k])
            masked = int(np.isnan(e).sum())
            print(f"    {k:<24} {np.nanmedian(e):6.2f} / {np.nanpercentile(e, 95):6.2f} / {np.nanmax(e):6.2f} deg  "
                  f"[{rest[k]:.2f}]" + (f"  ({masked} folded frames masked)" if masked else ""))


# ---------------------------------------------------------------------------
# 3. Change-of-basis sanity
# ---------------------------------------------------------------------------

def _facing(man: dict[str, np.ndarray]) -> np.ndarray:
    """Manny's facing from joint positions: +X is his left, +Z up, so forward is
    up x left (Z x X = Y)."""
    left = man["thigh_l"] - man["thigh_r"]
    left[:, 2] = 0.0
    fwd = np.cross(np.array([0.0, 0.0, 1.0]), left)
    return fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)


def check_change_of_basis(rt: Retargeter, rep: Report) -> None:
    print("\n3. Change-of-basis sanity (synthetic poses)")
    # The source data itself: SOMA's T-pose faces +Z with its left hand on +X.
    rest = source_fk(synthetic_clip(1))
    raw_eye = rest["LeftEye"][0] @ TO_COMP  # TO_COMP is its own inverse: back to source axes
    raw_head = rest["Head"][0] @ TO_COMP
    raw_lh = rest["LeftHand"][0] @ TO_COMP
    rep.check(raw_eye[2] > raw_head[2] and raw_lh[0] > 0,
              "SOMA source: eyes ahead in +Z, left hand on +X (Y up, +Z fwd, +X left)")

    # a) Walk forward 100 cm along +Z.
    T = 11
    clip = synthetic_clip(T)
    hips = clip.index("Hips")
    clip.local_pos[:, hips, 2] += np.linspace(0.0, 100.0, T)
    man = manny_fk(rt.retarget_clip(clip))
    d = man["pelvis"][-1] - man["pelvis"][0]
    rep.check(d[1] > 90.0 and abs(d[0]) < 1e-6 and abs(d[2]) < 1e-6,
              f"source walks +Z 100 cm -> Manny pelvis moves ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}) cm (+Y, scaled to leg length)")
    f = _facing(man)[0]
    rep.check(f[1] > 0.999, f"Manny faces +Y: facing ({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})")
    toe = man["ball_l"][0] - man["foot_l"][0]
    rep.check(toe[1] > 0.0, f"Manny's toes are ahead of the ankle in +Y ({toe[1]:.2f} cm)")
    rep.check(man["hand_l"][0][0] > 0 and man["hand_r"][0][0] < 0,
              "source left hand -> Manny hand_l on +X, right on -X (no left/right swap)")

    # b) Turn 90 deg to the character's left: about source +Y, +Z (fwd) -> +X (left).
    clip = synthetic_clip(1)
    clip.local_rot[hips] = R.from_euler("Y", [90.0], degrees=True)
    f = _facing(manny_fk(rt.retarget_clip(clip)))[0]
    rep.check(f[0] > 0.999, f"source turns to its left -> Manny faces +X: ({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})"
              " (an inverted rotation would face -X)")

    # c) Raise the left arm straight up: LeftArm about source +Z by +90 deg takes +X to +Y.
    clip = synthetic_clip(1)
    rest_man = manny_fk(rt.retarget_clip(synthetic_clip(1)))
    clip.local_rot[clip.index("LeftArm")] = R.from_euler("Z", [90.0], degrees=True)
    man = manny_fk(rt.retarget_clip(clip))
    arm = man["lowerarm_l"][0] - man["upperarm_l"][0]
    arm /= np.linalg.norm(arm)
    rep.check(arm[2] > 0.99, f"source raises left arm -> upperarm_l points up: ({arm[0]:.3f}, {arm[1]:.3f}, {arm[2]:.3f})")
    moved = max(float(np.linalg.norm(man[b][0] - rest_man[b][0]))
                for b in ("upperarm_r", "lowerarm_r", "hand_r", "middle_01_r"))
    rep.check(moved < 1e-6, f"right arm untouched by the left-arm raise: {moved:.2e} cm")


# ---------------------------------------------------------------------------
# 4. Head channel
# ---------------------------------------------------------------------------

def head_channel_error(motion: RetargetedMotion, rest_head: R) -> float:
    """Worst angle, over the clip, between the head the face curves put on the
    MetaHuman and the head the streamed body pose puts on Manny (independent FK).
    The decode inverts the measured behaviour stated in live_link_face_protocol:
    float32 on the wire, HEAD_DEG_PER_UNIT per unit, composed pitch * roll * yaw,
    headYaw + = -Z, headPitch + = +X, headRoll + = -Y."""
    _pos, glob = manny_fk_full(motion)
    sent = glob["head"] * rest_head.inv()
    angles = np.zeros((motion.num_frames, 3))
    for k in range(motion.num_frames):
        c = head_rotation_to_curves(motion.head_rotation[k])
        yaw, pitch, roll = (float(np.float32(c[n])) * HEAD_DEG_PER_UNIT
                            for n in ("headYaw", "headPitch", "headRoll"))
        angles[k] = (pitch, -roll, -yaw)
    decoded = R.from_euler("XYZ", angles, degrees=True)
    return float(np.degrees((decoded.inv() * sent).magnitude()).max())


# ---------------------------------------------------------------------------
# 5. Wire format
# ---------------------------------------------------------------------------

class _CaptureClient:
    """Stands in for the encoder's SimpleUDPClient: records instead of sending."""
    address: str = ""
    args: list = []

    def send_message(self, address: str, args: list) -> None:
        self.address, self.args = address, list(args)


def check_wire(rt: Retargeter, rep: Report) -> None:
    print("\n5. Wire format")
    motion = rt.retarget_clip(synthetic_clip(2))
    bones, _head = motion.sample(0.5 / motion.fps)   # what the conductor sends: a sampled pose
    encoder = LiveLinkPoseOSCEncoder()
    capture = _CaptureClient()
    encoder.client = capture
    encoder.send(bones, present=True)
    order = [b.name for b in bones]
    # conductor.py sends raw_bones + left_bones + right_bones, each hand being
    # FINGER_BONE_NAMES filtered by side (pose_solver._solve_one_hand).
    expected = (list(BONE_NAMES) + [n for n in FINGER_BONE_NAMES if n.endswith("_l")]
                + [n for n in FINGER_BONE_NAMES if n.endswith("_r")])
    rep.check(capture.address == "/mediapipe/pose", f"OSC address {capture.address!r}")
    rep.check(len(capture.args) == 1 + 7 * 60, f"{len(capture.args)} floats per packet (expected 421)")
    rep.check(order == expected, "60 bones in conductor.py's order (22 body, 19 left hand, 19 right hand)")


# ---------------------------------------------------------------------------
# 5b. Player (remote_kimodo_service.player)
# ---------------------------------------------------------------------------

class _PacketLog:
    """Stands in for SimpleUDPClient inside a real Sender: every packet the player sends."""
    def __init__(self) -> None:
        self.packets: list[tuple[float, list]] = []

    def send_message(self, address: str, args: list) -> None:
        self.packets.append((time.perf_counter(), list(args)))


def _capturing_sender() -> tuple[Sender, _PacketLog]:
    sender = Sender(face=False)
    log = _PacketLog()
    sender.pose.client = log
    return sender, log


def _short_motion(rt: Retargeter, frames: int, lift_deg: float = 0.0) -> RetargetedMotion:
    clip = synthetic_clip(frames)
    if lift_deg:
        clip.local_rot[clip.index("LeftArm")] = R.from_euler("z", np.full((frames, 1), lift_deg), degrees=True)
    return rt.retarget_clip(clip)


def check_player(rt: Retargeter, rep: Report) -> None:
    print("\n5b. Player (loop, hand-over at the end of a pass, packets)")
    a = _short_motion(rt, 9)                 # 0.3 s at 30 fps
    b = _short_motion(rt, 9, lift_deg=60.0)  # distinguishable: left arm raised
    starts: list[tuple[float, str]] = []

    sender, log = _capturing_sender()
    player = LoopPlayer(sender, loop=True, on_start=lambda c: starts.append((time.perf_counter(), c.label)))
    player.enqueue(a, "A")
    player.start()
    time.sleep(0.75)
    rep.check(player.passes >= 2 and player.current.label == "A",
              f"a lone clip loops: {player.passes} passes of A in 0.75 s")
    t_enq = time.perf_counter()
    player.enqueue(b, "B")
    time.sleep(0.5)
    b_start = next((t for t, lab in starts if lab == "B"), None)
    rep.check(b_start is not None and 0.0 <= b_start - t_enq <= a.duration_s + 0.05,
              f"queued clip takes over at the end of the current pass "
              f"({'never' if b_start is None else f'{(b_start - t_enq) * 1e3:.0f} ms after enqueue'}, pass is "
              f"{a.duration_s * 1e3:.0f} ms)")
    player.stop()
    lens = {len(args) for _t, args in log.packets}
    rep.check(lens == {421}, f"every packet is 421 floats ({len(log.packets)} packets, lengths {sorted(lens)})")
    rep.check(all(args[0] == 1.0 for _t, args in log.packets[:-1]) and log.packets[-1][1][0] == 0.0,
              "present=1 while playing, the last packet on stop is present=0")
    gaps = np.diff([t for t, _a in log.packets[:-1]])
    rep.check(len(gaps) > 0 and float(np.median(gaps)) < 1.5 / 60.0,
              f"sends at ~60 Hz (median gap {np.median(gaps) * 1e3:.1f} ms, worst lateness "
              f"{player.max_late_s * 1e3:.1f} ms)")

    sender, log = _capturing_sender()
    player = LoopPlayer(sender, loop=False)
    player.enqueue(a, "A")
    player.start()
    done = player.finished.wait(2.0)
    player.stop()
    rep.check(done, "loop=False: plays the clip once, then reports finished")

    # clear_pending(): the GUI's hook for Remove / Clear / reorder.
    sender, log = _capturing_sender()
    player = LoopPlayer(sender, loop=True)
    for label in "ABC":
        player.enqueue(a, label)
    player.start()
    time.sleep(0.1)
    drained = [c.label for c in player.clear_pending()]
    time.sleep(a.duration_s + 0.1)
    rep.check(drained == ["B", "C"] and player.current.label == "A" and player.passes >= 1 and not player.queued,
              f"clear_pending() takes the queued clips back in order ({drained}); the current one keeps looping")
    player.stop()


# ---------------------------------------------------------------------------
# 6. Client resilience (no Spark: stub servers and a closed port)
# ---------------------------------------------------------------------------

def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def check_client(rt: Retargeter, rep: Report) -> None:
    print("\n6. Client resilience")
    req = GenerationRequest.build("A person waves", seed=0, seconds=1.0)

    def outcome(fn) -> tuple[str, str, float]:
        t0 = time.perf_counter()
        try:
            fn()
            return "ok", "", time.perf_counter() - t0
        except (SparkUnavailable, GenerationFailed, ContractError) as e:
            return type(e).__name__, str(e), time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp:
        cache = ClipCache(Path(tmp) / "cache")
        # Windows retries a SYN to a closed localhost port and reports refused after
        # 2.05 s (measured), so the timeout must be longer for "refused" to show.
        refused = Lane(KimodoClient(f"http://127.0.0.1:{_closed_port()}", timeout=5.0), cache, rt)

        # The player keeps sending through every failure below.
        sender, log = _capturing_sender()
        player = LoopPlayer(sender, loop=True)
        player.enqueue(_short_motion(rt, 9), "A")
        player.start()
        before = len(log.packets)

        kind, msg, _ = outcome(lambda: refused.fetch(req))
        rep.check(kind == "SparkUnavailable" and "refused" in msg, f"closed port -> {kind}: {msg[:70]}")
        with FakeKimodo(mode="close") as fake:
            kind, msg, _ = outcome(lambda: Lane(KimodoClient(fake.url), cache, rt).fetch(req))
        rep.check(kind == "SparkUnavailable", f"accept-and-close (tunnel up, service down) -> {kind}: {msg[:60]}")
        with FakeKimodo(mode="slow", delay_s=3.0) as fake:
            kind, msg, dt = outcome(lambda: Lane(KimodoClient(fake.url, timeout=0.5), cache, rt).fetch(req))
        rep.check(kind == "SparkUnavailable" and dt < 2.0, f"no answer -> {kind} after {dt:.2f} s (timeout 0.5 s)")
        with FakeKimodo(mode="error") as fake:
            kind, msg, _ = outcome(lambda: Lane(KimodoClient(fake.url), cache, rt).fetch(req))
        rep.check(kind == "GenerationFailed" and "500" in msg, f"HTTP 500 -> {kind}: {msg[:60]}")
        with FakeKimodo(mode="garbage") as fake:
            kind, msg, _ = outcome(lambda: Lane(KimodoClient(fake.url), cache, rt).fetch(req))
        rep.check(kind == "ContractError" and not cache.entries(), f"malformed NPZ -> {kind}, nothing cached")

        time.sleep(0.1)
        rep.check(player._thread.is_alive() and len(log.packets) > before + 10,
                  f"player kept sending through all of it ({len(log.packets) - before} packets)")
        player.stop()

        with FakeKimodo(mode="ok") as fake:
            lane = Lane(KimodoClient(fake.url), cache, rt)
            kind, msg, _ = outcome(lambda: lane.fetch(req))
            health = lane.client.health()
            n_live = fake.requests
        rep.check(kind == "ok" and len(cache.entries()) == 1, f"a good answer is played and cached ({kind})")
        rep.check(health.get("loaded") is True, "/health answers through the client")
        kind, msg, _ = outcome(lambda: refused.fetch(req))
        rep.check(kind == "ok", f"cache hit with the Spark unreachable -> {kind} (network not touched)")
        rep.check(n_live == 2, f"the stub saw exactly one generate + one health ({n_live} requests)")

        # fetch_detailed(): the GUI's hook. fetch() is a wrapper around it, so the checks
        # above cover its failure paths; this covers what it reports.
        req2 = GenerationRequest.build("A person jumps", seed=3, seconds=1.0)
        with FakeKimodo(mode="ok") as fake:
            lane = Lane(KimodoClient(fake.url), cache, rt)
            miss = lane.fetch_detailed(req2)
            hit = lane.fetch_detailed(req2)
            n_live = fake.requests
        rep.check(miss.result is not None and miss.result.generation_s is not None and miss.path is not None
                  and miss.path.exists() and miss.meta.get("prompt") == req2.prompt
                  and hit.result is None and hit.path is None and hit.meta == miss.meta
                  and hit.label.endswith("(cache)") and n_live == 1,
                  "fetch_detailed: a miss reports the request's timings and the cache file; a hit reports "
                  "the same NPZ meta and makes no request")

    cases = {"A person walks /s 25": (25, 0, 270), "A person walks /seed 7 /t 4": (100, 7, 120),
             "walks and/or runs": (100, 0, 270)}
    ok = all((r.steps, r.seed, r.num_frames) == exp and "/" not in r.prompt.replace("and/or", "")
             for line, exp in cases.items() for r in [parse_prompt(line, 9.0, 100, 0, contract.DEFAULT_MODEL)])
    bad = []
    for line in ("walks /x 3", "walks /t 11", "walks /s zero", "/s 5"):
        try:
            parse_prompt(line, 9.0, 100, 0, contract.DEFAULT_MODEL)
        except ValueError:
            bad.append(line)
    rand = {parse_prompt("walks /seed random", 9.0, 100, 0, contract.DEFAULT_MODEL).seed for _ in range(5)}
    rep.check(ok and len(bad) == 4 and len(rand) > 1,
              "inline prompt options: /s, /seed N|random, /t parsed; unknown, too long, non-numeric, empty refused")


# ---------------------------------------------------------------------------
# 7. NPZ contract
# ---------------------------------------------------------------------------

def check_contract(rt: Retargeter, paths: list[Path], rep: Report) -> None:
    print("\n7. NPZ contract")
    sk = rt.source
    picked: dict[str, Path] = {}
    for p in paths:
        _c, conv = to_standard_convention(read_bvh(p), sk)
        picked.setdefault(conv, p)
        if len(picked) == 2:
            break
    for conv, p in sorted(picked.items()):
        raw = read_bvh(p)
        data = contract.pack(clip_from_bvh(raw, {"prompt": p.stem}))
        clip = contract.unpack(data)
        motion, adapted = retarget_npz(clip, rt)
        ref = rt.retarget_clip(to_standard_convention(raw, sk)[0])
        dq = float(np.degrees(np.max((R.from_quat(motion.local_quats.reshape(-1, 4)).inv()
                                      * R.from_quat(ref.local_quats.reshape(-1, 4))).magnitude())))
        dp = float(np.abs(motion.pelvis_pos - ref.pelvis_pos).max())
        rep.check(adapted.convention == conv and dq < 1e-3 and dp < 1e-3,
                  f"{p.name} ({conv}) via NPZ == via BVH: {dq:.5f} deg, pelvis {dp:.5f} cm "
                  f"(FK self-check {adapted.fk_error_cm:.4f} cm)")

    # Real Kimodo output from the Spark (kimodo 1.0.0, 2026-10-07): the convention, the
    # root and the units are Kimodo's own, not the ones clip_from_bvh writes.
    real = contract.unpack(REAL_FIXTURE.read_bytes())
    motion, adapted = retarget_npz(real, rt)
    src = {n: real.posed_joints[:, i] * 100.0 @ TO_COMP.T for i, n in enumerate(real.bone_order_names)}
    err = compare(manny_fk(motion), src, motion)
    worst = max(float(np.nanmax(err[f"{a}->{b}"])) for a, b in LIMB_SEGMENTS + FINGER_SEGMENTS)
    rep.check(adapted.convention == "standard" and adapted.fk_error_cm < 1e-3 and adapted.root_vs_hips_cm < 1e-3,
              f"real Kimodo NPZ ({REAL_FIXTURE.name}): {adapted.convention} convention, FK self-check "
              f"{adapted.fk_error_cm:.5f} cm, root_positions = Hips to {adapted.root_vs_hips_cm:.5f} cm")
    # float32 off the wire: ~6e-4 deg measured, against 1e-5 for float64 BVH data.
    rep.check(worst < TOL_DIR_DEG_FLOAT32,
              f"real Kimodo NPZ: 40 Manny bones vs Kimodo's own posed_joints, worst {worst:.5f} deg")

    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        keys = {k: (z[k].shape, z[k].dtype.kind) for k in z.files}
    T = clip.num_frames
    expected = {"global_rot_mats": ((T, 77, 3, 3), "f"), "root_positions": ((T, 3), "f"),
                "posed_joints": ((T, 77, 3), "f"), "foot_contacts": ((T, 4), "f"), "fps": ((), "f"),
                "bone_order_names": ((77,), "U"), "meta_json": ((), "U")}
    rep.check(keys == expected, "documented keys, shapes and dtypes, nothing else")
    hips_top = float(clip.posed_joints[:, 0, 1].max())
    rep.check(0.5 < hips_top < 2.0, f"positions are metres (highest Hips {hips_top:.2f} m)")

    def refused(mutate) -> bool:
        c = contract.unpack(data)
        mutate(c)
        try:
            retarget_npz(contract.unpack(contract.pack(c)), rt)
            return False
        except ContractError:
            return True

    def drop_key() -> bool:
        buf = io.BytesIO()
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            np.savez_compressed(buf, **{k: z[k] for k in z.files if k != "posed_joints"})
        try:
            contract.unpack(buf.getvalue())
            return False
        except ContractError:
            return True

    checks = {
        "transposed rotations": lambda c: setattr(c, "global_rot_mats", np.swapaxes(c.global_rot_mats, -1, -2)),
        "positions in cm instead of m": lambda c: setattr(c, "posed_joints", c.posed_joints * 100.0),
        "joints in another order": lambda c: setattr(c, "bone_order_names", c.bone_order_names[::-1]),
        "NaN in a rotation": lambda c: c.global_rot_mats.__setitem__((0, 3, 0, 0), np.nan),
    }
    results = {name: refused(fn) for name, fn in checks.items()}
    results["a missing key"] = drop_key()
    results["truncated bytes"] = _raises(lambda: contract.unpack(data[: len(data) // 2]))
    rep.check(all(results.values()), "refused: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in results.items()))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ContractError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.clips or sorted(p for d in CLIP_DIRS for p in d.glob("*.bvh"))

    rt = Retargeter(load_soma77())
    rep = Report()
    rest = check_rest(rt, rep)
    check_change_of_basis(rt, rep)
    check_wire(rt, rep)
    check_player(rt, rep)
    check_client(rt, rep)
    check_contract(rt, paths, rep)
    check_clips(rt, paths, rest, rep)
    print(f"\n{'ALL PASS' if not rep.failures else f'{rep.failures} FAILURE(S)'} (tests 1-7; geometry offsets are a report)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
