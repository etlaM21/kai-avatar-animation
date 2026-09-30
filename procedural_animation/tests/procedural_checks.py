"""
Offline acceptance checks for the procedural retarget - no GPU, no Kimodo, no engine.

    .\\mirroring\\venv\\Scripts\\python.exe procedural_animation\\tests\\procedural_checks.py
    .\\mirroring\\venv\\Scripts\\python.exe procedural_animation\\tests\\procedural_checks.py clip.bvh ...

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

Ground truth is deliberately independent of retarget.py: Manny positions come
from an Unreal-style FK written here over the streamed BoneTransforms (rotators
and offsets straight from pose_solver's rig dump, unstreamed bones held at bind),
and the source side from bvh_reader FK. Correspondence is by anatomical JOINT
POSITION (ANATOMY below), not by the retarget's own bone->joint table, so a
mis-wired rotation shows up as a direction error.
"""

from __future__ import annotations

import argparse
import math
import sys
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

CLIP_DIRS = [MODULE_ROOT / "pipeline-network-editor" / "kimodo-gen", MODULE_ROOT / "kimodo" / "kimodo-gen"]

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
    check_clips(rt, paths, rest, rep)
    print(f"\n{'ALL PASS' if not rep.failures else f'{rep.failures} FAILURE(S)'} (tests 1-5; geometry offsets are a report)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
