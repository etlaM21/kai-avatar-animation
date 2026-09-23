"""
Epic Live Link Face UDP protocol encoder.

Reimplements the wire format used by Epic's iOS Live Link Face app - and
consumed by UE5's stock "Live Link Face" plugin - based on the protocol
as documented by the open-source JimWest/PyLiveLinkFace project (archived
2026-08-13). Deliberately does not depend on that package: it's
unmaintained, and this only needs the wire format itself, not its
internal smoothing/filtering (the Conductor owns transformation here).

Packet layout (all multi-byte fields except `version` are big-endian /
network byte order):
    version        : uint32, little-endian (constant, 6)
    uuid           : 37 bytes, ascii, "$" + a UUID string
    name_length    : int32
    name           : name_length bytes, ascii
    frame_number   : uint32   \\ a simplified stand-in timecode,
    sub_frame      : uint32   /  regenerated from wall-clock time each send
    fps            : uint32
    denominator    : uint32
    blend_count    : uint8 (always 61)
    blend_shapes   : 61 x float32

Note on frame_number/sub_frame: the original library builds these from a
full SMPTE Timecode object (the `timecode` pip package). This is a
simplified equivalent that avoids that extra dependency - Unreal treats
these as informational frame-timing metadata rather than something it
strictly validates, but if you ever see timecode-related oddities in the
Live Link panel, swapping this for the `timecode` package's approach is
the first thing to try.
"""

from __future__ import annotations

import datetime
import struct
import uuid as uuid_module

from scipy.spatial.transform import Rotation

NUM_CHANNELS = 61

# Wire position (0-60), matching Epic's expected order. Indices 0-51 are
# the standard ARKit blendshape names - MediaPipe's FaceLandmarker
# outputs these exact camelCase names when blendshape output is enabled,
# so no renaming is needed for those. Indices 52-60 (head/eye rotation)
# are not part of MediaPipe's blendshape set and must be supplied
# separately: head rotation via head_rotation_to_curves() below, eyes not
# at all yet (see conductor.py).
CHANNEL_ORDER: list[str] = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
    "headYaw", "headPitch", "headRoll",
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]
assert len(CHANNEL_ORDER) == NUM_CHANNELS

# What the receiving MetaHuman does with headYaw/Pitch/Roll, MEASURED on UE 5.8
# (tests/ue_head_probe.py --measure) rather than taken from ARKit documentation:
#   - the curves set neck_02 + head to an ABSOLUTE component-space rotation. The
#     torso underneath is ignored, and the Body stream's own head never reaches
#     the visible head at all.
#   - 50 deg per unit on every axis, linear to at least 75 deg.
#   - composed pitch * roll * yaw (yaw applied first). Fitted to 0.000 deg; the
#     other five orders miss by 10-31 deg.
HEAD_DEG_PER_UNIT = 50.0


def head_rotation_to_curves(rotation: Rotation) -> dict[str, float]:
    """The head's rotation off the rig's rest head, in component space (X = the
    character's left, Y = forward, Z = up), as Live Link Face head curves.

    Absolute, so this must be the head's FULL orientation, torso included - see
    PoseSolver.head_rotation. Measured signs: headYaw + turns the head to the
    character's left (-Z), headPitch + looks up (+X), headRoll + tips the top of
    the head to the character's right (-Y). Intrinsic 'XYZ' Euler angles are
    exactly Rx * Ry * Rz, the measured composition; they only degenerate at
    +-90 deg of roll."""
    pitch, roll, yaw = rotation.as_euler("XYZ", degrees=True)
    return {"headYaw": -yaw / HEAD_DEG_PER_UNIT,
            "headPitch": pitch / HEAD_DEG_PER_UNIT,
            "headRoll": -roll / HEAD_DEG_PER_UNIT}


class LiveLinkFaceEncoder:
    """Encodes a channel-name -> value dict into an Epic Live Link Face packet."""

    def __init__(self, name: str = "PythonConductor_Face", fps: int = 60) -> None:
        self.name = name
        self.fps = fps
        self._uuid = "$" + str(uuid_module.uuid1())

    def encode(self, values: dict[str, float]) -> bytes:
        """`values` should map CHANNEL_ORDER names to floats; any missing
        channel defaults to 0.0."""
        blend_shapes = [values.get(name, 0.0) for name in CHANNEL_ORDER]

        version_packed = struct.pack("<I", 6)
        uuid_packed = self._uuid.encode("utf-8")
        name_packed = self.name.encode("utf-8")
        name_length_packed = struct.pack("!i", len(name_packed))

        now = datetime.datetime.now()
        frame_number = (
            now.hour * 3600 + now.minute * 60 + now.second
        ) * self.fps + int(now.microsecond / 1_000_000 * self.fps)
        sub_frame = 0
        frames_packed = struct.pack("!II", frame_number, sub_frame)
        frame_rate_packed = struct.pack("!II", self.fps, 1)

        data_packed = struct.pack(f"!B{NUM_CHANNELS}f", NUM_CHANNELS, *blend_shapes)

        return (
            version_packed
            + uuid_packed
            + name_length_packed
            + name_packed
            + frames_packed
            + frame_rate_packed
            + data_packed
        )