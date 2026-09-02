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

NUM_CHANNELS = 61

# Wire position (0-60), matching Epic's expected order. Indices 0-51 are
# the standard ARKit blendshape names - MediaPipe's FaceLandmarker
# outputs these exact camelCase names when blendshape output is enabled,
# so no renaming is needed for those. Indices 52-60 (head/eye rotation)
# are not part of MediaPipe's blendshape set and must be supplied
# separately (see conductor.py).
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
