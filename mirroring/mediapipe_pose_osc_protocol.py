"""
MediaPipe Pose -> OSC payload builder.

Mirrors live_link_face_protocol.py's role for the pose channel: this
module only knows how to turn a channel-name -> value dict into the
arguments for one OSC message. It does not own a socket or a python-osc
client - Conductor owns both channels' network I/O, so all actual
sending happens in one obvious place rather than being spread across
modules.

Wire format (address + args, sent via python-osc's SimpleUDPClient):
    address: /mediapipe/pose
    args:    [timestamp_ms, present (1.0/0.0), num_landmarks,
              <landmark>_x, <landmark>_y, <landmark>_z,
              <landmark>_visibility, <landmark>_presence,
              ... for all 33 landmarks, in PoseLandmark enum order]
    -> 3 + 33 * 5 = 168 args per frame, ALWAYS the same count - even
       while holding last-valid data with present=0.0, so the receiving
       side (the future C++ LiveLink Source plugin) can always expect
       the same fixed shape and doesn't need to handle a variable-length
       packet.

Important scope note: this only carries landmark *positions*, not bone
rotations. That's not something this module (or Conductor) solves -
turning positions into an actual animated rig on Manny's skeleton is a
separate, not-yet-built step. See conductor.py's answer to "how do we
get a rig from this" for where that fits in the pipeline.
"""

from __future__ import annotations

from mediapipe_pose_capture import PoseLandmark

_SUFFIXES = ("x", "y", "z", "visibility", "presence")


def pose_landmark_channel_prefix(landmark: PoseLandmark) -> str:
    """PoseLandmark.LEFT_SHOULDER -> "leftShoulder".

    A shared naming rule: both POSE_CHANNEL_ORDER below and Conductor's
    per-frame dict-building call this same function, so the two can
    never drift out of sync with each other by one of them being edited
    without the other. Purely cosmetic/readability - has no effect on
    the actual bytes sent over the wire.
    """
    parts = landmark.name.lower().split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Ordered, human-readable channel names: "nose_x", "nose_y", ...,
# "rightFootIndex_presence" - 165 entries, built programmatically so 33
# landmarks x 5 fields can't be hand-typed out of order by mistake.
POSE_CHANNEL_ORDER: list[str] = [
    f"{pose_landmark_channel_prefix(landmark)}_{suffix}"
    for landmark in PoseLandmark
    for suffix in _SUFFIXES
]
assert len(POSE_CHANNEL_ORDER) == len(PoseLandmark) * len(_SUFFIXES)


class PoseOSCEncoder:
    """Builds the (address, args) pair for one pose OSC message."""

    def __init__(self, address: str = "/mediapipe/pose") -> None:
        self.address = address

    def build_args(
        self,
        values: dict[str, float],
        present: bool,
        timestamp_ms: int,
    ) -> list[float]:
        """`values` should map POSE_CHANNEL_ORDER names to floats; any
        channel missing from the dict defaults to 0.0, so even a
        not-yet-fully-populated dict (e.g. before the very first valid
        detection) still produces a correctly-shaped, if all-zero,
        args list rather than an error."""
        landmark_args = [values.get(name, 0.0) for name in POSE_CHANNEL_ORDER]
        return [
            float(timestamp_ms),
            1.0 if present else 0.0,
            float(len(PoseLandmark)),
            *landmark_args,
        ]