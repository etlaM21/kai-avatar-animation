"""
LiveLink Pose JSON Protocol Encoder.

Packages computed skeletal bone transforms into a JSON payload for transmission
over UDP to Unreal Engine's MediaPipeLiveLink plugin.
"""

from __future__ import annotations

import json
from pose_solver import BoneTransform


class LiveLinkPoseJSONEncoder:
    """Encodes bone transforms and tracking metadata into a JSON UDP byte payload."""

    def __init__(self, subject_name: str = "MediaPipePose") -> None:
        self.subject_name = subject_name

    def encode(self, bone_transforms: list[BoneTransform], present: bool = True) -> bytes:
        """
        Builds the JSON payload matching MediaPipeLiveLinkSource.cpp's parser:
        {
            "name": "MediaPipePose",
            "present": 1.0,
            "bones": [
                {
                    "name": "pelvis",
                    "position": {"x": 0.0, "y": 0.0, "z": 95.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }, ...
            ]
        }
        """
        payload = {
            "name": self.subject_name,
            "present": 1.0 if present else 0.0,
            "bones": [
                {
                    "name": bone.name,
                    "position": bone.position,
                    "rotation": bone.rotation,
                }
                for bone in bone_transforms
            ],
        }

        # Encode directly to UTF-8 bytes for raw UDP transmission
        return json.dumps(payload).encode("utf-8")