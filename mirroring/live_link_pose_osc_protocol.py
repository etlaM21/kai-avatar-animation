from __future__ import annotations
from pythonosc.udp_client import SimpleUDPClient
from pose_solver import BoneTransform

class LiveLinkPoseOSCEncoder:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9001) -> None:
        self.client = SimpleUDPClient(ip, port)
        self.address = "/mediapipe/pose"

    def send(self, bone_transforms: list[BoneTransform], present: bool = True) -> None:
        args = [1.0 if present else 0.0]
        
        for bone in bone_transforms:
            # 3 position floats
            args.extend([bone.position["x"], bone.position["y"], bone.position["z"]])
            # 4 rotation floats
            args.extend([bone.rotation["x"], bone.rotation["y"], bone.rotation["z"], bone.rotation["w"]])
            
        self.client.send_message(self.address, args)