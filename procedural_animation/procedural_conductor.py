"""Play a Kimodo SOMA77 BVH clip into Unreal through the existing mirroring encoders.

    .\\mirroring\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --bvh clip.bvh
    .\\mirroring\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --bvh clip.bvh --loop
    .\\mirroring\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --bvh clip.bvh --loop --root in-place
    .\\mirroring\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --bvh clip.bvh --dry-run

Sends exactly what conductor.py sends, to the same default ports, so Unreal needs no
change: 60 BoneTransforms (22 body + 19 + 19 fingers, 421 floats with the present
flag) over OSC to 9001, and a Live Link Face packet to 11111 carrying the head
rotation (neutral blendshapes). Run it INSTEAD of conductor.py, not alongside it
(procedural-animation.md sec. 4.4, level 1).

Both of Kimodo's BVH conventions play (save_motion_bvh(..., standard_tpose=True or
False)); native clips are converted on load, see source_skeletons.to_standard_convention.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

from . import MIRRORING_DIR  # noqa: F401  (puts mirroring on sys.path)
from .bvh_reader import read_bvh
from .retarget import STREAMED_BONES, Retargeter, RetargetedMotion
from .source_skeletons import load_soma77, to_standard_convention

from live_link_face_protocol import LiveLinkFaceEncoder, head_rotation_to_curves  # read-only
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder  # read-only

POSE_OSC_PORT = 9001         # conductor.py's defaults
LIVE_LINK_FACE_PORT = 11111


def recenter(motion: RetargetedMotion, target_xy: np.ndarray) -> None:
    """Shift the clip horizontally so it starts at target_xy; the root motion within
    the clip is kept. Height is left alone (it is what keeps the feet on the floor)."""
    motion.pelvis_pos[:, :2] += target_xy - motion.pelvis_pos[0, :2]


def hold_in_place(motion: RetargetedMotion) -> None:
    """Drop the clip's horizontal root motion: the pelvis stays at its first-frame
    position (height kept), so a travelling clip loops without jumping back."""
    motion.pelvis_pos[:, :2] = motion.pelvis_pos[0, :2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh", type=Path, required=True, help="Kimodo SOMA77 clip (either BVH convention)")
    ap.add_argument("--rate", type=float, default=60.0, help="send rate in Hz (source frames are slerped)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--loop", action="store_true", help="loop the clip until Ctrl+C")
    ap.add_argument("--no-recenter", action="store_true",
                    help="keep the clip's absolute root position instead of starting on Manny's rest spot")
    ap.add_argument("--root", choices=("travel", "in-place"), default="travel",
                    help="travel: root moves as generated; in-place: no horizontal root motion")
    ap.add_argument("--pose-ip", default="127.0.0.1")
    ap.add_argument("--pose-port", type=int, default=POSE_OSC_PORT)
    ap.add_argument("--face-ip", default="127.0.0.1")
    ap.add_argument("--face-port", type=int, default=LIVE_LINK_FACE_PORT)
    ap.add_argument("--no-face", action="store_true", help="don't send the Live Link Face head channel")
    ap.add_argument("--dry-run", action="store_true", help="retarget and time the loop, but send nothing")
    args = ap.parse_args(argv)

    skeleton = load_soma77()
    try:
        clip, convention = to_standard_convention(read_bvh(args.bvh), skeleton)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    retargeter = Retargeter(skeleton)
    motion = retargeter.retarget_clip(clip)
    if not args.no_recenter:
        recenter(motion, retargeter.manny_rest_pos["pelvis"][:2])
    if args.root == "in-place":
        hold_in_place(motion)
    print(f"{args.bvh.name} ({convention} convention): {motion.num_frames} frames @ {motion.fps:.1f} fps "
          f"({motion.duration_s:.2f} s), retargeted in {time.perf_counter() - t0:.2f} s")

    pose_encoder = None if args.dry_run else LiveLinkPoseOSCEncoder(ip=args.pose_ip, port=args.pose_port)
    face_encoder = LiveLinkFaceEncoder(fps=int(round(args.rate)))
    face_socket = None if (args.dry_run or args.no_face) else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    face_target = (args.face_ip, args.face_port)
    if not args.dry_run:
        print(f"streaming pose -> {args.pose_ip}:{args.pose_port}"
              + ("" if args.no_face else f", head -> {args.face_ip}:{args.face_port}")
              + (" (looping, Ctrl+C to stop)" if args.loop else ""))

    period = 1.0 / args.rate
    sent = 0
    last_bones = None
    start = time.perf_counter()
    next_tick = start
    try:
        while True:
            clip_t = (time.perf_counter() - start) * args.speed
            if clip_t > motion.duration_s:
                if not args.loop:
                    break
                start = time.perf_counter()
                clip_t = 0.0
            bones, head = motion.sample(clip_t)
            assert len(bones) == len(STREAMED_BONES) == 60
            last_bones = bones
            if pose_encoder is not None:
                pose_encoder.send(bones, present=True)
            if face_socket is not None:
                face_socket.sendto(face_encoder.encode(head_rotation_to_curves(head)), face_target)
            sent += 1

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # fell behind: don't try to catch up in a burst
    except KeyboardInterrupt:
        pass
    finally:
        # Same "tracking lost" signal conductor.py sends when it has no pose: the
        # last pose, present=0.
        if pose_encoder is not None and last_bones is not None:
            pose_encoder.send(last_bones, present=False)
        if face_socket is not None:
            face_socket.close()

    print(f"{'would have sent' if args.dry_run else 'sent'} {sent} frames "
          f"({1 + 7 * len(STREAMED_BONES)} floats each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
