"""60 Hz player: the current clip loops; a queued clip takes over when the current pass
ends. Feeds the existing mirroring encoders unchanged.

Deliberately plain for v1 (agreed with Malte, 2026-10-07): the retargeted data is
played exactly as it comes - no re-centring, no heading alignment, no crossfade, no
stage box, no idle loop. A new clip may start somewhere else and facing elsewhere;
that is expected. Those layers come once the base system is proven.

What is NOT optional (CLAUDE.md, procedural invariants 6 and 9): this thread never
waits on the network, the disk or the retarget - clips arrive already retargeted
through a queue - and every pose packet is the full 421 floats with present=1 while
playing.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable

from procedural_animation.retarget import STREAMED_BONES, RetargetedMotion

from live_link_face_protocol import LiveLinkFaceEncoder, head_rotation_to_curves  # read-only
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder  # read-only

POSE_OSC_PORT = 9001         # conductor.py's defaults
LIVE_LINK_FACE_PORT = 11111


class Sender:
    """What conductor.py sends: the pose over OSC, the head over Live Link Face
    (neutral blendshapes). dry_run sends nothing but counts."""

    def __init__(self, pose_ip: str = "127.0.0.1", pose_port: int = POSE_OSC_PORT,
                 face_ip: str = "127.0.0.1", face_port: int = LIVE_LINK_FACE_PORT,
                 face: bool = True, dry_run: bool = False, rate: float = 60.0) -> None:
        self.pose = None if dry_run else LiveLinkPoseOSCEncoder(ip=pose_ip, port=pose_port)
        self.face_encoder = LiveLinkFaceEncoder(fps=int(round(rate)))
        self.face_socket = None if (dry_run or not face) else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.face_target = (face_ip, face_port)
        self.sent = 0

    def send(self, bones, head) -> None:
        if self.pose is not None:
            self.pose.send(bones, present=True)
        if self.face_socket is not None:
            self.face_socket.sendto(self.face_encoder.encode(head_rotation_to_curves(head)), self.face_target)
        self.sent += 1

    def send_lost(self, bones) -> None:
        """conductor.py's 'tracking lost' signal: the last pose, present=0."""
        if self.pose is not None and bones is not None:
            self.pose.send(bones, present=False)

    def close(self) -> None:
        if self.face_socket is not None:
            self.face_socket.close()
            self.face_socket = None


@dataclass
class Clip:
    motion: RetargetedMotion
    label: str


class LoopPlayer:
    def __init__(self, sender: Sender, rate: float = 60.0, speed: float = 1.0, loop: bool = True,
                 on_start: Callable[[Clip], None] | None = None) -> None:
        self.sender = sender
        self.period = 1.0 / rate
        self.speed = speed
        self.loop = loop
        self.on_start = on_start
        self._pending: queue.Queue[Clip] = queue.Queue()
        self._stop = threading.Event()
        self._skip = threading.Event()
        self.finished = threading.Event()    # loop=False: everything queued has played once
        self.current: Clip | None = None
        self.passes = 0                      # completed passes of the current clip
        self.max_late_s = 0.0                # worst tick lateness, for the timing check
        self._last_bones = None
        self._thread = threading.Thread(target=self._run, name="LoopPlayer", daemon=True)

    # -- control, from any thread -------------------------------------------
    def enqueue(self, motion: RetargetedMotion, label: str) -> None:
        self.finished.clear()
        self._pending.put(Clip(motion, label))

    def skip(self) -> None:
        """Cut to the next queued clip now instead of at the end of the pass."""
        self._skip.set()

    @property
    def queued(self) -> int:
        return self._pending.qsize()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        self.sender.send_lost(self._last_bones)
        self.sender.close()

    # -- the 60 Hz thread ------------------------------------------------------
    def _take_next(self, now: float) -> bool:
        try:
            clip = self._pending.get_nowait()
        except queue.Empty:
            return False
        self.current, self._clip_start, self.passes = clip, now, 0
        if self.on_start is not None:
            self.on_start(clip)
        return True

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if self.current is None:
                self._take_next(now)
            elif self._skip.is_set():
                self._skip.clear()
                self._take_next(now)

            if self.current is not None:
                t = (now - self._clip_start) * self.speed
                if t > self.current.motion.duration_s:
                    self.passes += 1
                    if self._take_next(now):
                        t = 0.0
                    elif self.loop:
                        self._clip_start, t = now, 0.0
                    else:
                        self.current = None
                        self.finished.set()
                if self.current is not None:
                    bones, head = self.current.motion.sample(t)
                    assert len(bones) == len(STREAMED_BONES) == 60
                    self._last_bones = bones
                    self.sender.send(bones, head)

            next_tick += self.period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                self.max_late_s = max(self.max_late_s, -delay)
                next_tick = time.perf_counter()  # fell behind: don't catch up in a burst
