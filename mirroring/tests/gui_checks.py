"""
Offline checks for gui.py's lane switch - no camera, no Unreal, no Spark, no window.

    ..\\venv\\Scripts\\python.exe tests\\gui_checks.py

Both lanes send to the same ports (OSC 9001, Live Link Face 11111), so the GUI must
never let them overlap. That rule lives in gui.LaneController; nothing else would
notice if a later change broke it - Unreal would just flicker between the webcam pose
and the generated one. These checks drive the real LaneController, the real
LoopPlayer and the real GenerationWorker against stand-ins that record every packet:

1. Handoff, no overlap: Tracking -> Prompt -> Tracking. Once the new lane has sent its
   first packet, the old one sends nothing more; each stopped lane's last packet is
   present=0.
2. Refusal: a webcam lane that will not stop (a hung camera thread) blocks the switch -
   the procedural player is never created and the lane stays Tracking.
3. In flight: a generation still running when the lane changes finishes into the cache
   but is never played; a request still waiting behind it is cancelled.
4. The handoff packets themselves (real UDP): the webcam's last pose with present=0,
   421 floats; a Live Link Face packet with neutral blendshapes and the held head.
5. Queue editing (LoopPlayer.clear_pending via LaneController.rearrange) and a player
   restart that carries the current and queued clips across, in order.
6. Generation-time text, stay in place.

Exit code 1 on any failure.
"""
from __future__ import annotations

import queue
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gui  # noqa: E402  (puts the module root on sys.path too)
from gui import (  # noqa: E402
    CANCELLED, FAILED, GENERATING, PROMPT, READY, TRACKING, WAITING, GenerationWorker, Item, LaneController,
    generation_time_text, held_in_place, send_tracking_handoff,
)
from live_link_face_protocol import CHANNEL_ORDER  # noqa: E402
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder  # noqa: E402
from pose_solver import BoneTransform  # noqa: E402
from procedural_animation.procedural_conductor import Lane  # noqa: E402
from procedural_animation.retarget import STREAMED_BONES, Retargeter  # noqa: E402
from procedural_animation.source_skeletons import load_soma77  # noqa: E402
from remote_kimodo_service.clip_cache import ClipCache  # noqa: E402
from remote_kimodo_service.fake_kimodo_server import FakeKimodo  # noqa: E402
from remote_kimodo_service.kimodo_client import KimodoClient  # noqa: E402
from remote_kimodo_service.kimodo_contract import GenerationRequest  # noqa: E402
from remote_kimodo_service.player import LoopPlayer, Sender  # noqa: E402

MODULE_ROOT = ROOT.parent
REAL_FIXTURE = MODULE_ROOT / "procedural_animation" / "tests" / "fixtures" / "kimodo_real_turn_around_2s.npz"


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, msg: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            self.failures += 1


# ---- stand-ins ------------------------------------------------------------------

class FakePipeline:
    """PipelineController's surface as LaneController uses it. While 'running' it
    appends a present=1 'tracking' packet to the shared log every 5 ms."""

    def __init__(self, log: list, refuse_stop: bool = False) -> None:
        self.log = log
        self.refuse_stop = refuse_stop
        self.conductor = None
        self.params = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_running(self) -> bool:
        return self.thread_alive

    def start(self, params) -> None:
        self._stop.clear()
        self.conductor = object()
        self.params = params

        def run() -> None:
            while not self._stop.is_set():
                self.log.append(("tracking", 1.0))
                time.sleep(0.005)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 0.3) -> bool:
        if self._thread is None:
            return True
        if self.refuse_stop:
            return False
        self._stop.set()
        self._thread.join(join_timeout)
        self._thread, self.conductor = None, None
        return True


class _OscLog:
    """Stands in for SimpleUDPClient inside a real Sender."""
    def __init__(self, log: list) -> None:
        self.log = log

    def send_message(self, _address: str, args: list) -> None:
        self.log.append(("prompt", float(args[0])))


def make_player_factory(log: list, created: list):
    def make() -> LoopPlayer:
        sender = Sender(face=False)
        sender.pose.client = _OscLog(log)
        player = LoopPlayer(sender)
        created.append(player)
        return player
    return make


def segments(log: list) -> list[tuple[str, list[float]]]:
    """[(tag, [present, ...]), ...] - consecutive packets of the same lane."""
    out: list[tuple[str, list[float]]] = []
    for tag, present in list(log):
        if out and out[-1][0] == tag:
            out[-1][1].append(present)
        else:
            out.append((tag, [present]))
    return out


# ---- 1 + 2 -----------------------------------------------------------------------

def check_handoff(motion, rep: Report) -> None:
    print("\n1. Handoff, no overlap")
    log: list = []
    created: list = []
    pipeline = FakePipeline(log)
    lanes = LaneController(pipeline, make_player_factory(log, created),
                           send_handoff=lambda _c: log.append(("tracking", 0.0)))
    pipeline.start({})
    time.sleep(0.1)
    err = lanes.to_prompt()
    rep.check(err is None and lanes.lane == PROMPT and lanes.tracking_was_running,
              f"Tracking -> Prompt ({err or 'ok'}); the webcam was running, so it will resume")
    rep.check(lanes.deliver(lanes.epoch, motion, "A"), "a clip of the current epoch reaches the player")
    time.sleep(0.25)
    err = lanes.to_tracking({})
    rep.check(err is None and lanes.lane == TRACKING and pipeline.thread_alive and lanes.player is None,
              f"Prompt -> Tracking ({err or 'ok'}); webcam lane restarted, player gone")
    time.sleep(0.1)
    pipeline.stop()

    segs = segments(log)
    tags = [t for t, _p in segs]
    rep.check(tags == ["tracking", "prompt", "tracking"],
              f"the senders never interleave: {' -> '.join(f'{t}x{len(p)}' for t, p in segs)}")
    if len(segs) == 3:
        rep.check(segs[0][1][-1] == 0.0 and all(p == 1.0 for p in segs[0][1][:-1]),
                  "the webcam lane's last packet before the switch is the present=0 handoff")
        rep.check(segs[1][1][-1] == 0.0 and all(p == 1.0 for p in segs[1][1][:-1]),
                  f"the player sends present=1 while playing ({len(segs[1][1]) - 1} packets), then present=0 on stop")
    rep.check(len(created) == 1 and not created[0]._thread.is_alive(), "the player thread has ended")

    print("\n2. Refusal")
    log2: list = []
    created2: list = []
    stuck = FakePipeline(log2, refuse_stop=True)
    lanes2 = LaneController(stuck, make_player_factory(log2, created2), send_handoff=lambda _c: None)
    stuck.start({})
    err = lanes2.to_prompt()
    rep.check(err is not None and lanes2.lane == TRACKING and not created2,
              f"a webcam thread that will not stop blocks the switch: {err}")
    stuck.refuse_stop = False
    stuck.stop()


# ---- 3 ---------------------------------------------------------------------------

def check_in_flight(rt: Retargeter, rep: Report) -> None:
    print("\n3. A generation in flight across a lane switch")
    log: list = []
    created: list = []
    events: queue.Queue = queue.Queue()
    lanes = LaneController(FakePipeline(log), make_player_factory(log, created), send_handoff=lambda _c: None)
    lanes.to_prompt()

    def wait_for(kind: str, item_id: int, timeout: float = 10.0):
        deadline = time.perf_counter() + timeout
        seen = []
        while time.perf_counter() < deadline:
            try:
                k, payload = events.get(timeout=0.1)
            except queue.Empty:
                continue
            seen.append((k, payload))
            pid = payload[0] if isinstance(payload, tuple) else payload
            if k == kind and pid == item_id:
                return payload, seen
        return None, seen

    with tempfile.TemporaryDirectory() as tmp, FakeKimodo(delay_s=1.0) as fake:
        cache = ClipCache(Path(tmp) / "cache")
        lane = Lane(KimodoClient(fake.url), cache, rt)
        worker = GenerationWorker(lambda: lane, lanes, lambda k, p=None: events.put((k, p)), lambda: False)
        req1 = GenerationRequest.build("A person waves", seed=0, seconds=2.0)
        req2 = GenerationRequest.build("A person jumps", seed=0, seconds=2.0)
        first = Item(id=1, title=req1.prompt, epoch=lanes.epoch, request=req1)
        second = Item(id=2, title=req2.prompt, epoch=lanes.epoch, request=req2)
        worker.submit(first)
        worker.submit(second)
        got, _ = wait_for("generating", 1)
        rep.check(got is not None, "the first request is generating (fake Spark, 1 s)")
        err = lanes.to_tracking(None)
        rep.check(err is None and lanes.lane == TRACKING, "switched to Tracking while it is in flight")
        fetched, _ = wait_for("fetched", 1)
        rep.check(fetched is not None and fetched[3] is False,
                  "the in-flight request finishes but is NOT delivered to a player")
        rep.check(len(cache.entries()) == 1, f"... and lands in the cache ({len(cache.entries())} clip)")
        cancelled, _ = wait_for("cancelled", 2)
        rep.check(cancelled is not None and fake.requests == 1,
                  f"the request waiting behind it is cancelled, never sent (Spark saw {fake.requests} request)")
        rep.check(not any(tag == "prompt" for tag, _p in log), "nothing was ever sent by the procedural lane")

        # Back in the procedural lane, the same prompt is a cache hit and plays.
        lanes.to_prompt()
        third = Item(id=3, title=req1.prompt, epoch=lanes.epoch, request=req1)
        worker.submit(third)
        fetched, _ = wait_for("fetched", 3)
        ok = fetched is not None and fetched[3] is True and fetched[2].get("source") == "cache"
        rep.check(ok and fake.requests == 1,
                  f"the same prompt later: a cache hit, delivered to the player "
                  f"({fetched[2] if fetched else 'no event'})")
        lanes.shutdown()


# ---- 4 ---------------------------------------------------------------------------

def _udp_listener() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2.0)
    return s


def check_handoff_packets(rep: Report) -> None:
    print("\n4. Handoff packets (real UDP)")
    from pythonosc.osc_message import OscMessage

    pose_rx, face_rx = _udp_listener(), _udp_listener()
    params = {"pose_ip": "127.0.0.1", "pose_port": pose_rx.getsockname()[1],
              "face_ip": "127.0.0.1", "face_port": face_rx.getsockname()[1]}
    bones = [BoneTransform(name=n, rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                           position={"x": float(k), "y": 0.0, "z": 0.0}) for k, n in enumerate(STREAMED_BONES)]
    head = {"headYaw": 0.2, "headPitch": -0.1, "headRoll": 0.05}
    own_encoder = LiveLinkPoseOSCEncoder(port=params["pose_port"])
    stub = SimpleNamespace(pose_smoother=SimpleNamespace(_state={b.name: b for b in bones}),
                           head_smoother=SimpleNamespace(_state=dict(head)),
                           pose_encoder=own_encoder)
    send_tracking_handoff(stub, params)
    try:
        args = OscMessage(pose_rx.recv(65536)).params
        face = face_rx.recv(65536)
    except socket.timeout:
        rep.check(False, "handoff packets received")
        return
    finally:
        pose_rx.close()
        face_rx.close()
    rep.check(len(args) == 421 and args[0] == 0.0 and args[1] == 0.0 and abs(args[1 + 7 * 59] - 59.0) < 1e-6,
              f"pose: {len(args)} floats, present={args[0]}, the last sent pose in send order")
    values = struct.unpack("!61f", face[-61 * 4:])
    by_name = dict(zip(CHANNEL_ORDER, values))
    head_ok = all(abs(by_name[k] - v) < 1e-6 for k, v in head.items())
    rest_zero = all(v == 0.0 for k, v in by_name.items() if k not in head)
    rep.check(head_ok and rest_zero, "face: the held head curves, every blendshape neutral (0)")
    rep.check(own_encoder.client._sock.fileno() == -1, "the stopped conductor's pose socket is closed")


# ---- 5 ---------------------------------------------------------------------------

def check_queue_edit(motion, rep: Report) -> None:
    print("\n5. Queue editing and player restart")
    log: list = []
    created: list = []
    lanes = LaneController(FakePipeline(log), make_player_factory(log, created), send_handoff=lambda _c: None)
    lanes.to_prompt()
    clips = {name: held_in_place(motion) for name in "ABCD"}   # distinct objects, as the GUI hands over
    for name in "ABCD":
        lanes.deliver(lanes.epoch, clips[name], name)
    time.sleep(0.1)
    player = lanes.player
    rep.check(player.current is not None and player.current.label == "A" and player.queued == 3,
              f"A plays, B C D queued ({player.queued})")
    lanes.rearrange(lambda pending: [pending[2], pending[0]])      # D first, C removed
    labels = [c.label for c in player.clear_pending()]
    rep.check(labels == ["D", "B"], f"rearrange: remove C, move D up -> {labels}")
    for name in "BC":
        lanes.deliver(lanes.epoch, clips[name], name)

    epoch = lanes.epoch
    err = lanes.restart_player(carry=True)
    time.sleep(0.1)
    new = lanes.player
    pending = [c.label for c in new.clear_pending()]
    rep.check(err is None and new is not player and not player._thread.is_alive() and lanes.epoch == epoch,
              "restart with new settings: old player stopped and replaced, same session")
    rep.check(new.current is not None and new.current.label == "A" and pending == ["B", "C"],
              f"... carrying the current clip (A, from its start) and the queue in order ({pending})")
    err = lanes.restart_player(carry=False)
    rep.check(err is None and lanes.epoch == epoch + 1 and lanes.player.current is None and not lanes.player.queued,
              "Stop playback: a fresh, empty player in a new session")
    rep.check(not lanes.deliver(epoch, clips["A"], "late"), "a clip from the old session is refused")
    lanes.shutdown()


# ---- 6 ---------------------------------------------------------------------------

def check_text_and_hold(motion, rep: Report) -> None:
    print("\n6. Generation-time text, stay in place")
    now = 100.0
    cases = [
        (Item(1, "t", 0, state=WAITING), 2, "waiting (2nd in line)"),
        (Item(1, "t", 0, state=GENERATING, started_at=96.8), None, "generating… 3.2 s"),
        (Item(1, "t", 0, state=READY, source="spark", gen_s=5.8, transfer_s=0.4, queue_s=0.0), None,
         "5.8 s gen + 0.4 s link"),
        (Item(1, "t", 0, state=READY, source="cache", gen_s=7.276, created="2026-10-07T17:41:50"), None,
         "7.3 s gen · cache 10-07 17:41"),
        (Item(1, "t", 0, state=READY, source="server cache", gen_s=2.9), None, "2.9 s gen · server cache"),
        (Item(1, "t", 0, state=READY, source="file"), None, "file"),
        (Item(1, "t", 0, state=READY, source="cache"), None, "gen time unknown · cache"),
        (Item(1, "t", 0, state=FAILED, error="Spark not reachable: refused"), None, "Spark not reachable: refused"),
        (Item(1, "t", 0, state=CANCELLED), None, "—"),
    ]
    bad = [(want, got) for item, pos, want in cases if (got := generation_time_text(item, now, pos)) != want]
    rep.check(not bad, f"every state reads as intended{'' if not bad else f': {bad}'}")

    held = held_in_place(motion)
    p0, p1 = motion.pelvis_pos, held.pelvis_pos
    rep.check(np.ptp(p1[:, :2], axis=0).max() == 0.0 and np.array_equal(p1[:, 2], p0[:, 2])
              and np.ptp(p0[:, :2], axis=0).max() > 1.0 and held.local_quats is motion.local_quats,
              f"stay in place: pelvis X/Y pinned to frame 0 (the clip travels "
              f"{np.ptp(p0[:, :2], axis=0).max():.0f} cm), height and rotations untouched, original unchanged")


def main() -> int:
    rt = Retargeter(load_soma77())
    lane = Lane(KimodoClient("http://127.0.0.1:1"), ClipCache(Path(tempfile.gettempdir()) / "unused"), rt)
    motion, _conv = lane.motion_from_npz(REAL_FIXTURE.read_bytes())
    rep = Report()
    check_handoff(motion, rep)
    check_in_flight(rt, rep)
    check_handoff_packets(rep)
    check_queue_edit(motion, rep)
    check_text_and_hold(motion, rep)
    print(f"\n{'ALL PASS' if not rep.failures else f'{rep.failures} FAILURE(S)'} (gui checks 1-6)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
