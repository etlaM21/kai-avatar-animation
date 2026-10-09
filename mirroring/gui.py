"""
gui.py - Tkinter control surface for both motion lanes.

Two tabs, and the tab IS the lane switch:

    Tracking   the webcam lane: drives an unmodified Conductor (conductor.py), exactly
               as this GUI always has.
    Prompt     the procedural lane: a front end to procedural_conductor's pieces
               (Lane, parse_prompt, LoopPlayer) - a prompt goes to the Spark (or the
               cache), is retargeted and played into Unreal.

Both lanes send to the same ports (OSC 9001, Live Link Face 11111), so only one may
ever send at a time. LaneController below owns that rule: the old sender is stopped
completely (thread joined, camera released, sockets closed) before the new one starts,
and a sender that will not stop blocks the switch instead of overlapping it.

This is still a WRAPPER. It reimplements no tracking, solving, retargeting or network
logic. The webcam lane is reached through:
    - live attribute assignment (conductor.face_smoother.alpha, etc.) for parameters
      that are safe to change on a running instance, and
    - constructor kwargs, for everything that must be baked in at startup (camera,
      model paths, ports/IPs, detection thresholds) - changing any of those tears the
      pipeline down and builds a fresh one.
Conductor owns the camera and calls cv2.imshow()/cv2.waitKey() itself, so getting
frames into a Tk widget and stopping the pipeline cleanly both go through a small set
of monkeypatches installed on the shared cv2 module before the first Conductor is
ever constructed - see install_cv2_patches(). conductor.py, pose_solver.py and the
mediapipe_*_capture.py modules are unmodified; `python conductor.py --debug --camera 1`
behaves exactly as before.

The procedural lane is reached through procedural_conductor.Lane.fetch_detailed() and
player.LoopPlayer.clear_pending(), two small hooks added for this GUI (each explained
where it is defined); the CLI behaves exactly as before.

Threads (CLAUDE.md, "gui.py"): Tk widgets are only ever touched on the main thread.
Everything else - the conductor loop, the 60 Hz player, generation, the Spark health
check, and every lane switch or stop that joins a thread - runs on its own thread and
reports back through one event queue the main thread drains via root.after.

Run with the module's venv interpreter, from this directory (the .task models are
looked up relative to it):
    ..\\venv\\Scripts\\python.exe gui.py
"""
from __future__ import annotations

import dataclasses
import os
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
from PIL import Image, ImageTk

# Patched below, before any Conductor is constructed - draw_landmarks is
# looked up as an attribute of this module object at call time, so patching
# it here reaches conductor.py's calls too without touching that file.
from mediapipe.tasks.python.vision import drawing_utils

from conductor import Conductor, LIVE_LINK_FACE_PORT, POSE_OSC_PORT
# Disabled, kept on purpose: see CLAUDE.md, "head_pose_capture.py - disabled".
# from head_pose_capture import HeadPoseCapture
from live_link_face_protocol import LiveLinkFaceEncoder  # read-only
from live_link_pose_osc_protocol import LiveLinkPoseOSCEncoder  # read-only
from mediapipe_holistic_capture import MediaPipeHolisticCapture

# The procedural lane lives beside mirroring/, at the module root (its own
# procedural_animation/__init__.py does the mirror image of this for mirroring/).
MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from procedural_animation.procedural_conductor import (  # noqa: E402
    DEFAULT_SEED, Fetched, Lane, load_bvh_motion, parse_prompt,
)
from procedural_animation.retarget import STREAMED_BONES, Retargeter, RetargetedMotion  # noqa: E402
from procedural_animation.source_skeletons import load_soma77  # noqa: E402
from remote_kimodo_service import kimodo_contract as contract  # noqa: E402
from remote_kimodo_service.clip_cache import DEFAULT_CACHE_DIR, CacheEntry, ClipCache  # noqa: E402
from remote_kimodo_service.kimodo_client import (  # noqa: E402
    DEFAULT_TIMEOUT_S, DEFAULT_URL, GenerationFailed, KimodoClient, SparkUnavailable,
)
from remote_kimodo_service.kimodo_contract import ContractError  # noqa: E402
from remote_kimodo_service.player import Clip, LoopPlayer, Sender  # noqa: E402

# ---- palette --------------------------------------------------------------
# Dark, technical, flat. Defined once here rather than scattered through the
# layout code below.
BG = "#1b1e23"
BG_PANEL = "#22262d"
BG_INPUT = "#14161a"
FG = "#c9d1d9"
FG_DIM = "#7d8590"
ACCENT = "#3fb950"
ACCENT_DIM = "#2a4a30"
WARN = "#d29922"
ERROR = "#f85149"
BORDER = "#30363d"
FONT_UI = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)
FONT_MONO_BOLD = ("Consolas", 9, "bold")
FONT_HEADER = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 11, "bold")

DEFAULT_HOLISTIC_MODEL = "holistic_landmarker.task"

# Smoothing sliders run 0 (raw) .. SMOOTHING_MAX; the smoothers take an EMA alpha,
# which runs the other way. Never 1.0: that is alpha 0, a channel that never moves.
SMOOTHING_MAX = 0.99


def smoothing_to_alpha(smoothing: float) -> float:
    return 1.0 - min(max(smoothing, 0.0), SMOOTHING_MAX)
# DEFAULT_HEAD_POSE_MODEL = "face_landmarker.task"  # disabled - see CLAUDE.md


# Prompt-tab defaults. Each one is the procedural CLI's default, so the GUI and
# `procedural_conductor` behave alike - except loop, which is the REPL's (it always
# loops); the CLI's one-shot mode plays once.
DEFAULT_LOOP = True
DEFAULT_SPEED = 1.0
DEFAULT_RATE = 60.0
DEFAULT_SPARK_URL = os.environ.get("KIMODO_URL", DEFAULT_URL)
DEFAULT_SPARK_TIMEOUT_S = float(os.environ.get("KIMODO_TIMEOUT", DEFAULT_TIMEOUT_S))
DEFAULT_CACHE = Path(os.environ.get("KIMODO_CACHE_DIR", DEFAULT_CACHE_DIR))


# ---- cv2 / drawing_utils monkeypatches -------------------------------------
#
# Conductor is always constructed with show_debug=True (see PipelineController
# below) so it produces an annotated frame every loop - if show_debug were
# False it would never call _draw_debug at all and the preview pane would go
# black. These patches intercept the cv2 calls that _draw_debug and __init__
# make instead of letting them open a real window.
_real_draw_landmarks = drawing_utils.draw_landmarks


def install_cv2_patches(frame_queue: "queue.Queue[object]", stop_event: threading.Event) -> None:
    """Patches the shared cv2 module so Conductor's window/imshow/waitKey
    calls redirect into this GUI instead of opening an OS window.

    Must run before the first Conductor() is constructed - its __init__
    already calls cv2.namedWindow/resizeWindow when show_debug=True.
    """

    def _no_op(*_args, **_kwargs) -> None:
        return None

    def _patched_imshow(_window_name, frame) -> None:
        # maxsize=1, drop-when-full: the GUI must never apply backpressure
        # to the capture loop. Keep the newest frame, not the oldest.
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _patched_wait_key(_delay: int = 1) -> int:
        # Conductor.run()'s ONLY external exit path is `key == 27` (Esc)
        # returned from this call - there is no separate stop flag it
        # checks. Rather than always returning -1 (which would make the
        # pipeline unstoppable from here), report Esc once stop_event is
        # set, so the wrapper reuses Conductor's own existing break path
        # instead of inventing a new one.
        return 27 if stop_event.is_set() else -1

    cv2.namedWindow = _no_op
    cv2.resizeWindow = _no_op
    cv2.imshow = _patched_imshow
    cv2.waitKey = _patched_wait_key


def set_skeleton_overlay_enabled(enabled: bool) -> None:
    """Toggles the expensive per-frame landmark/mesh drawing that
    _draw_debug does. Skeleton drawing on a 4K frame is genuinely costly -
    this is a real performance control, not just cosmetic. The cv2.putText
    status text in _draw_debug is untouched (cheap, left on always)."""
    drawing_utils.draw_landmarks = _real_draw_landmarks if enabled else (lambda *a, **k: None)


# ---- webcam lane lifecycle ----------------------------------------------------

class PipelineController:
    """Owns construction, threading and teardown of one Conductor instance.

    Restart-required parameters are only ever applied by tearing the whole
    thing down and building a fresh Conductor (and fresh capture objects,
    for simplicity/robustness - construction cost is only paid on an
    explicit Apply & Restart). Live parameters are applied by assigning
    straight onto the attributes of the currently running `conductor`.
    """

    def __init__(self, frame_queue: "queue.Queue[object]") -> None:
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()
        self.conductor: Conductor | None = None
        self.params: dict | None = None      # what the current/last Conductor was built with
        self._thread: threading.Thread | None = None
        self._construction_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self.conductor is not None and self._thread is not None and self._thread.is_alive()

    @property
    def thread_alive(self) -> bool:
        """Running OR still constructing (model load, camera open) - either way it may
        send, so a lane switch must stop it."""
        return self._thread is not None and self._thread.is_alive()

    def take_construction_error(self) -> str | None:
        err, self._construction_error = self._construction_error, None
        return err

    def start(self, params: dict) -> None:
        if self.is_running:
            raise RuntimeError("pipeline already running")
        self.stop_event.clear()
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.conductor = None
        self.params = params
        self._construction_error = None
        self._thread = threading.Thread(target=self._run, args=(params,), name="conductor", daemon=True)
        self._thread.start()

    def _run(self, params: dict) -> None:
        try:
            holistic_capture = MediaPipeHolisticCapture(
                model_path=params["holistic_model"],
                min_face_detection_confidence=params["h_min_face_detection"],
                min_face_landmarks_confidence=params["h_min_face_landmarks"],
                min_pose_detection_confidence=params["h_min_pose_detection"],
                min_pose_landmarks_confidence=params["h_min_pose_landmarks"],
                min_hand_landmarks_confidence=params["h_min_hand_landmarks"],
            )
            # head_pose_capture is disabled - see CLAUDE.md.
            # head_pose_capture = HeadPoseCapture(
            #     model_path=params["head_pose_model"],
            #     min_face_detection_confidence=params["hp_min_face_detection"],
            #     min_face_presence_confidence=params["hp_min_face_presence"],
            #     min_tracking_confidence=params["hp_min_tracking"],
            # )
            conductor = Conductor(
                holistic_capture,
                # head_pose_capture,
                camera_index=params["camera_index"],
                camera_width=params["camera_width"],
                camera_height=params["camera_height"],
                torso_lean_offset_deg=params["torso_lean_offset_deg"],
                face_ip=params["face_ip"],
                face_port=params["face_port"],
                pose_ip=params["pose_ip"],
                pose_port=params["pose_port"],
                face_smoothing_alpha=params["face_smoothing_alpha"],
                pose_smoothing_alpha=params["pose_smoothing_alpha"],
                show_debug=True,  # must stay True - see module docstring
            )
        except Exception as exc:  # noqa: BLE001 - reported to the GUI thread, not swallowed
            self._construction_error = str(exc)
            return

        self.conductor = conductor
        conductor.run()  # blocks until stop_event drives waitKey to return Esc

    def stop(self, join_timeout: float = 2.0) -> bool:
        """Returns True if the pipeline thread actually stopped in time. Blocks for up
        to join_timeout: never call it on the Tk thread."""
        if self._thread is None:
            return True
        self.stop_event.set()
        self._thread.join(timeout=join_timeout)
        stopped = not self._thread.is_alive()
        if stopped:
            self.conductor = None
            self._thread = None
        return stopped


def _close_osc_client(client) -> None:
    """python-osc's SimpleUDPClient has no close() and keeps its socket private; without
    this the socket lives until garbage collection. Harmless if the attribute moves."""
    sock = getattr(client, "_sock", None)
    if sock is not None:
        sock.close()


def send_tracking_handoff(conductor: Conductor, params: dict) -> None:
    """Called once the webcam lane has stopped, before the procedural lane starts.

    Conductor.run() sends nothing when it stops (it only sends present=0 while running
    and tracking is lost), so without this Unreal would keep the last webcam pose
    flagged present=1 until the first generated clip arrives - and the MetaHuman's face
    would keep the last webcam expression. This sends the same "tracking lost" signal
    the player sends on its own stop:
      - the pose the conductor last sent (all 60 bones, from its pose smoother's state,
        in send order), once more with present=0;
      - one Live Link Face packet with neutral blendshapes that keeps the head curves
        the conductor last sent, so the head does not snap while the body holds.
    Reads two private attributes of the stopped conductor (pose_smoother._state,
    head_smoother._state) - the same kind of read-only reach-in as the calibration
    button's conductor._last_valid_world_landmarks. Also closes the conductor's pose
    socket, which Conductor.run() leaves open."""
    bones = list(conductor.pose_smoother._state.values())
    if len(bones) == len(STREAMED_BONES):          # empty if no frame was ever processed
        encoder = LiveLinkPoseOSCEncoder(ip=params["pose_ip"], port=params["pose_port"])
        try:
            encoder.send(bones, present=False)
        finally:
            _close_osc_client(encoder.client)
    head = dict(conductor.head_smoother._state)    # headYaw/Pitch/Roll as last sent
    if head:
        packet = LiveLinkFaceEncoder().encode(head)  # every other channel -> 0.0 = neutral
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(packet, (params["face_ip"], params["face_port"]))
    _close_osc_client(conductor.pose_encoder.client)


# ---- procedural lane: the clips the Prompt tab shows -------------------------

# Item states. Queue: waiting -> generating -> ready -> playing. Then history.
WAITING, GENERATING, READY, PLAYING = "waiting", "generating", "ready", "playing"
PLAYED, NOT_PLAYED, FAILED, CANCELLED = "played", "not played", "failed", "cancelled"
QUEUE_STATES = (WAITING, GENERATING, READY)
# Events can arrive out of order (the 60 Hz thread may start a clip before the
# generation thread's "ready" is processed): a state only ever moves forward.
_STATE_RANK = {WAITING: 0, GENERATING: 1, READY: 2, PLAYING: 3,
               PLAYED: 4, NOT_PLAYED: 4, FAILED: 4, CANCELLED: 4}


@dataclass(eq=False)
class Item:
    """One requested clip, from the moment it is typed to the end of the session.
    Owned by the main thread; the worker only reads request/file/epoch/cancelled."""
    id: int
    title: str
    epoch: int                          # LaneController.epoch it was queued in
    request: contract.GenerationRequest | None = None
    file: Path | None = None            # a BVH / NPZ played offline (Library, Play file...)
    details: str = ""
    state: str = WAITING
    note: str = ""                      # shown instead of the bare state, e.g. "not played - cached"
    started_at: float | None = None     # perf_counter when generation started
    source: str = ""                    # "spark", "server cache", "cache", "file"
    gen_s: float | None = None          # GPU time on the Spark (original, for cached clips)
    transfer_s: float | None = None     # fresh requests only
    queue_s: float | None = None
    created: str | None = None          # when the Spark generated it (meta "created")
    error: str = ""
    held: bool = False                  # played with "stay in place"
    motion: RetargetedMotion | None = None   # the copy handed to the player; freed in history
    cancelled: bool = False             # set on the main thread, read by the worker

    def advance(self, state: str) -> bool:
        if _STATE_RANK[state] < _STATE_RANK[self.state]:
            return False
        self.state = state
        return True


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _short_created(created: str | None) -> str:
    """'2026-10-07T17:41:50' -> '10-07 17:41'."""
    if not created or len(created) < 16:
        return ""
    return f"{created[5:10]} {created[11:16]}"


def generation_time_text(item: Item, now: float, line_pos: int | None = None) -> str:
    """The generation-time column. Cached clips show the ORIGINAL generation time,
    which the Spark wrote into the NPZ's meta_json - no cache change was needed."""
    if item.state == WAITING:
        return f"waiting ({_ordinal(line_pos)} in line)" if line_pos else "waiting"
    if item.state == GENERATING:
        return f"generating… {now - (item.started_at or now):.1f} s"
    if item.state == FAILED:
        return item.error
    if not item.source:
        return "—"
    if item.source == "file":
        return "file"
    if item.gen_s is None:
        return f"gen time unknown · {item.source}"
    text = f"{item.gen_s:.1f} s gen"
    if item.source == "spark":
        if item.transfer_s is not None:
            text += f" + {item.transfer_s:.1f} s link"
        if item.queue_s and item.queue_s >= 0.05:
            text += f" + {item.queue_s:.1f} s queued"
        return text
    stamp = _short_created(item.created)
    return f"{text} · {item.source}{' ' + stamp if stamp else ''}"


def timing_from_fetched(f: Fetched) -> dict:
    """Item fields from what Lane.fetch_detailed() returned."""
    r, meta = f.result, f.meta
    if r is None:
        return dict(source="cache", gen_s=meta.get("generation_s"), created=meta.get("created"))
    if r.server_cache:
        # The Spark's own result cache answered (a retry after a client timeout): its
        # header says 0 s, the NPZ still carries the real generation time.
        return dict(source="server cache", gen_s=meta.get("generation_s"), created=meta.get("created"))
    return dict(source="spark", gen_s=r.generation_s if r.generation_s is not None else meta.get("generation_s"),
                transfer_s=r.transfer_s, queue_s=r.queue_s, created=meta.get("created"))


def held_in_place(motion: RetargetedMotion) -> RetargetedMotion:
    """'Stay in place': a copy with no horizontal root motion - the pelvis keeps its
    first-frame X/Y, its height is untouched (that keeps the feet on the floor), so a
    travelling clip loops without jumping back. The same rule as hold_in_place() in the
    frozen procedural_conductor_bvh.py (`--root in-place`), not imported from there
    because that copy must not grow dependants. Rotations are untouched, so the head
    channel is unaffected. A playback layer beyond v1's 'plain' - approved by Malte
    (2026-10-09) as an option, off by default."""
    pelvis = motion.pelvis_pos.copy()
    pelvis[:, :2] = pelvis[0, :2]
    return dataclasses.replace(motion, pelvis_pos=pelvis)


# ---- procedural lane: the lane switch, generation and health threads -----------
# Tk-free on purpose: tests/gui_checks.py drives these with stub senders.

TRACKING, PROMPT, SWITCHING = "tracking", "prompt", "switching"


class LaneController:
    """The one place that decides who sends to 9001/11111.

    lane is TRACKING, PROMPT or SWITCHING. A switch stops the old sender completely
    (thread joined) before the new one is created; if the old one will not stop, the
    switch is refused and the old lane stays. epoch numbers a playback session: every
    switch and every Stop bumps it, and a clip only reaches the player if it was queued
    in the current epoch - which is how a generation still in flight when the lane
    changes finishes into the cache but is never played.

    switch/stop methods block (they join threads): call them from a background thread,
    never the Tk thread. `lock` guards (lane, player, epoch); it is only ever held for
    a few statements, never across a join."""

    def __init__(self, pipeline, make_player: Callable[[], LoopPlayer],
                 send_handoff: Callable[[object], None]) -> None:
        self.pipeline = pipeline               # PipelineController (or a test stub)
        self.make_player = make_player
        self.send_handoff = send_handoff       # send_tracking_handoff, bound to the params
        self.lock = threading.Lock()
        self.lane = TRACKING
        self.player: LoopPlayer | None = None
        self.epoch = 0
        self.tracking_was_running = False      # restart the webcam on return to Tracking?

    # -- switches ----------------------------------------------------------
    def to_prompt(self) -> str | None:
        """Tracking -> Prompt. Returns None, or a one-line reason the switch was refused."""
        with self.lock:
            if self.lane != TRACKING:
                return f"cannot switch to the procedural lane while {self.lane}"
            self.lane = SWITCHING
        was_running = self.pipeline.thread_alive
        conductor = self.pipeline.conductor
        if not self.pipeline.stop():
            with self.lock:
                self.lane = TRACKING
            return "the webcam thread did not stop within 2 s (camera hung?) - procedural lane not started"
        self.tracking_was_running = was_running
        handoff_error = None
        if conductor is not None:
            try:
                self.send_handoff(conductor)
            except OSError as e:
                handoff_error = f"handoff packet not sent ({e})"
        try:
            player = self.make_player()
            player.start()
        except Exception as e:  # noqa: BLE001 - bad IP/port: report, never a traceback
            with self.lock:
                self.lane = TRACKING
            return f"procedural player could not start: {type(e).__name__}: {e}"
        with self.lock:
            self.player = player
            self.epoch += 1
            self.lane = PROMPT
        return handoff_error

    def to_tracking(self, params: dict | None) -> str | None:
        """Prompt -> Tracking. params: restart the webcam lane with these, or None to
        leave it stopped. Returns None, or a one-line reason the switch was refused."""
        with self.lock:
            if self.lane != PROMPT:
                return f"cannot switch to the webcam lane while {self.lane}"
            self.lane = SWITCHING
            self.epoch += 1
            player = self.player
        if player is not None and not self._stop_player(player):
            with self.lock:
                self.lane = PROMPT
            return "the procedural player did not stop within 2 s - webcam lane not started"
        with self.lock:
            self.player = None
            self.lane = TRACKING
        if params is not None:
            self.pipeline.start(params)
        return None

    @staticmethod
    def _stop_player(player: LoopPlayer) -> bool:
        """LoopPlayer.stop() joins with a timeout but does not say whether the thread
        ended; its thread is private, so this reads it (a stop that silently failed
        would mean two senders)."""
        player.stop()
        return not player._thread.is_alive()

    # -- inside the procedural lane --------------------------------------------
    def restart_player(self, carry: bool) -> str | None:
        """Replace the player (new send settings, or Stop playback). carry=True hands
        the current clip (from its start) and the queued ones to the new player;
        carry=False is Stop: everything stops, and the epoch moves on so a clip still
        generating will not start playing afterwards."""
        with self.lock:
            if self.lane != PROMPT or self.player is None:
                return "the procedural lane is not active"
            old = self.player
            self.lane = SWITCHING
            if not carry:
                self.epoch += 1
        clips = []
        if carry:
            # Drain FIRST: with nothing pending the 60 Hz thread cannot take a new clip,
            # so `current` is stable from here on. Read the other way round, a clip taken
            # in between would be in neither list - lost.
            pending = old.clear_pending()
            current = old.current
            clips = ([current] if current is not None else []) + pending
        if not self._stop_player(old):
            with self.lock:
                self.lane = PROMPT
            return "the procedural player did not stop within 2 s"
        try:
            new = self.make_player()
            for clip in clips:
                new.enqueue(clip.motion, clip.label)
            new.start()
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.player = None
                self.lane = PROMPT
            return f"procedural player could not start: {type(e).__name__}: {e}"
        with self.lock:
            self.player = new
            self.lane = PROMPT
        return None

    def deliver(self, epoch: int, motion: RetargetedMotion, label: str) -> bool:
        """From the generation thread: play this clip if its session is still current."""
        with self.lock:
            if self.lane != PROMPT or self.player is None or epoch != self.epoch:
                return False
            self.player.enqueue(motion, label)
            return True

    def rearrange(self, arrange: Callable[[list[Clip]], list[Clip]]) -> None:
        """Remove / reorder clips already handed to the player (LoopPlayer.clear_pending,
        then enqueue the result). If a pass ends in the instant between the two, the
        current clip just plays one more pass (or, with loop off, the next clip starts a
        tick later) - nothing is lost or played twice."""
        with self.lock:
            if self.player is None:
                return
            for clip in arrange(self.player.clear_pending()):
                self.player.enqueue(clip.motion, clip.label)

    def shutdown(self) -> None:
        """Window close: stop whichever lane is sending."""
        with self.lock:
            player, self.player = self.player, None
            self.lane = SWITCHING
            self.epoch += 1
        if player is not None:
            self._stop_player(player)
        self.pipeline.stop()


class GenerationWorker:
    """procedural_conductor's `generation` thread, for the GUI: ONE request in flight,
    FIFO. cache -> HTTP -> validate -> retarget -> cache write (all inside
    Lane.fetch_detailed) -> LaneController.deliver. Every outcome is posted as an event;
    nothing here ever raises out of the thread or touches Tk."""

    def __init__(self, get_lane: Callable[[], Lane], lanes: LaneController,
                 post: Callable[[str, object], None], stay_in_place: Callable[[], bool]) -> None:
        self.get_lane = get_lane
        self.lanes = lanes
        self.post = post
        self.stay_in_place = stay_in_place
        self._jobs: queue.Queue[Item | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="generation", daemon=True)
        self._thread.start()

    def submit(self, item: Item) -> None:
        self._jobs.put(item)

    def _run(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            self._process(item)

    def _process(self, item: Item) -> None:
        if item.cancelled or item.epoch != self.lanes.epoch:
            self.post("cancelled", item.id)
            return
        self.post("generating", (item.id, time.perf_counter()))
        lane = self.get_lane()
        try:
            if item.request is not None:
                fetched = lane.fetch_detailed(item.request)
                motion, timing = fetched.motion, timing_from_fetched(fetched)
            elif item.file.suffix.lower() == ".bvh":
                motion, timing = load_bvh_motion(item.file, lane.retargeter), dict(source="file")
            else:  # a cached NPZ from the Library, or one picked by hand
                data = item.file.read_bytes()
                motion, _conv = lane.motion_from_npz(data)
                meta = contract.unpack(data).meta
                timing = dict(source="cache", gen_s=meta.get("generation_s"), created=meta.get("created"))
        except SparkUnavailable as e:
            self.post("failed", (item.id, f"Spark not reachable: {e}"))
            self.post("spark_down", str(e))
            return
        except (GenerationFailed, ContractError, ValueError, OSError) as e:
            self.post("failed", (item.id, str(e)))
            return
        except Exception as e:  # noqa: BLE001 - one line in the GUI, never a dead worker
            self.post("failed", (item.id, f"{type(e).__name__}: {e}"))
            return
        if self.stay_in_place():
            motion = held_in_place(motion)
            timing["held"] = True
        delivered = not item.cancelled and self.lanes.deliver(item.epoch, motion, item.title)
        self.post("fetched", (item.id, motion, timing, delivered))


class HealthPoller:
    """The Spark indicator: KimodoClient.health() every INTERVAL_S while `active` is
    set (the Prompt tab is showing), on its own thread. health() has its own 3 s
    timeout - longer than the 2.05 s Windows takes to report a refused localhost
    connect - so a stopped tunnel reads as "refused", not as a timeout."""

    INTERVAL_S = 5.0

    def __init__(self, get_client: Callable[[], KimodoClient], post: Callable[[str, object], None]) -> None:
        self.get_client = get_client
        self.post = post
        self.active = threading.Event()
        self._poke = threading.Event()
        threading.Thread(target=self._run, name="health", daemon=True).start()

    def poke(self) -> None:
        self._poke.set()

    def _run(self) -> None:
        while True:
            self.active.wait()
            client = self.get_client()
            try:
                self.post("health", (True, client.health(), client.base_url))
            except (SparkUnavailable, GenerationFailed) as e:
                self.post("health", (False, str(e), client.base_url))
            self._poke.wait(self.INTERVAL_S)
            self._poke.clear()


# ---- small reusable widgets -------------------------------------------------

class CollapsibleSection(ttk.Frame):
    """A LabelFrame-like section that can be expanded/collapsed by clicking
    its header. Used for Advanced, which is collapsed by default."""

    def __init__(self, parent: tk.Widget, title: str, *, start_expanded: bool = False) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self._expanded = tk.BooleanVar(value=start_expanded)

        header = ttk.Frame(self, style="Panel.TFrame")
        header.pack(fill="x")
        self._toggle_btn = ttk.Label(
            header, text=self._arrow(), style="Header.TLabel", cursor="hand2", width=2,
        )
        self._toggle_btn.pack(side="left")
        title_lbl = ttk.Label(header, text=title, style="Header.TLabel", cursor="hand2")
        title_lbl.pack(side="left")
        for widget in (header, self._toggle_btn, title_lbl):
            widget.bind("<Button-1>", self._toggle)

        self.body = ttk.Frame(self, style="Panel.TFrame")
        if start_expanded:
            self.body.pack(fill="x", pady=(4, 0))

    def _arrow(self) -> str:
        return "▾" if self._expanded.get() else "▸"

    def _toggle(self, _event=None) -> None:
        self._expanded.set(not self._expanded.get())
        self._toggle_btn.configure(text=self._arrow())
        if self._expanded.get():
            self.body.pack(fill="x", pady=(4, 0))
        else:
            self.body.forget()


class LabeledSlider(ttk.Frame):
    """A titled 0..1-ish slider with a monospace live numeric readout,
    applying changes immediately via `on_change`."""

    def __init__(self, parent, label: str, *, from_: float, to: float, initial: float,
                 on_change, resolution: float = 0.01) -> None:
        super().__init__(parent, style="Panel.TFrame")
        self._on_change = on_change
        top = ttk.Frame(self, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="Body.TLabel").pack(side="left")
        self.value_lbl = ttk.Label(top, text=f"{initial:.2f}", style="Mono.TLabel", width=5, anchor="e")
        self.value_lbl.pack(side="right")

        self.var = tk.DoubleVar(value=initial)
        self.scale = ttk.Scale(
            self, from_=from_, to=to, orient="horizontal", variable=self.var,
            style="Accent.Horizontal.TScale", command=self._handle,
        )
        self.scale.pack(fill="x", pady=(2, 6))
        self._resolution = resolution

    def _handle(self, raw_value: str) -> None:
        value = round(float(raw_value) / self._resolution) * self._resolution
        self.value_lbl.configure(text=f"{value:.2f}")
        self._on_change(value)

    def set(self, value: float) -> None:
        self.var.set(value)
        self.value_lbl.configure(text=f"{value:.2f}")


def make_scrollable(parent: tk.Widget) -> ttk.Frame:
    """Wraps a vertically-scrolling Canvas+Scrollbar around a plain ttk.Frame and
    returns that inner frame - callers pack/grid their own widgets into it exactly as
    if it were unscrolled. Needed because Advanced, once expanded, can be taller than
    the window. The caller grids the returned frame's container at row 0."""
    container = ttk.Frame(parent, style="Panel.TFrame")
    container.grid(row=0, column=0, sticky="nsew")

    canvas = tk.Canvas(container, bg=BG_PANEL, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas, style="Panel.TFrame")
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _sync_scrollregion(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_inner_width(event) -> None:
        canvas.itemconfig(inner_id, width=event.width)

    inner.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_inner_width)

    def _on_mousewheel(event) -> None:
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_wheel(_event) -> None:
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_wheel(_event) -> None:
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", _bind_wheel)
    canvas.bind("<Leave>", _unbind_wheel)

    return inner


def sync_tree(tree: ttk.Treeview, rows: list[tuple[str, tuple, tuple]]) -> None:
    """Make the Treeview show exactly `rows` (iid, values, tags), in order, updating in
    place so the selection survives the 5 Hz refresh."""
    wanted = {iid for iid, _v, _t in rows}
    for iid in tree.get_children():
        if iid not in wanted:
            tree.delete(iid)
    for index, (iid, values, tags) in enumerate(rows):
        if tree.exists(iid):
            tree.item(iid, values=values, tags=tags)
            if tree.index(iid) != index:
                tree.move(iid, "", index)
        else:
            tree.insert("", index, iid=iid, values=values, tags=tags)


# ---- main application --------------------------------------------------------

class App:
    FPS_WINDOW = 30
    MIN_WIDTH = 900
    MIN_HEIGHT = 240
    INITIAL_WIDTH = 1150
    # Assumed until the first frame reveals the real capture aspect.
    DEFAULT_ASPECT = 16 / 9
    # Room left on screen for the title bar and taskbar when a tall (4:3) capture
    # would push the window past the bottom edge.
    SCREEN_MARGIN_PX = 80
    POLL_MS = 33
    # The Prompt tab's lists need more room than a 16:9 video fit leaves at the default
    # width (~470 px measured, one visible queue row). Its own height; Tracking refits.
    PROMPT_MIN_HEIGHT = 760
    PROMPT_REFRESH_EVERY = 6     # ticks: the Prompt tab's counters/progress at ~5 Hz

    def __init__(self) -> None:
        self.frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self.controller = PipelineController(self.frame_queue)
        install_cv2_patches(self.frame_queue, self.controller.stop_event)
        # Everything a background thread wants the UI to know arrives here.
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Mocap Conductor")
        self.root.configure(bg=BG)
        # The HEIGHT is derived, not chosen: it is whatever makes the video pane the
        # capture's own shape at the current width (_fit_height_to_video), so a 4:3
        # camera gets a taller window than a 16:9 one and neither is letterboxed.
        # Only the width is the user's to drag; a free height would just bring the
        # bars back. The Prompt tab lives in the same window height.
        self.root.geometry(f"{self.INITIAL_WIDTH}x700")
        self.root.minsize(self.MIN_WIDTH, 1)
        self.root.resizable(True, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_style()

        self._frame_times: deque[float] = deque(maxlen=self.FPS_WINDOW)
        self._capture_aspect: float | None = None
        # What the window was last fitted for, so a fit only runs when the capture
        # aspect or the window width actually changed.
        self._fitted_aspect: float | None = None
        self._fitted_width: int | None = None
        self._fit_pending: str | None = None
        self._last_frame = None  # redrawn when the pane resizes between frames
        self._photo_image: ImageTk.PhotoImage | None = None  # keep a reference alive
        self._skeleton_enabled = tk.BooleanVar(value=True)
        self._fps_enabled = tk.BooleanVar(value=True)
        self._dirty_while_running = False
        self._calibration_applied = False  # so one finished calibration writes the spinbox once
        self._busy = False                 # a lane switch / stop / restart is running in the background
        self._closing = False
        self._ignore_tab_event = False
        self._tick = 0

        # ---- procedural lane --------------------------------------------------
        self._retargeter = Retargeter(load_soma77())
        self._lane = Lane(KimodoClient(DEFAULT_SPARK_URL, timeout=DEFAULT_SPARK_TIMEOUT_S),
                          ClipCache(DEFAULT_CACHE), self._retargeter)
        self._player_settings: dict = {}
        self.lanes = LaneController(self.controller, self._make_player, self._send_handoff)
        self.items: dict[int, Item] = {}
        self._next_item_id = 1
        self.queue_ids: list[int] = []     # waiting/generating/ready, in play order
        self.history_ids: list[int] = []   # newest first
        self.playing_id: int | None = None
        self._playing_since: float = 0.0
        self._playing_speed: float = DEFAULT_SPEED
        self._end_signalled = False        # loop off: present=0 sent after the last clip
        self._early_start: tuple | None = None   # a "started" event that overtook its "fetched"
        self._library: list[CacheEntry] = []
        self._library_loading = False
        # Read by threads (the worker, the player's on_start): plain floats/bools,
        # written only by the main thread.
        self._stay_in_place = False
        self._speed_now = DEFAULT_SPEED

        self._build_layout()
        set_skeleton_overlay_enabled(self._skeleton_enabled.get())

        self.worker = GenerationWorker(lambda: self._lane, self.lanes, self._post, lambda: self._stay_in_place)
        self.health = HealthPoller(lambda: self._lane.client, self._post)

        self.root.bind("<Configure>", self._on_root_configure)
        self.root.after_idle(self._fit_height_to_video)
        self.root.after(self.POLL_MS, self._poll)

    def _post(self, kind: str, payload: object = None) -> None:
        """Thread-safe: the only way a background thread talks to the UI."""
        self.events.put((kind, payload))

    def _in_background(self, name: str, fn: Callable[[], object], done: Callable[[object], None]) -> None:
        """Run a blocking call (thread joins, mostly) off the Tk thread; `done(result)`
        runs on the Tk thread afterwards."""
        def run() -> None:
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001 - one line, never a traceback
                result = f"{type(e).__name__}: {e}"
            self._post("call", (done, result))
        threading.Thread(target=run, name=name, daemon=True).start()

    # ---- style --------------------------------------------------------

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Body.TLabel", background=BG_PANEL, foreground=FG, font=FONT_UI)
        style.configure("Dim.TLabel", background=BG_PANEL, foreground=FG_DIM, font=FONT_UI)
        style.configure("Header.TLabel", background=BG_PANEL, foreground=FG, font=FONT_HEADER)
        style.configure("Mono.TLabel", background=BG_PANEL, foreground=ACCENT, font=FONT_MONO_BOLD)
        style.configure("Status.TLabel", background=BG, foreground=FG, font=FONT_MONO)
        style.configure(
            "TLabelframe", background=BG_PANEL, foreground=FG, bordercolor=BORDER,
        )
        style.configure(
            "TLabelframe.Label", background=BG_PANEL, foreground=FG, font=FONT_HEADER,
        )
        style.configure(
            "TButton", background=BG_INPUT, foreground=FG, bordercolor=BORDER,
            focusthickness=0, padding=6, font=FONT_UI,
        )
        style.map("TButton", background=[("active", BORDER), ("disabled", BG_PANEL)],
                  foreground=[("disabled", FG_DIM)])
        style.configure("Small.TButton", padding=(6, 2))
        style.configure(
            "Accent.TButton", background=ACCENT_DIM, foreground=ACCENT, padding=6, font=FONT_HEADER,
        )
        style.map("Accent.TButton", background=[("active", ACCENT), ("disabled", BG_PANEL)],
                  foreground=[("active", BG), ("disabled", FG_DIM)])
        style.configure(
            "TEntry", fieldbackground=BG_INPUT, foreground=FG, insertcolor=FG, bordercolor=BORDER,
        )
        style.map("TEntry", fieldbackground=[("disabled", BG_PANEL)], foreground=[("disabled", FG_DIM)])
        style.configure(
            "TSpinbox", fieldbackground=BG_INPUT, foreground=FG, insertcolor=FG,
            bordercolor=BORDER, arrowcolor=FG,
        )
        style.configure("TCheckbutton", background=BG_PANEL, foreground=FG, font=FONT_UI)
        style.map("TCheckbutton", background=[("active", BG_PANEL)])
        style.configure(
            "Accent.Horizontal.TScale", background=BG_PANEL, troughcolor=BG_INPUT,
        )
        style.configure(
            "Vertical.TScrollbar", background=BG_INPUT, troughcolor=BG_PANEL,
            bordercolor=BORDER, arrowcolor=FG, relief="flat",
        )
        style.map("Vertical.TScrollbar", background=[("active", BORDER)])
        style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=BG_INPUT, foreground=FG_DIM, padding=(14, 5),
                        font=FONT_HEADER, bordercolor=BORDER)
        style.map("TNotebook.Tab", background=[("selected", BG_PANEL), ("disabled", BG)],
                  foreground=[("selected", ACCENT), ("disabled", BORDER)])
        style.configure("Treeview", background=BG_INPUT, fieldbackground=BG_INPUT, foreground=FG,
                        bordercolor=BORDER, font=FONT_UI, rowheight=20)
        style.map("Treeview", background=[("selected", BORDER)], foreground=[("selected", FG)])
        style.configure("Treeview.Heading", background=BG_PANEL, foreground=FG_DIM, font=FONT_HEADER,
                        bordercolor=BORDER, relief="flat")
        style.configure("Pass.Horizontal.TProgressbar", troughcolor=BG_INPUT, background=ACCENT,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)

    # ---- layout --------------------------------------------------------

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="TFrame")
        root_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._root_frame = root_frame

        # Who owns the ports right now, always visible whatever the tab.
        self.lane_strip = tk.Label(root_frame, text="", bg=BG, fg=FG_DIM, font=FONT_MONO_BOLD, anchor="w")
        self.lane_strip.pack(side="bottom", fill="x", pady=(6, 0))
        self._lane_strip_text = ""

        self.notebook = ttk.Notebook(root_frame)
        self.notebook.pack(fill="both", expand=True)
        self.tracking_tab = ttk.Frame(self.notebook, style="TFrame")
        self.prompt_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.tracking_tab, text="  Tracking  ")
        self.notebook.add(self.prompt_tab, text="  Prompt  ")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        tab = self.tracking_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)
        self._build_video_pane(tab)
        self._build_controls_pane(tab)

        self._build_prompt_tab(self.prompt_tab)

    def _build_video_pane(self, parent: tk.Widget) -> None:
        left = ttk.Frame(parent, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
        self._video_pane = left  # its rows around the canvas are the fit's overhead

        display_row = ttk.Frame(left, style="Panel.TFrame")
        display_row.pack(fill="x", padx=8, pady=(8, 0))
        # Camera first: it decides what the pane below shows at all. Changing it
        # needs a restart, like every other constructor parameter.
        ttk.Label(display_row, text="Camera", style="Body.TLabel").pack(side="left")
        self.camera_index_var = tk.IntVar(value=0)
        ttk.Spinbox(display_row, from_=0, to=9, width=3, textvariable=self.camera_index_var,
                    command=self._mark_restart_dirty).pack(side="left", padx=(6, 16))
        self.camera_index_var.trace_add("write", lambda *_: self._mark_restart_dirty())
        ttk.Checkbutton(
            display_row, text="Debug overlay", variable=self._skeleton_enabled,
            command=lambda: set_skeleton_overlay_enabled(self._skeleton_enabled.get()),
        ).pack(side="left")
        ttk.Checkbutton(
            display_row, text="Show FPS", variable=self._fps_enabled,
        ).pack(side="left", padx=(12, 0))

        # height=1: the canvas takes its height from the window (expand), never the
        # other way round. With Tk's default requested height, a short 16:9 window
        # would squeeze the status row below it out of view instead.
        self.canvas = tk.Canvas(left, bg=BG_INPUT, highlightthickness=0, height=1)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self._placeholder_text_id = self.canvas.create_text(
            0, 0, text="NO SIGNAL", fill=FG_DIM, font=("Segoe UI", 14), anchor="center",
        )
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        status_row = ttk.Frame(left, style="Panel.TFrame")
        status_row.pack(fill="x", padx=8, pady=(0, 8))
        self.status_var = tk.StringVar(value="stopped")
        ttk.Label(status_row, textvariable=self.status_var, style="Body.TLabel").pack(side="left")
        self.fps_var = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self.fps_var, style="Mono.TLabel").pack(side="right")
        # Tracking status next to the FPS - the same POSE/FACE state the debug overlay
        # burns into the frame, readable here even with the overlay text tiny or off.
        self.tracking_lbls: dict[str, ttk.Label] = {}
        for key in ("face", "pose"):  # packed right-to-left: reads POSE  FACE  fps
            lbl = ttk.Label(status_row, text="", style="Mono.TLabel")
            lbl.pack(side="right", padx=(0, 12))
            self.tracking_lbls[key] = lbl

    def _on_canvas_resize(self, event) -> None:
        self.canvas.coords(self._placeholder_text_id, event.width // 2, event.height // 2)
        # Redraw at the new size straight away: an image scaled for a larger pane
        # would otherwise overhang the edges - i.e. be cropped - until the next frame.
        if self._last_frame is not None:
            self._render_frame(self._last_frame)

    # ---- window height follows the capture aspect ----------------------------

    def _tracking_tab_shown(self) -> bool:
        return self.notebook.select() == str(self.tracking_tab)

    def _on_root_configure(self, event) -> None:
        # Configure fires for every child widget too, and for the height changes the
        # fit itself makes; only a new WIDTH (a user drag) needs a refit.
        if event.widget is self.root and event.width != self._fitted_width:
            self._schedule_fit()

    def _schedule_fit(self) -> None:
        # Debounced: a drag delivers a burst of Configure events.
        if self._fit_pending is not None:
            self.root.after_cancel(self._fit_pending)
        self._fit_pending = self.root.after(60, self._fit_height_to_video)

    def _fit_height_to_video(self, passes_left: int = 8) -> None:
        """Resize the window so the video pane has exactly the capture's aspect at
        the current width: no letterbox bars, and nothing cropped, since the frame
        is still only ever scaled to fit (_render_frame).

        Window height = the pane height the aspect asks for + everything stacked
        around the pane (tab header, lane strip, option rows, status row, padding).
        That overhead is taken from Tk's REQUESTED sizes, which don't depend on the
        current window size, plus the notebook's tab header, which is constant once
        laid out. Measuring it as window height - pane height instead oscillated:
        straight after a geometry change the window reports its new height while the
        pane still reports the old one. The pane WIDTH is measured, since how the
        width splits between the video and controls columns is grid's call; it only
        moves when the window is narrowed here, which is re-checked on the next pass.
        If the fitted window would run off the bottom of the screen (4:3 at a wide
        window), it gets narrower instead - the image shrinks, it never crops.

        Only while the Tracking tab shows: the Prompt tab keeps the window as it is,
        and the canvas's size is meaningless while it is unmapped."""
        if self._fit_pending is not None:
            self.root.after_cancel(self._fit_pending)
        self._fit_pending = None
        if not self._tracking_tab_shown():
            return
        aspect = self._capture_aspect or self.DEFAULT_ASPECT
        self._fitted_aspect = aspect
        self.root.update_idletasks()  # requested sizes are computed at idle
        pane_w = self.canvas.winfo_width()
        win_w, win_h = self.root.winfo_width(), self.root.winfo_height()
        tab_header = self.notebook.winfo_height() - self.tracking_tab.winfo_height()
        if pane_w <= 1 or tab_header <= 0:  # not laid out yet
            self._fit_pending = self.root.after(30, self._fit_height_to_video)
            return

        overhead = (self.root.winfo_reqheight() - self.notebook.winfo_reqheight() + tab_header
                    + self._video_pane.winfo_reqheight() - self.canvas.winfo_reqheight()
                    + 10)  # the video pane's top padding inside the tab
        max_h = self.root.winfo_screenheight() - self.SCREEN_MARGIN_PX
        new_w, new_h = win_w, overhead + round(pane_w / aspect)
        if new_h > max_h and win_w > self.MIN_WIDTH:
            # Narrow by what the excess height costs in pane width. Only part of a
            # window-width change reaches the pane, so this undershoots; the next
            # pass measures again and tightens it.
            new_w = max(self.MIN_WIDTH, win_w - round((new_h - max_h) * aspect))
        new_h = max(self.MIN_HEIGHT, min(new_h, max_h))

        self._fitted_width = new_w
        if (new_w, new_h) != (win_w, win_h):
            self.root.geometry(f"{new_w}x{new_h}")
            if new_w != win_w and passes_left > 0:
                # Narrowed: the pane width this fit assumed is now stale. Re-check
                # once Tk has laid out the new width.
                self._fit_pending = self.root.after(30, lambda: self._fit_height_to_video(passes_left - 1))

    def _fit_prompt_height(self) -> None:
        self.root.update_idletasks()
        max_h = self.root.winfo_screenheight() - self.SCREEN_MARGIN_PX
        want = min(max(self.root.winfo_height(), self.PROMPT_MIN_HEIGHT), max_h)
        if want != self.root.winfo_height():
            self.root.geometry(f"{self.root.winfo_width()}x{want}")

    def _build_controls_pane(self, parent: tk.Widget) -> None:
        right = ttk.Frame(parent, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew", pady=(10, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Group content (in particular Advanced, once expanded) can be taller
        # than the window - scroll that area, but keep Start/Stop/Apply&Restart
        # pinned below it, always reachable regardless of scroll position.
        scroll_body = make_scrollable(right)

        self._build_smoothing_group(scroll_body)
        self._build_calibration_group(scroll_body)
        self._build_advanced_group(scroll_body)

        bottom = ttk.Frame(right, style="Panel.TFrame")
        bottom.grid(row=1, column=0, sticky="ew")

        run_row = ttk.Frame(bottom, style="Panel.TFrame")
        run_row.pack(fill="x", padx=8, pady=12)
        self.start_btn = ttk.Button(run_row, text="Start", style="Accent.TButton", command=self._on_start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(run_row, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.apply_restart_btn = ttk.Button(
            bottom, text="Apply & Restart", command=self._on_apply_restart, state="disabled",
        )
        self.apply_restart_btn.pack(fill="x", padx=8, pady=(0, 4))
        self.dirty_lbl = ttk.Label(bottom, text="", style="Dim.TLabel")
        self.dirty_lbl.pack(fill="x", padx=8, pady=(0, 8))

    def _build_smoothing_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Smoothing")
        group.pack(fill="x", padx=8, pady=(8, 4))
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)

        # The sliders show SMOOTHING - 0 = raw, higher = smoother - and send the
        # smoothers their EMA alpha, which runs the other way (1 = raw). Stops at
        # SMOOTHING_MAX: alpha 0 would freeze the channel on its first value forever.
        ttk.Label(body, text="0 = raw, higher = smoother but laggier", style="Dim.TLabel").pack(
            fill="x", pady=(0, 4))
        self.face_alpha_slider = LabeledSlider(
            body, "Face", from_=0.0, to=SMOOTHING_MAX, initial=0.5, on_change=self._set_face_alpha,
        )
        self.face_alpha_slider.pack(fill="x")

        # Pose alpha also smooths both hands - PoseSmoother keys purely by
        # bone name and merges body + left hand + right hand into one call,
        # so there is no separate hand-smoothing knob to expose. It also
        # smooths the face channel's head rotation (conductor slaves it).
        self.pose_alpha_slider = LabeledSlider(
            body, "Pose & Hands", from_=0.0, to=SMOOTHING_MAX, initial=0.5, on_change=self._set_pose_alpha,
        )
        self.pose_alpha_slider.pack(fill="x")

    def _build_calibration_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Calibration")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)

        # The button first: it is what gets used, the offset is its readout / override.
        self.calibrate_btn = ttk.Button(body, text="Calibrate upright", command=self._on_calibrate)
        self.calibrate_btn.pack(fill="x")

        row = ttk.Frame(body, style="Panel.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Torso lean offset (deg)", style="Body.TLabel").pack(side="left")
        self.torso_lean_var = tk.DoubleVar(value=0.0)
        spin = ttk.Spinbox(
            row, from_=-45.0, to=45.0, increment=0.5, width=6, textvariable=self.torso_lean_var,
            command=self._on_torso_lean_changed,
        )
        spin.pack(side="right")
        spin.bind("<Return>", lambda _e: self._on_torso_lean_changed())
        spin.bind("<FocusOut>", lambda _e: self._on_torso_lean_changed())

    def _build_advanced_group(self, parent: tk.Widget) -> None:
        section = CollapsibleSection(parent, "Advanced", start_expanded=False)
        section.pack(fill="x", padx=8, pady=4)
        body = section.body

        # --- resolution ---
        res_row = ttk.Frame(body, style="Panel.TFrame")
        res_row.pack(fill="x", pady=(4, 8))
        ttk.Label(res_row, text="Width", style="Dim.TLabel").pack(side="left")
        self.camera_width_var = tk.StringVar(value="")
        ttk.Entry(res_row, textvariable=self.camera_width_var, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(res_row, text="Height", style="Dim.TLabel").pack(side="left")
        self.camera_height_var = tk.StringVar(value="")
        ttk.Entry(res_row, textvariable=self.camera_height_var, width=6).pack(side="left", padx=(4, 0))
        for var in (self.camera_width_var, self.camera_height_var):
            var.trace_add("write", lambda *_: self._mark_restart_dirty())

        # --- model paths ---
        self.holistic_model_var = tk.StringVar(value=DEFAULT_HOLISTIC_MODEL)
        # self.head_pose_model_var = tk.StringVar(value=DEFAULT_HEAD_POSE_MODEL)  # disabled - see CLAUDE.md
        self._add_labeled_entry(body, "Holistic model", self.holistic_model_var)
        # self._add_labeled_entry(body, "Head-pose model", self.head_pose_model_var)

        # --- network: shared by BOTH lanes (they send to the same receivers) ---
        ttk.Label(body, text="Targets (both lanes)", style="Header.TLabel").pack(
            fill="x", pady=(8, 2), anchor="w",
        )
        self.face_ip_var = tk.StringVar(value="127.0.0.1")
        self.face_port_var = tk.IntVar(value=LIVE_LINK_FACE_PORT)
        self.pose_ip_var = tk.StringVar(value="127.0.0.1")
        self.pose_port_var = tk.IntVar(value=POSE_OSC_PORT)
        self._add_labeled_entry(body, "Face IP", self.face_ip_var)
        self._add_labeled_entry(body, "Face port", self.face_port_var)
        self._add_labeled_entry(body, "Pose IP", self.pose_ip_var)
        self._add_labeled_entry(body, "Pose port", self.pose_port_var)

        # --- detection thresholds ---
        ttk.Label(body, text="Detection thresholds", style="Header.TLabel").pack(
            fill="x", pady=(8, 2), anchor="w",
        )
        self.h_min_face_detection_var = tk.DoubleVar(value=0.5)
        self.h_min_face_landmarks_var = tk.DoubleVar(value=0.5)
        self.h_min_pose_detection_var = tk.DoubleVar(value=0.5)
        self.h_min_pose_landmarks_var = tk.DoubleVar(value=0.5)
        self.h_min_hand_landmarks_var = tk.DoubleVar(value=0.5)
        # Head-pose thresholds disabled with head_pose_capture - see CLAUDE.md.
        # self.hp_min_face_detection_var = tk.DoubleVar(value=0.5)
        # self.hp_min_face_presence_var = tk.DoubleVar(value=0.5)
        # self.hp_min_tracking_var = tk.DoubleVar(value=0.5)
        for label, var in (
            ("Holistic: face detect", self.h_min_face_detection_var),
            ("Holistic: face landmarks", self.h_min_face_landmarks_var),
            ("Holistic: pose detect", self.h_min_pose_detection_var),
            ("Holistic: pose landmarks", self.h_min_pose_landmarks_var),
            ("Holistic: hand landmarks", self.h_min_hand_landmarks_var),
            # ("Head pose: face detect", self.hp_min_face_detection_var),
            # ("Head pose: face presence", self.hp_min_face_presence_var),
            # ("Head pose: tracking", self.hp_min_tracking_var),
        ):
            self._add_threshold_spinbox(body, label, var)

    def _add_labeled_entry(self, parent: tk.Widget, label: str, var) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Dim.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=var, width=16).pack(side="right")
        var.trace_add("write", lambda *_: self._mark_restart_dirty())

    def _add_threshold_spinbox(self, parent: tk.Widget, label: str, var: tk.DoubleVar) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, style="Dim.TLabel").pack(side="left")
        ttk.Spinbox(
            row, from_=0.0, to=1.0, increment=0.05, width=6, textvariable=var,
            command=self._mark_restart_dirty,
        ).pack(side="right")
        var.trace_add("write", lambda *_: self._mark_restart_dirty())

    # ---- Prompt tab layout --------------------------------------------------

    def _build_prompt_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=3)
        left.rowconfigure(5, weight=2)

        # -- the prompt --
        entry_row = ttk.Frame(left, style="Panel.TFrame")
        entry_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        self.prompt_var = tk.StringVar()
        entry = ttk.Entry(entry_row, textvariable=self.prompt_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, ipady=3)
        entry.bind("<Return>", lambda _e: self._on_submit_prompt())
        self.prompt_entry = entry
        self.generate_btn = ttk.Button(entry_row, text="Generate", style="Accent.TButton",
                                       command=self._on_submit_prompt)
        self.generate_btn.pack(side="left", padx=(6, 0))
        self.prompt_msg = ttk.Label(left, text="inline options work too: /s 25  /t 4  /seed 7  /seed random",
                                    style="Dim.TLabel")
        self.prompt_msg.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))

        # -- now playing: the one thing that must stand out --
        card = tk.Frame(left, bg=BG_INPUT, highlightthickness=2, highlightbackground=BORDER)
        card.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.np_card = card
        top = tk.Frame(card, bg=BG_INPUT)
        top.pack(fill="x", padx=8, pady=(6, 0))
        self.np_badge = tk.Label(top, text="", font=FONT_MONO_BOLD, padx=6)
        self.np_badge.pack(side="left")
        self.np_pass = tk.Label(top, text="", bg=BG_INPUT, fg=FG_DIM, font=FONT_MONO)
        self.np_pass.pack(side="right")
        self.np_title = tk.Label(card, text="", bg=BG_INPUT, fg=FG, font=FONT_TITLE, anchor="w",
                                 justify="left", wraplength=560)
        self.np_title.pack(fill="x", padx=8, pady=(4, 0))
        self.np_details = tk.Label(card, text="", bg=BG_INPUT, fg=FG_DIM, font=FONT_UI, anchor="w")
        self.np_details.pack(fill="x", padx=8)
        self.np_progress = ttk.Progressbar(card, style="Pass.Horizontal.TProgressbar", maximum=1000)
        self.np_progress.pack(fill="x", padx=8, pady=(4, 8))

        # -- queue --
        self.queue_tree = self._make_item_tree(left, row=3, title="Queue (play order)", height=5)
        qbtns = ttk.Frame(left, style="Panel.TFrame")
        qbtns.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 6))
        for text, cmd in (("▲", lambda: self._on_move(-1)), ("▼", lambda: self._on_move(1)),
                          ("Remove", self._on_remove), ("Clear", self._on_clear_queue)):
            ttk.Button(qbtns, text=text, style="Small.TButton", command=cmd).pack(side="left", padx=(0, 4))
        self.stop_play_btn = ttk.Button(qbtns, text="■ Stop playback", style="Small.TButton",
                                        command=self._on_stop_playback)
        self.stop_play_btn.pack(side="right")
        self.next_btn = ttk.Button(qbtns, text="Next ▶▶", style="Small.TButton", command=self._on_next)
        self.next_btn.pack(side="right", padx=(0, 4))

        # -- history --
        self.history_tree = self._make_item_tree(left, row=5, title="History (this session)", height=4)
        self.history_tree.bind("<Double-1>", lambda _e: self._on_replay())
        hbtns = ttk.Frame(left, style="Panel.TFrame")
        hbtns.grid(row=6, column=0, sticky="ew", padx=8, pady=(2, 8))
        ttk.Button(hbtns, text="Replay", style="Small.TButton", command=self._on_replay).pack(side="left")

        # -- right: settings, Spark, library --
        right = ttk.Frame(tab, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew", pady=(10, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        body = make_scrollable(right)
        self._build_spark_indicator(body)
        self._build_generation_group(body)
        self._build_playback_group(body)
        self._build_library_group(body)
        self._build_prompt_advanced(body)

        for tree in (self.queue_tree, self.history_tree):
            tree.tag_configure(PLAYING, foreground=ACCENT)
            tree.tag_configure(READY, foreground=FG)
            tree.tag_configure(GENERATING, foreground=WARN)
            tree.tag_configure(WAITING, foreground=FG_DIM)
            tree.tag_configure(FAILED, foreground=ERROR)
            tree.tag_configure(PLAYED, foreground=FG_DIM)
            tree.tag_configure(NOT_PLAYED, foreground=FG_DIM)
            tree.tag_configure(CANCELLED, foreground=FG_DIM)
        self._render_now_playing()

    def _make_item_tree(self, parent: tk.Widget, row: int, title: str, height: int) -> ttk.Treeview:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=row, column=0, sticky="nsew", padx=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text=title, style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 2))
        tree = ttk.Treeview(frame, columns=("state", "prompt", "params", "time"), show="headings",
                            height=height, selectmode="browse")
        for col, text, width, stretch in (("state", "", 92, False), ("prompt", "prompt", 220, True),
                                          ("params", "seed · length · steps", 130, False),
                                          ("time", "generation time", 190, False)):
            tree.heading(col, text=text, anchor="w")
            tree.column(col, width=width, stretch=stretch, anchor="w")
        tree.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=sb.set)
        return tree

    def _build_spark_indicator(self, parent: tk.Widget) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", padx=8, pady=(10, 4))
        self.spark_dot = tk.Canvas(row, width=12, height=12, bg=BG_PANEL, highlightthickness=0)
        self.spark_dot.pack(side="left", padx=(0, 6))
        self._spark_oval = self.spark_dot.create_oval(1, 1, 11, 11, fill=FG_DIM, outline="")
        self.spark_lbl = ttk.Label(row, text="Spark: not checked yet", style="Body.TLabel", wraplength=330)
        self.spark_lbl.pack(side="left", fill="x", expand=True)

    def _spin_row(self, parent: tk.Widget, label: str, var, default_text: str, *, from_: float, to: float,
                  increment: float, width: int = 7) -> ttk.Spinbox:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Body.TLabel").pack(side="left")
        ttk.Label(row, text=default_text, style="Dim.TLabel").pack(side="right", padx=(6, 0))
        spin = ttk.Spinbox(row, from_=from_, to=to, increment=increment, width=width, textvariable=var)
        spin.pack(side="right")
        return spin

    def _build_generation_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Generation (next prompt)")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)
        self.seconds_var = tk.DoubleVar(value=contract.DEFAULT_SECONDS)
        self._spin_row(body, "Length (s)", self.seconds_var,
                       f"default {contract.DEFAULT_SECONDS:g}, max {contract.MAX_SECONDS:g}",
                       from_=0.5, to=contract.MAX_SECONDS, increment=0.5)
        self.steps_var = tk.IntVar(value=contract.DEFAULT_STEPS)
        self._spin_row(body, "Denoising steps", self.steps_var, f"default {contract.DEFAULT_STEPS}",
                       from_=1, to=1000, increment=1)
        seed_row = ttk.Frame(body, style="Panel.TFrame")
        seed_row.pack(fill="x", pady=2)
        ttk.Label(seed_row, text="Seed", style="Body.TLabel").pack(side="left")
        ttk.Label(seed_row, text=f"default {DEFAULT_SEED}", style="Dim.TLabel").pack(side="right", padx=(6, 0))
        self.seed_random_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(seed_row, text="random", variable=self.seed_random_var,
                        command=self._on_seed_random).pack(side="right", padx=(6, 0))
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        self.seed_entry = ttk.Entry(seed_row, textvariable=self.seed_var, width=9)
        self.seed_entry.pack(side="right")
        ttk.Label(body, text="a fixed seed makes a repeated prompt a cache hit", style="Dim.TLabel").pack(
            fill="x", pady=(2, 0))

    def _build_playback_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Playback")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)
        self.loop_var = tk.BooleanVar(value=DEFAULT_LOOP)
        ttk.Checkbutton(body, text="Loop the current clip until the next one is ready (default on)",
                        variable=self.loop_var, command=self._on_loop_changed).pack(anchor="w")
        self.stay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Stay in place: no horizontal root motion (default off)",
                        variable=self.stay_var, command=self._on_stay_changed).pack(anchor="w", pady=(2, 0))
        ttk.Label(body, text="   applies to clips that finish generating after the change",
                  style="Dim.TLabel").pack(anchor="w")
        self.speed_var = tk.DoubleVar(value=DEFAULT_SPEED)
        spin = self._spin_row(body, "Speed", self.speed_var, f"default {DEFAULT_SPEED:g}, from the next clip",
                              from_=0.1, to=4.0, increment=0.1, width=5)
        spin.configure(command=self._on_speed_changed)
        spin.bind("<Return>", lambda _e: self._on_speed_changed())
        spin.bind("<FocusOut>", lambda _e: self._on_speed_changed())

    def _build_library_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Library (cached clips, play offline)")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)
        tree = ttk.Treeview(body, columns=("prompt", "params", "time"), show="headings", height=6,
                            selectmode="browse")
        for col, text, width, stretch in (("prompt", "prompt", 150, True), ("params", "seed · steps", 80, False),
                                          ("time", "gen time", 70, False)):
            tree.heading(col, text=text, anchor="w")
            tree.column(col, width=width, stretch=stretch, anchor="w")
        tree.pack(fill="x")
        tree.bind("<Double-1>", lambda _e: self._on_library_play())
        self.library_tree = tree
        btns = ttk.Frame(body, style="Panel.TFrame")
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Queue selected", style="Small.TButton", command=self._on_library_play).pack(side="left")
        ttk.Button(btns, text="Refresh", style="Small.TButton", command=self._refresh_library).pack(
            side="left", padx=(4, 0))
        ttk.Button(btns, text="Play file…", style="Small.TButton", command=self._on_play_file).pack(side="right")

    def _build_prompt_advanced(self, parent: tk.Widget) -> None:
        section = CollapsibleSection(parent, "Advanced", start_expanded=False)
        section.pack(fill="x", padx=8, pady=4)
        body = section.body
        self.spark_url_var = tk.StringVar(value=DEFAULT_SPARK_URL)
        self.spark_timeout_var = tk.DoubleVar(value=DEFAULT_SPARK_TIMEOUT_S)
        self.cache_dir_var = tk.StringVar(value=str(DEFAULT_CACHE))
        self.model_var = tk.StringVar(value=contract.DEFAULT_MODEL)
        self.rate_var = tk.DoubleVar(value=DEFAULT_RATE)
        self.send_face_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=False)
        for label, var in (("Spark URL", self.spark_url_var), ("Request timeout (s)", self.spark_timeout_var),
                           ("Cache directory", self.cache_dir_var), ("Model", self.model_var),
                           ("Send rate (Hz)", self.rate_var)):
            row = ttk.Frame(body, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, style="Dim.TLabel").pack(side="left")
            ttk.Entry(row, textvariable=var, width=24).pack(side="right")
            var.trace_add("write", lambda *_: self._mark_prompt_dirty())
        for text, var in (("Send head channel (Live Link Face)", self.send_face_var),
                          ("Dry run: play, but send nothing", self.dry_run_var)):
            ttk.Checkbutton(body, text=text, variable=var, command=self._mark_prompt_dirty).pack(anchor="w")
        ttk.Label(body, text="Targets: Tracking › Advanced (shared by both lanes)", style="Dim.TLabel").pack(
            anchor="w", pady=(4, 0))
        self.prompt_apply_btn = ttk.Button(body, text="Apply", command=self._on_prompt_apply, state="disabled")
        self.prompt_apply_btn.pack(fill="x", pady=(6, 0))
        self.prompt_dirty_lbl = ttk.Label(body, text="", style="Dim.TLabel")
        self.prompt_dirty_lbl.pack(fill="x")

    # ---- restart-required dirty tracking --------------------------------

    def _mark_restart_dirty(self) -> None:
        if not self.controller.is_running:
            return  # not running yet - Start will just pick up current values
        self._dirty_while_running = True
        self.apply_restart_btn.configure(state="normal")
        self.dirty_lbl.configure(text="changes pending restart", foreground=WARN)

    def _clear_restart_dirty(self) -> None:
        self._dirty_while_running = False
        self.apply_restart_btn.configure(state="disabled")
        self.dirty_lbl.configure(text="")

    def _mark_prompt_dirty(self) -> None:
        self.prompt_apply_btn.configure(state="normal")
        self.prompt_dirty_lbl.configure(text="changes pending Apply", foreground=WARN)

    # ---- live parameter handlers ------------------------------------------

    def _set_face_alpha(self, smoothing: float) -> None:
        if self.controller.conductor is not None:
            self.controller.conductor.face_smoother.alpha = smoothing_to_alpha(smoothing)

    def _set_pose_alpha(self, smoothing: float) -> None:
        if self.controller.conductor is not None:
            self.controller.conductor.pose_smoother.alpha = smoothing_to_alpha(smoothing)

    def _on_torso_lean_changed(self) -> None:
        try:
            value = float(self.torso_lean_var.get())
        except (tk.TclError, ValueError):
            return
        if self.controller.conductor is not None:
            self.controller.conductor.pose_solver.torso_lean_offset_deg = value

    def _on_calibrate(self) -> None:
        conductor = self.controller.conductor
        if conductor is None:
            messagebox.showinfo("Calibrate upright", "Start the pipeline first.")
            return
        landmarks = conductor._last_valid_world_landmarks
        if not landmarks:
            messagebox.showinfo("Calibrate upright", "No pose detected yet - step into frame first.")
            return
        # Timed: a countdown to get back into position, then averaged over 30 frames.
        # The countdown is drawn into the video pane by the conductor itself, so it is
        # visible from across the room; _poll_calibration picks up the result.
        conductor.start_calibration()

    # ---- start / stop / restart (webcam lane) ----------------------------------

    def _collect_params(self) -> dict | None:
        def parse_optional_int(raw: str) -> int | None:
            raw = raw.strip()
            return int(raw) if raw else None

        try:
            params = dict(
                camera_index=int(self.camera_index_var.get()),
                camera_width=parse_optional_int(self.camera_width_var.get()),
                camera_height=parse_optional_int(self.camera_height_var.get()),
                holistic_model=self.holistic_model_var.get().strip(),
                # head_pose_model=self.head_pose_model_var.get().strip(),  # disabled - see CLAUDE.md
                face_ip=self.face_ip_var.get().strip(),
                face_port=int(self.face_port_var.get()),
                pose_ip=self.pose_ip_var.get().strip(),
                pose_port=int(self.pose_port_var.get()),
                face_smoothing_alpha=smoothing_to_alpha(float(self.face_alpha_slider.var.get())),
                pose_smoothing_alpha=smoothing_to_alpha(float(self.pose_alpha_slider.var.get())),
                torso_lean_offset_deg=float(self.torso_lean_var.get()),
                h_min_face_detection=float(self.h_min_face_detection_var.get()),
                h_min_face_landmarks=float(self.h_min_face_landmarks_var.get()),
                h_min_pose_detection=float(self.h_min_pose_detection_var.get()),
                h_min_pose_landmarks=float(self.h_min_pose_landmarks_var.get()),
                h_min_hand_landmarks=float(self.h_min_hand_landmarks_var.get()),
                # hp_min_face_detection=float(self.hp_min_face_detection_var.get()),
                # hp_min_face_presence=float(self.hp_min_face_presence_var.get()),
                # hp_min_tracking=float(self.hp_min_tracking_var.get()),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Could not parse a field: {exc}")
            return None

        for label, path in (("Holistic model", params["holistic_model"]),
                             # ("Head-pose model", params["head_pose_model"]),  # disabled - see CLAUDE.md
                             ):
            if not Path(path).exists():
                messagebox.showerror("Model not found", f"{label} not found at: {path}")
                return None
        return params

    def _on_start(self) -> None:
        if self._busy or self.lanes.lane != TRACKING:
            return
        params = self._collect_params()
        if params is None:
            return
        self.start_btn.configure(state="disabled")
        self.status_var.set("starting...")
        self.controller.start(params)
        self.root.after(200, self._check_startup)

    def _check_startup(self) -> None:
        error = self.controller.take_construction_error()
        if error is not None:
            self.status_var.set("stopped")
            self.start_btn.configure(state="normal")
            messagebox.showerror("Pipeline failed to start", error)
            return
        if self.controller.is_running:
            self.stop_btn.configure(state="normal")
            self.status_var.set("running")
            self._clear_restart_dirty()
            return
        if not self.controller.thread_alive:
            return  # stopped again (lane switch) before it finished starting
        # Still constructing (model load / camera open) - keep polling.
        self.root.after(200, self._check_startup)

    def _on_stop(self) -> None:
        self.stop_btn.configure(state="disabled")
        self.status_var.set("stopping...")
        self._busy = True

        def done(stopped) -> None:
            self._busy = False
            self.status_var.set("stopped" if stopped is True else "stop timed out - camera may still be held")
            self.start_btn.configure(state="normal")
            self._clear_restart_dirty()
        self._in_background("stop", self.controller.stop, done)

    def _on_apply_restart(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        self.apply_restart_btn.configure(state="disabled")
        self.status_var.set("restarting...")
        self._busy = True

        def restart() -> bool:
            if not self.controller.stop():
                return False
            self.controller.start(params)
            return True

        def done(ok) -> None:
            self._busy = False
            if ok is True:
                self.root.after(200, self._check_startup)
            else:
                self.status_var.set("stop timed out - camera may still be held")
        self._in_background("restart", restart, done)

    # ---- the lane switch ---------------------------------------------------------

    def _send_handoff(self, conductor: Conductor) -> None:
        send_tracking_handoff(conductor, self.controller.params or {
            "pose_ip": self.pose_ip_var.get(), "pose_port": int(self.pose_port_var.get()),
            "face_ip": self.face_ip_var.get(), "face_port": int(self.face_port_var.get())})

    def _collect_player_settings(self) -> dict | None:
        try:
            s = dict(pose_ip=self.pose_ip_var.get().strip(), pose_port=int(self.pose_port_var.get()),
                     face_ip=self.face_ip_var.get().strip(), face_port=int(self.face_port_var.get()),
                     rate=float(self.rate_var.get()), send_face=bool(self.send_face_var.get()),
                     dry_run=bool(self.dry_run_var.get()), loop=bool(self.loop_var.get()),
                     speed=self._speed_now)
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Could not parse a field: {exc}")
            return None
        if not s["rate"] > 0:
            messagebox.showerror("Invalid input", "Send rate must be above 0 Hz.")
            return None
        return s

    def _make_player(self) -> LoopPlayer:
        """Called by LaneController on the switch thread, with the settings collected on
        the Tk thread before the switch began."""
        s = self._player_settings
        sender = Sender(s["pose_ip"], s["pose_port"], s["face_ip"], s["face_port"],
                        face=s["send_face"], dry_run=s["dry_run"], rate=s["rate"])
        player = LoopPlayer(sender, rate=s["rate"], speed=s["speed"], loop=s["loop"])

        def on_start(clip: Clip) -> None:
            # Runs on the 60 Hz thread, right after the clip is taken and before its
            # first sample: the one moment a speed change cannot make the clip jump.
            player.speed = self._speed_now
            self._post("started", (id(clip.motion), time.perf_counter(), player.speed))
        player.on_start = on_start
        return player

    def _on_tab_changed(self, _event=None) -> None:
        if self._ignore_tab_event or self._closing:
            return
        want = PROMPT if self.notebook.select() == str(self.prompt_tab) else TRACKING
        if want == self.lanes.lane:
            return
        if self._busy:   # a stop/restart is still joining a thread: switch afterwards
            self._select_tab(self.lanes.lane)
            return
        self._begin_switch(want)

    def _select_tab(self, lane: str) -> None:
        self._ignore_tab_event = True
        self.notebook.select(self.prompt_tab if lane == PROMPT else self.tracking_tab)
        self.root.update_idletasks()
        self._ignore_tab_event = False

    def _set_tabs_enabled(self, enabled: bool) -> None:
        for tab in (self.tracking_tab, self.prompt_tab):
            if not enabled and self.notebook.select() == str(tab):
                continue
            self.notebook.tab(tab, state="normal" if enabled else "disabled")

    def _begin_switch(self, want: str) -> None:
        if want == PROMPT:
            settings = self._collect_player_settings()
            if settings is None:
                self._select_tab(TRACKING)
                return
            self._player_settings = settings
            fn = self.lanes.to_prompt
        else:
            # Resume the webcam only if it was running (or starting) when we left it.
            params = self._collect_params() if self.lanes.tracking_was_running else None
            fn = lambda: self.lanes.to_tracking(params)  # noqa: E731
        self._busy = True
        self._set_tabs_enabled(False)
        self._render_lane_strip()
        self._in_background("lane-switch", fn, lambda err: self._end_switch(want, err))

    def _end_switch(self, want: str, error) -> None:
        self._busy = False
        self._set_tabs_enabled(True)
        lane = self.lanes.lane
        if lane != want:
            self._select_tab(lane)
            messagebox.showerror("Lane switch refused", str(error))
        elif error:
            self.prompt_msg.configure(text=str(error), foreground=WARN)
        if lane == PROMPT and want == PROMPT:
            self.status_var.set("stopped (procedural lane active)")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._clear_restart_dirty()
            self._show_placeholder("PROCEDURAL LANE ACTIVE")
            self._fit_prompt_height()
            self.health.active.set()
            self.health.poke()
            self._refresh_library()
            self.prompt_entry.focus_set()
        elif lane == TRACKING and want == TRACKING:
            self.health.active.clear()
            # The clip stopped; nothing queued in that session will play.
            self._retire_session()
            if self.controller.thread_alive:
                self.status_var.set("starting...")
                self.start_btn.configure(state="disabled")
                self.root.after(200, self._check_startup)
            else:
                self.status_var.set("stopped")
                while True:   # a frame from before the switch must not reappear
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
            self._show_placeholder("NO SIGNAL")
            self._schedule_fit()
        self._render_lane_strip()
        self._refresh_prompt_views()

    # ---- Prompt tab actions ----------------------------------------------------

    def _new_item(self, title: str, **kw) -> Item:
        item = Item(id=self._next_item_id, title=title, epoch=self.lanes.epoch, **kw)
        self._next_item_id += 1
        self.items[item.id] = item
        return item

    def _enqueue_item(self, item: Item) -> None:
        if self.lanes.lane != PROMPT:
            return
        self.queue_ids.append(item.id)
        self.worker.submit(item)
        self._refresh_prompt_views()

    @staticmethod
    def _request_details(req: contract.GenerationRequest) -> str:
        return f"{req.seed} · {req.num_frames / contract.KIMODO_FPS:g} s · {req.steps}"

    def _on_submit_prompt(self) -> None:
        line = self.prompt_var.get().strip()
        if not line:
            return
        try:
            seed = "random" if self.seed_random_var.get() else int(self.seed_var.get())
            req = parse_prompt(line, float(self.seconds_var.get()), int(self.steps_var.get()), seed,
                               self.model_var.get().strip())
        except (ValueError, tk.TclError) as e:
            self.prompt_msg.configure(text=f"✗ {e}", foreground=ERROR)
            return
        self._enqueue_item(self._new_item(req.prompt, request=req, details=self._request_details(req)))
        self.prompt_msg.configure(text=f"→ {req.prompt!r} seed {req.seed}, {req.num_frames} frames, "
                                       f"{req.steps} steps", foreground=FG_DIM)
        self.prompt_var.set("")

    def _on_seed_random(self) -> None:
        self.seed_entry.configure(state="disabled" if self.seed_random_var.get() else "normal")

    def _on_loop_changed(self) -> None:
        # LoopPlayer reads .loop every tick: live, no restart.
        with self.lanes.lock:
            if self.lanes.player is not None:
                self.lanes.player.loop = bool(self.loop_var.get())

    def _on_stay_changed(self) -> None:
        self._stay_in_place = bool(self.stay_var.get())

    def _on_speed_changed(self) -> None:
        try:
            speed = float(self.speed_var.get())
        except (tk.TclError, ValueError):
            return
        if speed > 0:
            self._speed_now = speed   # picked up by on_start, i.e. from the next clip

    def _selected(self, tree: ttk.Treeview) -> Item | None:
        sel = tree.selection()
        return self.items.get(int(sel[0])) if sel else None

    def _on_move(self, step: int) -> None:
        """Reorder clips that are ready (already in the player's queue). Generating and
        waiting ones keep their place: they become ready in submission order."""
        item = self._selected(self.queue_tree)
        if item is None or item.state != READY:
            return
        ready = [i for i in self.queue_ids if self.items[i].state == READY]
        k = ready.index(item.id)
        j = k + step
        if not 0 <= j < len(ready):
            return
        ready[k], ready[j] = ready[j], ready[k]
        rest = [i for i in self.queue_ids if self.items[i].state != READY]
        self.queue_ids = ready + rest
        self._sync_player_queue()
        self._refresh_prompt_views()

    def _on_remove(self) -> None:
        item = self._selected(self.queue_tree)
        if item is not None:
            self._drop_from_queue([item.id])

    def _on_clear_queue(self) -> None:
        self._drop_from_queue(list(self.queue_ids))

    def _drop_from_queue(self, ids: list[int]) -> None:
        """Waiting: never generated. Generating: finishes into the cache, never plays.
        Ready: taken back out of the player."""
        drop: set[int] = set()
        for i in ids:
            item = self.items[i]
            item.cancelled = True
            if item.state == READY:
                if item.motion is not None:
                    drop.add(id(item.motion))   # before _to_history frees it
                item.advance(NOT_PLAYED)
                item.note = "removed"
                self._to_history(i)
            elif item.state == WAITING:
                item.advance(CANCELLED)
                self._to_history(i)
            else:   # generating: stays listed until it lands, then goes to history
                item.note = "removed - finishes into cache"
        self._sync_player_queue(drop)
        self._refresh_prompt_views()

    def _sync_player_queue(self, drop: set[int] = frozenset()) -> None:
        """Make the player's pending clips match the READY items, in list order, minus
        `drop` (ids of removed clips' motions). A clip the worker delivered a moment ago
        (its 'fetched' event not processed yet) is unknown here and keeps its place at
        the end. A drained clip keeps its motion alive, so its id() cannot be reused."""
        order = [id(self.items[i].motion) for i in self.queue_ids
                 if self.items[i].state == READY and self.items[i].motion is not None]

        def arrange(clips: list[Clip]) -> list[Clip]:
            by_motion = {id(c.motion): c for c in clips}
            out = [by_motion[m] for m in order if m in by_motion]
            ordered = set(order)
            out += [c for c in clips if id(c.motion) not in ordered and id(c.motion) not in drop]
            return out
        self.lanes.rearrange(arrange)

    def _on_next(self) -> None:
        with self.lanes.lock:
            if self.lanes.player is not None:
                self.lanes.player.skip()

    def _on_stop_playback(self) -> None:
        if self._busy or self.lanes.lane != PROMPT:
            return
        self._busy = True
        self._render_lane_strip()

        def done(err) -> None:
            self._busy = False
            self._retire_session()
            if err:
                self.prompt_msg.configure(text=f"✗ {err}", foreground=ERROR)
            self._render_lane_strip()
            self._refresh_prompt_views()
        self._in_background("stop-playback", lambda: self.lanes.restart_player(carry=False), done)

    def _on_replay(self) -> None:
        old = self._selected(self.history_tree)
        if old is None:
            return
        if old.request is not None:   # a cache hit by now, unless it failed
            item = self._new_item(old.title, request=old.request, details=old.details)
        else:
            item = self._new_item(old.title, file=old.file, details=old.details)
        self._enqueue_item(item)

    def _on_library_play(self) -> None:
        sel = self.library_tree.selection()
        if not sel:
            return
        entry = self._library[int(sel[0])]
        m = entry.meta
        details = f"{m.get('seed', '?')} · {m.get('num_frames', 0) / contract.KIMODO_FPS:g} s · {m.get('steps', '?')}"
        self._enqueue_item(self._new_item(str(m.get("prompt", entry.path.name)), file=entry.path, details=details))

    def _on_play_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Play a Kimodo BVH or a cached NPZ",
            filetypes=[("Kimodo clips", "*.bvh *.npz"), ("BVH", "*.bvh"), ("NPZ", "*.npz")])
        if path:
            p = Path(path)
            self._enqueue_item(self._new_item(p.name, file=p, details="file"))

    def _refresh_library(self) -> None:
        if self._library_loading:
            return
        self._library_loading = True
        cache = self._lane.cache
        self._in_background("library", cache.entries, self._show_library)

    def _show_library(self, entries) -> None:
        self._library_loading = False
        if not isinstance(entries, list):
            return
        self._library = list(reversed(entries))   # newest first
        rows = []
        for k, e in enumerate(self._library):
            m = e.meta
            gen = m.get("generation_s")
            rows.append((str(k), (m.get("prompt", e.path.name), f"{m.get('seed', '?')} · {m.get('steps', '?')}",
                                  f"{gen:.1f} s" if isinstance(gen, (int, float)) else "—"), ()))
        # iids are list positions here, so rebuild rather than sync
        self.library_tree.delete(*self.library_tree.get_children())
        for iid, values, tags in rows:
            self.library_tree.insert("", "end", iid=iid, values=values, tags=tags)

    def _on_prompt_apply(self) -> None:
        if self._busy:
            return
        try:
            url = self.spark_url_var.get().strip()
            timeout = float(self.spark_timeout_var.get())
            cache_dir = Path(self.cache_dir_var.get().strip())
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Could not parse a field: {exc}")
            return
        settings = self._collect_player_settings()
        if settings is None:
            return
        # The worker and the health thread read self._lane per job: swapping the whole
        # object is atomic, and a request already in flight finishes on the old one.
        self._lane = Lane(KimodoClient(url, timeout=timeout), ClipCache(cache_dir), self._retargeter)
        self.prompt_apply_btn.configure(state="disabled")
        self.prompt_dirty_lbl.configure(text="")
        self.health.poke()
        self._refresh_library()
        restart_keys = ("pose_ip", "pose_port", "face_ip", "face_port", "rate", "send_face", "dry_run")
        changed = any(settings[k] != self._player_settings.get(k) for k in restart_keys)
        if changed and self.lanes.lane == PROMPT:
            self._player_settings = settings
            self._busy = True

            def done(err) -> None:
                self._busy = False
                if err:
                    self.prompt_msg.configure(text=f"✗ {err}", foreground=ERROR)
                self._render_lane_strip()
            self._in_background("player-restart", lambda: self.lanes.restart_player(carry=True), done)
        else:
            self._player_settings = settings

    # ---- Prompt tab state --------------------------------------------------------

    def _to_history(self, item_id: int) -> None:
        if item_id in self.queue_ids:
            self.queue_ids.remove(item_id)
        if self.playing_id == item_id:
            self.playing_id = None
        if item_id not in self.history_ids:
            self.history_ids.insert(0, item_id)
        item = self.items[item_id]
        if item.state not in (GENERATING, WAITING, READY, PLAYING):
            item.motion = None   # the history keeps the facts, not the frames

    def _retire_session(self) -> None:
        """The playback session ended (lane switch or Stop): the clip stopped, nothing
        queued will play. A generation in flight finishes into the cache (its 'fetched'
        event comes back with delivered=False)."""
        if self.playing_id is not None:
            self.items[self.playing_id].advance(PLAYED)
            self._to_history(self.playing_id)
        for i in list(self.queue_ids):
            item = self.items[i]
            item.cancelled = True
            if item.state == READY:
                item.advance(NOT_PLAYED)
            elif item.state == WAITING:
                item.advance(CANCELLED)
            else:
                item.note = "not played - finishes into cache"
            self._to_history(i)
        self._end_signalled = False

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "call":
            done, result = payload
            done(result)
        elif kind == "generating":
            item_id, t0 = payload
            item = self.items[item_id]
            if item.advance(GENERATING):
                item.started_at = t0
        elif kind == "fetched":
            item_id, motion, timing, delivered = payload
            item = self.items[item_id]
            for k, v in timing.items():
                setattr(item, k, v)
            if delivered:
                item.motion = motion
                early = self._early_start
                if early is not None and early[0] == id(motion):
                    # The 60 Hz thread started it before this event was processed.
                    self._early_start = None
                    self._mark_started(item, early[1], early[2])
                elif item.advance(READY) and item_id in self.queue_ids:
                    # Ready clips sit before the ones still generating/waiting: that is
                    # the order the player will take them in.
                    self.queue_ids.remove(item_id)
                    n_ready = sum(1 for i in self.queue_ids if self.items[i].state in (READY,))
                    self.queue_ids.insert(n_ready, item_id)
            else:
                item.advance(NOT_PLAYED)
                item.note = "not played - cached" if item.source in ("spark", "server cache") else "not played"
                self._to_history(item_id)
            if item.source in ("spark", "server cache"):
                self._refresh_library()
        elif kind == "failed":
            item_id, message = payload
            item = self.items[item_id]
            item.error = message
            item.advance(FAILED)
            self._to_history(item_id)
        elif kind == "cancelled":
            item = self.items[payload]
            item.advance(CANCELLED)
            self._to_history(payload)
        elif kind == "started":
            motion_id, t_start, speed = payload
            item = next((it for it in self.items.values() if it.motion is not None and id(it.motion) == motion_id),
                        None)
            if item is None:
                # Delivered, but its "fetched" event is still behind this one.
                self._early_start = payload
            else:
                self._mark_started(item, t_start, speed)
        elif kind == "health":
            self._render_health(*payload)
        elif kind == "spark_down":
            self._render_health(False, payload, self._lane.client.base_url)
            self.health.poke()

    def _mark_started(self, item: Item, t_start: float, speed: float) -> None:
        if self.playing_id is not None and self.playing_id != item.id:
            self.items[self.playing_id].advance(PLAYED)
            self._to_history(self.playing_id)
        item.advance(PLAYING)
        if item.id in self.queue_ids:
            self.queue_ids.remove(item.id)
        self.playing_id = item.id
        self._playing_since, self._playing_speed = t_start, speed
        self._end_signalled = False

    def _render_health(self, ok: bool, info, url: str) -> None:
        if ok:
            busy = bool(info.get("busy"))
            colour = WARN if busy else ACCENT
            text = (f"Spark reachable · {info.get('model', '?')} on {info.get('device', '?')}"
                    f" · {'busy (generating)' if busy else 'idle'}")
            if not info.get("loaded", True):
                colour, text = WARN, "Spark reachable, model still loading"
        else:
            colour = ERROR
            text = f"Spark not reachable ({url}): {str(info)[:140]}"
        self.spark_dot.itemconfigure(self._spark_oval, fill=colour)
        self.spark_lbl.configure(text=text)

    def _check_end_of_clip(self) -> None:
        """Loop off: after the last clip's pass, LoopPlayer stops sending, leaving the
        final frame flagged present=1. Send the 'tracking lost' signal once instead, the
        same as a stop. Main thread, but the 60 Hz thread is idle at this point (no
        clip); should a clip be enqueued in the same instant, Unreal sees one present=0
        packet between two present=1 ones."""
        player = self.lanes.player
        if (self.lanes.lane != PROMPT or player is None or self._end_signalled
                or not player.finished.is_set() or player.current is not None or player.queued):
            return
        player.sender.send_lost(player._last_bones)   # private: the pose it last sent
        self._end_signalled = True
        if self.playing_id is not None:
            self.items[self.playing_id].advance(PLAYED)
            self._to_history(self.playing_id)

    # ---- Prompt tab rendering ------------------------------------------------------

    def _item_row(self, item: Item, now: float, line_pos: int | None) -> tuple[str, tuple, tuple]:
        if item.state == PLAYING:
            state = "▶ PLAYING"
        elif item.note:
            state = item.note
        else:
            state = item.state.upper() if item.state in QUEUE_STATES else item.state
        return (str(item.id), (state, item.title, item.details,
                               generation_time_text(item, now, line_pos)), (item.state,))

    def _refresh_prompt_views(self) -> None:
        now = time.perf_counter()
        rows, line = [], 0
        for i in self.queue_ids:
            item = self.items[i]
            pos = None
            if item.state in (WAITING, GENERATING):
                line += 1
                pos = line
            rows.append(self._item_row(item, now, pos))
        sync_tree(self.queue_tree, rows)
        sync_tree(self.history_tree, [self._item_row(self.items[i], now, None) for i in self.history_ids])
        self._render_now_playing()

    def _render_now_playing(self) -> None:
        item = self.items.get(self.playing_id) if self.playing_id is not None else None
        player = self.lanes.player
        if item is None or player is None:
            self.np_card.configure(highlightbackground=BORDER)
            self.np_badge.configure(text="■ NOTHING PLAYING", bg=BG_INPUT, fg=FG_DIM)
            self.np_title.configure(text="Type a prompt above, or queue a clip from the Library."
                                    if self.lanes.lane == PROMPT else "The procedural lane is not active.",
                                    fg=FG_DIM)
            self.np_details.configure(text="")
            self.np_pass.configure(text="")
            self.np_progress.configure(value=0)
            return
        self.np_card.configure(highlightbackground=ACCENT)
        self.np_badge.configure(text="▶ NOW PLAYING", bg=ACCENT, fg=BG)
        self.np_title.configure(text=item.title, fg=FG)
        self.np_details.configure(text=f"seed · length · steps  {item.details}     "
                                       f"{generation_time_text(item, time.perf_counter())}"
                                       f"{'     stay in place' if item.held else ''}")
        duration = item.motion.duration_s if item.motion is not None else 0.0
        if duration > 0:
            t = (time.perf_counter() - self._playing_since) * self._playing_speed
            t = t % duration if player.loop else min(t, duration)
            self.np_progress.configure(value=1000 * t / duration)
            self.np_pass.configure(text=f"pass {player.passes + 1} · {t:4.1f} / {duration:.1f} s"
                                        f"{' · loop' if player.loop else ''}")

    def _render_lane_strip(self) -> None:
        if self._busy:
            text, colour = "… switching lanes / stopping - nothing new starts until the old sender has stopped", WARN
        elif self.lanes.lane == TRACKING:
            if self.controller.is_running:
                text, colour = (f"▶ SENDING  webcam lane  →  pose {self.pose_ip_var.get()}:{self.pose_port_var.get()}"
                                f" · face {self.face_ip_var.get()}:{self.face_port_var.get()}"), ACCENT
            else:
                text, colour = "■ sending nothing  (webcam lane stopped)", FG_DIM
        elif self.lanes.lane == PROMPT:
            s = self._player_settings
            target = f"pose {s.get('pose_ip')}:{s.get('pose_port')}" + (
                f" · face {s.get('face_ip')}:{s.get('face_port')}" if s.get("send_face") else " · face channel off")
            player = self.lanes.player
            if s.get("dry_run"):
                text, colour = "■ procedural lane, DRY RUN - sending nothing", WARN
            elif player is not None and player.current is not None:
                text, colour = f"▶ SENDING  procedural lane  →  {target}", ACCENT
            else:
                text, colour = "■ procedural lane active, nothing playing", FG_DIM
        else:
            text, colour = "… switching lanes", WARN
        if text != self._lane_strip_text:
            self._lane_strip_text = text
            self.lane_strip.configure(text=text, fg=colour)

    # ---- closing ------------------------------------------------------------

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.root.title("Mocap Conductor - stopping…")
        self._in_background("shutdown", self.lanes.shutdown, lambda _r: self.root.destroy())

    # ---- frame pump / FPS / event pump ---------------------------------------------

    def _poll(self) -> None:
        if self._closing:
            # Still pump events: the shutdown thread reports back through them.
            self._drain_events()
            self.root.after(self.POLL_MS, self._poll)
            return
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None and self.lanes.lane == TRACKING:
            self._frame_times.append(time.perf_counter())
            self._render_frame(frame)

        # Detect a pipeline thread that died without going through Stop
        # (e.g. camera unplugged) and reflect it in the UI instead of
        # silently leaving Stop enabled for a thread that no longer exists.
        if self.status_var.get() == "running" and not self.controller.is_running and not self._busy:
            self.status_var.set("stopped (pipeline exited unexpectedly)")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._clear_restart_dirty()

        self._poll_calibration()
        self._poll_tracking()
        self._drain_events()
        self._check_end_of_clip()
        self._tick += 1
        if self._tick % self.PROMPT_REFRESH_EVERY == 0:
            self._render_lane_strip()
            if not self._tracking_tab_shown():
                self._refresh_prompt_views()
        self.root.after(self.POLL_MS, self._poll)

    def _drain_events(self) -> None:
        changed = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(kind, payload)
            changed = changed or kind != "call"
        if changed and not self._closing:
            self._refresh_prompt_views()

    def _poll_tracking(self) -> None:
        conductor = self.controller.conductor
        for key, lbl in self.tracking_lbls.items():
            if conductor is None or not self.controller.is_running:
                lbl.configure(text="")
                continue
            tracking = getattr(conductor, f"{key}_tracking", False)
            lbl.configure(text=f"{key.upper()} {'TRACKING' if tracking else 'SEARCHING'}",
                          foreground=ACCENT if tracking else ERROR)

    def _poll_calibration(self) -> None:
        """Mirror a running timed calibration into the lean spinbox once it lands.
        The countdown itself is drawn into the video pane by the conductor - this only
        picks up the averaged result, on the main thread like every other widget write."""
        conductor = self.controller.conductor
        state = getattr(conductor, "calibration_state", None) if conductor else None
        if state is not None and state.phase == "done" and not self._calibration_applied:
            self.torso_lean_var.set(round(state.lean_offset_deg, 1))
            self._calibration_applied = True
        elif state is None or state.phase != "done":
            self._calibration_applied = False

    def _render_frame(self, frame) -> None:
        self._last_frame = frame
        h, w = frame.shape[:2]
        self._capture_aspect = w / h
        if self._fitted_aspect is None or abs(self._capture_aspect - self._fitted_aspect) > 1e-3:
            # New camera or resolution: refit, then this and later frames fill the pane.
            self._fit_height_to_video()

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_w / w, canvas_h / h)
        draw_w, draw_h = max(1, int(w * scale)), max(1, int(h * scale))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((draw_w, draw_h), Image.BILINEAR)

        # Drawn as a Tk label next to the status text, not burned into the
        # frame - conductor's own debug overlay already puts status text in
        # the top-left corner (_draw_debug), and stacking ours on top of it
        # there was unreadable.
        if self._fps_enabled.get() and len(self._frame_times) >= 2:
            fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
            self.fps_var.set(f"{fps:5.1f} fps")
        else:
            self.fps_var.set("")

        self._photo_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo_image, anchor="center")

    def _show_placeholder(self, text: str) -> None:
        self._last_frame = None
        self.canvas.delete("all")
        self._placeholder_text_id = self.canvas.create_text(
            self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
            text=text, fill=FG_DIM, font=("Segoe UI", 14), anchor="center")

    # ---- entry point -----------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
