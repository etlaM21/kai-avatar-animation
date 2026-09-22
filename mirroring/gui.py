"""
gui.py - Tkinter control surface for the mocap pipeline (conductor.py).

This is a WRAPPER. It does not reimplement any tracking, smoothing or
solving logic - it drives an unmodified Conductor instance and only reaches
into it through:
    - live attribute assignment (conductor.face_smoother.alpha, etc.) for
      parameters that are safe to change on a running instance, and
    - constructor kwargs, for everything that must be baked in at startup
      (camera, model paths, ports/IPs, detection thresholds) - changing any
      of those tears the pipeline down and builds a fresh one.

Conductor owns the camera and calls cv2.imshow()/cv2.waitKey() itself, so
getting frames into a Tk widget and stopping the pipeline cleanly both go
through a small set of monkeypatches installed on the shared cv2 module
before the first Conductor is ever constructed - see install_cv2_patches()
below for exactly what is patched and why. conductor.py, pose_solver.py and
the mediapipe_*_capture.py modules are unmodified; `python conductor.py
--debug --camera 1` behaves exactly as before.

Run with the project's venv interpreter, same as conductor.py:
    .\\venv\\Scripts\\python.exe gui.py
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

# Patched below, before any Conductor is constructed - draw_landmarks is
# looked up as an attribute of this module object at call time, so patching
# it here reaches conductor.py's calls too without touching that file.
from mediapipe.tasks.python.vision import drawing_utils

from conductor import Conductor, LIVE_LINK_FACE_PORT, POSE_OSC_PORT
from head_pose_capture import HeadPoseCapture
from mediapipe_holistic_capture import MediaPipeHolisticCapture

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

DEFAULT_HOLISTIC_MODEL = "holistic_landmarker.task"
DEFAULT_HEAD_POSE_MODEL = "face_landmarker.task"


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


# ---- pipeline lifecycle -----------------------------------------------------

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
        self._thread: threading.Thread | None = None
        self._construction_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self.conductor is not None and self._thread is not None and self._thread.is_alive()

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
        self._construction_error = None
        self._thread = threading.Thread(target=self._run, args=(params,), daemon=True)
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
            head_pose_capture = HeadPoseCapture(
                model_path=params["head_pose_model"],
                min_face_detection_confidence=params["hp_min_face_detection"],
                min_face_presence_confidence=params["hp_min_face_presence"],
                min_tracking_confidence=params["hp_min_tracking"],
            )
            conductor = Conductor(
                holistic_capture,
                head_pose_capture,
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
        """Returns True if the pipeline thread actually stopped in time."""
        if self._thread is None:
            return True
        self.stop_event.set()
        self._thread.join(timeout=join_timeout)
        stopped = not self._thread.is_alive()
        if stopped:
            self.conductor = None
            self._thread = None
        return stopped


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


# ---- main application --------------------------------------------------------

class App:
    VIDEO_MIN_W, VIDEO_MIN_H = 400, 300
    FPS_WINDOW = 30

    def __init__(self) -> None:
        self.frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self.controller = PipelineController(self.frame_queue)
        install_cv2_patches(self.frame_queue, self.controller.stop_event)

        self.root = tk.Tk()
        self.root.title("Mocap Conductor")
        self.root.configure(bg=BG)
        self.root.geometry("1150x700")
        self.root.minsize(900, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_style()

        self._frame_times: deque[float] = deque(maxlen=self.FPS_WINDOW)
        self._capture_aspect: float | None = None
        self._photo_image: ImageTk.PhotoImage | None = None  # keep a reference alive
        self._skeleton_enabled = tk.BooleanVar(value=True)
        self._fps_enabled = tk.BooleanVar(value=True)
        self._dirty_while_running = False
        self._calibration_applied = False  # so one finished calibration writes the spinbox once

        self._build_layout()
        set_skeleton_overlay_enabled(self._skeleton_enabled.get())

        self.root.after(33, self._poll_frame_queue)

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
        style.configure(
            "Accent.TButton", background=ACCENT_DIM, foreground=ACCENT, padding=6, font=FONT_HEADER,
        )
        style.map("Accent.TButton", background=[("active", ACCENT), ("disabled", BG_PANEL)],
                  foreground=[("active", BG), ("disabled", FG_DIM)])
        style.configure(
            "TEntry", fieldbackground=BG_INPUT, foreground=FG, insertcolor=FG, bordercolor=BORDER,
        )
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

    # ---- layout --------------------------------------------------------

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="TFrame")
        root_frame.pack(fill="both", expand=True, padx=10, pady=10)
        root_frame.columnconfigure(0, weight=3)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(0, weight=1)

        self._build_video_pane(root_frame)
        self._build_controls_pane(root_frame)

    def _build_video_pane(self, parent: tk.Widget) -> None:
        left = ttk.Frame(parent, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        display_row = ttk.Frame(left, style="Panel.TFrame")
        display_row.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Checkbutton(
            display_row, text="Debug overlay", variable=self._skeleton_enabled,
            command=lambda: set_skeleton_overlay_enabled(self._skeleton_enabled.get()),
        ).pack(side="left")
        ttk.Checkbutton(
            display_row, text="Show FPS", variable=self._fps_enabled,
        ).pack(side="left", padx=(12, 0))

        self.canvas = tk.Canvas(left, bg=BG_INPUT, highlightthickness=0)
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

    def _on_canvas_resize(self, event) -> None:
        self.canvas.coords(self._placeholder_text_id, event.width // 2, event.height // 2)

    def _build_controls_pane(self, parent: tk.Widget) -> None:
        right = ttk.Frame(parent, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Group content (in particular Advanced, once expanded) can be taller
        # than the window - scroll that area, but keep Start/Stop/Apply&Restart
        # pinned below it, always reachable regardless of scroll position.
        scroll_body = self._make_scrollable(right)

        self._build_camera_group(scroll_body)
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

    def _make_scrollable(self, parent: tk.Widget) -> ttk.Frame:
        """Wraps a vertically-scrolling Canvas+Scrollbar around a plain
        ttk.Frame and returns that inner frame - callers pack/grid their own
        widgets into it exactly as if it were unscrolled. Needed because
        Advanced, once expanded, can be taller than the window."""
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

    def _build_camera_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Camera")
        group.pack(fill="x", padx=8, pady=(8, 4))

        row = ttk.Frame(group, style="Panel.TFrame")
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Index", style="Body.TLabel").pack(side="left")
        self.camera_index_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.camera_index_var,
                            command=self._mark_restart_dirty)
        spin.pack(side="left", padx=(6, 0))
        self.camera_index_var.trace_add("write", lambda *_: self._mark_restart_dirty())

    def _build_smoothing_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Smoothing")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)

        self.face_alpha_slider = LabeledSlider(
            body, "Face", from_=0.01, to=1.0, initial=0.5, on_change=self._set_face_alpha,
        )
        self.face_alpha_slider.pack(fill="x")

        # Pose alpha also smooths both hands - PoseSmoother keys purely by
        # bone name and merges body + left hand + right hand into one call,
        # so there is no separate hand-smoothing knob to expose.
        self.pose_alpha_slider = LabeledSlider(
            body, "Pose & Hands", from_=0.01, to=1.0, initial=0.5, on_change=self._set_pose_alpha,
        )
        self.pose_alpha_slider.pack(fill="x")

    def _build_calibration_group(self, parent: tk.Widget) -> None:
        group = ttk.LabelFrame(parent, text="Calibration")
        group.pack(fill="x", padx=8, pady=4)
        body = ttk.Frame(group, style="Panel.TFrame")
        body.pack(fill="x", padx=8, pady=8)

        row = ttk.Frame(body, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Torso lean offset (deg)", style="Body.TLabel").pack(side="left")
        self.torso_lean_var = tk.DoubleVar(value=0.0)
        spin = ttk.Spinbox(
            row, from_=-45.0, to=45.0, increment=0.5, width=6, textvariable=self.torso_lean_var,
            command=self._on_torso_lean_changed,
        )
        spin.pack(side="right")
        spin.bind("<Return>", lambda _e: self._on_torso_lean_changed())
        spin.bind("<FocusOut>", lambda _e: self._on_torso_lean_changed())

        self.calibrate_btn = ttk.Button(body, text="Calibrate upright", command=self._on_calibrate)
        self.calibrate_btn.pack(fill="x", pady=(8, 0))

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
        self.head_pose_model_var = tk.StringVar(value=DEFAULT_HEAD_POSE_MODEL)
        self._add_labeled_entry(body, "Holistic model", self.holistic_model_var)
        self._add_labeled_entry(body, "Head-pose model", self.head_pose_model_var)

        # --- network ---
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
        self.hp_min_face_detection_var = tk.DoubleVar(value=0.5)
        self.hp_min_face_presence_var = tk.DoubleVar(value=0.5)
        self.hp_min_tracking_var = tk.DoubleVar(value=0.5)
        for label, var in (
            ("Holistic: face detect", self.h_min_face_detection_var),
            ("Holistic: face landmarks", self.h_min_face_landmarks_var),
            ("Holistic: pose detect", self.h_min_pose_detection_var),
            ("Holistic: pose landmarks", self.h_min_pose_landmarks_var),
            ("Holistic: hand landmarks", self.h_min_hand_landmarks_var),
            ("Head pose: face detect", self.hp_min_face_detection_var),
            ("Head pose: face presence", self.hp_min_face_presence_var),
            ("Head pose: tracking", self.hp_min_tracking_var),
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

    # ---- live parameter handlers ------------------------------------------

    def _set_face_alpha(self, value: float) -> None:
        if self.controller.conductor is not None:
            self.controller.conductor.face_smoother.alpha = value

    def _set_pose_alpha(self, value: float) -> None:
        if self.controller.conductor is not None:
            self.controller.conductor.pose_smoother.alpha = value

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

    # ---- start / stop / restart --------------------------------------------

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
                head_pose_model=self.head_pose_model_var.get().strip(),
                face_ip=self.face_ip_var.get().strip(),
                face_port=int(self.face_port_var.get()),
                pose_ip=self.pose_ip_var.get().strip(),
                pose_port=int(self.pose_port_var.get()),
                face_smoothing_alpha=float(self.face_alpha_slider.var.get()),
                pose_smoothing_alpha=float(self.pose_alpha_slider.var.get()),
                torso_lean_offset_deg=float(self.torso_lean_var.get()),
                h_min_face_detection=float(self.h_min_face_detection_var.get()),
                h_min_face_landmarks=float(self.h_min_face_landmarks_var.get()),
                h_min_pose_detection=float(self.h_min_pose_detection_var.get()),
                h_min_pose_landmarks=float(self.h_min_pose_landmarks_var.get()),
                h_min_hand_landmarks=float(self.h_min_hand_landmarks_var.get()),
                hp_min_face_detection=float(self.hp_min_face_detection_var.get()),
                hp_min_face_presence=float(self.hp_min_face_presence_var.get()),
                hp_min_tracking=float(self.hp_min_tracking_var.get()),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid input", f"Could not parse a field: {exc}")
            return None

        for label, path in (("Holistic model", params["holistic_model"]),
                             ("Head-pose model", params["head_pose_model"])):
            if not Path(path).exists():
                messagebox.showerror("Model not found", f"{label} not found at: {path}")
                return None
        return params

    def _on_start(self) -> None:
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
        # Still constructing (model load / camera open) - keep polling.
        self.root.after(200, self._check_startup)

    def _on_stop(self) -> None:
        self.stop_btn.configure(state="disabled")
        self.status_var.set("stopping...")
        self.root.update_idletasks()
        stopped = self.controller.stop()
        self.status_var.set("stopped" if stopped else "stop timed out - camera may still be held")
        self.start_btn.configure(state="normal")
        self._clear_restart_dirty()

    def _on_apply_restart(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        self.apply_restart_btn.configure(state="disabled")
        self.status_var.set("restarting...")
        self.root.update_idletasks()
        self.controller.stop()
        self.controller.start(params)
        self.root.after(200, self._check_startup)

    def _on_close(self) -> None:
        self.controller.stop()
        self.root.destroy()

    # ---- frame pump / FPS -------------------------------------------------

    def _poll_frame_queue(self) -> None:
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            self._frame_times.append(time.perf_counter())
            self._render_frame(frame)

        # Detect a pipeline thread that died without going through Stop
        # (e.g. camera unplugged) and reflect it in the UI instead of
        # silently leaving Stop enabled for a thread that no longer exists.
        if self.status_var.get() == "running" and not self.controller.is_running:
            self.status_var.set("stopped (pipeline exited unexpectedly)")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._clear_restart_dirty()

        self._poll_calibration()
        self.root.after(33, self._poll_frame_queue)

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
        h, w = frame.shape[:2]
        self._capture_aspect = w / h

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

    # ---- entry point -----------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
