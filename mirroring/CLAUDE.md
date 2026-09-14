# project_kaspar — MediaPipe → Unreal Engine live mocap

Real-time markerless mocap: webcam → MediaPipe → OSC → a custom Live Link source
driving the UE5 Mannequin (Manny).

## Environment

Windows / PowerShell. There is a venv at `.\venv`.

**Always invoke the venv interpreter directly. Never rely on activation** — venv
activation does not reliably persist between separate shell commands, and a command
that silently falls back to system Python will fail in confusing ways (MediaPipe and
scipy are only installed in the venv).

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1
.\venv\Scripts\python.exe -m pip install <pkg>
```

## Architecture

```
webcam (cv2, MJPG, up to 4K)
  └─ conductor.py            owns the single shared camera + both detectors,
     │                       smoothing (EMA for face, NLERP for pose quats),
     │                       debug overlay, and all sockets
     ├─ mediapipe_face_capture.py  → FaceFrame  (blendshapes, head yaw/pitch/roll)
     │    └─ live_link_face_protocol.py → UDP 11111 (Epic's stock Live Link Face)
     └─ mediapipe_pose_capture.py  → PoseFrame  (33 world landmarks)
          └─ pose_solver.py        → 22 BoneTransforms
               └─ live_link_pose_osc_protocol.py → OSC 9001
                    └─ MediaPipeLiveLink (UE5 C++ ILiveLinkSource, a dumb pipe)
```

MediaPipe is used through the **Tasks API** (`mp.Image`, `.task` model bundles).
The legacy `mp.solutions` namespace and `mediapipe.framework.formats.landmark_pb2`
**do not exist** in the installed version. Drawing helpers come from
`mediapipe.tasks.python.vision` (`drawing_utils`, `drawing_styles`,
`PoseLandmarksConnections`, `FaceLandmarksConnections`) and take raw landmark lists
natively — no protobuf conversion.

## pose_solver.py — treat with care

This file took a long, painful debugging pass to get right. Do not refactor it
opportunistically, and do not "simplify" the parts described below.

How it works: for each bone, compute the minimal **swing** rotation taking the bone's
rest direction to the direction measured from the performer, then convert to
parent-local with `local = parent_global⁻¹ * global`. Everything happens in Unreal's
component space.

`PoseSolver` also solves Manny's 19-bone-per-hand finger rig (`solve_hands()`,
`FINGER_CHAIN`), using the exact same technique one layer further down the same
chain. It was originally a separate `hand_solver.py` reading `PoseSolver`'s public
state from outside (to avoid duplicating the swing formula for hand_l/hand_r across
files) — folded back in once that separation stopped paying for itself, since
`solve_hands()` now reads hand_l/hand_r's live global rotation straight out of
`_solve_body_globals()`, the same computation `solve()` itself uses, with no
duplication at all. `FINGER_CHAIN` is VERIFIED against the same `RefSkeleton` dump
of `SKM_Manny_Simple` as `BIND_POSES` — don't hand-edit it either. One accepted,
disclosed approximation: each metacarpal's aim is measured `WRIST → MCP`, but MediaPipe
has no landmark at the per-finger palm offset its REST direction actually starts from,
costing a few degrees on metacarpals specifically. Every phalanx joint (the ones that
actually drive curl/splay) solves to 0.000° in the bind-pose round-trip check.

Things that were tried and are WRONG — do not reintroduce:

- Computing a delta against a hand-written T-pose reference table and composing it as
  `BIND_POSE * delta`. Not frame-correct: the delta lives in landmark space while
  `BIND_POSE` is a parent-local rotation in component space. Produced a body tilt that
  no sign flip could fix.
- Any "handedness conversion" such as `{-x,-y,-z,w}`. That is the quaternion conjugate,
  i.e. the inverse rotation. Landmark space is component space rotated 90° about Z —
  same handedness — so quaternions carry over directly.
- Hand-deriving `BIND_POSES` / `BIND_POSITIONS`. They are verified against a
  `RefSkeleton` dump of `SKM_Manny_Simple`. If the mesh changes, re-dump; don't guess.

Non-obvious invariants:

- `spine_03`, `spine_05` and `neck_02` exist in Manny but are NOT streamed. They are
  modelled internally anyway (`FULL_CHAIN`), because Unreal applies each streamed local
  transform relative to the bone's REAL parent. Omitting `spine_03` alone rotates the
  whole upper body by ~11°.
- `_convert_landmarks_to_ue_space` mirrors X (`Right = -lm.x`). MediaPipe reports the
  performer's LEFT side with positive `lm.x`.
- Twist about each bone's axis is deliberately unconstrained. MediaPipe gives joint
  positions, which cannot observe forearm pronation.
- MediaPipe reports a vertically standing performer as leaning ~18° forward. This is a
  known bias, corrected by `torso_lean_offset_deg` / `calibrate_neutral()`, not by
  changing the solve.

## Acceptance tests for any solver change

Any change to `pose_solver.py` must keep all three passing. They need no camera —
drive them from recorded landmark dumps.

1. **Rest-pose identity.** Feed the rig's own reference pose in; every one of the 22
   output rotations must equal `BIND_POSES` (22/22).
2. **Bind-pose FK is upright.** Forward-kinematic `BIND_POSES` + `BIND_POSITIONS` with
   no mocap: head above feet, thigh > calf > foot descending, feet symmetric.
   (`foot_l` must land at `(14.09, -0.99, 8.24)`, matching the rig's stored
   `ik_foot_l`.)
3. **Absolute direction error on a real capture.** Compare each bone's 3D direction
   against the performer's, in absolute terms — NOT as an angle-from-spine, which is
   invariant to body roll and will hide a tilt bug. Current baseline: RMS ≈ 0.6° on the
   rest-pose capture, ≈ 2.0° on the kettlebell capture.

The same three, adapted, apply to `solve_hands()`: rest-pose identity (38 finger
rotations must equal each bone's own bind rotation), bind-pose FK sane/symmetric
L/R, and a synthetic bind-pose round-trip showing 0.000° error on every phalanx
joint (metacarpals alone carry a few degrees — see above, not a regression).

## Code standards

- Python 3, type hints, dataclasses for frame/transform structs.
- Comments explain *why*, especially where a non-obvious convention is load-bearing.
- Prefer measuring over guessing: when a pose looks wrong, compute the error against
  ground truth before changing any math.

## gui.py — Tkinter control surface

A single-window control surface for the mocap pipeline. It is a **wrapper**:
`gui.py` is the only new file, and it drives the existing pipeline rather than
reimplementing any of it.

### Hard constraint: do not modify the existing scripts

`conductor.py`, `pose_solver.py`, `mediapipe_*_capture.py` and the protocol modules
must keep working unchanged when run from the command line exactly as they do today.
`python conductor.py --debug --camera 1` must behave identically after this feature
lands.

The one permitted exception, if and only if the wrapper approach below proves
unworkable: add an **optional** `frame_sink: Callable[[np.ndarray], None] | None = None`
parameter to `Conductor.__init__`, called with the annotated frame in `_draw_debug`.
It must default to `None` and preserve current behaviour exactly when unset. Propose
this before doing it — do not change existing files silently.

### Getting frames out of the conductor

`Conductor` owns the camera and calls `cv2.imshow` itself, so the wrapper has to
intercept. Preferred approach, which touches no project file:

- Before constructing `Conductor`, monkeypatch in the `gui` module: `cv2.imshow` to
  push the frame onto a `queue.Queue(maxsize=1)` instead of opening a window;
  `cv2.waitKey` to return `-1`; `cv2.namedWindow` / `cv2.resizeWindow` to no-ops.
- Construct `Conductor(show_debug=True, ...)` ALWAYS. If `show_debug` is `False` the
  conductor never produces an annotated frame and the video pane goes black — the UI's
  debug toggle must therefore be implemented some other way (see below), not by
  flipping `show_debug`.

Use a `maxsize=1` queue and drop frames when full. The GUI must never apply
backpressure to the capture loop; a laggy UI must not become laggy tracking.

### Threading

`Conductor.run()` blocks. Run it on a `threading.Thread(daemon=True)`. Tkinter is not
thread-safe: **all** widget updates happen on the main thread via `root.after(...)`
polling the frame queue. Never touch a widget from the pipeline thread.

Stopping: `Conductor.run()` loops until its own exit condition. Signal it with a
`threading.Event` the wrapper checks, or set the flag the conductor already uses —
whichever exists. Always `join()` with a timeout before restarting, or you leak camera
handles and the next start fails with a device-busy error.

### Live vs restart-required parameters

This split is the core of the design. Get it right before building widgets.

**Live** — adjustable while running, by assigning to the live object's attributes:
- face / pose / hand smoothing factors (the `Smoother` alpha values)
- torso lean offset (`pose_solver.torso_lean_offset_deg`), plus a "Calibrate upright"
  button calling `pose_solver.calibrate_neutral(...)`
- debug overlay toggle, FPS display toggle

**Restart-required** — baked in when the MediaPipe task or camera is constructed:
- camera index, capture width/height
- every detection/tracking/presence confidence threshold
- OSC and UDP ports, target IPs
- model selection (lite/full/heavy)

Restart-required controls must visibly indicate that they need a restart. Either
disable them while running, or mark them dirty and enable an "Apply & Restart" button.
Do not silently accept a change that has no effect — that is the single most confusing
failure mode for this kind of panel.

### Debug overlay toggle

Since `show_debug` must stay `True`, implement the toggle by monkeypatching
`mediapipe.tasks.python.vision.drawing_utils.draw_landmarks` to a no-op while the
toggle is off. Leave the status text (`cv2.putText`) alone — it is cheap and useful.
Note in a comment that skeleton drawing on a 4K frame is genuinely expensive, so this
toggle is a real performance control, not just cosmetic.

### FPS counter

Measured in the wrapper from frames arriving on the queue, not from anything inside the
conductor. Show a rolling average over ~30 frames, not the instantaneous value.
Draw it into the video pane when the debug toggle is on.

### Layout

- Window height **max 720 px**. Resizable. Set a sensible `minsize`.
- Roughly half the window is the webcam view. Preserve the capture aspect ratio when
  scaling (letterbox rather than stretch); a distorted preview makes the tracking look
  broken when it is not.
- Controls in the other half, grouped: Camera · Smoothing · Calibration · Advanced.
- **Advanced** is a collapsible section, collapsed by default, holding ports, IPs,
  thresholds and model selection. A plain expand/collapse toggle is fine.

### Styling

Dark, technical, no icons or decoration. Flat colours, one accent for active state,
monospace for numeric readouts. `ttk` does not honour background colours on all
platforms with the default theme — use `ttk.Style().theme_use("clam")` as the base
before restyling, or plain `tk` widgets where `ttk` fights you. Define the palette as
module-level constants; do not scatter hex codes through the layout code.

### Dependencies

Displaying an OpenCV frame in Tkinter needs `Pillow` (`PIL.ImageTk`). Add it to the
project's requirements. Convert BGR→RGB before handing the frame to PIL, or the preview
comes out blue.

### Acceptance checks

1. `python conductor.py --debug --camera 1` still works, unchanged.
2. Changing a smoothing slider visibly changes responsiveness without a restart.
3. Changing the camera index and restarting picks up the new camera, with no
   device-busy error on the old one.
4. Toggling the debug overlay off measurably raises the FPS counter at 4K.
5. Closing the window terminates the pipeline thread and releases the camera — no
   orphaned `python.exe`.
