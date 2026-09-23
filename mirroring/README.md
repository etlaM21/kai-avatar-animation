# project_kaspar — MediaPipe → Unreal Engine live mocap

**Project:** project_kaspar

**Status:** the live-capture lane is built and working end to end — one webcam
drives face (ARKit blendshapes + head rotation), full body pose, and finger
tracking on Unreal Engine 5.8's "Manny" mannequin in real time, and from there
a MetaHuman (`BP_NewMetaHumanCharacter`, §6). Head rotation reaches the
MetaHuman through the Live Link Face channel, driven by the solver's face-mesh
head basis and verified in the engine. The second, generated-motion lane
(Kimodo/SOMA, §9) has not been started.

**Target engine:** Unreal Engine 5.8

## 1. Goal

Drive a character from two independent motion sources at once:

- **Live capture** *(this module, done)* — a webcam feed processed through Google
  MediaPipe (face + body + hands)
- **Generated motion** *(not started)* — NVIDIA Kimodo/SOMA output, streamed as an
  offline gesture/motion library rather than real-time generation

Both are meant to reach the same character through one coherent custom LiveLink
pipeline. Only the live-capture lane exists in code today; everything below
describes it as actually implemented, not as planned.

## 2. Current pipeline

```
webcam (cv2, MJPG, up to 4K)
  └─ conductor.py            owns the one shared camera, the detector, the
     │                       solver, EMA/NLERP smoothing, debug overlay, and
     │                       both UDP sockets. Nothing else touches the camera
     │                       or a socket directly.
     │
     ├─ mediapipe_holistic_capture.py   HolisticLandmarker, ONE inference pass
     │     │                            covering pose + face + both hands
     │     ├─ PoseFrame   (33 world landmarks)
     │     ├─ FaceFrame   (52 ARKit blendshapes + face mesh landmarks)
     │     └─ HandsFrame  (21 landmarks x 2 hands, world + image space)
     │
     ├─ head_pose_capture.py           DISABLED (commented out, kept on purpose -
     │                                 see CLAUDE.md). It never detected the face
     │                                 at performance distance; head rotation now
     │                                 comes from pose_solver's face-mesh basis.
     │
     ├─ pose_solver.py                 pure math - no MediaPipe/camera code,
     │     │                           just landmarks in, BoneTransforms out
     │     ├─ PoseSolver.solve(pose, face_mesh, ...)            → 22 body bones
     │     │                           + PoseSolver.head_rotation (face channel)
     │     └─ PoseSolver.solve_hands(pose_lm, L_hand, R_hand)    → 19+19 finger bones
     │          │
     │          └─ live_link_pose_osc_protocol.py → OSC /mediapipe/pose, UDP 9001
     │               └─ MediaPipeLiveLink (UE5 C++ ILiveLinkSource - see §5)
     │
     └─ live_link_face_protocol.py     Holistic blendshapes + head curves from
           │                           head_rotation_to_curves(PoseSolver.head_rotation)
           └─ UDP 11111 (raw, not OSC) → Epic's stock "Live Link Face" plugin
```

Per frame, pose is solved and sent **before** the face packet, because the
face packet's head channels come from that same solve.

`mediapipe_pose_capture.py` is not a detector any more - after the Holistic
migration it's just the shared data model (`PoseLandmark` enum, `PoseFrame`
dataclass) that `pose_solver.py`, `mediapipe_holistic_capture.py` and
`conductor.py` all import, so none of them can drift out of sync on landmark
order or field names.

### Wire formats

- **Face → UDP 11111.** Not OSC - Epic's own proprietary 61-float packet (the
  same one the iOS Live Link Face app emits), reimplemented in
  `live_link_face_protocol.py`. 52 ARKit blendshapes + headYaw/Pitch/Roll + 6
  always-zero eye-rotation channels (MediaPipe gives eye-look blendshapes, not
  a separate eye bone rotation). Consumed by the stock Live Link Face plugin -
  zero custom UE code on this side. On a MetaHuman these head curves are the
  ONLY route to the visible head (the body stream's `neck_02`/`head` never
  reach it). Measured in UE 5.8 with `tests/ue_head_probe.py --measure`:
  the curves set `neck_02` + `head` to an **absolute** component-space
  rotation (the torso underneath is ignored), 50° per unit on every axis,
  linear to at least 75°; `headYaw` + turns the head to the character's left,
  `headPitch` + looks up, `headRoll` + tips the top of the head to the
  character's right; composed pitch · roll · yaw (yaw applied first).
  `head_rotation_to_curves()` encodes exactly that, from the head's full
  rotation (torso included). Smoothed with the pose alpha, so it can't lag the
  body stream by a different amount, and held independently of the
  blendshapes, so a face-mesh dropout doesn't freeze it.
- **Pose + hands → OSC `/mediapipe/pose`, UDP 9001.** One message per frame:
  `[present (1.0/0.0), then 7 floats per bone (position x/y/z, rotation
  x/y/z/w)]` for all 60 bones (22 body + 19 left-hand + 19 right-hand,
  fixed order) = `1 + 60*7 = 421` floats, **always** that length - even while
  holding last-valid data with `present=0.0` - so the C++ side never has to
  handle a variable-length packet.

### Running it

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1
```

Needs `holistic_landmarker.task` next to `conductor.py` (download link in
`mediapipe_holistic_capture.py`'s docstring). `face_landmarker.task` is only
needed if `head_pose_capture.py` is ever reinstated; `pose_landmarker_full.task`
is a leftover from before the Holistic migration - nothing loads either today.
In the debug window, press `c` while standing upright to calibrate (a 3 s
countdown, then lean and neutral head pose averaged over 30 frames), `Esc` to
quit. MediaPipe's forward-lean bias is not constant across setups (~18° in
early sessions, ~7° in the current recordings) - always calibrate.

### Recording + offline solver checks

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\rest.npz
.\venv\Scripts\python.exe tests\solver_checks.py              # runs on every recordings\*.npz
```

`--record` dumps the raw (pre-solve, pre-smoothing) landmarks of the session on
exit (`landmark_recorder.py`); `head_ypr_deg` is still written, as NaN, so every
recording keeps one layout. `recordings\trims.json` cuts the walk to and from
the laptop out of each clip. `tests/solver_checks.py` needs no camera. Asserted
(exit code 1 on failure): rest-pose identity, bind-pose FK, timed calibration,
occlusion gating (including the `solve()` + `solve_hands()` call pattern
`conductor` uses), ground locking, the face-channel head round-trip through the
wire with the torso turned (2e), and that the head channels actually move on
real captures (2f). Reported: absolute per-bone direction error on each
recording. Ground truth uses an anatomical MediaPipe→Manny joint mapping - and,
for the head channel, the MetaHuman's measured behaviour - written out
independently of the solver's own tables, so a mis-mapped bone shows up as error
instead of passing by construction.

### Unreal probe (`tests/ue_head_probe.py`)

Streams known Body + Live Link Face inputs to a running editor, stage by stage
(rest, turned head, turned torso, each head curve ±, all three combined).
`--measure` reads the MetaHuman's bone rotations back over the editor's Python
remote execution (Project Settings → Python → Enable Remote Execution) and
prints them against the rest stage, so questions about what Unreal does with a
signal are answered with numbers rather than by eye. Stop `conductor.py` first -
both send to the same ports.

Or drive the same pipeline from a GUI instead of the terminal:

```powershell
.\venv\Scripts\python.exe gui.py
```

### GUI control surface (`gui.py`)

A single-window Tkinter wrapper around `Conductor` - it does not reimplement
any tracking/smoothing/solving logic, only drives a `Conductor` instance
through its constructor and public attributes. `python conductor.py --debug
--camera 1` keeps working from the command line exactly as before.

- **Live pane** shows the same annotated frame `conductor.py`'s own debug
  window would. The window **height follows the camera's aspect**: at a given
  width a 4:3 camera gets a taller window than a 16:9 one, so the frame fills
  its pane with no letterbox bars and is never cropped (only ever scaled to
  fit, and redrawn when the pane resizes). Only the width is user-resizable;
  a window that would run off the screen is narrowed instead. Above the pane:
  the **camera index** (restart-required), then the debug-overlay and FPS
  toggles. Below it: run status, **POSE / FACE tracking status** (green/red,
  the same state the overlay burns into the frame), and a rolling FPS readout
  measured from frames actually received.
- **Live controls** (applied by assigning straight onto the running
  `Conductor`, no restart): face and pose **smoothing**, shown as 0 = raw up
  to 0.99 = smoothest and sent as the inverted EMA alpha (pose smoothing also
  covers both hands and the face channel's head rotation), the
  **Calibrate upright** button (`pose_solver.begin_calibration()`, same as
  `c` in the plain debug window: a 3 s countdown drawn big into the video
  pane, then the lean and the neutral head pose averaged over 30 frames), and
  the torso lean offset it fills in, which can also be set by hand. The lean is
  measured relative to Manny's own rest torso (5.8° off vertical), so
  calibrating on a performer who matches the rig is a no-op.
- **Restart-required controls**, grouped under a collapsible, scrollable
  **Advanced** section (collapsed by default): capture resolution, model
  path, face/pose target IP:port, and Holistic confidence thresholds.
  Changing any of these (or the camera index) while running marks the panel
  dirty and enables **Apply & Restart** instead of silently doing nothing.
- **How it gets frames out and stops cleanly**, since `Conductor` owns the
  camera and calls `cv2.imshow`/`cv2.waitKey` itself: `gui.py` monkeypatches
  the shared `cv2` module (`imshow` → push onto a `maxsize=1` queue,
  drop-when-full; `waitKey` → returns Esc once a `threading.Event` is set,
  reusing `Conductor.run()`'s own existing exit path instead of inventing a
  new one; `namedWindow`/`resizeWindow` → no-ops) before ever constructing a
  `Conductor`, and runs `Conductor.run()` on a daemon thread, joined with a
  timeout on Stop/restart/window-close so the camera is never left held by
  an orphaned thread.
- `mediapipe_holistic_capture.py` takes the confidence thresholds as optional
  constructor kwargs, all defaulting to `0.5`. (`head_pose_capture.py` has the
  same, unused while it is disabled; its GUI fields are commented out with it.)
- Needs `Pillow` (`PIL.ImageTk`), added to `requirements.txt`.

### Orphaned files, kept but not live

- `head_pose_capture.py` - **disabled on purpose, not dead**: commented out at
  every call site (`conductor.py`, `gui.py`) with a pointer to CLAUDE.md. Its
  standalone FaceLandmarker never detected the face at performance distance,
  but it is the seed of the face-crop experiment (§9): run on a crop around
  the pose-derived head, it would recover `output_facial_transformation_matrixes`,
  the one thing Holistic can't provide.
- `mediapipe_pose_osc_protocol.py` - imported by `conductor.py` but only
  reachable through dead code (a `'''...'''`-quoted block after an unconditional
  `return`-equivalent). Predates `pose_solver.py`/OSC-bone-transform encoding.
- `live_link_pose_json_protocol.py` - not imported anywhere. An alternate
  JSON-based pose wire format that was never wired into `conductor.py`.
- `ue_plugin/` (this repo) - the plugin's original home; superseded by a
  synced copy at `K:\KaiTracking\Plugins\MediaPipeLiveLink`, which is the one
  the project's `.uproject` now actually builds (see §5). Edit the C++ there,
  not here - this copy no longer participates in the build.

## 3. Why two transport lanes instead of one

**Face rides a shortcut that already exists.** Epic's stock "Live Link Face"
plugin listens for a fixed, proprietary 61-float UDP packet - the same format
the iOS Live Link Face app emits, based on ARKit's standardized 52-blendshape
set. Because that schema is standardized, a fixed listener can exist at all.

**Body (and now hands) have no equivalent shortcut.** There's no standardized
schema for full-body skeletons the way there is for ARKit blendshapes - rig
joint counts and naming vary per project. Epic doesn't ship a plug-and-play
body listener, so pose+fingers (and, eventually, Kimodo/SOMA) both need a real
custom `ILiveLinkSource` regardless of wire format.

## 4. Why OSC for the custom lane

Not a universal "OSC beats JSON" claim:

1. **Kimodo/SOMA already emits OSC** (per the original design survey) -
   standardizing the pose lane on OSC too means one parser in the plugin
   instead of two, once that lane exists.
2. **Ecosystem compatibility** - OSC is the common language of
   real-time/show-control tooling (lighting, TouchDesigner, etc.), which may
   matter for a live theatrical context later.

JSON was a fully legitimate alternative (DollarsMoCap's trial plugin streams
body data as plain JSON with no measurable parsing bottleneck) - Epic ships
`Json`/`JsonUtilities` exactly as readily as `OSC`. If Kimodo's OSC output
weren't already a given, JSON would be an equally reasonable choice.

## 5. Custom `ILiveLinkSource` plugin - what's actually built

Lives at `K:\KaiTracking\Plugins\MediaPipeLiveLink` (this repo's `ue_plugin/`
copy is stale - see §2). Simpler than the DollarsMoCap-derived design
originally sketched here:

- **One class, `FMediaPipeLiveLinkSource`**, implements `ILiveLinkSource` and
  `FGCObject` (to keep its `UOSCServer` alive against the garbage collector) -
  no hand-rolled `FUdpSocketBuilder` + `FRunnable` worker thread. Epic's own
  `OSC` module owns the socket and threading; the plugin just binds
  `OnOscMessageReceivedNative`.
- **All the retargeting math happens in Python, not C++.** `pose_solver.py`
  sends already-fully-solved local bone quaternions; the plugin does nothing
  more than `FTransform(FQuat(...), FVector(...))` per bone and pushes the
  frame - it never composes against a bind pose itself. (An earlier version of
  this doc described incoming quaternions as bind-pose deltas composed in
  C++ - that's not what got built, and not what `pose_solver.py`'s own history
  found correct either; see its module docstring.)
- **Multiple Subjects per Source, auto-created:** `CreateSubject` is idempotent
  via a `TSet` + critical section, matching the original plan - this is how a
  future Kimodo/SOMA subject would coexist with `MediaPipePose` under the same
  Source.
- **Tracking-valid flag** carried via `FLiveLinkSkeletonStaticData::PropertyNames`
  + `FLiveLinkAnimationFrameData::PropertyValues` (`"present"`), not a fake bone.
- **Target skeleton is Manny, not a MetaHuman directly** - bone names/parents
  are hardcoded (`MediaPipeBoneNames`/`MediaPipeBoneParents` in
  `MediaPipeLiveLinkSource.cpp`, 60 entries: 22 body + 38 fingers), verified
  against a `RefSkeleton` dump rather than guessed. `ExpectedArgs` is computed
  from that array's length, not hardcoded, so extending it (as the finger
  bones did) needed no other logic change.
- **Build.cs**, for reference: Public - `Core`, `LiveLinkInterface`,
  `LiveLink`, `LiveLinkAnimationCore`, `OSC`. Private - `CoreUObject`,
  `Engine`, `Slate`, `SlateCore`, `Sockets`, `Networking`, `InputCore`.

## 6. Downstream: Manny → MetaHuman

Live Link Pose drives Manny 1:1 (no retargeting). In the `KaiTracking` UE
project, `BP_NewMetaHumanCharacter` carries three skeletal mesh components,
as seen from the editor during this session's measurements:

- `MannySource` - `SKM_Manny_Simple` on the Live Link AnimBP, the retarget
  source.
- `Body` - the MetaHuman body on `ABP_MH_LiveLink`, following Manny (a 45°
  torso turn on Manny lands as exactly 45° on the MetaHuman pelvis and spine).
- `Face` - the MetaHuman face on `ABP_Face`, which copies the Body's pose and
  takes its blendshapes **and head rotation** from Live Link Face.

**Who owns the head.** On the MetaHuman the visible head is the Face
component, and the Body stream's head rotation never reaches it - measured:
with the head turned 30° on Manny and the face curves at zero, the MetaHuman's
`neck_02`/`head` stay at 0.0°. Head rotation is therefore sent over the Live
Link Face channel (§2, wire formats). No Epic asset was edited for this; the
body stream keeps sending its own head for Manny-only and non-MetaHuman
targets. The alternative - changing `ABP_Face`'s layered-blend branch root so
the Body's head passes through - works too, but is a per-user edit to an Epic
asset (see CLAUDE.md).

Still open: the retarget's solver choice is a live-performance tradeoff -
**Full Body IK** gives the best proportion correction but is the most
expensive per frame; **Body Mover + Limb IK Solvers** trades some correction
for runtime cost. Not benchmarked. One Manny → MetaHuman setup serves both
MediaPipe and, later, Kimodo, since both drive the same Manny proxy.

## 7. Reference material

- **JimWest/MeFaMo and JimWest/PyLiveLinkFace** - face capture reference. Both
  archived by the owner on 2026-08-13. They use the legacy
  `mediapipe.python.solutions.face_mesh` API, deprecated in favor of the Tasks
  API after 2023 - this codebase uses the Tasks API throughout
  (`HolisticLandmarker`/`FaceLandmarker`, see CLAUDE.md).
- **DollarsMoCap (Sunnyview Inc.) trial plugin** - reviewed both its wire
  protocol and full C++ source; the patterns actually adopted (multi-subject
  auto-creation, the tracking-valid property, hardcoding Manny) are called out
  in §5, alongside where the built plugin diverged from it.
- MediaPipe's PyPI package currently supports **Python 3.9-3.12** only - no
  3.13/3.14 wheels.

## 8. Delivered so far

- **Face**: HolisticLandmarker blendshapes + head yaw/pitch/roll → Epic's stock
  Live Link Face plugin. The head curves come from the solver's face-mesh head
  basis (`PoseSolver.head_rotation` → `head_rotation_to_curves()`), converted
  to what the MetaHuman was measured to do with them (§2). Verified end to end
  in UE 5.8: turned and leaning torsos with turned heads put the MetaHuman's
  head on target to 0.000°. This replaces the second FaceLandmarker pass
  (`head_pose_capture.py`, now disabled), which at performance distance
  detected the face on 31 of 652 frames (Holistic: 597) - Live Link Face's
  head rotation had been pinned at zero for the whole project.
- **Body**: `pose_solver.py`'s `solve()` - per-bone minimal-swing rotation
  from rest direction to measured direction, converted to parent-local in
  Unreal's component space. 22 bones, verified against three acceptance tests
  (see CLAUDE.md).
- **Head (body stream)**: `neck_01`/`head` get a full orientation (yaw, pitch
  and roll) from the Holistic face mesh (cheek extremes + eye corners for the
  side axis, forehead→chin for up), split 40/60 between neck and head, held
  across face dropouts. "Calibrate upright" also records the neutral head pose,
  which both lanes share: after calibration the head sits in line with the
  torso. Chosen over the pose model's own ear/nose points by measurement: those
  caught ~8° of a ~37° head turn. On a MetaHuman the Live Link Face lane owns
  the visible head (§6); this body-stream head drives Manny.
- **Hands (wrist orientation)**: `hand_l`/`hand_r` come from a full palm basis
  (wrist→middle MCP, plus the palm normal from the index/pinky MCPs), built with
  the identical formula from the rig's rest hand so geometric offsets cancel.
  This is what makes wrist ROLL observable at all - a swing aim leaves rotation
  about the bone axis free, and the fingers were inheriting the torso's roll.
  Half the wrist's twist is passed back to `lowerarm_*`, so pronation comes from
  the forearm instead of snapping at the wrist (`FOREARM_TWIST_SHARE`). Falls
  back to the old pose-INDEX aim per hand when that hand isn't tracked.
- **Ground locking**: the pelvis height is derived each frame so the lower foot
  sits on the rig's own floor (the ball of the foot at 0.75 cm in bind), instead
  of being pinned at `pelvis_default_height_cm`. MediaPipe's world landmarks are
  hip-centred, so hip height carries no information and the pinned version let
  the feet sink up to 2.7 cm while merely standing, more with any crouch. The
  rest pose still reproduces the bind pelvis height exactly.
  `PoseSolver(ground_lock=False)` restores the old behaviour. Known limits: a
  real jump reads as grounded, and while both feet are occluded the height comes
  from their held pose.
- **Occlusion gating**: MediaPipe never reports "I can't see that limb" - it
  invents a plausible landmark and lowers `visibility`/`presence`. Each aimed
  bone is gated on the lowest of those across its own landmarks: below 0.5 it
  holds its last trusted pose *relative to the torso* (so an occluded limb
  still travels with the body), and resumes above 0.65, easing back over 8
  frames (live, too: the hold used to step twice per frame under `conductor`,
  halving that, until fixed and covered by a check). Entering the hold is instant - the held pose is where the character
  already is, while blending in would show a frame of the invented one. On the
  recordings this engages on feet/calves for 17-22% of frames and cuts their
  frame-to-frame jitter ~25%; arms and torso never drop below threshold.
- **Fingers**: `pose_solver.py`'s `solve_hands()` - the identical technique
  extended one layer past hand_l/hand_r, verified against a real
  `RefSkeleton` dump of Manny's finger rig (19 bones/hand: metacarpal + 3
  phalanges x 4 fingers, 3 phalanges for the thumb). Every phalanx joint
  round-trips to 0.000° error in the bind-pose test; metacarpals carry a
  few degrees from a disclosed, accepted approximation (no MediaPipe landmark
  sits at each finger's individual palm offset - see `pose_solver.py`).
- **Custom UE5 plugin** (`MediaPipeLiveLink`, §5) - built on Epic's `OSC`
  module, dynamic bone count, multi-subject-ready, driving Manny live.
- **Debug overlay** - full pose/face-mesh/hand landmark drawing, tracking
  status per channel, live torso-lean readout + one-key calibration.
- **GUI control surface** (`gui.py`, §2) - Tkinter wrapper driving a
  `Conductor`: window height fitted to the camera's aspect (no letterbox, no
  crop), camera index in the video toolbar, POSE/FACE tracking status next to
  the FPS, smoothing sliders reading 0 = raw, calibration up front, a
  scrollable restart-required Advanced section, and clean Start/Stop with no
  orphaned camera handles.
- **Unreal head probe** (`tests/ue_head_probe.py`, §2) - known inputs in,
  MetaHuman bone rotations read back out of the editor.

## 9. Open items / next steps

- **Head is still a bit wobbly live.** The conversion itself is exact (§8), so
  the jitter comes from the face-mesh basis frame to frame and/or smoothing;
  measure it on a recording before tuning anything.
- **Ear/nose fallback head aim - now load-bearing.** Head rotation depends
  entirely on the face mesh, which is lost when the performer turns away. Blend
  to an aim from the pose model's `LEFT_EAR`/`RIGHT_EAR`/`NOSE` as face
  confidence drops, reusing the occlusion-gating hysteresis. Record a 360° turn
  first so the dropout point is measured.
- **Lean calibration with a turned torso.** `measure_torso_lean_deg()` reads
  lean along the camera's forward axis, so calibrating while turned
  under-counts Manny's 5.8° rest lean (0.35° at 20°, 1.7° at 45°). Calibrate
  square to the camera until this is fixed.
- **MetaHuman-side effects of the head curves** (measured, not ours): the
  Body mesh's `neck_01` also reacts to them (up to 13° for a 50° pitch) while
  the Face mesh's `neck_01` barely moves (~2°), a possible seam at the neck;
  and the pelvis/spine shift by up to 1.6°.
- **Root translation.** MediaPipe world landmarks are hip-centred, so the
  character can never step, shift or jump. Needs image-space landmarks or a
  separate position estimate - an architecture decision, not a tune.
- **Distortion layer** - tracking artefacts as art-directable effects,
  downstream of a clean solver (see CLAUDE.md for the plan).
- **Face crop experiment** - reinstate `head_pose_capture.py` on a crop around
  the pose-derived head, only if the landmark-derived basis proves fragile.
- **Knees converge** - the character's stance is narrower than the
  performer's (~0.65 knee/hip ratio). Not investigated.
- **Kimodo/SOMA lane**: not started - no code in this module talks to it yet.
- **Finger twist**: unconstrained, same limitation as body twist - MediaPipe
  gives joint positions, not rotations, so finger roll can't be observed.
  (Wrist roll can, from the palm plane - §8.)
- **Cleanup candidates**: the orphaned files in §2 (`mediapipe_pose_osc_protocol.py`,
  `live_link_pose_json_protocol.py`, `pose_landmarker_full.task`, this repo's
  `ue_plugin/` copy) could be removed once nobody needs them as reference.
- Benchmark Full Body IK vs. Body Mover + Limb IK Solvers for the MetaHuman
  retarget (§6).
