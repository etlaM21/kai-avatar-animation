# project_kaspar — MediaPipe live mocap + Kimodo generated motion → Unreal Engine

**Project:** project_kaspar

**Status (2026-10-07):** both lanes work end to end.

- **Live capture** (§2–8): one webcam drives face (ARKit blendshapes + head
  rotation), full body pose, and finger tracking on Unreal Engine 5.8's "Manny"
  mannequin in real time, and from there a MetaHuman (`BP_NewMetaHumanCharacter`,
  §6). Head rotation reaches the MetaHuman through the Live Link Face channel,
  driven by the solver's face-mesh head basis and verified in the engine.
- **Generated motion** (§9): type a prompt on the laptop; NVIDIA Kimodo generates
  the motion on the DGX Spark; the clip is cached, retargeted onto Manny on Windows
  and played into Unreal through the **same** two channels. Verified driving the
  MetaHuman. Playback is deliberately plain for now (loop, hard cut between clips).

**Target engine:** Unreal Engine 5.8

## 1. Goal

Drive a character from two independent motion sources at once:

- **Live capture** *(done)* — a webcam feed processed through Google
  MediaPipe (face + body + hands)
- **Generated motion** *(Phase 1 done)* — NVIDIA Kimodo (SOMA skeleton) clips,
  generated on demand on the DGX Spark from a text prompt, cached on disk so they
  also form an offline library. Real-time streaming generation (ARDY) is a later
  phase.

Both reach the same character through one coherent custom Live Link pipeline: they
meet at a single seam, a list of 60 Manny `BoneTransform`s per frame plus a head
rotation, and everything downstream of that seam is shared. Everything below
describes what is actually implemented, not what is planned. Run one lane at a
time: both send to the same ports (mixing them is a later phase).

## 2. Current pipeline (live lane)

(The generated-motion lane is §9. Both lanes meet at the same seam, the 60
`BoneTransform`s + head rotation handed to the two encoders below, so §2's wire
formats and §5–6 apply to both.)

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
..\venv\Scripts\python.exe conductor.py --debug --camera 1
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
..\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\rest.npz
..\venv\Scripts\python.exe tests\solver_checks.py              # runs on every recordings\*.npz
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
..\venv\Scripts\python.exe gui.py
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
  but it is the seed of the face-crop experiment (§10): run on a crop around
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

1. **Kimodo/SOMA already emitted OSC** in the earlier service experiments
   (`pipeline-network-osc/`) - standardizing the pose lane on OSC meant one parser
   in the plugin instead of two. As built, the generated-motion lane (§9) does not
   stream from the GPU machine at all: it retargets on Windows and sends the very
   same `/mediapipe/pose` packet as the webcam lane, so the plugin has exactly one
   format to parse.
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
MediaPipe and Kimodo (§9), since both drive the same Manny proxy.

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
- **Generated-motion lane, Phase 0 + 1** (§9) - Kimodo SOMA77 clips retargeted onto
  Manny with the solver's own technique (bone directions exact against the source,
  head and root motion included); saved BVH clips play offline; prompts typed on the
  laptop are generated on the DGX Spark, cached, checked and played into Unreal
  through the unchanged encoders, plugin and MetaHuman setup. 55 offline acceptance
  checks, including a real Kimodo clip.

## 9. Generated-motion lane: Kimodo on the DGX Spark

A text prompt typed on the laptop becomes motion on the MetaHuman, with the GPU work
on the DGX Spark and every piece of Manny knowledge on Windows. This section follows
one prompt through every stage. Code lives in two folders at the module root:
`remote_kimodo_service/` (service, NPZ contract, client, cache, adapter, player) and
`procedural_animation/` (BVH reader, skeleton tables, the retarget, the CLI, tests).
`remote_kimodo_service/README.md` is the short operator's guide; this is the full
explanation.

### 9.1 The whole picture

```
 DGX Spark "kaspar" (GB10, aarch64)          Windows laptop (mirroring venv, no torch)                     Unreal 5.8
┌─────────────────────────────────┐        ┌───────────────────────────────────────────────────────┐     ┌──────────────┐
│ kimodo_service.py  (FastAPI)    │        │ procedural_conductor.py                               │     │ MediaPipe-   │
│  127.0.0.1:8765                 │  SSH   │  main thread: REPL  ─┐                                │     │ LiveLink     │
│  Kimodo-SOMA-RP-v1.1, warm      │ tunnel │                      ▼ prompt queue                   │ OSC │ plugin       │
│  asyncio.Lock: one at a time    │◄──────►│  "generation" thread:                                 │9001 │  → Manny     │
│  POST /generate → NPZ bytes     │ HTTP   │   clip_cache ─hit─┐                                   │────►│  → IK Retarg.│
│  GET  /health                   │        │   kimodo_client ──┴► kimodo_contract.unpack           │     │  → MetaHuman │
│  in-memory result cache (32)    │        │   → kimodo_adapter (FK self-check, Root, m→cm)        │ UDP │              │
│  timing_log.txt                 │        │   → Retargeter.retarget_globals → RetargetedMotion    │11111│ Live Link    │
└─────────────────────────────────┘        │   → clip_cache.put → player queue                     │────►│ Face → head  │
                                           │  "LoopPlayer" thread, 60 Hz:                          │     │ (ABP_Face)   │
                                           │   sample(t) → 60 BoneTransforms + head rotation       │     └──────────────┘
                                           │   → LiveLinkPoseOSCEncoder / LiveLinkFaceEncoder      │
                                           └───────────────────────────────────────────────────────┘
```

Design rules behind this shape (measured reasons in
`docs/remote-kimodo-streaming.md`):

- **Windows asks, the Spark answers.** Every connection starts on the laptop, so no
  firewall rule is needed anywhere and it works from any network.
- **The network never touches the real-time path.** A whole clip is fetched, checked
  and retargeted before the player ever sees it. A slow or dead link delays the *next*
  clip; it can never make the current one stutter. Unreal only receives packets from
  `127.0.0.1`, exactly as with the webcam lane.
- **The Spark sends raw Kimodo data and knows nothing about Manny.** The retarget,
  its tables and its tests all stay on Windows. Streaming swizzled per-bone
  quaternions from the GPU machine straight into Unreal is what failed in the earlier
  attempts (`procedural-animation.md` §3).
- **Every clip is cached.** If the Spark is down, the cache is the library.

### 9.2 Running it

**On the Spark** (once per session, in your own SSH login):

```bash
cd ~/project_kaspar/modules/kai-avatar-animation && git pull
cd remote_kimodo_service
../pipeline-network-osc/venv/bin/python kimodo_service.py
```

It loads the model (~25 s) and prints `Model loaded`. It runs in the foreground and
stops when you log out or press `Ctrl+C`, on purpose: no tmux, no systemd, no
auto-restart. The existing `pipeline-network-osc/venv` already has Kimodo 1.0.0
(installed from the aarch64-patched `~/kimodo-src`), CUDA torch and FastAPI;
`remote_kimodo_service/requirements.txt` explains a fresh setup.

**On the laptop, the tunnel** (a second PowerShell window, left open):

```powershell
ssh -N -L 8765:127.0.0.1:8765 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 etlam@100.83.6.8
```

The service listens only on the Spark's own loopback, so nothing on the Spark's
network can reach it. `-L` makes that port appear on the laptop as `127.0.0.1:8765`,
carried inside the SSH connection. After the password it prints nothing; that is the
working state. `curl.exe -s http://127.0.0.1:8765/health` checks the whole chain.

**On the laptop, play** (from the module root; stop `conductor.py` first):

```powershell
# REPL
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor
# one shot: generate (or take from cache), play once, exit
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --prompt "A person waves both arms /s 25"
# offline
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --bvh assets\kimodo\clips\kimodo-gen\wave.bvh --loop
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --clip remote_kimodo_service\cache\<file>.npz --loop
```

| Flag | Default | |
|---|---|---|
| `--prompt` / `--bvh` / `--clip` | none = REPL | the source; mutually exclusive |
| `--seconds` | 9 | clip length, max 10 (Kimodo's limit per prompt) |
| `--steps` | 100 | denoising steps |
| `--seed` | 0 | an integer or `random` |
| `--spark-url` | `http://127.0.0.1:8765` (env `KIMODO_URL`) | the tunnel's end |
| `--timeout` | 60 s (env `KIMODO_TIMEOUT`) | per request |
| `--cache-dir` | `remote_kimodo_service\cache` (env `KIMODO_CACHE_DIR`) | gitignored |
| `--loop` | off | one-shot / offline: loop until `Ctrl+C` (the REPL always loops) |
| `--rate`, `--speed` | 60 Hz, 1.0 | send rate, playback speed |
| `--pose-ip/--pose-port`, `--face-ip/--face-port` | 127.0.0.1:9001, :11111 | conductor.py's defaults |
| `--no-face`, `--dry-run` | | skip the head channel; send nothing at all |

Exit codes for one-shot: 0 played, 2 error (bad prompt, refused clip), 3 Spark
unreachable and the clip not cached.

### 9.3 Stage 1 — the prompt (Windows, main thread)

A REPL line that starts with `/` is a command:

| Command | |
|---|---|
| `/list` | the cached clips, numbered |
| `/play N` | queue cached clip N (works offline) |
| `/next` | cut to the queued clip now instead of at the end of the current pass |
| `/status` | what is playing (pass number), how many clips are queued / generating |
| `/health` | ask the Spark service |
| `/help`, `/quit` | (`Ctrl+C` also quits) |

Anything else is a prompt. Options can sit anywhere in it, also in `--prompt`:
`/s N` steps, `/seed N` or `/seed random`, `/t N` seconds. `A person walks /s 25 /t 4`
asks for 4 s at 25 steps. Only standalone `/word` tokens count as options, so
"and/or" stays in the prompt; an unknown option is refused with a one-line message.

`parse_prompt` turns the line into a `GenerationRequest` (`kimodo_contract.py`):
the prompt with whitespace collapsed, the seed (`random` is drawn **here**, on the
laptop, so the request and its cache key are known before anything is sent), the
frame count `round(seconds × 30)`, the steps and the model name
(`Kimodo-SOMA-RP-v1.1`). The seed defaults to a fixed 0, so typing the same prompt
again is an instant cache hit; `/seed random` gives variation.

The request goes onto a FIFO queue and the prompt returns immediately: typing never
waits for generation.

### 9.4 Stage 2 — cache first (Windows, "generation" thread)

One background thread works through the queue, **one request at a time**. (The
service only generates one clip at a time anyway; several requests in flight would
just wait at its lock and time out there.)

The cache key is the SHA-256 of a canonical JSON of contract version, model, prompt,
seed, frame count and steps. The same function computes it on both sides of the
network. Files are named readably,
`<prompt-slug>__s<seed>__<frames>f__<steps>st__<first 12 hex of the key>.npz`, and
are found by the key suffix. A hit is unpacked, validated and retargeted without
touching the network at all.

### 9.5 Stage 3 — the request (`kimodo_client.py`)

On a miss: `POST /generate` with `{prompt, seed, num_frames, steps, model}` as JSON,
stdlib `urllib` only (the mirroring venv has no HTTP library and needs none). The
request is synchronous with a 60 s timeout; nothing else waits on it.

Every failure leaves the client as one of two exceptions with a one-line message,
never a traceback:

| What happened | How it shows up | Message |
|---|---|---|
| Tunnel not running | connection refused (on Windows only after 2.05 s: the OS retries the SYN, measured) | `connection refused ... is the SSH tunnel running?` |
| Tunnel up, service not running or still loading | ssh accepts locally, then closes: `RemoteDisconnected` / reset, **not** refused | `accepted and closed the connection - tunnel up, service not running?` |
| No answer in time | socket timeout | `no answer from ... within 60 s` |
| Transfer broke off | incomplete read / reset mid-body | `transfer ... broke off` |
| Service said no | HTTP 4xx/5xx, with the service's `detail` | `service answered 500: generation failed: ...` |

In the REPL, a failure prints that line, playback continues untouched, and cached
clips stay available (`/list`, `/play N`).

### 9.6 Stage 4 — generation (`kimodo_service.py`, on the Spark)

- **Startup:** `HF_HOME` is set to `/opt/huggingface_cache` (the shared cache that
  already holds the Kimodo weights and the LLM2Vec / Llama-3-8B text encoder) before
  Kimodo is imported; then `load_model("Kimodo-SOMA-RP-v1.1", device="cuda")` once,
  inside FastAPI's lifespan hook, so the port only accepts requests once the model is
  warm. It refuses to bind `0.0.0.0`: only `127.0.0.1` (default, for the tunnel) or
  the Tailscale IP.
- **Validation** (pydantic): prompt 1–1000 characters, `num_frames` 1–300 (10 s),
  steps 1–1000. A different `model` than the one loaded → 409. `first_frame` (the
  chaining constraint, reserved in the schema) → 501 in v1. A missing seed is drawn
  by the service and returned in `X-Seed`.
- **The lock:** an `asyncio.Lock` serialises generations, so a running generation is
  never interrupted or overlapped; a second request waits, and that wait is measured.
- **Server-side result cache:** the last 32 clips stay in memory under the same key.
  If the client timed out while the GPU kept working (a thread cannot be cancelled),
  the retry of the same request gets the finished clip instead of a second
  generation.
- **Generation** runs in a worker thread: Python, NumPy, torch and CUDA are seeded
  with the request's seed, then
  `model(prompts=[prompt], num_frames=…, num_denoising_steps=…, post_processing=True)`
  under `torch.no_grad()`. `post_processing=True` stays on: it is Kimodo's foot-skate
  cleanup and constraint enforcement. CUDA is synchronised before the clock stops, so
  the time logged is the real GPU time.
- **What Kimodo returns** (read from its installed source and checked on real
  output): it predicts on its 30-joint `somaskel30`, post-processes the local
  rotations, then converts to the 77-joint `somaskel77` in one forward-kinematics pass
  (`output_to_SOMASkeleton77`). The fingers come from a fixed relaxed-hands rest pose
  (Kimodo does not animate hands). `global_rot_mats` and `posed_joints` therefore
  come from the same FK, in Kimodo's **standard T-pose** convention (identity = the
  T-pose); `root_positions` is `posed_joints[:, root]`, the Hips; `foot_contacts` are
  booleans, 6 per frame (the docs say 4). Kimodo's frame: right-handed, Y up, +Z
  forward, metres, 30 fps.
- **Packing:** the arrays are written with `kimodo_contract.pack` (below), unpacked
  once more as a self-test so the service never sends what the client would refuse,
  and returned as `application/octet-stream` with headers `X-Generation-Seconds`,
  `X-Queue-Seconds`, `X-Seed` and `X-Cache` (hit/miss). One line per request goes to
  the console and to `timing_log.txt`.

### 9.7 Stage 5 — the NPZ contract (`kimodo_contract.py`)

numpy-only and with no package imports, so the **same file** is imported on the Spark
and on Windows. One clip of T frames:

| Key | Shape, dtype | |
|---|---|---|
| `global_rot_mats` | (T, 77, 3, 3) float32 | global joint rotations, Kimodo's frame, standard T-pose convention |
| `root_positions` | (T, 3) float32 | metres |
| `posed_joints` | (T, 77, 3) float32 | metres; global joint positions, used to prove the rotations' meaning |
| `foot_contacts` | (T, N) float32 | 0/1; N = 6 from Kimodo 1.0.0 |
| `fps` | () float32 | 30 |
| `bone_order_names` | (77,) unicode | Kimodo's joint order = the BVH joints after `Root` |
| `meta_json` | () unicode | prompt, seed, num_frames, steps, model, generation time, Kimodo version, created, contract version |

`np.savez_compressed`, about 890 KB for 9 s. Loaded with `allow_pickle=False`, so
nothing in a file can execute code. `unpack` checks structure only: every key
present, shapes, 77 names, finite values, positive fps, valid JSON, matching contract
version. Anything else raises `ContractError`.

### 9.8 Stage 6 — the adapter and the self-check (`kimodo_adapter.py`)

The retarget was built and tested against Kimodo's own T-pose **BVH**
(`assets/kimodo/soma_skeleton/somaskel77_standard_tpose.bvh`, read by
`source_skeletons.load_soma77`). The adapter bridges the three differences between
that skeleton and the NPZ:

1. **The `Root` wrapper.** The BVH has 78 joints, `Root` first; Kimodo's 77 are the
   rest, in the same order (checked name by name). `Root` gets an identity rotation.
2. **Units.** The NPZ is in metres, the BVH skeleton in centimetres, so positions are
   multiplied by 100. Both the rest offsets and the frames have to be in the same
   unit; the retarget's root scaling (9.9) cancels a factor applied to both, but not
   one applied to only one of them.
3. **The root.** The Hips joint's own position (`posed_joints[:, Hips]`), the same
   quantity the BVH path uses.

Then the **self-check**: forward kinematics over the BVH's standard T-pose bone
offsets, with the received rotations, starting from the received Hips position, must
land on Kimodo's own `posed_joints` within 0.5 cm. If it doesn't, the rotations are
tried as Kimodo's *native* rest convention (the one `save_motion_bvh(...,
standard_tpose=False)` writes; converted with Kimodo's own per-joint offset table,
`G_std = G_native · Oⱼᵀ`). If neither fits, the clip is refused with both errors
in the message, and is **never cached**. This one check catches a transposed
rotation, a unit slip, a reordered skeleton or a convention change in a future Kimodo
version. On real output it misses by 0.00003 cm (float32 rounding), and
`root_positions` equals the Hips exactly.

### 9.9 Stage 7 — the retarget (`procedural_animation/retarget.py`)

The same technique as `pose_solver`, applied to a source skeleton instead of
landmarks: global orientations in Unreal component space against **Manny's measured
rest pose**, then parent-local rotations over Manny's full chain. No local rotation
is ever copied across rigs, and no quaternion component is ever swapped or negated.

**Change of basis.** Kimodo: +X = the character's left, Y up, +Z forward. Manny's
component space: +X = the character's left, +Y forward, Z up. So
`C = [[1,0,0],[0,0,1],[0,1,0]]`, i.e. (x, y, z) → (x, z, y). It is a reflection
(det −1) and its own inverse, and rotations change basis as `C · R · C`; never
`C · Rᵀ · C` (that is the inverse rotation, the bug of an earlier attempt).

**The mapping**, per source skeleton (`source_skeletons.SOMA77_TO_MANNY`), all 60
streamed bones:

| Manny | SOMA joint | Note |
|---|---|---|
| `pelvis`, `spine_01`, `spine_02`, `spine_04` | `Hips`, `Spine1`, `Spine2`, `Chest` | |
| `neck_01`, `head` | `Neck1`, `Head` | SOMA's `Neck2` bend lands on these two |
| `clavicle/upperarm/lowerarm/hand_l` | `LeftShoulder/LeftArm/LeftForeArm/LeftHand` | and `_r` |
| `thigh/calf/foot/ball_l` | `LeftLeg/LeftShin/LeftFoot/LeftToeBase` | **SOMA's `LeftLeg` is the thigh** |
| `<finger>_metacarpal, _01.._03` | `<Finger>1..4` | SOMA's `<Finger>1` is the metacarpal |
| `thumb_01..03` | `Thumb1..3` | both rigs' first thumb bone is the metacarpal |

`spine_03`, `spine_05` and `neck_02` are not streamed, so Unreal holds them at bind
relative to their parent; the retarget models them exactly the same way, or every
child's local rotation would be computed against a parent Unreal doesn't have.

**The alignment, once at start-up.** For each Manny bone `b` driven by source joint
`j`, `A[b]` re-poses Manny's bind orientation into the source's rest pose (A-pose →
T-pose) with the minimal swing that turns Manny's rest bone direction onto the source
bone's rest direction:

```
A[b]       = swing(manny_rest_dir[b] → C · src_rest_dir[j]) · manny_rest_global[b]
G_manny[b] = (C · G_src[j](t) · C) · G_ue_rest[j]⁻¹ · A[b]       per frame
local[b]   = G_manny[parent(b)]⁻¹ · G_manny[b]                    over Manny's full chain
```

For SOMA in the standard convention the source rest globals are identity, so
`G_ue_rest` drops out. Manny's rest directions come from `pose_solver`'s own FK of
the verified rig dump (a child joint where one exists, otherwise the bone's own +X
axis, −X on the right side). `hand_*` aims at `middle_01` like SOMA's
`Hand → Middle2`. The **head** is the one exception (`FULL_FRAME_BONES`): both rest
heads look straight ahead, so the alignment is the identity map of rest frames
rather than a swing. Swinging Manny's head axis onto SOMA's `Head → HeadEnd`, which
leans back 6.5° in SOMA's skull geometry, tilted every head up by that much
(measured).

Bone directions come out exact by construction: every Manny bone points where the
source bone points, in absolute component-space terms.

**The pelvis position.** The source root (cm) goes through `C`, is scaled by Manny's
rest pelvis height over SOMA's (95.90 / 100.0 = 0.959, so stride length matches
Manny's legs), and is anchored so the source's rest lands exactly on Manny's rest
pelvis (0, 2.28, 95.90) cm. A source walking 100 cm forward moves Manny 95.9 cm along
+Y. This is real root motion, which the webcam lane cannot give.

**The head for the MetaHuman.** `head_rotation = G_manny[head] · manny_rest_global[head]⁻¹`,
the same quantity `PoseSolver.head_rotation` holds in the webcam lane, so it goes
through `head_rotation_to_curves()` unchanged (§2, wire formats).

The result is a `RetargetedMotion`: per frame 60 parent-local quaternions in wire
order (22 body, 19 left hand, 19 right hand), the pelvis position and the head
rotation. A 9 s clip is self-checked and retargeted in ~0.06 s, on the generation thread. The clip is
written to the cache only now, after it has passed all of the above, and handed to
the player's queue.

### 9.10 Stage 8 — playback (`remote_kimodo_service/player.py`)

`LoopPlayer` runs on its own thread and does nothing but keep time and send:

- **Clock:** every 1/60 s (`time.perf_counter` + sleep to the next tick). If it ever
  falls behind it resets instead of sending a catch-up burst. Worst lateness is
  measured and printed on exit: 0.0 ms so far, including while clips were being
  retargeted on the other thread.
- **Sampling:** `RetargetedMotion.sample(t)` slerps every bone's local rotation
  between the two neighbouring 30 fps source frames, lerps the pelvis and slerps the
  head, so 30 fps clips play smoothly at 60 Hz. `--speed` scales `t`.
- **Queue, v1 (deliberately plain):** the current clip **loops**. A clip that arrives
  waits until the current pass ends and then takes over as a hard cut (`/next` cuts
  at once). Clips play exactly as retargeted: no re-centring, no heading alignment,
  no crossfade, no idle loop, so a new clip may start elsewhere and facing elsewhere.
  Those layers come next (§10). Before the first clip arrives nothing is sent.
  One-shot mode plays its clip once (or loops with `--loop`) and exits.
- **Sending** (`Sender`, the same encoders `conductor.py` uses, imported read-only):
  - **Pose → OSC `/mediapipe/pose`, UDP 9001:** `present = 1.0`, then for each of
    the 60 bones position x/y/z + rotation x/y/z/w = **421 floats**, always. The pelvis
    carries the retargeted root position; every other body bone its bind offset
    (`pose_solver.BIND_POSITIONS`), every finger its rig offset (`FINGER_CHAIN`).
  - **Head → Live Link Face, UDP 11111:** a 61-channel packet with all blendshapes
    at 0 (Kimodo has no face) and `headYaw/Pitch/Roll` from
    `head_rotation_to_curves(head_rotation)`.
  - **On stop:** the last pose once more with `present = 0.0`, the same
    "tracking lost" signal `conductor.py` sends.

### 9.11 Stage 9 — Unreal

Nothing in Unreal knows the motion is generated. The `MediaPipeLiveLink` plugin
receives the same packet as from the webcam lane and drives Manny (§5); the IK
Retargeter carries Manny onto the MetaHuman body (§6); the Live Link Face packet
drives the MetaHuman's visible head through `ABP_Face` exactly as for the webcam
(§6, "Who owns the head"). No plugin, AnimBP or MetaHuman asset was changed.

### 9.12 Measured

On the GB10 (`kaspar`), 2026-10-07, a 9 s clip (270 frames), through the SSH tunnel
over a direct Tailscale connection:

| Steps | Generation | Transfer (~890 KB) |
|---|---|---|
| 100 (default) | 5.8 s warm (7.3 s on the first request after start) | 0.3–1.3 s |
| 50 | 2.9 s | |
| 25 | 1.5 s | |
| 10 | 0.7 s | |

Each REPL line prints the split (`generating + queued + transfer`), so a slow day
shows whether the GPU or the link is at fault. `/health` round trip: 76 ms. Cache
hit: no network, ~0.07 s to unpack, self-check and retarget (measured on a 9 s clip). Accuracy on real Kimodo output: FK
self-check 0.00003 cm; 40 Manny bone directions within 0.0006° of Kimodo's own joint
positions; the face-channel head decodes to the body's head exactly.

### 9.13 Tests

```powershell
.\venv\Scripts\python.exe procedural_animation\tests\procedural_checks.py
```

No GPU, no Spark, no engine; 55 checks, exit code 1 on any failure. As in
`solver_checks.py`, ground truth is independent of the retarget's own tables
(anatomical joint correspondence, an Unreal-style FK written in the test).

| # | What |
|---|---|
| 1 | Rest identity: SOMA's T-pose in → every Manny bone points along the source, pelvis on Manny's rest pelvis, neutral head |
| 2 | Direction truth on all 243 Kimodo BVH clips (both conventions), every frame |
| 3 | Change of basis: walk +Z → Manny +Y; a left turn faces +X (an inverted rotation would face −X); left arm raise lifts `upperarm_l` only |
| 4 | Head channel: the Live Link Face curves decode to the sent body's head, every frame |
| 5 | Wire format: `/mediapipe/pose`, 421 floats, conductor.py's bone order |
| 5b | Player, real time: loops, hand-over at the end of a pass, 421-float packets, present 1 → 0 on stop, ~60 Hz |
| 6 | Client resilience against stub servers: refused, accept-and-close, timeout, HTTP 500, malformed NPZ; nothing cached, the player keeps sending; a cache hit never touches the network; prompt options |
| 7 | NPZ contract: NPZ path == BVH path (both conventions), exact keys/shapes/dtypes, metres; transposed, cm, reordered, NaN, missing key, truncated → refused; a **real Kimodo clip** (`tests/fixtures/kimodo_real_turn_around_2s.npz`) passes the self-check and direction truth |

`remote_kimodo_service/fake_kimodo_server.py` is a stdlib stand-in for the service
(clips built from the BVH library, plus failure modes); the tests use it, and it can
replace the Spark for offline REPL work:
`.\venv\Scripts\python.exe -m remote_kimodo_service.fake_kimodo_server --delay 3`.

### 9.14 Known limits

- **Head yaw beyond 75°.** The face-channel head curves are absolute; Unreal was only
  measured linear to 75°, and generated motion turns the whole body (7 of 243 library
  clips reach ±180°, where the Euler yaw also wraps). Open problem, §10.
- **No transitions yet:** hard cuts, no re-centring (§9.10).
- **Hands** are Kimodo's static relaxed pose; **no face** animation (neutral
  blendshapes); max **10 s** per prompt; 30 fps source.
- **The Spark is shared** (another user is logged in): a busy GPU shows up as
  generation time, and a queued request as `queued` time.

## 10. Open items / next steps

Generated-motion lane:

- **Head yaw beyond 75° on the face channel.** The Live Link Face head curves are
  absolute component-space rotation, measured linear only up to 75°. Generated
  motion turns the whole body: of 243 library clips, 13 exceed 75° of head yaw and
  7 reach ±180°, where the Euler yaw also wraps (`headYaw` jumps from +3.6 to −3.6 in
  one frame). What the MetaHuman does there is unmeasured, and the offline head test
  decodes with the same linear model, so it passes regardless. Next: a
  `tests/ue_head_probe.py` stage at 90/135/180°, then decide (cap the yaw, keep the
  heading near the camera, or rely on Unreal if it stays linear).
- **Playback layers deferred from v1:** re-centring each clip on where the last one
  ended, with heading alignment (the face channel's head rotation must be turned by
  the same yaw, or head and body disagree); a per-bone slerp crossfade (~0.3 s); an
  idle loop; a stage box.
- **Steps vs quality:** judge in Unreal which step count is good enough (25 steps is
  4× faster than 100) and pick the default.
- **Chaining** (`first_frame`, reserved in the request schema): a full-body keyframe
  constraint on frame 0 so consecutive clips continue without a cut.
- **Later phases:** ARDY-Core streaming (prompt changes mid-motion), a mixer with the
  webcam lane ("AI body, webcam face and hands"), ARDY-SOMA when released, a
  "Procedural" tab in `gui.py` (`procedural-animation.md` §6).

Live lane:

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
  separate position estimate - an architecture decision, not a tune. (The
  generated-motion lane already has real root motion, §9.9.)
- **Distortion layer** - tracking artefacts as art-directable effects,
  downstream of a clean solver (see CLAUDE.md for the plan).
- **Face crop experiment** - reinstate `head_pose_capture.py` on a crop around
  the pose-derived head, only if the landmark-derived basis proves fragile.
- **Knees converge** - the character's stance is narrower than the
  performer's (~0.65 knee/hip ratio). Not investigated.
- **Finger twist**: unconstrained, same limitation as body twist - MediaPipe
  gives joint positions, not rotations, so finger roll can't be observed.
  (Wrist roll can, from the palm plane - §8.)
- **Cleanup candidates**: the orphaned files in §2 (`mediapipe_pose_osc_protocol.py`,
  `live_link_pose_json_protocol.py`, `pose_landmarker_full.task`, this repo's
  `ue_plugin/` copy) could be removed once nobody needs them as reference.
- Benchmark Full Body IK vs. Body Mover + Limb IK Solvers for the MetaHuman
  retarget (§6).
