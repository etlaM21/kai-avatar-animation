# project_kaspar — MediaPipe → Unreal Engine live mocap

**Project:** project_kaspar

**Status:** the live-capture lane is built and working end to end — one webcam
drives face (ARKit blendshapes + head rotation), full body pose, and finger
tracking on Unreal Engine 5.8's "Manny" mannequin in real time. The second,
generated-motion lane (Kimodo/SOMA, §9) has not been started.

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
  └─ conductor.py            owns the one shared camera, both detectors, both
     │                       solvers, EMA/NLERP smoothing, debug overlay, and
     │                       both UDP sockets. Nothing else touches the camera
     │                       or a socket directly.
     │
     ├─ mediapipe_holistic_capture.py   HolisticLandmarker, ONE inference pass
     │     │                            covering pose + face + both hands
     │     ├─ PoseFrame   (33 world landmarks)
     │     ├─ FaceFrame   (52 ARKit blendshapes + face mesh landmarks)
     │     └─ HandsFrame  (21 landmarks x 2 hands, world + image space)
     │
     ├─ head_pose_capture.py           a second, much cheaper FaceLandmarker
     │     └─ HeadPoseFrame (yaw/pitch/roll only)  — HolisticLandmarkerResult
     │        has no transformation-matrix output, so this is the only source
     │        for head rotation; conductor.py stitches it into FaceFrame.
     │
     ├─ pose_solver.py                 pure math - no MediaPipe/camera code,
     │     │                           just landmarks in, BoneTransforms out
     │     ├─ PoseSolver.solve(pose_landmarks)                  → 22 body bones
     │     └─ PoseSolver.solve_hands(pose_lm, L_hand, R_hand)    → 19+19 finger bones
     │          │
     │          └─ live_link_pose_osc_protocol.py → OSC /mediapipe/pose, UDP 9001
     │               └─ MediaPipeLiveLink (UE5 C++ ILiveLinkSource - see §5)
     │
     └─ live_link_face_protocol.py     FaceFrame + stitched head rotation
           └─ UDP 11111 (raw, not OSC) → Epic's stock "Live Link Face" plugin
```

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
  zero custom UE code on this side.
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

Needs `holistic_landmarker.task` and `face_landmarker.task` next to
`conductor.py` (download links in each capture module's docstring).
`pose_landmarker_full.task` is a leftover from before the Holistic migration -
nothing loads it any more. In the debug window, press `c` while standing
upright to calibrate away MediaPipe's ~18° forward-lean bias, `Esc` to quit.

### Orphaned files, kept but not live

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

## 6. Downstream: Manny → MetaHuman retargeting - not yet built

Today, Live Link Pose drives Manny directly (1:1 skeleton match, no
retargeting). Taking it further to a MetaHuman is still just a plan:

- A separate Actor Blueprint would hold both a Manny (source) and MetaHuman
  (target) Skeletal Mesh Component, linked by an `IKRetargeter` asset built
  from a source IK Rig (Manny) and target IK Rig (MetaHuman).
- Solver choice is a live-performance-relevant tradeoff: **Full Body IK** gives
  the best proportion correction (foot/hand placement) but is the most
  expensive per frame; a cheaper **Body Mover + Limb IK Solvers** stack trades
  some correction quality for runtime cost. Not benchmarked.
- One Manny → MetaHuman retarget setup would serve both MediaPipe and Kimodo,
  since both would drive the same Manny proxy upstream.

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

- **Face**: HolisticLandmarker blendshapes + a second slim FaceLandmarker pass
  for head yaw/pitch/roll (HolisticLandmarkerResult has no transformation-matrix
  equivalent) → Epic's stock Live Link Face plugin. Working.
- **Body**: `pose_solver.py`'s `solve()` - per-bone minimal-swing rotation
  from rest direction to measured direction, converted to parent-local in
  Unreal's component space. 22 bones, verified against three acceptance tests
  (see CLAUDE.md).
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

## 9. Open items / next steps

- **Kimodo/SOMA lane**: not started - no code in this module talks to it yet.
- **Manny → MetaHuman retargeting** (§6): not built - Manny is the final
  driven skeleton today.
- **Finger twist**: unconstrained, same limitation as body twist - MediaPipe
  gives joint positions, not rotations, so forearm pronation and finger roll
  can't be observed either.
- **Cleanup candidates**: the orphaned files in §2 (`mediapipe_pose_osc_protocol.py`,
  `live_link_pose_json_protocol.py`, `pose_landmarker_full.task`, this repo's
  `ue_plugin/` copy) could be removed once nobody needs them as reference.
- Benchmark Full Body IK vs. Body Mover + Limb IK Solvers, once MetaHuman
  retargeting is actually being built.
