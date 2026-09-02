# MediaPipe + Kimodo → MetaHuman LiveLink Architecture

**Project:** project_kaspar

**Status:** Architecture settled, `ILiveLinkSource` plugin implementation not yet started

**Target engine:** Unreal Engine 5.8

## 1. Goal

Drive a MetaHuman from two independent motion sources at once:

- **Live capture** — a webcam feed processed through Google MediaPipe (face + body)
- **Generated motion** — NVIDIA Kimodo/SOMA output, streamed as an offline gesture/motion library rather than real-time generation (Kimodo is diffusion-based and clip-output, evaluated as unsuitable for real-time generation directly)

Both need to reach the same MetaHuman through one coherent, custom LiveLink pipeline, replacing the earlier Dummy + IK Retargeter approach that was built around Kimodo/SOMA alone.

## 2. High-level data flow

```
 MediaPipe Face          MediaPipe Pose         Kimodo / SOMA
 (webcam, ARKit              (webcam,           (generated motion,
  blendshapes)             33 landmarks)             offline)
        |                        |                        |
        v                        v                        v
 Live Link Face UDP        OSC over UDP             OSC over UDP
(fixed 61-float,          (custom pose +           (custom SOMA
 Epic's own protocol)      SOMA schema)              schema)
        |                        \                       /
        v                         \                     /
 Stock "Live Link              Custom LiveLink Source (C++)
  Face" plugin                  ILiveLinkSource + FRunnable
 (ships with engine)            OSC receive thread, one plugin,
        |                       multiple Subjects by name
        |                                  |
        +----------------+-----------------+
                          v
                  LiveLink Client (UE5)
                Subjects, roles, buffering
                          |
                          v
              Live Link Pose node → drives
             "Manny" (Third Person Mannequin)
                  animation blueprint directly
                     (1:1 skeleton match,
                      no retargeting needed here)
                          |
                          v
              Runtime IK Retargeter (Blueprints)
                  Manny (source) → MetaHuman
                       (target), IK Rig +
                    IK Retargeter asset pair
                          |
                          v
                      MetaHuman
              (face + body driven live)
```

## 3. Why two transport lanes instead of one

**Face rides a shortcut that already exists.** Epic's stock "Live Link Face" plugin listens for a fixed, proprietary 61-float UDP packet — the same format the iOS Live Link Face app emits, based on ARKit's standardized 52-blendshape set. Because that schema is standardized, a fixed listener can exist at all. `mefamo.py`/`PyLiveLinkFace` (JimWest) work by spoofing that exact packet, so face capture needs zero custom UE-side code.

**Body has no equivalent shortcut.** There's no standardized schema for full-body skeletons the way there is for ARKit blendshapes — rig joint counts and naming vary per project. Epic doesn't ship a plug-and-play body listener, so body (MediaPipe Pose) and generated motion (Kimodo/SOMA) both require a real custom `ILiveLinkSource` plugin regardless of wire format.

## 4. Why OSC for the custom lane

Not a universal "OSC beats JSON" claim — the decision was driven by two concrete factors:

1. **Kimodo/SOMA already emits OSC.** Standardizing MediaPipe Pose on OSC too means one parser in the plugin instead of two.
2. **Ecosystem compatibility** — OSC is the common language of real-time/show-control tooling (lighting, TouchDesigner, etc.), which may matter for a live theatrical context later.

JSON was seriously considered and is a fully legitimate alternative — confirmed by reviewing DollarsMoCap's trial plugin, which streams body data as plain JSON over UDP (port 12351) with no measurable parsing bottleneck at real-world frame rates. Epic ships `Json`/`JsonUtilities` exactly as readily as the `OSC` module, so neither costs extra as a dependency. If Kimodo's OSC output weren't already a given, JSON would be an equally reasonable choice.

## 5. Custom `ILiveLinkSource` plugin — design, informed by DollarsMoCap's source

Reviewed DollarsMoCap's (Sunnyview Inc.) UE 5.8 trial plugin source directly. Patterns adopted:

- **Class shape:** one class implementing both `ILiveLinkSource` and `FRunnable`. `ReceiveClient()` starts a worker thread (`FRunnableThread::Create`); destructor sets an atomic "inactive" flag, calls `Stop()`, joins the thread, closes the socket.
- **Socket:** `FUdpSocketBuilder`, non-blocking, polling loop with a short sleep — simple, proven sufficient.
- **Thread-safety pattern (adopt as-is):** raw bytes are received on the worker thread; a `TSharedPtr<FThreadSafeBool>` "is this object still alive" flag is captured into the `AsyncTask(ENamedThreads::GameThread, ...)` lambda before touching `this`, guarding against the source being destroyed while a task is still queued (PIE stop, hot reload, source removal).
- **Multiple Subjects per Source:** one socket, one plugin instance, can host several independently named LiveLink Subjects — a subject is auto-created (`CreateSubject`, idempotent via a `TSet` + critical section) the first time a new subject name is seen in an incoming frame. This is exactly how MediaPipe Pose and Kimodo/SOMA will coexist as two Subjects under one Source.
- **Tracking-valid flag:** carried properly via `FLiveLinkSkeletonStaticData::PropertyNames` + `FLiveLinkAnimationFrameData::PropertyValues`, not as a fake bone. Reuse this mechanism for any confidence/valid flags MediaPipe or Kimodo need to signal.
- **Target skeleton is Manny, not MetaHuman directly.** Since IK Retargeting happens downstream in Blueprints (Manny → MetaHuman), the plugin only ever needs to correctly drive Manny — a single, fixed, always-present skeleton. This settles an earlier open question: hardcoding Manny's real bone names/parent hierarchy and rest pose in C++ (as DollarsMoCap does) is the *correct* low-risk choice here, not a shortcut to avoid, precisely because Manny never changes and the target-skeleton variability (different MetaHuman body presets) is fully absorbed by the IK Retargeter, not the plugin.
- **Rest-pose handling:** incoming quaternions are treated as deltas relative to a known rest pose, composed as `FinalRotation = BindPose * DeltaRotation`. Same math a Retarget Pose asset would do internally — just applied by hand here since the plugin's only target is the fixed Manny skeleton.
- **Build.cs dependencies to mirror:**
  - Public: `Core`, `LiveLinkInterface`, `LiveLink`, `LiveLinkAnimationCore`, `OSC` (swap in for DollarsMoCap's `Json`/`JsonUtilities`)
  - Private: `CoreUObject`, `Engine`, `Sockets`, `Networking`, `Projects`, `InputCore` (`Slate`/`SlateCore` only if a custom connection-settings panel is built, otherwise skippable)
- **A rough edge *not* to copy:** DollarsMoCap's receive buffer null-terminates at `ReceivedData[BytesRead]` with no bounds check against the buffer size — fine at their current payload size, but worth guarding properly.

## 6. Downstream: Manny → MetaHuman retargeting

- Live Link Pose node feeds Manny's Animation Blueprint directly (1:1 skeleton match, no retargeting at this stage).
- A separate Actor Blueprint holds both a Manny (source) and MetaHuman (target) Skeletal Mesh Component, linked by an `IKRetargeter` asset built from a source IK Rig (Manny) and target IK Rig (MetaHuman), each with an editable Retarget Pose.
- Solver choice is a live-performance-relevant tradeoff: **Full Body IK** gives the best proportion correction (foot/hand placement) but is the most expensive per frame; a cheaper **Body Mover + Limb IK Solvers** stack trades some correction quality for runtime cost, and is likely the better fit for a live show's frame budget. Not yet benchmarked.
- One Manny → MetaHuman retarget setup serves both MediaPipe and Kimodo, since both drive the same Manny proxy upstream.

## 7. Reference material

- **JimWest/MeFaMo and JimWest/PyLiveLinkFace** — face capture reference. Both repositories were archived by the owner on 2026-08-13 (read-only, no further updates). They use the legacy `mediapipe.python.solutions.face_mesh` API, which Google deprecated in favor of the Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`/`PoseLandmarker`) after 2023. New code (the pose sender) uses the Tasks API instead.
- **DollarsMoCap (Sunnyview Inc.) trial plugin** — reviewed both the wire protocol (via packet capture) and the full C++ plugin source, as detailed in §5 above.
- MediaPipe's PyPI package currently supports **Python 3.9–3.12** only — no 3.13/3.14 wheels. A dedicated 3.11/3.12 environment is recommended, kept separate from any UE5 Python environment.

## 8. Delivered so far

- `mediapipe_pose_sender.py` — MediaPipe PoseLandmarker (Tasks API) webcam capture, sends world-space landmarks (metric, hip-centered) as a single OSC message per frame to a configurable host/port. Uses a `PoseLandmark` enum mirroring BlazePose's 33-joint topology.

## 9. Open items / next steps

- Write the C++ `ILiveLinkSource` + `ILiveLinkSourceFactory` plugin pair (body: Manny skeleton static/frame data + OSC receive loop; face: none needed, already solved via the stock plugin).
- Solve landmark-position → bone-local-rotation math for the MediaPipe Pose sender (or push raw landmark positions and do the conversion plugin-side) — not yet designed.
- Define the OSC address/schema for MediaPipe Pose and for Kimodo/SOMA under the shared custom Source (e.g. `/mediapipe/pose`, `/kimodo/pose`), including a per-subject tracking-valid flag mirroring DollarsMoCap's `present` field.
- Benchmark Full Body IK vs. Body Mover + Limb IK Solvers for the Manny → MetaHuman retarget step under live-performance frame budget.
- Decide the actual rest pose / bind pose values for Manny to hardcode into the plugin (mirroring DollarsMoCap's approach, adapted to whatever pose MediaPipe/Kimodo output assumes as neutral).