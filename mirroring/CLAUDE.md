# project_kaspar — MediaPipe → Unreal Engine live mocap

Real-time markerless mocap: webcam → MediaPipe → OSC → a custom Live Link source
driving the UE5 Mannequin (Manny), and from there a MetaHuman.

`README.md` describes the system as built. This file is the working guide: how to
run things, what is load-bearing, what must not be re-broken, and what is still open.

## Environment

Windows / PowerShell. There is a venv at `.\venv`.

**Always invoke the venv interpreter directly. Never rely on activation** — venv
activation does not reliably persist between separate shell commands, and a command
that silently falls back to system Python will fail in confusing ways (MediaPipe and
scipy are only installed in the venv).

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1
.\venv\Scripts\python.exe gui.py
.\venv\Scripts\python.exe tests\solver_checks.py
.\venv\Scripts\python.exe -m pip install <pkg>
```

In the debug window: `c` calibrates (see below), `Esc` quits.

## Architecture

```
webcam (cv2, MJPG, up to 4K)
  └─ conductor.py            owns the single shared camera + the detector,
     │                       smoothing, debug overlay, recording, all sockets
     ├─ mediapipe_holistic_capture.py   ONE inference pass → pose + face + hands
     │    ├─ PoseFrame  (33 world landmarks, with visibility/presence)
     │    ├─ FaceFrame  (52 ARKit blendshapes + 478 face mesh landmarks)
     │    └─ HandsFrame (21 landmarks × 2 hands)
     ├─ head_pose_capture.py            DISABLED — commented out, see below
     ├─ pose_solver.py        pure math: landmarks in, 60 BoneTransforms out
     │    │                   ALSO the single source of head orientation
     │    ├─ live_link_pose_osc_protocol.py  → OSC 9001 → MediaPipeLiveLink (UE5)
     │    └─ live_link_face_protocol.py      → UDP 11111 (Epic's Live Link Face)
     │                                          blendshapes + head yaw/pitch/roll
     └─ landmark_recorder.py            optional --record dump for offline checks
```

Solver inputs, all optional except the first — each missing input degrades one
part of the solve rather than failing:

```python
solve(pose_world_landmarks, face_landmarks, image_size, left_hand, right_hand)
solve_hands(pose_world_landmarks, left_hand, right_hand)
```

MediaPipe is used through the **Tasks API** (`mp.Image`, `.task` model bundles).
The legacy `mp.solutions` namespace and `mediapipe.framework.formats.landmark_pb2`
**do not exist** in the installed version. Drawing helpers come from
`mediapipe.tasks.python.vision`.

## Head rotation — who owns it, and why it is routed this way

This is the least obvious part of the system. Read before touching anything
head-related in Python or in Unreal.

### The Unreal finding

On a MetaHuman the **visible head is the Face skeletal mesh component, not the Body**.
`ABP_Face` has three graphs; the first one decides everything:

```
Body & Face:   Copy Pose From Mesh (Body)  → Base Pose  ┐
               Input Pose (Face's own anim)→ Blend Pose ├→ Layered blend per bone → InputPose
                                             weight 1.0 ┘
Head Movement IK:  Control Rig, gated behind `HeadControlSwitch > 0.5`
                   AND `Enable Head Movement IK`. MetaHuman Animator only —
                   a passthrough for us. NOTE: if Live Link Face ever delivers a
                   curve named HeadControlSwitch, this engages and becomes a second
                   writer to the head.
RigLogic:      facial solve. Does not touch head rotation.
```

The layered blend wins for every bone in its Layer Setup. With `head` inside that
filter, the Body's head rotation is **discarded** and replaced by whatever the Face
component's own animation carries — ARKit head rotation when Live Link Face is
connected, bind pose (identity) when it is not. That is why the MetaHuman's head only
moved with the ARKit receiver enabled, and why *disabling* the receiver did not help:
the node still evaluates, it just writes identity.

### The decision

**Send head rotation through the Live Link Face channel**, which the MetaHuman already
routes to the head correctly, rather than rewiring Epic's asset to listen to the body.

Rationale, in order of weight:

1. **Distribution.** K.ai users should only have to add a Retarget Pose node to the
   Body AnimBP. No MetaHuman asset gets edited, so nothing breaks when Epic
   restructures MetaHumans again (they did so substantially at 5.6).
2. It repairs the Live Link Face head channel, which was independently broken
   (see below) — one change, two fixes.
3. Both head signals then derive from the same basis, so they cannot disagree.

The body stream **keeps** its own head rotation. Manny-only and non-MetaHuman targets
depend on it, and it costs nothing to send.

### The alternative, documented but not taken

Change the `Layered blend per bone` Layer Setup branch root from `head` to the facial
root bone (`FACIAL_C_FacialRoot`). Facial bones still come from the face animation;
`head` passes through from the Body. One property, and it works — but it is a
per-user, per-MetaHuman edit to an Epic asset. Keep this as a README note for users
who specifically want body-driven head.

## head_pose_capture.py — disabled, kept on purpose

**Comment the module out; do not delete it.** It is the seed of the face-crop
experiment described under Open items.

Why it was dropped:

- Measured: that standalone `FaceLandmarker` **never detected the face at performance
  distance** — it only fired while walking up to the laptop, in *both* recordings.
  Live Link Face's head yaw/pitch/roll was therefore pinned at its initial zero for
  the whole project. A dead signal into a working path.
- It cost a full extra inference pass per frame.

Why it might come back: `FaceLandmarker` supports
`output_facial_transformation_matrixes` — a 4×4 head pose fitted against MediaPipe's
canonical face model. **`HolisticLandmarker` has no such option** (its options are the
confidence thresholds plus `output_face_blendshapes` and `output_segmentation_masks`).
If the landmark-derived basis ever proves fragile under extreme pitch or partial
occlusion, running this module on a crop around the pose-derived head position
recovers that fitted matrix. That is the only capability Holistic cannot provide.

Leave the imports, the class and the call sites commented with a pointer to this
section, so the reinstatement path stays obvious.

## pose_solver.py — treat with care

This file took a long, painful debugging pass to get right. Do not refactor it
opportunistically.

**Core technique.** For each bone, compute the minimal **swing** rotation taking the
bone's rest direction to the direction measured from the performer, then convert to
parent-local with `local = parent_global⁻¹ * global`. Everything happens in Unreal's
component space (X = performer's left, Y = forward, Z = up).

Where a full orientation is observable, a basis is used instead of a swing, because
a swing leaves roll about the bone axis free:

| Part | How it is oriented |
|---|---|
| Torso, spine | Body frame from hips + shoulders |
| Arms, legs | Swing from rest direction to measured direction |
| `neck_01` / `head` | Full basis from the face mesh, split 40/60 (`NECK_SHARE`) |
| `hand_l` / `hand_r` | Full basis from the palm plane |
| Fingers | Swing, in the **hand's** frame (not the torso's) |
| `lowerarm_*` | Swing + half the wrist's twist (`FOREARM_TWIST_SHARE`) |
| Pelvis height | Derived so the lower foot sits on the floor (`ground_lock`) |

**Things that were tried and are WRONG — do not reintroduce:**

- Computing a delta against a hand-written T-pose reference table and composing it as
  `BIND_POSE * delta`. Not frame-correct: the delta lives in landmark space while
  `BIND_POSE` is a parent-local rotation in component space.
- Any "handedness conversion" such as `{-x,-y,-z,w}`. That is the quaternion conjugate,
  i.e. the inverse rotation. Landmark space is component space rotated 90° about Z —
  same handedness — so quaternions carry over directly.
- Hand-deriving `BIND_POSES` / `BIND_POSITIONS` / `FINGER_CHAIN`. They are verified
  against a `RefSkeleton` dump of `SKM_Manny_Simple`. If the mesh changes, re-dump.
- Welding `neck_01`/`head` into `TORSO_BONES`. That is what killed head rotation.
- Putting `neck_02` in `INTERNAL_BONES`. Unreal holds it at bind **relative to
  `neck_01`**, so once `neck_01` moves independently, `neck_02` must follow it by FK.
- Aiming the thumb `WRIST→CMC`. Manny's `thumb_01` **is** the thumb metacarpal, so it
  runs `CMC→MCP`. The old mapping sheared the whole thumb by 18–32°.
- Measuring the wrist's twist against the forearm directly. `hand_l`'s bind local
  contains a −67.8° roll, which then counts as twist and tips the arm by 34°. Measure
  against where the forearm *would* carry the hand at bind.
- Measuring torso lean against **vertical**. Manny's own hip→shoulder line is 5.8° off
  vertical, so calibrating an upright performer left the character 5.8° off its bind
  pose. Lean is measured relative to the rig's rest torso.

**Non-obvious invariants:**

- `spine_03`, `spine_05` and `neck_02` exist in Manny but are NOT streamed. They are
  modelled internally anyway (`FULL_CHAIN`), because Unreal applies each streamed local
  transform relative to the bone's REAL parent. Omitting `spine_03` alone rotates the
  whole upper body by ~11°.
- `_convert_landmarks_to_ue_space` mirrors X (`Right = -lm.x`). MediaPipe reports the
  performer's LEFT side with positive `lm.x`. Mirroring is verified in the data: a
  raised `RIGHT_WRIST` drives `hand_r` on the character's right.
- Twist about each **arm** bone's axis is unobservable from joint positions alone. The
  wrist is the exception now that the palm plane is measured.
- Metacarpals carry a few degrees of disclosed error: MediaPipe has no landmark at a
  metacarpal's base, so they are aimed `WRIST→MCP`. Every phalanx solves exactly.
- MediaPipe never reports "I can't see that limb" — it invents a landmark and lowers
  `visibility`/`presence`. Hence the gating.
- The face mesh is image-normalised only. Scaling x and z by width and y by height
  makes it metrically consistent (verified: z×1.0 keeps the face most rigid).

### Head rotation for the face channel

The head basis already exists — `neck_01`/`head` are solved from the face mesh. The
face channel needs the **same orientation re-expressed**, not a second solve. Reuse it.

**The conversion is not a passthrough.** The solver's head rotation is relative to the
rig's rest torso; Live Link Face expects ARKit-style head pose, which is relative to
the camera. Composing the torso orientation back in is the missing step. Get this
wrong and the head will look correct while standing square to the camera and drift as
soon as the performer turns their body — exactly the failure mode that is hardest to
spot live.

The calibration already captures a neutral head pose (see below). Use it as the zero
reference for the face channel too, so "neutral" means the same thing in both paths.

Order of work: convert, verify offline against `recordings\*.npz`, and only then look
at it in Unreal. All of this is testable without a camera or an engine.

## Acceptance tests

```powershell
.\venv\Scripts\python.exe tests\solver_checks.py            # all recordings\*.npz
.\venv\Scripts\python.exe tests\solver_checks.py some.npz   # one capture
```

No camera needed. Checks 1–2 are **asserted** (exit code 1 on failure); check 3 is a
**report** on real captures. Any solver change must keep all of 1–2 passing.

1. **Rest-pose identity** — the rig's own bind pose in, bind pose out: 22/22 body
   bones, 30/30 phalanx and thumb bones, metacarpals within their disclosed 12.5°.
   Plus: straight-ahead face mesh is a no-op; a turned head (yaw 30, pitch −15,
   roll 10) round-trips at 0.0000° after Unreal-style FK; wrist roll of ±45/−60°
   round-trips exactly.
1b. **Anatomical report** — the same rest pose with landmarks at the rig's REAL joint
   positions. Not asserted; this is what caught the thumb and hand mappings.
2. **Bind-pose FK** — upright, symmetric, `foot_l` at `(14.09, -0.99, 8.24)`.
2b. **Timed calibration** on a synthetic clock — phase sequence, countdown length,
   sample count, and a no-op on the rig's own rest pose.
2c. **Occlusion gating** — follows confident landmarks, holds low-confidence ones
   without ever showing the invented pose, resumes monotonically, respects hysteresis.
2d. **Ground locking** — bind pelvis height preserved, foot stays on the floor through
   a crouch, `ground_lock=False` still ignores it.
3. **Absolute direction error on real captures** — per-bone 3D direction vs the
   performer, in absolute terms, NOT as an angle-from-spine (which is invariant to
   body roll and hides tilt bugs).

**To add with the head-channel change:**

2e. **Face-channel head round-trip** — a synthetic head pose through the solver and
   out through `live_link_face_protocol`, recovered to the same yaw/pitch/roll.
   Include frames where the **torso is turned**, not only square-on; a passthrough bug
   passes trivially when the body faces the camera.
2f. **Not-zero regression** — assert the face channel's head values actually vary on a
   recording with head motion. The old failure was silent: three channels reading zero
   forever, with everything else working.

**Ground truth in the checks is deliberately independent of the solver's own aim
tables.** Checking a bone against the same landmark pair the solver aims it by passes
by construction and hides a mis-mapped bone — that is exactly how the thumb bug
survived. Keep `BODY_TRUTH` / `FINGER_TRUTH` / `HAND_JOINT_AT` anatomical.

### Current numbers (rest.npz / motion.npz)

| Metric | Baseline | Now |
|---|---|---|
| Body bone directions | 0.07° / 0.18° | 0.00° / 0.00° (trusted frames) |
| All finger bones | 11.2° / 10.9° | 0.00° / 0.00° |
| Palm orientation | 19–26° / 25–47° | 3.0–3.8° / 3.1–4.1° |
| Head turn | dead (welded to chest) | exact; −30…+35° yaw, −19…+43° pitch |
| Lowest foot point | −2.0…+39.6 cm | 0.75 cm, every frame |
| Legs held (occluded) | n/a | 17–22% of frames |

The `hand_l`/`hand_r` rows read 10–25°, but they are measured against the *pose*
model's INDEX point, which disagrees with the dedicated hand model by 15–23° RMS.
The report prints that floor next to them. Not solver error.

### Recording new captures

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\name.npz
```

Raw landmarks (pre-solve, pre-smoothing), so recordings stay useful across solver
rewrites. Add a trim window to `recordings\trims.json` — the performer walking to and
from the laptop is in every clip and is not performance.

## Calibration

`c` in the debug window, or the GUI button: a 3 s countdown drawn large into the video
pane, then the torso lean and the neutral head pose averaged over 30 frames.
Single frames of the same standing clip spread 5.4–7.5°, so one snapshot is not
enough. The state machine is in `PoseSolver.begin_calibration()` /
`update_calibration()`; `conductor` drives it per frame, the GUI polls the result on
the main thread. `calibrate_neutral()` still does a single-frame calibration.

MediaPipe's forward-lean bias is **not constant across setups** — measured ~18° in
earlier sessions and ~7° in the current recordings. Always calibrate; never hardcode.

## gui.py — Tkinter control surface

A wrapper that drives an unmodified `Conductor`; it reimplements no pipeline logic.
`python conductor.py --debug --camera 1` must keep behaving identically. It
monkeypatches `cv2.imshow`/`waitKey`/`namedWindow` before constructing `Conductor`,
runs `run()` on a daemon thread, and updates widgets only on the main thread via
`root.after`. Frame queue is `maxsize=1`, drop-when-full: **the GUI must never apply
backpressure to the capture loop.**

Live controls (no restart): smoothing alphas, torso lean, calibrate, overlay/FPS
toggles. Restart-required (baked in at construction): camera index/resolution, model
paths, ports/IPs, confidence thresholds — these mark the panel dirty and enable
**Apply & Restart** rather than silently doing nothing.

## Open items

### Next up

- **Head rotation through the face channel.** Described above. Comment out
  `head_pose_capture.py`, feed Live Link Face's yaw/pitch/roll from the existing
  face-mesh head basis, add tests 2e and 2f.
- **Ear/nose fallback head aim — now load-bearing.** Head rotation depends entirely on
  the face mesh, and the face mesh is lost when the performer turns away. The pose
  model still reports `LEFT_EAR`/`RIGHT_EAR`/`NOSE` with lowered visibility; blend to
  an aim from those as face confidence drops, reusing the existing occlusion-gating
  hysteresis rather than adding a second mechanism. Record a 360° turn first so the
  dropout point is measured, not guessed.

### For you (Malte)

- **Test the new solver in UE.** Branch `feature/solver-fixes` is not merged.
  Press "Calibrate upright" once, then: turn and nod your head, rotate your wrists
  palm-up/palm-down, spread and curl your fingers, crouch, and lean out of frame.
- **Record an occlusion clip** if you want the gating thresholds tuned harder: lean
  over the desk until the legs leave frame, step half out of shot, put one arm behind
  your back. Current thresholds come from the incidental leg dropouts in `motion.npz`.

### Not started

- **Distortion layer.** The tracking artefacts (dead head, twisted fingers) are wanted
  as *art-directable* effects, not as bugs. Decision taken: keep the solver clean and
  correct, and add distortion downstream — geometric distortions in Unreal (Control Rig
  node or Transform (Modify) Bone, **after** the retarget, so the IK Retargeter cannot
  correct them away), and tracking-failure distortions (jitter, dropout, confidence
  collapse) as an optional module between `pose_solver` and the OSC encoder, off by
  default. Drive both live over a second OSC address space on the existing
  `UOSCServer`. Tag the current commit before fixing anything else — some artefacts are
  emergent and will not reproduce procedurally.
- **Root translation.** MediaPipe world landmarks are hip-centred, so all global
  movement is discarded *by construction*: the character can never step, shift or
  jump, and ground locking reads a real jump as grounded. This needs image-space
  landmarks or a separate position estimate — an architecture decision, not a tune.
  Probably the biggest remaining gap for a live performance piece.
- **Face crop experiment.** Reinstate `head_pose_capture.py` on a crop around the
  pose-derived head position, to recover `output_facial_transformation_matrixes` at
  performance distance. Only worth it if the landmark-derived basis proves fragile.
- **Knees converge** — the character's stance is narrower than the performer's
  (~0.65 knee/hip ratio measured earlier). Not investigated this session.
- **Manny → MetaHuman retargeting** (README §6), and benchmarking Full Body IK vs
  Body Mover + Limb IK.
- **Kimodo/SOMA lane** — no code talks to it yet.
- **Cleanup candidates**: `mediapipe_pose_osc_protocol.py` (reachable only through
  dead code), `live_link_pose_json_protocol.py` (unused), `pose_landmarker_full.task`
  (nothing loads it), this repo's stale `ue_plugin/` copy.

## Code standards

- Python 3, type hints, dataclasses for frame/transform structs.
- Comments explain *why*, especially where a non-obvious convention is load-bearing,
  and record what a number was measured from rather than asserting it.
- **Prefer measuring over guessing**: when a pose looks wrong, compute the error
  against ground truth before changing any math. Every fix in this session was found
  that way, and two of them (the thumb, the lean reference) were invisible until the
  ground truth was made independent of the solver.