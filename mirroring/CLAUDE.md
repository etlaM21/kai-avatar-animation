# project_kaspar — MediaPipe → Unreal Engine live mocap

Real-time markerless mocap: webcam → MediaPipe → OSC → a custom Live Link source
driving the UE5 Mannequin (Manny).

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
  └─ conductor.py            owns the single shared camera + both detectors,
     │                       smoothing, debug overlay, recording, all sockets
     ├─ mediapipe_holistic_capture.py   ONE inference pass → pose + face + hands
     │    ├─ PoseFrame  (33 world landmarks, with visibility/presence)
     │    ├─ FaceFrame  (52 ARKit blendshapes + 478 face mesh landmarks)
     │    └─ HandsFrame (21 landmarks × 2 hands)
     ├─ head_pose_capture.py            second FaceLandmarker, head yaw/pitch/roll
     │                                  → Live Link Face. SEE WARNING BELOW.
     ├─ pose_solver.py        pure math: landmarks in, 60 BoneTransforms out
     │    └─ live_link_pose_osc_protocol.py → OSC 9001 → MediaPipeLiveLink (UE5)
     ├─ live_link_face_protocol.py      → UDP 11111 (Epic's stock Live Link Face)
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

### For you (Malte)

- **Test the new solver in UE.** Branch `feature/solver-fixes` is not merged.
  Press "Calibrate upright" once, then: turn and nod your head, rotate your wrists
  palm-up/palm-down, spread and curl your fingers, crouch, and lean out of frame.
- **Decide who owns head rotation in UE.** The body stream now sends real head
  rotation. If the MetaHuman Face AnimBP also drives the head from the ARKit solve,
  they will fight — and since Live Link Face's head rotation is currently always zero
  (see below), it may simply pin the head. Disable one of them.
- **Decide about `head_pose_capture.py`.** Measured: that standalone FaceLandmarker
  **never detects the face at performance distance** (it only fired while walking up
  to the laptop, in both recordings), so Live Link Face's head yaw/pitch/roll has been
  stuck at its initial zero the whole time. Options: (a) feed those three channels
  from the same face-mesh basis the body solve uses, (b) delete the module and save an
  inference pass per frame, or both.
- **Record an occlusion clip** if you want the gating thresholds tuned harder: lean
  over the desk until the legs leave frame, step half out of shot, put one arm behind
  your back. Current thresholds come from the incidental leg dropouts in `motion.npz`.

### Not started

- **Step 4, GUI polish** (skipped on request): the video pane letterboxes heavily, the
  camera Index field reads 0, and FPS is burned into the frame — a panel label with
  min/avg would read better.
- **Root translation.** MediaPipe world landmarks are hip-centred, so all global
  movement is discarded *by construction*: the character can never step, shift or
  jump, and ground locking reads a real jump as grounded. This needs image-space
  landmarks or a separate position estimate — an architecture decision, not a tune.
  Probably the biggest remaining gap for a live performance piece.
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
