# UnityPythonMediaPipeBodyPose: How Root Translation Is Done

Source analysed: `K:\Kaspar_Project\_relatedWork\UnityPythonMediaPipeBodyPose`
(Unity 2021.3.17f1 + Python `mediapipe` legacy `mp.solutions.pose`; the README says the project is superseded by "Tracking4All").

## TL;DR

The repo **does** get root translation from MediaPipe alone. It does it with **one `cv2.solvePnP` call per frame** in `mediapipebody/body.py:50-81`:

- **3D object points:** `pose_world_landmarks`. These are metric, hip-centred and camera-aligned.
- **2D image points:** `pose_landmarks` × image width/height, in pixels.
- **Camera matrix:** made up (`f = 0.6 × image width`, principal point at the image centre, zero distortion).

The PnP solve returns a translation vector `t`. That is the position of the hip-centre origin in camera space, in metres. **The rotation from PnP is thrown away.** Every world landmark is then shifted by that translation, which places the whole skeleton in camera space. Unity only displays those points, scales them ×10 and lightly smooths them. There is no avatar or retargeting in this repo, only spheres and line renderers.

So the trick is: **MediaPipe's world landmarks give the metric size of the body, the 2D landmarks give its pixel size and position, and PnP turns that size ratio into depth (and image offset into lateral/vertical position).**

**For K.ai, see §7.** The method was run offline on `mirroring\recordings\rest.npz` and `motion.npz`, which already contain everything it needs. Standing still, the root jitters 0.3 cm sideways and 1.3 cm in depth. The chapter covers the proposed design, the measurement plan and which other open items this repo does or does not help with.

---

## 1. Data flow

```
Webcam ─(CaptureThread)─► frame ─(BodyThread)─► mp.solutions.pose
                                                   │
               pose_landmarks (normalised 2D) ─────┤
               pose_world_landmarks (metric 3D) ───┤
                                                   ▼
                            compute_real_world_landmarks()  ← solvePnP
                                                   │
       "FREE|i|x|y|z"     (33 lines, root-translated)  ─┐
       "ANCHORED|i|x|y|z" (33 lines, hip-centred raw)  ─┴─► Windows named pipe \\.\pipe\UnityMediaPipeBody
                                                                    │
                                                    PipeServer.cs (Unity) ─► 33 sphere transforms + lines
```

- Python side: `mediapipebody/body.py`, `main.py`, `global_vars.py`
- Unity side: `UnityMediaPipeBody/Assets/PipeServer.cs`, `Landmark.cs`, `CameraController.cs`
- Each pipe message is a `uint32` length prefix followed by a UTF-8 text block (`body.py:155-157`, `PipeServer.cs:261-262`).
- Unity picks one of the two streams with the `anchoredBody` bool. The scene has `anchoredBody: 0`, so it uses the **FREE** (root-translated) stream.

## 2. The core algorithm (`body.py`)

### 2.1 Inputs (`body.py:139-146`)

```python
model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks.landmark])          # metres, hip-centred, NEGATED
image_points = np.float32([[l.x * image.shape[1], l.y * image.shape[0]] for l in image_landmarks.landmark])  # pixels
```

- `pose_world_landmarks` are in metres with the origin at the hip midpoint. In practice their axes follow the camera view (x ≈ image right, y ≈ image down, z ≈ depth). They are **not** canonicalised to the body's facing, so body rotation is already in them. (Measured for K.ai in §7.2: PnP finds only 2–5° of residual rotation.)
- All three axes are negated. More on this in §3.
- All 33 points are used. No `visibility` or `presence` filtering is applied.

### 2.2 Pseudo intrinsics + PnP (`body.py:50-68`)

```python
focal_length = frame_width * .6
center = (frame_width/2, frame_height/2)
camera_matrix = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
distortion = zeros(4,1)

success, rvec, tvec = cv2.solvePnP(objectPoints=world_landmarks, imagePoints=image_landmarks,
                                   cameraMatrix=camera_matrix, distCoeffs=distortion,
                                   flags=cv2.SOLVEPNP_SQPNP)
```

- `f = 0.6·W` means an assumed horizontal FOV of `2·atan(0.5/0.6) ≈ 80°`, which is typical for a webcam. The comment says real calibration would improve quality.
- `SOLVEPNP_SQPNP` is a global, non-iterative solver. It needs no initial guess and does not use the previous frame's result (`useExtrinsicGuess` is not used).

### 2.3 Applying only the translation (`body.py:69-78`)

```python
transformation = np.eye(4)
transformation[0:3, 3] = translation_vector.squeeze()       # rotation deliberately NOT inserted
model_points_hom = np.concatenate((world_landmarks, np.ones((33, 1))), axis=1)
world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
```

`inv([I | t]) = [I | -t]`, so this is just:

```
world_points[i] = model_points[i] - t        (= -p_mp[i] - t)
```

- `rvec` is computed and then ignored. The world landmarks already contain the body's orientation relative to the camera, so only the root offset is needed.
- The result is 33×4 (homogeneous). Only columns 0–2 are sent.
- If PnP fails (`tvec is None` → `AttributeError`), the function falls back to the un-translated landmarks (`body.py:79-81`).

### 2.4 Why this produces a usable root position

For a pinhole camera, a body of metric size `S` that appears `s` pixels large is at depth `tz ≈ f·S/s`. Its lateral offset is `tx ≈ (u−cx)·tz/f = (u−cx)·S/s`, and `ty` works the same way. PnP is a least-squares version of this over all 33 points. Consequences:

| Component | Depends on | Notes |
|---|---|---|
| **Depth `tz`** | assumed `f` **and** MediaPipe's metric scale `S` | A wrong `f` scales depth linearly. Expect depth to be the noisiest axis. |
| **Lateral `tx` / vertical `ty`** | MediaPipe's metric scale `S`, *not* `f` | The `f` term cancels, so sideways and vertical movement are fairly robust even with the fake intrinsics. |
| Absolute scale | MediaPipe's built-in "average human" scale | A child or a very tall person shifts depth estimates. |

This is why the method gives a plausible root motion without calibration. The depth axis is where it degrades first. (Both rows were confirmed on K.ai's recordings, §7.2.)

## 3. Coordinate conventions (MediaPipe/OpenCV → Unity)

The negation `[-x, -y, -z]` on the model points, followed by `- t`, gives approximately:

```
FREE_point ≈ -(p_mp + t) = -p_camera
```

In OpenCV camera space (x right, y down, z forward), `-p` breaks down as follows:

- **Flip Y.** This is the standard OpenCV (right-handed, y-down) → Unity (left-handed, y-up) conversion `(x, -y, z)`.
- **Rotate 180° about Unity's Y axis.** This adds `x → -x` and `z → -z`.

As a result, the Unity origin acts as the real webcam position, the webcam looks along Unity −Z, and the person stands at negative Z facing +Z. A Unity camera behind the origin looking toward −Z sees the same, non-mirrored view as the webcam. (Moving to image-right gives `-tx`, which is the Unity camera's right when it looks along −Z.)

**Caveat:** negating all three axes of the model before PnP is a point reflection (det = −1), which no rotation can reproduce. It works only because the rotation from PnP is discarded and the body is fairly flat in depth compared with its height and width. PnP then finds a rotation close to Rz(180°), and the translation stays sensible. A cleaner approach is to run PnP on the **un-negated** MediaPipe points and convert the final camera-space result to Unity explicitly (see §6).
*Measured on K.ai's recordings (§7.2):* the negated variant moves the translation by 12–26 cm and makes the reprojection error about 10 px worse. Do not copy the negation.

Mirroring: `cv2.flip(image, 1)` is commented out (`body.py:105`). `Landmark.cs` has an optional `#define FLIP` that swaps left/right indices for anyone who wants a mirror view.

## 4. Unity side (`PipeServer.cs`)

Scene values from `SampleScene.unity`: `anchoredBody=0`, `multiplier=10`, `landmarkScale=0.5`, `maxSpeed=50`, `samplesForPose=1`. The CameraController uses `distance=44.76` and `offset=(0,-5.5,0)`.

1. **Receive thread** (`Run`, `:243-289`): parses the `FREE` or `ANCHORED` lines and adds them into `positionsBuffer[i]` (sum plus count).
2. **Averaging** (`UpdateBody`, `:208-214`): once a landmark has `samplesForPose` samples, the target is `sum / count × multiplier`. With `samplesForPose = 1` this is just the latest sample ×10, so metres become decimetre-scale Unity units.
3. **One-time calibration** (`Calibrate`, `:162-167`, called once at `:216-223`):
   `calibrationOffset = -(LEFT_HIP + RIGHT_HIP)/2` from the **first** frame. Every landmark gets this offset, so the person's starting hip position becomes the local origin and **later root motion is relative to where they first stood**. It is never re-run. The CameraController is also anchored to the NOSE at that moment.
4. **Smoothing / outlier clamp** (`:225-228`):
   `localPosition = MoveTowards(current, target + calibrationOffset, Time.deltaTime * maxSpeed)`. This is a per-landmark speed limit (50 units/s, about 5 m/s real). It does crude smoothing and suppresses jitter spikes in the PnP depth.
5. **Head**: an experimental orientation built from face-landmark normals (`:231-240`). It is unrelated to the root.

There is **no skeleton, no IK and no bone rotations**. The "body" is 33 independently moving spheres under a `parent` transform. So this repo shows **how to get the root position**, not how to drive a rigged avatar with it.

## 5. Weaknesses and bugs spotted (worth avoiding)

- **BGR passed to MediaPipe**: frames from `cv2.VideoCapture` go straight into `pose.process(image)` without `cv2.cvtColor(..., COLOR_BGR2RGB)`. MediaPipe expects RGB, which may slightly lower detection quality. (K.ai's `conductor.py` converts correctly.)
- **No visibility weighting**: occluded or out-of-frame landmarks (often feet and hands) still feed PnP and can drag the root, especially in depth.
- **No temporal filtering on `t`**: PnP is solved from scratch every frame (no `useExtrinsicGuess`, no One-Euro/Kalman filter). Smoothing only happens in Unity through `MoveTowards`.
- **Fake intrinsics**: depth accuracy depends on `f = 0.6·W`. A webcam with a very different FOV (for example a wide-angle 120° camera) will produce scaled depth.
- **Thread races**: Python reads `capture.frame` without a lock and may process the same frame twice. Unity's `positionsBuffer` is written on the pipe thread and read/reset on the main thread without synchronisation.
- **Calibration is one-shot**: there is no re-centre key, and the first frame (possibly a poor detection) becomes the origin permanently.
- **Windows-only transport** (named pipe).

## 6. Takeaways for Kaspar / kai-avatar-animation

A minimal re-implementation of the idea, with the fixes from §5 applied. **Superseded by §7.** That chapter is measured on K.ai's own recordings and uses K.ai's real axis conventions. In short: use the torso and face points rather than all visible ones, and map to Manny component space rather than Unity.

```python
# per frame, after MediaPipe Pose
world = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark], np.float32)   # metres, hip-centred
img   = np.array([[l.x * W, l.y * H] for l in res.pose_landmarks.landmark], np.float32)       # pixels
vis   = np.array([l.visibility for l in res.pose_landmarks.landmark])
m     = vis > 0.5                                                   # drop occluded points (keep >= 6)

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)     # calibrate, or f ≈ 0.6*W as fallback
ok, rvec, tvec = cv2.solvePnP(world[m], img[m], K, None, flags=cv2.SOLVEPNP_SQPNP)

root_cam = tvec.ravel()                                             # hip-centre in OpenCV camera space (m)
root_cam = one_euro_filter(root_cam)                                # smooth, depth needs the strongest filtering
root_unity = np.array([root_cam[0], -root_cam[1], root_cam[2]])     # OpenCV -> Unity (flip Y), then place relative
                                                                    # to a "camera" transform in the Unity scene
```

Then, on the avatar:

- **Root/hips position** = `root_unity`, taken relative to a calibration pose (re-centre on demand rather than on the first frame only). For ground contact, optionally lock or project the Y component onto the floor plane and use foot landmarks or a foot-lock step.
- **Bone rotations**: compute from `pose_world_landmarks` (hip-centred, so they are unaffected by the translation). PnP's `rvec` is not needed, as the repo demonstrates.
- Expect lateral and vertical motion to be good and depth to be noisy and scale-biased. Calibrating the webcam (`cv2.calibrateCamera`) or at least using the real FOV is the cheapest accuracy win for the depth axis.

---

## 7. Applications for K.ai tracking

**Sources.** I read `CLAUDE.md`, `TODO.md`, `mirroring\README.md`, both `docs\*.md`, `remote_kimodo_service\README.md`, the `_old` READMEs and the live-lane code: `conductor.py`, `pose_solver.py`, `mediapipe_holistic_capture.py`, `mediapipe_pose_capture.py`, `landmark_recorder.py` and `live_link_pose_osc_protocol.py`. `REA_FINDINGS.md` was not opened. Dollars MONO is cited only through CLAUDE.md's "Lessons from Dollars MONO". The MediaPipeLiveLink plugin (`K:\KaiTracking\...`) was **not** read; it is outside the folders I was allowed to look at. Malte confirmed on 2026-10-15 that the Unreal side does apply the pelvis translation (§7.3).

**Method.** Everything marked *measured* was computed offline with the module's own `venv` on `mirroring\recordings\rest.npz` (397 frames, standing) and `motion.npz` (652 frames), 1920×1080, inside the `trims.json` windows unless noted. The probe scripts live in Claude's temporary scratchpad and are **not** in the repo. They are easy to turn into a checked-in report like `tests\solver_checks.py` on request. Nothing in the module was changed except this file.

**Status.** Everything in this chapter is a **proposal or a measurement, not an implementation**. Per CLAUDE.md ("Architecture first", "pose_solver.py — treat with care"), each item needs Malte's approval before any code is written.

### 7.1 Which open items the Unity repo helps with

| Open item (where it is listed) | Does the repo help? | Section |
|---|---|---|
| **Root translation, live lane** (CLAUDE.md "Not started"; README §10; TODO §6) | **Yes, directly.** It is the missing depth source CLAUDE.md asks for, and K.ai already records everything it needs. | 7.2–7.5 |
| "A real jump reads as grounded" (README §8, ground locking) | Partly. PnP's vertical signal could detect jumps, but it is too pose-biased to replace ground lock. | 7.6 |
| Lean bias "not constant across setups" (CLAUDE.md Calibration) | Indirectly. A hypothesis that camera pitch is part of it; it matters for root translation either way. | 7.7 |
| Leg gate / occlusion (Dollars proposal; TODO §5–6) | Partly. The PnP reprojection residual is a strong candidate occlusion signal. | 7.8 |
| Head still wobbly (CLAUDE.md Next up) | **No** (measured). Face-mesh PnP has the same jitter as the current basis, which supports trying the One Euro filter first. | 7.9 |
| Face-crop experiment (CLAUDE.md "Not started") | Partly. Face-mesh PnP gives a perspective head pose without a second inference pass. | 7.9 |
| Ear/nose fallback head | No. The repo's head uses the same pose ear/eye/nose points that K.ai measured as under-responsive (~8° of a 37° turn, README §8). | — |
| Knees converge, curved spine, neck share, lean calibration with a turned torso, MetaHuman neck seam, every procedural-lane and GUI item | No. | — |

### 7.2 Measured: the repo's method on K.ai's own recordings

`PoseFrame.landmarks` (image-normalised) and `PoseFrame.world_landmarks` come out of the **same** `HolisticLandmarker` result (`mediapipe_holistic_capture.py:159-165`). `landmark_recorder.py` stores both (`pose_image`, `pose_world`) plus `frame_size`. So nothing new has to be captured to evaluate or run the method.

**Stability while standing still** (`rest.npz`, 294 frames, f = 0.6·W, SQPNP, all 33 points):

| Axis (camera frame) | Std over the clip | Frame-to-frame RMS |
|---|---|---|
| tx (image right) | 0.32 cm | 0.04 cm |
| ty (image down) | 0.29 cm | 0.05 cm |
| tz (depth) | 1.31 cm | 0.20 cm |

**Theory from §2.4 confirmed:**

| Check | f = 0.6·W | f = 0.8·W | Verdict |
|---|---|---|---|
| Lateral `tx` mean (rest) | 11.6 cm | 11.4 cm | independent of `f` ✔ |
| Depth `tz` mean (rest) | 180.4 cm | 236.9 cm | ratio 1.31 ≈ 0.8/0.6 = 1.33, i.e. scales with `f` ✔ |
| Residual PnP rotation (rest / motion) | 2.1° / 4.7° (max 13.8°) | 3.6° / 4.0° | world landmarks are camera-aligned ✔ (so `rvec` can be ignored) |

**The Unity repo's negated model** (§3), against the plain one on the same frames: the translation moves by 12.5 cm on average (max 16.5) on rest and 15.2 cm (max 26.1) on motion, and the reprojection error rises by about 10 px. **Do not negate.**

**Which points to use.** This changes the design compared with the Unity repo:

| Point set (f = 0.6·W) | rest: depth std | motion: depth std | motion: depth frame-to-frame RMS | motion: reprojection RMS |
|---|---|---|---|---|
| All 33 (Unity repo) | 1.3 cm | 10.6 cm | 0.94 cm | 20.5 px |
| All with visibility & presence ≥ 0.5 | 1.3 cm | 8.9 cm | 1.19 cm | 19.6 px |
| **Torso + face only: 0–12, 23, 24 (15 pts)** | **1.2 cm** | **7.5 cm** | **0.53 cm** | **11.9 px** |

The torso + face subset sits at a near-constant offset from the all-points solve: (+1.5, −0.7, +3.0) cm on rest, with a standard deviation of only 0.2–0.4 cm. That offset disappears into the calibration origin (§7.3 step 7).

Why the limbs hurt is shown by the per-landmark reprojection residuals, after fitting PnP on the torso + face subset (§7.8). MediaPipe's metric 3D legs do not project onto its own 2D legs: the ankles miss by about 72 px even while standing still, at a shoulder width of about 210 px.

**Depth error that depends on the pose** (motion.npz). The correlation between reprojection error and depth is **−0.69**. Frames with reprojection ≥ 16 px read **11.6 cm closer** (166.6 cm, n = 267) than frames below 16 px (178.2 cm, n = 286). I can't tell whether the performer actually stepped, because the recording has no ground truth. At t = 8.6 s and t = 10.6 s the shoulder width in pixels is identical (216 px) while depth differs by 14 cm. That points to a bias, not a step. → Use reprojection error as a quality weight, and record ground truth (§7.4).

**The approach walk** (outside the trim windows: the performer walks up to the laptop). Depth falls from 180 → 136 cm (rest) and 181 → 158 → 118 cm (motion) while the shoulder width grows from 207 → 551 px. So depth follows the approach in direction. It is not proof of accuracy: the performer bends over the laptop there, which brings the shoulders closer than the hips, and the legs leave the frame (23–28 of 33 points confident).

**Focal length hint.** Reprojection error over `f` has a very shallow minimum around 0.9·W (rest 12.18 px; 12.27 px at both 0.8·W and 1.0·W). The 3D-leg-vs-2D-leg consistency has the same tendency: projected hip→ankle divided by the 2D hip→ankle is 0.86 at 0.6·W, 0.96 at 0.8·W and 1.02 at 1.0·W. Both suggest the camera's horizontal FOV is nearer 53–58° than the repo's 80°. In that case the performer stood about **2.6–2.9 m** away rather than 1.8 m. This is not conclusive (MediaPipe's own limb proportions are a confounder). The standing distance and webcam of those two recordings are no longer known (Malte, 2026-10-15), so `f` cannot be pinned down from them. It has to come from a camera calibration (§7.3 step 2) or from the tape-marked `root_grid.npz` recording (§7.4). Note the webcam model with every new recording from now on.

**MediaPipe's metric body** (median over the clips): shoulder width 31.7–32.9 cm, hip width 21.1–21.2 cm, left hip→knee→ankle 74.7–76.7 cm. Manny's thigh + calf is 85.6 cm (`BIND_POSITIONS`), a ratio of **1.12–1.15**. This matters for the stride scale (§7.3 step 8).

**Cost**: SQPNP on 15 points takes a median of 15 µs (p95 28 µs). EPnP takes 28 µs, plain ITERATIVE 433 µs, and ITERATIVE warm-started from the previous frame 55 µs. Negligible next to Holistic.

### 7.3 Proposed design: a root estimator for the live lane

The seam already exists. In `PoseSolver.solve()` (`pose_solver.py:973`):

```python
pelvis_pos = hip_mid + np.array([0.0, 0.0, self.pelvis_default_height_cm])
```

`hip_mid` is ~0 because the world landmarks are hip-centred. That is exactly why the character never travels. The pelvis position is already streamed in every 421-float packet (`live_link_pose_osc_protocol.py`, 7 floats per bone), so **no wire-format, plugin or packet change is needed** to carry it.

**Unreal side: confirmed working** (Malte, 2026-10-15). The plugin and the Manny → MetaHuman retarget apply the streamed pelvis translation: Kimodo's generated root motion travels in Unreal. The webcam lane only shows none because it sends none. So root translation is purely a Python-side problem.

**Where it goes: `pose_solver.py`** (Malte's decision, 2026-10-15). The estimator is part of the solver, not a separate module and not a post-solve step in `Conductor`. It stays pure math, like the rest of the solver (no MediaPipe or camera code), and its output replaces `hip_mid`'s X/Y in the line above. Ground lock then keeps working on Z unchanged. What that implies, for whoever implements it later:

- `solve()` needs one more optional input: the pose's **image** landmarks (`PoseFrame.landmarks`). `image_size` is already passed for the face mesh, and the camera intrinsics would be solver settings, like `torso_lean_offset_deg`. Passing the image landmarks is one more argument at the existing call site in `Conductor._handle_pose()`. The image landmarks already come out of the same Holistic result.
- The calibration origin `p₀` (step 7) is captured inside `begin_calibration()` / `update_calibration()`, which would also need the image landmarks.
- `pose_solver.py` is the "treat with care" file (CLAUDE.md). With the feature **off by default** (like "stay in place"), or when no image landmarks are passed, `solve()` output must stay bit-identical, so every existing asserted check in `solver_checks.py` keeps passing unchanged.

Per frame:

1. **Points:** the torso + face subset (0–12, 23, 24), each kept only if visibility and presence are ≥ 0.5, with at least 6 points. **Not negated.**
2. **Intrinsics:** a real `K` (plus distortion) per camera **and resolution mode**. `conductor.open_camera()` picks anything from 4K down to 720p, and webcams often crop the sensor differently per mode, so the FOV can change with the mode. Get it once with `cv2.calibrateCamera` (checkerboard), store it per camera and mode, and fall back to `f = 0.6·W` (the repo) only if nothing is known.
3. **Solve:** `cv2.solvePnP(..., SOLVEPNP_SQPNP)`. A warm-started ITERATIVE refinement is optional (55 µs).
4. **Quality:** the reprojection RMS of the subset. Above a threshold the gain fades toward 0 (§7.2: 10–12 px is typical, frames ≥ 16 px read about 12 cm closer). The threshold is to be set from the ground-truth recording in §7.4.
5. **Axes:** the same mapping the solver already applies to landmarks (`_convert_landmarks_to_ue_space` followed by `PTS_TO_COMPONENT` gives `(lm.x, −lm.z, −lm.y)·100`). For the translation that is **`(X, Y, Z)_component = (tx, −tz, −ty)·100` cm**. A step to the performer's left is +X (the character's left) and a step toward the camera is +Y (the character's forward), which is consistent with the non-mirrored body the solver already sends. No new convention is involved, and no quaternion is touched.
6. **Levelling:** rotate the camera-frame translation by the camera pitch before step 5, or a camera tilted by θ turns every step toward it into a vertical change of `D·sinθ` (§7.7).
7. **Origin:** `Δ = p − p₀`, where `p₀` is averaged during the existing "Calibrate upright" (the same 30 frames that already average lean and head). The constant offset of the point subset (§7.2) cancels here. Add a re-centre action: the Unity repo's one-shot first-frame origin is a weakness (§5).
8. **Stride scale:** multiply by `Manny thigh+calf / performer's MediaPipe hip→knee→ankle` (taken from the calibration frames, ≈ 1.12–1.15 on the recordings). This is the live-lane counterpart of the procedural lane's 0.959 pelvis-height ratio (README §9.9), and it makes steps match Manny's legs, reducing foot sliding. It also cancels MediaPipe's "average human" scale assumption on X and Z, because `tx` and `ty` are proportional to that same scale. Depth then depends only on `f`, which is why step 2 matters.
9. **Filter:** a One Euro filter per axis, with Y (depth) filtered hardest. The `PoseSmoother` EMA already smooths the pelvis position on top of that; that is fine, but the two should be tuned together. A speed clamp (the Unity repo's `MoveTowards` at 5 m/s; Dollars' max speed 15) is a cheap outlier guard.
10. **Loss:** when the hips' confidence drops or the pose is invalid, **glide** the offset back to `p₀` rather than freezing it. This matches the Dollars shape summarised in CLAUDE.md (`delta = (hip − initial) × hipWeight`, glide to rest on loss). `present = 0` handling stays as it is.
11. **Z:** keep `_ground_locked_pelvis_z()` exactly as it is. PnP's vertical is only a jump detector (§7.6).
12. **GUI:** a "Root translation" toggle plus a gain slider on the Tracking tab, both live controls (no restart), off by default.

**Checks to add** (offline and independent of the estimator's own tables, as CLAUDE.md requires):

- **Synthetic round trip:** project a known skeleton with a known `K` and `t` → the estimator returns `t` (float tolerance) and maps it to the expected component axes. A step left gives +X and a step toward the camera gives +Y. This catches every sign and axis slip.
- **Rest jitter bound:** on `rest.npz`, lateral std below 0.5 cm and depth std below 2 cm (measured 0.3 / 1.2 cm with the subset).
- **Approach direction:** outside the trim windows of both recordings, Y increases monotonically (after filtering) as the performer walks up.
- **Off means off:** with the feature disabled, `solve()` output is bit-identical to today's.

### 7.4 Measurement plan (recordings that settle the open numbers)

All from `mirroring\`, with the existing recorder (raw landmarks, so they stay useful across rewrites):

```powershell
..\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\root_grid.npz
```

1. **`root_grid.npz`: ground truth for depth, lateral and pose bias.** Tape marks on the floor, measured from the camera: centre at 2.0, 2.5 and 3.0 m, plus ±0.75 m sideways at 2.5 m. At each mark, stand still for about 3 s, then raise both arms, then crouch. This gives absolute depth error (and `f`, if the camera is not calibrated), lateral scale, and the pose-dependent depth bias from §7.2 in numbers.
2. **`root_turn_offaxis.npz`:** at the +0.75 m mark, turn slowly to face the far wall, then back. This tests the off-axis hypothesis (§7.7).
3. **`root_jump.npz`:** a few jumps on the spot and a step forward and back. This covers the jump detector (§7.6) and camera pitch (§7.7).
4. The **occlusion clip** already on TODO §5 also serves §7.8.

Add trim windows to `recordings\trims.json`, as for the existing clips.

### 7.5 Root translation: why PnP rather than floor-plane rays

CLAUDE.md names "image-space hips/ankles + camera intrinsics/floor plane" as the candidate method. The floor-plane alternative intersects the ray through the ankle pixel with a known floor. It needs the camera's height and pitch, and it fails whenever the feet are occluded, out of frame or in the air. Those are exactly the frames where the legs are held today (17–22%). PnP on the torso + face subset needs no feet, works close to the camera, and takes 15 µs.

The two can be combined later: when both ankles are confident and grounded, the floor-ray depth is an independent check on PnP depth (or on `f`).

### 7.6 Ground lock and jumps

PnP's vertical (`−ty`) against the ground-locked pelvis height from the real solver:

| | ground lock range (std) | PnP range (std) | correlation | RMS difference |
|---|---|---|---|---|
| rest.npz | −0.6…+1.2 cm (0.36) | −0.7…+1.0 cm (0.29) | +0.45 | 0.3 cm |
| motion.npz | −4.2…+1.4 cm (0.84) | −9.2…+5.9 cm (3.75) | +0.68 | 3.2 cm |

Standing still the two agree to 0.3 cm. In motion, PnP swings about 4× wider than ground lock, which is the same pose-dependent bias as for depth. So **ground lock should stay the source of pelvis height**. PnP vertical can serve as a **jump detector**: airborne if the PnP hip rises clearly above that noise (to be set from `root_jump.npz`; the motion clip's noise reaches about +6 cm) *and* both ankles rise in the 2D image. While airborne, take Z from PnP; on landing, hand back to ground lock with a smoothstep.

### 7.7 Two hypotheses the root work will expose (unverified)

**Camera pitch.** The world landmarks are camera-aligned (measured, §7.2), so if the camera is tilted, the whole landmark frame is tilted. The torso lean calibration absorbs that for the body. The translation, however, would carry it as a vertical drift of `D·sinθ` per metre walked toward or away from the camera: 12 cm at θ = 7° and 31 cm at θ = 18° (the two lean offsets CLAUDE.md reports). Whether the reported lean bias is partly camera pitch is **unknown**: CLAUDE.md attributes it to MediaPipe, measured on a vertical subject.

A pitch estimate that needs no extra hardware: during calibration the performer takes one step straight back. On a level floor the hip moves horizontally, so the angle of `Δt` in the camera's y–z plane *is* the camera pitch. Recording 3 in §7.4 tests this.

**Off-axis viewing.** MediaPipe crops the person before inference. If the world landmarks are oriented relative to the **viewing ray** rather than the optical axis, a performer facing straight ahead at 0.75 m sideways and 2.5 m deep would read as turned by atan(0.3) ≈ 17°, up to about 40° at the edge of an 80° frame. Today the performer stays near the centre (the hip ray varies only 3–6° in the recordings), so this is untestable on existing data. Once root translation lets the character travel sideways, it would show as the body turning while it walks. If confirmed, the fix is analytic: rotate the world landmarks by the rotation from the optical axis to `K⁻¹·[u_hip, v_hip, 1]`. Recording 2 in §7.4 tests this.

### 7.8 Occlusion: the reprojection residual as an extra gate signal

After fitting PnP on the torso + face subset, project every 3D world landmark and compare it with its 2D landmark (motion.npz):

| Landmarks | residual, confidence ≥ 0.5 | residual, confidence < 0.5 | correlation with confidence |
|---|---|---|---|
| Ankles 27/28 | median 78 px | median 189 px (n = 124) | −0.81 |
| Feet 29–32 | median 74 px | median 200 px (n = 394) | −0.78 |
| Wrists 15/16 | median 30 px (rest 12 px) | none below 0.5 | — |

When MediaPipe invents a limb, its 2D and 3D guesses disagree much more. The residual measures something different from Dollars' left/right length ratio (CLAUDE.md), and both could feed the proposed continuous gate. Two caveats:

- Legs carry a large constant residual even when confident (72 px standing still), so a threshold must be relative to each landmark's own baseline, not absolute.
- On these clips the residual mostly repeats what `visibility`/`presence` already say. Whether it **adds** information (for instance by catching a confident but wrong leg) can only be shown on the occlusion clip from TODO §5.

### 7.9 Head: what the same tool does and does not fix (measured)

Applied to the face mesh: PnP of the 468 non-iris points against a rigid face model built from the performer's own first 30 still frames (no canonical-model download needed), compared with the solver's current orthographic face-mesh basis (`_face_mesh_rotation`):

| | solver basis | face PnP |
|---|---|---|
| rest: frame-to-frame rotation RMS | 1.66° | 1.74° |
| rest: high-frequency jitter (residual vs a 9-frame mean) | 1.16° | 1.21° |
| motion: frame-to-frame rotation RMS | 2.54° | 2.51° |
| motion: high-frequency jitter | 1.74° | 1.74° |
| motion: largest angle from mean | 51.3° | 50.5° |
| agreement (correlation of angle-from-mean) | rest 0.90, motion 0.99 | |

Conclusions:

- **The wobble is in the landmarks, not in the solve.** A perspective-correct, all-points rigid fit has the same jitter, so swapping the head solve would not help. This **supports the existing plan to try a One Euro filter on the face-mesh points first** (CLAUDE.md, "Lessons from Dollars MONO"). The rest clip gives a clean baseline to tune against: 1.2° of high-frequency jitter while standing still.
- **The orthographic basis is sound near the centre of the frame** (correlation 0.99 with the perspective fit).
- **For the face-crop experiment:** face-mesh PnP gives a full head rotation (and translation) from the Holistic mesh that already exists. That is the role `output_facial_transformation_matrixes` was going to play, without a second `FaceLandmarker` inference pass. It becomes relevant once the performer moves off-centre (§7.7), where an orthographic basis and a perspective fit should start to disagree.

### 7.10 Answered and still open

Answered by Malte on 2026-10-15:

| Question | Answer | Consequence |
|---|---|---|
| Standing distance and webcam in `rest.npz` / `motion.npz` | Not known any more | `f` (and so absolute depth) cannot be recovered from the existing recordings. It must come from a camera calibration or `root_grid.npz` (§7.2, §7.4). |
| Does Unreal apply the pelvis translation? | **Yes.** Kimodo's root motion travels; the webcam lane just sends none. | No plugin, packet or Unreal work needed. TODO §6's prerequisite is met. |
| Where does the estimator live? | **In `pose_solver.py`** | §7.3 updated: an optional image-landmark input to `solve()` and the calibration, off-by-default behaviour bit-identical. |

Still open:

1. **Default and controls:** should root translation be off by default, with a live toggle and gain on the GUI's Tracking tab (§7.3 step 12)?
2. **Recordings:** are the three clips in §7.4 feasible in your space (about 3.5 m of depth and 1.5 m of width visible)?
3. **Calibration:** checkerboard calibration (`cv2.calibrateCamera`, per camera and resolution mode) or the tape-marked `root_grid.npz` alone to fix `f`?
