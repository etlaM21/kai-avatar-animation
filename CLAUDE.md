# project_kaspar — MediaPipe + generated motion → Unreal Engine

Real-time markerless mocap and generated motion, both driving the UE5 Mannequin
(Manny) through a custom Live Link source, and from there a MetaHuman.

Two motion lanes, one seam:

- **Live lane (done):** webcam → MediaPipe → `pose_solver` → OSC 9001 / UDP 11111.
- **Procedural lane (in progress):** NVIDIA Kimodo on the DGX Spark → NPZ clip →
  retarget on Windows → the same OSC 9001 / UDP 11111 encoders.

`README.md` describes the live lane as built. `procedural-animation.md` is the
research and plan for generated motion. `remote-kimodo-streaming.md` is the
network analysis for reaching the Spark. This file is the working guide: how to
run things, what is load-bearing, what must not be re-broken, what is open.

## Working agreement

- **Verify names before relying on them.** Module, class and function names for the
  `procedural_animation/` and `remote_kimodo_service/` packages are not recorded in
  this file on purpose; read the code.
  Where this file and the code disagree, the code wins. Then fix this file.
- **Architecture first.** For anything new in the procedural lane (service, client,
  CLI), propose the design and wait for Malte's approval before writing code.
- **Measure, don't guess** (see Code standards).

## Environment

Windows / PowerShell. This file sits at the module root (`kai-avatar-animation`).
The live lane lives in `mirroring\` and its commands below run from there; the venv is
`mirroring\venv`. The procedural lane runs from the module root as modules:

```powershell
.\mirroring\venv\Scripts\python.exe -m procedural_animation.procedural_conductor           # REPL
.\mirroring\venv\Scripts\python.exe procedural_animation\tests\procedural_checks.py
```

In `mirroring\`:

**Always invoke the venv interpreter directly. Never rely on activation.** Venv
activation does not reliably persist between separate shell commands, and a command
that silently falls back to system Python will fail in confusing ways (MediaPipe and
scipy are only installed in the venv).

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1
.\venv\Scripts\python.exe gui.py
.\venv\Scripts\python.exe tests\solver_checks.py
.\venv\Scripts\python.exe -m pip install <pkg>
```

In the debug window: `c` calibrates, `Esc` quits.

The venv has **no torch and no GPU stack**, and must stay that way. Everything that
needs torch/CUDA runs on the Spark.

## Architecture

### Live lane

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

Solver inputs, all optional except the first; each missing input degrades one
part of the solve rather than failing:

```python
solve(pose_world_landmarks, face_landmarks, image_size, left_hand, right_hand)
solve_hands(pose_world_landmarks, left_hand, right_hand)
```

MediaPipe is used through the **Tasks API** (`mp.Image`, `.task` model bundles).
The legacy `mp.solutions` namespace and `mediapipe.framework.formats.landmark_pb2`
**do not exist** in the installed version. Drawing helpers come from
`mediapipe.tasks.python.vision`.

### Procedural lane

```
 Spark (spark-001, user etlam)               Windows laptop                                Unreal
┌──────────────────────────┐  HTTP / NPZ   ┌───────────────────────────────────────┐  localhost ┌────────┐
│ kimodo service (FastAPI) │ ◄───────────► │ CLI → kimodo client → clip cache      │  UDP 9001  │ Manny  │
│  warm model, Lock,       │  via SSH -L   │      → Retargeter (tested, Phase 0)   │  ────────► │  → MH  │
│  /generate  /health      │  tunnel       │      → motion player (60 Hz clock)    │  UDP 11111 │        │
│  bound to 127.0.0.1      │               │      → existing OSC + face encoders   │  ────────► │        │
└──────────────────────────┘               └───────────────────────────────────────┘            └────────┘
```

Generated motion enters the pipeline **at the same seam as the solver's output**: a
`list[BoneTransform]` in the 60-bone order. Everything downstream (encoders, plugin,
Manny AnimBP, IK Retargeter, `ABP_Face`) is reused untouched.

**Status:**

- Phase 0 done: BVH reader, source-skeleton tables, retarget and player; saved BVH
  clips play into Unreal through the existing pipeline.
- Phase 1 built and tested offline (2026-10-07): `remote_kimodo_service/` holds the
  Spark service, its `requirements.txt`, the NPZ contract, the client, the disk cache
  and the loop player; `procedural_animation/procedural_conductor.py` is the CLI
  (`procedural_conductor_bvh.py` is the frozen Phase 0 copy, BVH only). Generating on
  the real Spark since 2026-10-07; not yet judged in Unreal.
- ARDY streaming, the mixer with the webcam lane, and ARDY-SOMA are later
  (`procedural-animation.md` §6).

## Procedural lane — invariants

Do not break these. Each one comes from a failure in the earlier MetaHuman-direct
attempts (`procedural-animation.md` §3) or from the network measurements.

1. **No edits to `pose_solver.py`, `conductor.py`, the plugin or any Unreal asset**
   to make generated motion work. Import their tables, `BoneTransform` and encoders
   read-only. If something seems to require an edit there, stop and ask.
2. **Never swizzle quaternion components.** No `[x, z, -y, w]` tricks, no
   conjugates. The change of basis is `C · R · Cᵀ`, never `C · Rᵀ · C`. Source
   skeleton space (RH, Y up, +Z forward, metres) → Manny component space
   (X = character's left, Y = forward, Z = up, cm).
3. **Retarget global rotations against Manny's measured rest pose**, then
   `local = parent_global⁻¹ · global` over the **full** Manny chain (including the
   unstreamed `spine_03`, `spine_05`, `neck_02`). Never copy local rotations across rigs.
4. **Mapping tables are per source skeleton, never shared by name.** In SOMA,
   `LeftLeg` is the thigh and `LeftShin` the calf. In ARDY-Core, `LeftUpLeg` is the
   thigh and `LeftLeg` the calf.
5. **The service returns raw source data and knows nothing about Manny.** The NPZ
   holds global rotations, root positions (metres), foot contacts, fps, bone order
   names, and prompt/seed metadata. All Manny knowledge stays on Windows, next to
   the tests that cover it.
6. **The network never sits on the real-time path.** The 60 Hz player never waits on
   a request. While a request is in flight, the current clip finishes, then the idle
   loop (or a cached clip) plays. Never a frozen pose. Unreal only ever receives
   packets from `127.0.0.1`.
7. **Windows always initiates connections.** No inbound firewall rules, works from
   any network.
8. **Cache every returned clip on disk**, keyed by prompt + seed + frame count (+
   model name). If the Spark is unreachable, the cache is the library.
9. **Packet invariants:** the pose packet is always 421 floats; `present=1.0` while
   playing. The head for the MetaHuman goes through `head_rotation_to_curves()`
   unchanged, with `head_rotation = G_manny[head] · rest_global[head]⁻¹`.
10. **Services bind to `127.0.0.1` or the Tailscale IP, never `0.0.0.0`.**
11. **Kimodo `post_processing=True` stays on** (foot-skate cleanup, constraint
    enforcement).
12. **Units:** the Kimodo NPZ is in metres; the retarget skeleton (from the BVH) is in
    cm. The frames must be in the skeleton's units: the adapter multiplies by 100.
    (Setting `units_to_cm = 100` on the BVH skeleton does nothing: the root scale
    cancels a factor applied to rest and frames alike.) Kimodo's 77 joints are the BVH
    joints minus the `Root` wrapper, same order; Root gets an identity rotation, and
    the root position is the Hips joint's own (`posed_joints[:, Hips]`).
13. **The NPZ proves its own meaning.** It carries `posed_joints`; FK over the standard
    T-pose offsets with its rotations must reproduce them (0.5 cm), or the clip is
    refused and never cached. This is what catches a convention change, a transpose or
    a unit slip in a future Kimodo version. Native-convention rotations are recognised
    and converted, as for BVH files.

## Spark access

| | |
|---|---|
| Host | `spark-001`, Tailscale `100.83.6.8` |
| User | `etlam` |
| Node origin | Shared into Malte's tailnet from another account (`whatphilipcodes@`). That account's ACLs decide which ports are reachable. SSH is known to work |
| Location | Moved out of the Filmuniversität network (2026-10); stays out for the near future |
| Link | **Direct** since the move: `tailscale ping` 2026-10-07 `via 5.61.145.75:17793 in 35ms`, netcheck `MappingVariesByDestIP: false`. Before (university): relayed through DERP Frankfurt, ~45 ms with spikes, stall-then-burst on loss. Keep designing so a relay would still work |
| SSH auth | Password for Malte. Claude has a key, `~\.ssh\id_ed25519_spark` (installed 2026-10-07): `ssh -i ~/.ssh/id_ed25519_spark -o BatchMode=yes -o IdentitiesOnly=yes etlam@100.83.6.8`. **Read-only inspection only** - Malte installs packages and starts the service himself. Revoke by deleting the `claude-readonly@malte-laptop` line in `~/.ssh/authorized_keys` |
| Hostname | `kaspar` (Tailscale name still `spark-001`). Shared machine: user `philip` is also logged in, so the GPU may be busy with someone else's job |
| Hardware | NVIDIA GB10, driver 580.178.04, 121 GB unified memory (nvidia-smi shows no per-process VRAM), Ubuntu 24.04.5, aarch64 |
| Service port | `8765` (setting, not hard-coded); free as of 2026-10-07 |
| Python env | `~/project_kaspar/modules/kai-avatar-animation/pipeline-network-osc/venv` (Python 3.12.3): kimodo 1.0.0, torch 2.13.0 (CUDA 13 build), fastapi, uvicorn, pydantic, numpy. The new service runs with this interpreter; nothing to install. (`~/ardy/ardy` is the ARDY venv, torch cu132; `~/ComfyUI/venv` unrelated) |
| Kimodo checkout | `~/kimodo-src` at `1aece8c` (2026-07-13), **locally patched** for aarch64 (MotionCorrection: `sse2neon.h`, `SIMD.h`, `CMakeLists.txt`); installed with `pip install --no-build-isolation ~/kimodo-src`. The plain git URL does not build here |
| Repo on the Spark | `~/project_kaspar/modules/kai-avatar-animation`, old (`6d5ca17`, 2026-08-04), clean. New code arrives by `git pull` |
| HF cache | `/opt/huggingface_cache` (also set system-wide in `/etc/profile.d/hf_cache.sh`); has `Kimodo-SOMA-RP-v1.1`, the LLM2Vec / Llama-3-8B text encoder, `ARDY-Core-RP-20FPS-Horizon40` |
| Kimodo API facts (read from the installed source) | `model.fps` exists (30). `global_rot_mats` and `posed_joints` come from one `somaskel77.fk` over the **standard T-pose** convention (`save_motion_bvh` converts to native only when `standard_tpose=False`), so the FK self-check holds by construction. `root_positions` = `posed_joints[:, root_idx]` (Hips). `foot_contacts` is bool (T, **6**) - measured on real output; the docs say 4. Post-processing replaces the locals before the 77-joint FK, so it stays consistent |

**Default transport: SSH port forward.**

```powershell
ssh -N -L 8765:127.0.0.1:8765 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 etlam@100.83.6.8
```

The Windows client talks to `http://127.0.0.1:8765`; the Spark service stays bound to
`127.0.0.1`. The base URL is a setting. If the direct check below passes, switching
to `http://100.83.6.8:8765` needs no code change.

**Service lifetime: tied to the login session, on purpose.** Malte starts the service
by hand in his SSH session; it must end when he logs out. Do **not** use `tmux`,
`screen`, `nohup`, `disown` or a systemd unit, and do not add auto-restart. Because the
service can vanish at any time, the client must treat "connection refused / timeout"
as a normal state: print a clear message ("Spark service not reachable — start it and
the tunnel"), fall back to the cache, and never crash the player.

**Health and checks:**

```powershell
curl.exe -s http://127.0.0.1:8765/health          # through the tunnel
tailscale ping --c 5 100.83.6.8                    # "via DERP(fra)" = relayed
tailscale netcheck
# Direct-port check (only if you want to skip the tunnel). On the Spark:
#   python3 -m http.server 8765 --bind 100.83.6.8
curl.exe -s -o NUL -w "%{http_code} %{time_total}s\n" http://100.83.6.8:8765/
```

Re-run `netcheck` after the Spark moves to the colleague's home. If it still shows
DERP, forwarding UDP 41641 on that router usually enables a direct connection.

## Kimodo service (runs on the Spark)

Built as `remote_kimodo_service/kimodo_service.py`, rewritten from
`pipeline-network-osc/kimodo_service_handoff_osc_v3.py`. **Never edit the old Kimodo
services** (`pipeline-network-osc/`, `pipeline-network-editor/`, `kimodo/`); new Kimodo
work goes into `remote_kimodo_service/`. Its `requirements.txt` is what Malte installs
on the Spark; keep it complete. The design rules it follows:

- **Keep:** FastAPI, model loaded once and kept warm, `asyncio.Lock` (a request is
  never interrupted mid-generation), shared HF cache, timing log.
- **Delete:** all OSC streaming and all quaternion swizzling.
- `POST /generate` — request JSON: `prompt`, `seed`, frame count / seconds, and an
  optional `first_frame` field **in the schema but not implemented in v1** (reserved
  for chaining). Response: the clip as `np.savez_compressed` bytes.
- `GET /health` — model name, loaded yes/no.
- **Synchronous request, 60 s client timeout** for v1. Add job-ID polling only if the
  relay proves to be a problem.
- Log generation time and transfer time **separately** per request (client side
  measures total, service side measures generation), so a slow day shows whether the
  GPU or the link is at fault.
- Kimodo limits to remember: max 10 s per prompt, 30 fps, ~17 GB VRAM mostly from the
  text encoder, hands are a static relaxed pose (not animated), no face.

## Procedural CLI (Windows)

Extend the existing procedural script rather than adding a parallel entry point.
Target behaviour:

- `--prompt "..." [--seconds N] [--seed N]`: one-shot. Generate, cache, retarget,
  play into Unreal, exit.
- No `--prompt`: small REPL. Type a prompt, it generates in a **background thread**
  and queues the clip while input stays responsive. `/list` shows cached clips,
  `/quit` exits. Both run in the same loop.
- `--bvh <file>` and cached clips keep working fully offline.
- **Playback v1 (Malte, 2026-10-07): plain.** Clips play exactly as retargeted. The
  current clip loops; a queued clip takes over when the current pass ends (hard cut;
  `/next` cuts at once). **No re-centring, heading alignment, crossfade, stage box or
  idle loop yet** - a new clip may start elsewhere, facing elsewhere. Get the base system
  proven first; those layers come later (crossfade ~0.3 s per-bone slerp, idle loop).
- Inline prompt options: `/s N` steps (default 100), `/seed N|random` (default seed
  is fixed at 0, so a repeated prompt is a cache hit), `/t N` seconds (default 9, max 10).
- Cache: `remote_kimodo_service/cache/`, gitignored.
- Spark URL, timeout and cache directory are settings with defaults, not constants.
- `conductor.py` and the procedural script both send to 9001/11111. **Run one or the
  other**, not both (mixing is Phase 3).

## Head rotation — who owns it, and why it is routed this way

Resolved and verified end to end in UE 5.8 (turned and leaning torsos with turned
heads land on target to 0.000°). Read before touching anything head-related in Python
or in Unreal.

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
component's own animation carries: ARKit head rotation when Live Link Face is
connected, bind pose (identity) when it is not.

### The decision

**Send head rotation through the Live Link Face channel**, which the MetaHuman already
routes to the head, rather than rewiring Epic's asset.

1. **Distribution.** K.ai users only add a Retarget Pose node to the Body AnimBP. No
   MetaHuman asset gets edited, so nothing breaks when Epic restructures MetaHumans.
2. It repairs the Live Link Face head channel, which was independently broken.
3. Both head signals derive from the same basis, so they cannot disagree.

The body stream **keeps** its own head rotation (Manny-only and non-MetaHuman targets
depend on it). This applies to **both lanes**: generated motion also drives the
MetaHuman head via the face channel.

Measured behaviour of the curves on a MetaHuman (UE 5.8, `tests/ue_head_probe.py
--measure`): they set `neck_02` + `head` to an **absolute** component-space rotation,
50° per unit on every axis; `headYaw` + turns left, `headPitch` + looks up, `headRoll`
+ tips the head to the character's right; composed pitch · roll · yaw.

### The alternative, documented but not taken

Change the `Layered blend per bone` Layer Setup branch root from `head` to
`FACIAL_C_FacialRoot`. Works, but it is a per-user edit to an Epic asset. Keep as a
README note for users who want body-driven head.

## head_pose_capture.py — disabled, kept on purpose

**Comment the module out; do not delete it.** It is the seed of the face-crop
experiment under Open items.

- Measured: that standalone `FaceLandmarker` **never detected the face at performance
  distance** (31 of 652 frames, vs 597 for Holistic). Live Link Face's head
  yaw/pitch/roll was pinned at zero for the whole project until the solver-derived
  head replaced it.
- It cost a full extra inference pass per frame.
- Why it might come back: `FaceLandmarker` supports
  `output_facial_transformation_matrixes`; `HolisticLandmarker` does not. Running this
  module on a crop around the pose-derived head recovers that fitted matrix.

Leave the imports, the class and the call sites commented with a pointer to this
section.

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
  vertical. Lean is measured relative to the rig's rest torso.

**Non-obvious invariants:**

- `spine_03`, `spine_05` and `neck_02` exist in Manny but are NOT streamed. They are
  modelled internally anyway (`FULL_CHAIN`), because Unreal applies each streamed local
  transform relative to the bone's REAL parent. Omitting `spine_03` alone rotates the
  whole upper body by ~11°.
- `_convert_landmarks_to_ue_space` mirrors X (`Right = -lm.x`). MediaPipe reports the
  performer's LEFT side with positive `lm.x`.
- Twist about each **arm** bone's axis is unobservable from joint positions alone. The
  wrist is the exception, since the palm plane is measured.
- Metacarpals carry a few degrees of disclosed error: MediaPipe has no landmark at a
  metacarpal's base, so they are aimed `WRIST→MCP`. Every phalanx solves exactly.
- MediaPipe never reports "I can't see that limb"; it invents a landmark and lowers
  `visibility`/`presence`. Hence the occlusion gating.
- The face mesh is image-normalised only. Scaling x and z by width and y by height
  makes it metrically consistent.

### Head rotation for the face channel

The face channel re-expresses the **same** head orientation the solver already
computes (`PoseSolver.head_rotation` → `head_rotation_to_curves()`); there is no
second solve. The conversion is not a passthrough: the solver's head is relative to
the rig's rest torso, the curves are absolute in component space, so the torso
orientation is composed back in. The calibration's neutral head pose is the shared
zero reference for both lanes.

## Acceptance tests

### Live lane

```powershell
.\venv\Scripts\python.exe tests\solver_checks.py            # all recordings\*.npz
.\venv\Scripts\python.exe tests\solver_checks.py some.npz   # one capture
```

No camera needed. Asserted checks exit with code 1 on failure; real-capture checks
are reports. Any solver change must keep all asserted checks passing.

1. **Rest-pose identity** — the rig's bind pose in, bind pose out: 22/22 body bones,
   30/30 phalanx and thumb bones, metacarpals within their disclosed 12.5°. Plus:
   straight-ahead face mesh is a no-op; a turned head round-trips at 0.0000° after
   Unreal-style FK; wrist roll of ±45/−60° round-trips exactly.
1b. **Anatomical report** — rest pose with landmarks at the rig's REAL joint positions.
2. **Bind-pose FK** — upright, symmetric, `foot_l` at `(14.09, -0.99, 8.24)`.
2b. **Timed calibration** on a synthetic clock.
2c. **Occlusion gating** — follows confident landmarks, holds low-confidence ones
    without showing the invented pose, resumes monotonically, respects hysteresis.
2d. **Ground locking** — bind pelvis height preserved, foot stays on the floor through
    a crouch, `ground_lock=False` still ignores it.
2e. **Face-channel head round-trip** — synthetic head pose through the solver and out
    through `live_link_face_protocol`, recovered to the same yaw/pitch/roll, including
    frames with a **turned torso**.
2f. **Not-zero regression** — the face channel's head values vary on a recording with
    head motion.
3. **Absolute direction error on real captures** — per-bone 3D direction vs the
   performer, in absolute terms, NOT as an angle-from-spine.

**Ground truth is deliberately independent of the solver's own aim tables.** Checking
a bone against the same landmark pair the solver aims it by passes by construction and
hides a mis-mapped bone. Keep `BODY_TRUTH` / `FINGER_TRUTH` / `HAND_JOINT_AT`
anatomical.

### Procedural lane

Offline, no GPU, no engine, no Spark. Same philosophy: ground truth independent of the
retarget's own tables. The file is `procedural_animation\tests\procedural_checks.py`;
extend it rather than starting a new one. Its own numbering differs slightly: its 5 is
the wire format, 5b the player; 6 and 7 match the list below. Item 5's crossfade
continuity and stage box wait for those features (deferred, see Procedural CLI).

1. **Rest identity** — source rest pose in → Manny bind out, 22/22 body bones.
2. **Direction truth on real clips** — Manny FK bone direction vs the source's own FK
   direction for the anatomically matching joint pair, in absolute terms, on the
   existing `kimodo-gen/*.bvh` clips.
3. **Change-of-basis sanity** — a source walking forward (+Z) moves Manny along +Y;
   raising the source's left arm raises `upperarm_l`. Catches transpose/inverse and
   left/right swaps.
4. **Head channel** — `head_rotation_to_curves()` round-trips to the retargeted head on
   a clip with the torso turned.
5. **Player** — fixed 421-float packets, slerp continuity at clip boundaries (no
   per-bone jump above a threshold between consecutive sent frames), re-centring keeps
   the pelvis inside the stage box.
6. **Client resilience (new, needs no Spark)** — against a stub server or a closed
   port: timeout, connection refused and a malformed NPZ each leave the player running
   and fall back to the cache with a clear message. A cache hit never touches the
   network.
7. **NPZ contract** — a fixture NPZ with the documented keys, shapes and units (metres,
   `fps`, `bone_order_names`) loads through the client into the retargeter.

Then the Unreal check, stage by stage like `tests/ue_head_probe.py`: stream a handful
of known poses and read the MetaHuman bones back.

### Current numbers (live lane, rest.npz / motion.npz)

| Metric | Baseline | Now |
|---|---|---|
| Body bone directions | 0.07° / 0.18° | 0.00° / 0.00° (trusted frames) |
| All finger bones | 11.2° / 10.9° | 0.00° / 0.00° |
| Palm orientation | 19–26° / 25–47° | 3.0–3.8° / 3.1–4.1° |
| Head turn | dead (welded to chest) | exact; −30…+35° yaw, −19…+43° pitch |
| Lowest foot point | −2.0…+39.6 cm | 0.75 cm, every frame |
| Legs held (occluded) | n/a | 17–22% of frames |

The `hand_l`/`hand_r` rows read 10–25° because they are measured against the *pose*
model's INDEX point, which disagrees with the dedicated hand model by 15–23° RMS.
Not solver error.

### Recording new captures

```powershell
.\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\name.npz
```

Raw landmarks (pre-solve, pre-smoothing), so recordings stay useful across solver
rewrites. Add a trim window to `recordings\trims.json`; the walk to and from the
laptop is in every clip and is not performance.

## Calibration

`c` in the debug window, or the GUI button: a 3 s countdown, then the torso lean and
the neutral head pose averaged over 30 frames. Single frames of the same standing clip
spread 5.4–7.5°, so one snapshot is not enough. State machine:
`PoseSolver.begin_calibration()` / `update_calibration()`; `conductor` drives it per
frame, the GUI polls the result on the main thread.

MediaPipe's forward-lean bias is **not constant across setups** (~18° in early
sessions, ~7° recently). Always calibrate; never hardcode. Calibrate square to the
camera (a turned torso under-counts the lean; open item).

## gui.py — Tkinter control surface

A wrapper that drives an unmodified `Conductor`; it reimplements no pipeline logic.
`python conductor.py --debug --camera 1` must keep behaving identically. It
monkeypatches `cv2.imshow`/`waitKey`/`namedWindow` before constructing `Conductor`,
runs `run()` on a daemon thread, and updates widgets only on the main thread via
`root.after`. Frame queue is `maxsize=1`, drop-when-full: **the GUI must never apply
backpressure to the capture loop.**

Live controls (no restart): smoothing alphas, torso lean, calibrate, overlay/FPS
toggles. Restart-required: camera index/resolution, model paths, ports/IPs, confidence
thresholds. These mark the panel dirty and enable **Apply & Restart**.

A "Procedural" tab (prompt box, queue, connection status) is planned later and would
touch only `gui.py`.

## Open items

### Next up

- **Kimodo on-demand lane (Phase 1): first look in Unreal.** Generation through the
  tunnel works (2026-10-07). Real output: standard convention, FK self-check 0.00003 cm,
  `root_positions` = Hips exactly, 40 Manny bones within 0.0006° of Kimodo's own joints;
  a real clip is the fixture `procedural_animation\tests\fixtures\kimodo_real_turn_around_2s.npz`.
  Measured on the GB10, 270 frames (9 s): 100 steps 5.8 s warm (7.3 s first request),
  50 steps 2.9 s, 25 steps 1.5 s, 10 steps 0.7 s; transfer of the ~890 KB NPZ 0.3-1.3 s
  (direct link, through the SSH tunnel); `/health` round trip 76 ms. Next: play into
  Unreal (REPL), judge quality vs steps by eye.
- **Head yaw beyond 75° on the face channel (open problem, procedural lane).** The Live
  Link Face head curves are ABSOLUTE component-space rotation, measured in Unreal as
  linear only up to 75°. Generated motion turns the whole body: measured over 243
  existing Kimodo clips, 13 exceed 75° of head yaw and 7 reach ±180°, where the Euler
  yaw also wraps (`headYaw` jumps +3.6 → −3.6 in one frame). Unmeasured what the
  MetaHuman does there. Procedural test 4 decodes with the same linear model, so it
  passes regardless. Needs a `ue_head_probe.py` stage at 90/135/180° before deciding
  (cap the yaw, keep heading near the camera, or rely on Unreal if it is linear).
- **Playback layers deferred from v1:** re-centring + heading alignment (the face-channel
  `head_rotation` must be rotated by the same yaw, or the head disagrees with the body),
  per-bone crossfade, idle loop, stage box.
- **Ear/nose fallback head aim, now load-bearing.** Head rotation depends entirely on
  the face mesh, which is lost when the performer turns away. Blend to an aim from the
  pose model's `LEFT_EAR`/`RIGHT_EAR`/`NOSE` as face confidence drops, reusing the
  occlusion-gating hysteresis. Record a 360° turn first so the dropout point is
  measured, not guessed.
- **Head is still a bit wobbly live.** The conversion is exact, so the jitter comes
  from the face-mesh basis frame to frame and/or smoothing. Measure on a recording
  before tuning.

### For you (Malte)

- **Test the solver in UE** if not already done: calibrate upright once, then turn and
  nod, rotate wrists palm-up/down, spread and curl fingers, crouch, lean out of frame.
  Confirm the state of branch `feature/solver-fixes` (merged or not).
- **Record an occlusion clip** if you want the gating thresholds tuned harder: lean
  over the desk until the legs leave frame, step half out of shot, put one arm behind
  your back.
- **Run the 5-minute Spark network check** (Spark access section) and note whether
  non-SSH ports work, so we know whether the tunnel is needed.
- **Request Llama-3-8B-Instruct access on Hugging Face** before the ARDY phase.

### Not started

- **ARDY-Core streaming (Phase 2).** Autoregressive, ~33–63 ms per step, mid-motion
  prompt changes. Needs a persistent Windows-initiated connection and a 150–250 ms
  jitter buffer while relayed (not 2–3 frames). 20 fps source, slerp up to 60 Hz.
- **Mixer with the webcam lane (Phase 3).** Proxy that blends the conductor's packets
  with the generated pose per body region ("AI body, webcam face and hands").
- **ARDY-SOMA (Phase 4).** Unreleased; reuses the Kimodo SOMA table when it ships.
- **Chaining.** `first_frame` full-body keyframe constraint so consecutive clips need
  no crossfade.
- **Distortion layer.** The tracking artefacts (dead head, twisted fingers) are wanted
  as *art-directable* effects. Decision: keep the solver clean, add distortion
  downstream: geometric distortions in Unreal (Control Rig node or Transform (Modify)
  Bone **after** the retarget), tracking-failure distortions as an optional module
  between `pose_solver` and the OSC encoder, off by default; driven live over a second
  OSC address space on the existing `UOSCServer`. Tag the current commit before fixing
  anything else; some artefacts are emergent and will not reproduce procedurally.
- **Root translation (live lane).** MediaPipe world landmarks are hip-centred, so the
  live character can never step, shift or jump. Needs image-space landmarks or a
  separate position estimate. An architecture decision, not a tune. The procedural
  lane already provides real root motion.
- **Face crop experiment.** Reinstate `head_pose_capture.py` on a crop around the
  pose-derived head. Only if the landmark-derived basis proves fragile.
- **Knees converge.** The character's stance is narrower than the performer's (~0.65
  knee/hip ratio). Not investigated.
- **Lean calibration with a turned torso** (see Calibration).
- **MetaHuman neck seam.** The head curves also move the Body mesh's `neck_01` (up to
  13° for a 50° pitch) while the Face mesh's barely moves. Possible visible seam.
- **Benchmark Full Body IK vs Body Mover + Limb IK** for the Manny → MetaHuman retarget.
- **Cleanup candidates:** `mediapipe_pose_osc_protocol.py` (reachable only through dead
  code), `live_link_pose_json_protocol.py` (unused), `pose_landmarker_full.task`
  (nothing loads it), this repo's stale `ue_plugin/` copy. The live plugin is at
  `K:\KaiTracking\Plugins\MediaPipeLiveLink`; edit C++ there.

## Code standards

- Python 3, type hints, dataclasses for frame/transform structs.
- Comments explain *why*, especially where a non-obvious convention is load-bearing,
  and record what a number was measured from rather than asserting it.
- **Prefer measuring over guessing.** When a pose looks wrong, compute the error
  against ground truth before changing any math. Every fix in the solver was found
  that way, and two of them (the thumb, the lean reference) were invisible until the
  ground truth was made independent of the solver.
- Network code: explicit timeouts everywhere, no blocking call on the player thread,
  errors surfaced as readable one-line messages, never a traceback in the middle of a
  performance. Two measured traps: over `ssh -L`, a stopped service looks like
  "accepted and closed", not "refused" (refused = tunnel down); and on Windows a
  refused localhost connect takes 2.05 s, so timeouts shorter than that report a
  timeout instead.