# Procedural animation — Kimodo and ARDY into the mirroring pipeline

Research notes, comparison and integration plan. Written 2026-09-28 against
`nv-tlabs/kimodo` and `nv-tlabs/ardy` at their `main` heads, the ARDY paper
(arXiv 2607.08741), the Hugging Face model cards, and this repo's earlier Kimodo
attempts (`kimodo/`, `pipeline-network-editor/`, `pipeline-network-osc/`).

> **Status, 2026-10-07: Phases 0 and 1 are built and working.** Saved BVH clips and
> prompts generated on demand on the DGX Spark both drive the MetaHuman through the
> unchanged encoders, plugin and Unreal setup. How it works as built: `README.md` §9.
> What is load-bearing and what is open: `../CLAUDE.md`. This document stays the
> research and plan; where the build departs from it:
>
> | Plan | As built |
> |---|---|
> | `procedural/` package, `services/` folder (§4, §4.5) | `procedural_animation/` (BVH reader, tables, retarget, CLI, tests) and `remote_kimodo_service/` (Spark service, NPZ contract, adapter, client, cache, player), both at the module root |
> | `Spine2 → spine_03`, `Neck2 → neck_02` (§4.1) | `spine_03`, `spine_05`, `neck_02` get no source joint and stay at bind relative to their parent, exactly as Unreal treats unstreamed bones; `Spine2 → spine_02`, `Chest → spine_04` |
> | Swing alignment for every bone (§4.1) | Same, except the **head**: identity map of rest frames, because the swing tilted every head up 6.5° (SOMA's `Head → HeadEnd` leans back) |
> | NPZ: rotations, root, contacts, fps, names, metadata (§4.3) | Plus `posed_joints`, so the client proves by FK that the rotations mean what the retarget assumes; a clip that fails is refused and never cached |
> | Player: crossfade, chaining constraint, idle loop (§4.2, Phase 1) | v1 is deliberately plain: the current clip loops, the next one hard-cuts in at the end of the pass, no re-centring. Crossfade, heading alignment, idle loop and chaining (`first_frame`, reserved in the request schema) come next |
> | Kimodo on WSL2 / a local 4090 (§6) | On the DGX Spark (`kaspar`, GB10): 5.8 s for 9 s at 100 steps, 1.5 s at 25 steps, reached through an SSH tunnel |
> | `kimodo_service_handoff_osc_v3.py` as the base (§4.3) | Rewritten in `remote_kimodo_service/kimodo_service.py`; the old services are left untouched |
>
> New open problem found while building: the face channel's head yaw is absolute and
> was only measured linear to 75° in Unreal, while generated clips turn up to ±180°
> (`README.md` §10).

**Short answer.**
- **Kimodo:** usable today. It generates whole clips (≤ 10 s each) on the SOMA
  skeleton in about 3 s on a 4090.
- **ARDY:** also usable today, on its 27-joint **Core** skeleton. That skeleton
  is not the WIP one; only ARDY-**SOMA** is unreleased. ARDY is the one that
  streams in real time (33–63 ms per step), takes new prompts mid-motion and
  follows root waypoints. Its Core skeleton has no fingers and fewer spine
  joints.
- **Common requirement:** both need the same missing piece: a correct,
  rest-pose-aware retarget from their skeleton onto the 60 Manny bones that
  `mirroring` already streams. That retarget can be written as a new module
  beside the pipeline, reusing the existing encoders, plugin and Manny →
  MetaHuman setup unchanged. **No edits to `pose_solver.py`, `conductor.py`,
  the plugin or any Unreal asset are required.**

---

## 1. What each model is

### Kimodo

A kinematic motion **diffusion** model: it produces a complete, fixed-length
sequence per call.

| | |
|---|---|
| Repo / docs | `github.com/nv-tlabs/kimodo`, `research.nvidia.com/labs/sil/projects/kimodo/docs` |
| Models | `Kimodo-SOMA-RP-v1.1` (what this repo used), `Kimodo-SOMA-SEED-v1.1`, `Kimodo-G1-RP-v1`, `Kimodo-SMPLX-RP-v1` (R&D licence, "severe retargeting artifacts" per the docs) |
| Skeleton | Predicts on `somaskel30`, returns `somaskel77` (77 joints + `Root`, full fingers). **The fingers of the 77-joint output are filled from a fixed "relaxed hands" rest pose**. Hands are not animated. |
| Frame rate | 30 fps (all saved BVHs here: `Frame Time: 0.0333`) |
| Length | max **10 s per prompt**. Longer sequences come from `multi_prompt=True` (segments generated one after another and stitched with `num_transition_frames`) or from chaining calls with a full-body keyframe constraint on frame 0. |
| Control | text, full-body keyframes, 2D root path/waypoints, end-effector position/rotation (hands, feet) — sparse, < 20 keyframes per constraint type |
| Speed (measured here) | model load 24 s; 270 frames × 100 DDIM steps ≈ **2.5–2.7 s** on an RTX 4090 (`kimodo/kimodo-gen/gen-time.txt`); 3–11 s in the OSC service log (`pipeline-network-osc/pipeline_timing_log.txt`, different machine/steps) |
| VRAM | ~17 GB (mostly the LLM2Vec / Llama-3-8B text encoder); < 3 GB with `TEXT_ENCODER_DEVICE=cpu` |
| Platform | developed on Linux; this repo ran it under WSL2 and on the DGX Spark |
| Licence | code Apache-2.0; SOMA/G1 weights NVIDIA Open Model Agreement (commercial OK) |

Python API (verified in `kimodo/model` and the API reference):

```python
from kimodo.model.load_model import load_model
model = load_model("Kimodo-SOMA-RP-v1.1", device="cuda")
out = model(prompts="A person waves both arms", num_frames=270,
            num_denoising_steps=100, constraint_lst=[...],
            cfg_weight=[2.0, 2.0], multi_prompt=False,
            num_transition_frames=5, post_processing=True, return_numpy=True)
# out keys: local_rot_mats [B,T,J,3,3], global_rot_mats [B,T,J,3,3],
#           posed_joints [B,T,J,3], root_positions [B,T,3], smooth_root_pos,
#           foot_contacts [B,T,4], global_root_heading [B,T,2]
skel = model.output_skeleton          # SOMASkeleton77: bone_order_names, fk(), neutral_joints
```

**Coordinate system:** right-handed, **Y up, +Z forward**, metres.
`post_processing=True` does foot-skate cleanup and constraint enforcement; keep it on.

### ARDY

An **autoregressive** diffusion model (SIGGRAPH 2026). It generates a short
window, appends it to its history, and repeats, re-planning when the prompt or
the constraints change. This is the "real-time Kimodo".

| | |
|---|---|
| Repo | `github.com/nv-tlabs/ardy`, project page `research.nvidia.com/labs/sil/projects/ardy/` |
| Released models | `ARDY-Core-RP-20FPS-Horizon40`, `ARDY-Core-RP-20FPS-Horizon8`, `ARDY-G1-RP-25FPS-Horizon52`, `ARDY-G1-RP-25FPS-Horizon8` |
| Unreleased | **SOMA** — "coming soon". The registry already reserves it: `DEFAULT_HORIZON["soma"] = 60`, example name `ARDY-SOMA-RP-30FPS-Horizon60`, and `scripts/generate.py` already imports `SOMASkeleton30`. Expect 30 fps, same skeleton as Kimodo. |
| Skeleton (Core) | `cskel27`, 27 joints, **no fingers**, see the table below |
| Frame rate | 20 fps (Core), 25 fps (G1) |
| Horizon | frames generated per step, in 4-frame tokens: Horizon40 = 2 s windows (better quality/planning), Horizon8 = 0.4 s windows (lowest latency, most responsive) |
| Speed (paper) | **33 ms** average per step with 4 denoising steps, 63 ms with 10, RTX 4090 |
| Control | online text changes (re-plan on new input), 2D root waypoints/paths, target velocity, full-body keyframes, sparse joint positions/rotations, **long-horizon constraints beyond the current window** |
| VRAM | LLM2Vec text encoder ~14 GB (cuda/bf16), CPU option available. Standalone encoder server (`scripts/run_text_encoder_server.py`, default `http://127.0.0.1:9550/`) so the encoder is loaded once |
| Gated dependency | the text encoder needs a Hugging Face token **with access granted to `meta-llama/Meta-Llama-3-8B-Instruct`** |
| Platform | tested on Ubuntu 22.04 + RTX 4090, driver 575; builds a C++ extension (`MotionCorrection`, CMake ≥ 3.15). WSL2 is the realistic route on this machine, as with Kimodo. Optional TensorRT extra |
| Licence | code Apache-2.0; weights NVIDIA Open Model Agreement (commercial OK) |
| Known limits | kinematic only (foot skating, jitter), may not follow text consistently, single character, no scene awareness |

API: `load_model("core" | "core8" | ...)`, then `Ardy.autoregressive_step(num_frames,
num_denoising_steps, motion_mask, observed_motion, cfg_weight, texts | text_feat,
init_history_sequence, init_global_translation, init_first_heading_angle)`.
There is **no packaged streaming server**. The reference streaming loop is
`scripts/interactive_demo/generation.py::GenerationMixin._generate_step`:
history slicing, constraint masks, `motion_rep.inverse(...)` to get joints and
rotations back. Our streaming service would be built from that loop.

`cskel27` hierarchy (from `ardy/skeleton/definitions.py`):

```
Hips ─ Spine ─ Spine1 ─ Spine2 ─ Spine3 ─┬─ Neck ─ Head
                                         ├─ RightShoulder ─ RightArm ─ RightForeArm ─ RightHand ─┬ RightHandEnd
                                         │                                                       └ RightHandThumb1
                                         └─ LeftShoulder … (mirror)
Hips ─ RightUpLeg ─ RightLeg ─ RightFoot ─ RightToeBase      (and Left…)
```

`HandEnd` and `Thumb1` exist only to pin down the hand's orientation (a palm
plane). There are no finger chains.

> **Naming trap.** In SOMA, `LeftLeg` is the **thigh** and `LeftShin` the calf.
> In Core, `LeftUpLeg` is the thigh and `LeftLeg` the **calf**. Mapping tables
> must be per skeleton, never shared by name.

---

## 2. Comparison

| | Kimodo-SOMA | ARDY-Core (today) | ARDY-SOMA (future) |
|---|---|---|---|
| Generation | whole clip, offline-ish | streaming, per step | streaming |
| Latency to first frame | ~3 s (4090, 100 steps; fewer steps is faster) | ~33–63 ms per step | same |
| Can change prompt mid-motion | no; next clip only | **yes** | yes |
| Max length | 10 s per prompt (chainable) | unbounded | unbounded |
| Body joints → Manny | 22/22 body bones have a counterpart | spine has 4 joints for Manny's 5; no neck_02 | as Kimodo |
| Fingers | output present but **static relaxed pose** | none | presumably static, as Kimodo |
| Root translation + heading | yes | yes, plus waypoints and velocity control | yes |
| Face / blendshapes | none | none | none |
| Frame rate | 30 | 20 (needs upsampling) | 30 |
| Maturity | released, docs site, used here already | released, README-level docs, demo-driven API | not released |
| Fit for K.ai | a **gesture/motion library**: pre-generate or generate on demand, play back, crossfade | a **live, steerable** performer (LLM or operator changes prompts in real time) | the best of both |

Neither model animates the face or fingers. Both give a full body with root
motion, which is exactly the part the webcam lane cannot give (README §10, "Root
translation").

**Recommendation.** Build the Manny retarget layer and the player **once**,
table-driven per source skeleton. Bring it up on **Kimodo-SOMA first**: there
are already 80+ Kimodo BVH clips in this repo, so the whole retarget can be
developed and tested offline on Windows without a GPU. Then add
**ARDY-Core** as a streaming source through the same layer. When ARDY-SOMA
ships, it reuses the Kimodo SOMA table and only the source changes. The WIP
skeleton does not block ARDY: after a proper retarget, the choice of source
skeleton only affects fidelity (spine resolution, hands), not whether the
integration works.

---

## 3. Why the earlier MetaHuman-direct attempts failed

Reading `pipeline-network-osc/kimodo_service_handoff_osc*.py` and the git history
("semi-working joint remapping", "tried further FK remapping (no success)",
"ups are just pointing up"):

1. **Local rotations were copied across rigs.** SOMA-local rotations were sent
   straight onto the MetaHuman bones via a Blueprint `LivePoseMap` + Control
   Rig. A local rotation only means something relative to that rig's own bone
   axes and rest pose. SOMA's rest is a T-pose with its own bone rolls; the
   MetaHuman/Manny bind is an A-pose with Epic's bone rolls. No per-bone
   formula of the quaternion components can bridge that.
2. **Per-bone component swizzles were guessed** (`[x, z, -y, w]` for forearms,
   `[x, -z, y, w]` for upper arms, `[x, y, -z, -w]` for the rest, …). This is
   the "handedness conversion" anti-pattern that `CLAUDE.md` already bans:
   sign/swap tricks on quaternion components are not a change of basis.
3. **The global-space variant inverted every rotation.**
   `kimodo_service_handoff_global_fk_osc.py` computes `M @ R.T @ M.T`. The
   transpose is the inverse rotation; the change of basis is `M @ R @ M.T`.
   It then sent **global** rotations to a receiver that applies **local**
   ones.
4. **The FBX/editor route** (`pipeline-network-editor`, Blender → FBX →
   `ue_import_pipeline.py` → IK Retargeter) was correct in principle, since
   Epic's retargeter handles rest pose and proportions. But it only runs in
   Edit mode, costs seconds per clip, and cannot be live.

What carries over from those attempts:
- the warm-model FastAPI service with an `asyncio.Lock` (a performance is never
  interrupted mid-move)
- the WSL2 → Windows gateway-IP lookup
- the shared Hugging Face cache
- the timing log
- the clip library in `kimodo-gen/`

The `mirroring` lane later solved the same problem for MediaPipe: compute
**global** orientations in Unreal component space against **Manny's measured
rest pose**, then take `local = parent_global⁻¹ · global` over Manny's **full**
chain (including the unstreamed `spine_03`, `spine_05`, `neck_02`). The plan
below applies that same, already-verified technique to generated motion.

---

## 4. Integration architecture

Principle: generated motion enters the pipeline **at the same seam as the
solver's output**, as a `list[BoneTransform]` in the 60-bone order. Everything
downstream is reused untouched:
- `LiveLinkPoseOSCEncoder` → OSC 9001 → `MediaPipeLiveLink` → Manny → IK
  Retargeter → MetaHuman
- `LiveLinkFaceEncoder` + `head_rotation_to_curves()` → UDP 11111 → the
  MetaHuman's head

```
 WSL2 / Linux (GPU, torch)                   Windows, mirroring venv (no torch)
┌──────────────────────────┐   clip (npz)   ┌───────────────────────────────────────────┐
│ kimodo_service.py        │ ─────HTTP────► │ procedural/                               │
│  FastAPI, warm model     │                │  source_skeletons.py  SOMA77 / Core27     │
└──────────────────────────┘                │     tables: names, parents, rest, → Manny │
┌──────────────────────────┐  frames (20Hz) │  retarget.py   global rots → 60 Manny     │
│ ardy_service.py          │ ──UDP/ZMQ────► │     BoneTransforms + head_rotation        │
│  autoregressive loop,    │ ◄──prompt/HTTP │  motion_player.py  clock, slerp upsample, │
│  text-encoder server     │                │     crossfade, idle, root re-centering    │
└──────────────────────────┘                │  procedural_conductor.py  CLI / GUI hook  │
                                            └───────────────┬───────────────────────────┘
                                                            │ imports, unchanged:
                                  pose_solver (FULL_CHAIN, BIND_*, FINGER_CHAIN, BoneTransform)
                                  live_link_pose_osc_protocol.LiveLinkPoseOSCEncoder  → OSC 9001
                                  live_link_face_protocol.{LiveLinkFaceEncoder,
                                                           head_rotation_to_curves} → UDP 11111
```

Why the split: torch, CUDA and Llama stay inside the Linux services (as they
already do), and the `mirroring` venv gains no heavy dependencies. The services
send **source-skeleton data** (global rotation matrices + root position in the
model's own space). All Manny knowledge lives on the Windows side, next to the
tables it depends on.

### 4.1 The retarget (`procedural/retarget.py`) — the one piece of real math

For each Manny bone `b` driven by source joint `j`:

```
C           = change of basis, source (RH, Y-up, +Z fwd, X = character's left)
              → Manny component space (X = character's left, Y = fwd, Z = up):
              (x, y, z) → (x, z, y);  C = C⁻¹ = Cᵀ, det −1
G_src[j](t) = source global rotation (from global_rot_mats, or fk() of local_rot_mats)
G_ue[j](t)  = C · G_src[j](t) · C          # change of basis — NOT C·Gᵀ·C, NOT a swizzle
A[b]        = alignment: Manny's bind global, re-posed into the source rest pose.
              A[b] = swing(dir_manny_bind[b] → C·dir_src_rest[j]) · Manny_bind_global[b]
G_manny[b]  = G_ue[j](t) · G_ue_rest[j]⁻¹ · A[b]
local[b]    = G_manny[parent_full(b)]⁻¹ · G_manny[b]   over pose_solver.FULL_CHAIN
```

For SOMA the rest-pose globals are identity (identity local rotations *are*
the rest pose), so `G_ue_rest` drops out. The alignment `A` is what removes
the T-pose vs A-pose difference. It is computed **once** at start-up from the
two rest poses, not per frame and not hand-tuned. Its swing leaves bone roll
at the reference pose to the minimal rotation. The acceptance tests below
measure whether that is good enough, rather than assuming it.

Details:
- **Unstreamed Manny bones** (`spine_03`, `spine_05`, `neck_02`) either get
  their own source joint (SOMA: `Spine2 → spine_03`, `Neck2 → neck_02`) or
  stay at bind relative to their parent, exactly as `pose_solver` models
  them.
- **Core27 spine:** 4 joints for Manny's 5. Map `Hips/Spine/Spine1/Spine3 →
  pelvis/spine_01/spine_02/spine_04` and let the rest follow by FK. Core27 has
  one neck joint: split it with the same 40/60 idea as `NECK_SHARE`.
- **Hands:** use the source hand's full global rotation, so wrist roll comes
  through. Pass half the wrist twist back to `lowerarm_*` with
  `FOREARM_TWIST_SHARE`, the same reasoning as the solver.
- **Fingers:** Kimodo's static relaxed pose can be retargeted like any other
  bone (SOMA `Index1..4 → index_metacarpal, index_01..03`, etc.); Core27 has
  none. Default to Manny's finger bind pose, with the option to keep the live
  webcam hands (see 4.4).
- **Root / pelvis position:** source root (metres) × 100 → cm, through `C`,
  into the pelvis `BoneTransform.position`. This is what gives the character
  real locomotion; the webcam lane cannot. `motion_player` re-centres each
  clip on where the previous one ended (position and heading), and can clamp
  or wrap to a stage area.
- **Proportions:** SOMA and Manny limb ratios differ. Rotations transfer
  proportion-free; the IK Retargeter downstream already fixes Manny →
  MetaHuman proportions. Foot contacts (`foot_contacts`) are available if
  foot sliding on Manny needs a ground lock later.
- **Head for the MetaHuman:** `head_rotation = G_manny[head] · rest_global[head]⁻¹`,
  the same quantity `PoseSolver.head_rotation` holds. So
  `head_rotation_to_curves()` applies unchanged, including the measured
  absolute-component-space behaviour.

### 4.2 The player (`procedural/motion_player.py`)

- **Clock:** sends at 60 Hz (or the conductor's rate). It slerps between
  source frames, so 20 fps (ARDY-Core) and 30 fps (Kimodo) both look smooth.
- **Clip queue for Kimodo:** play → crossfade (per-bone slerp over ~0.3 s) →
  next clip; an idle loop when the queue is empty. For seamless chaining, the
  next generation request carries a **full-body keyframe constraint at frame
  0 = the current clip's last frame**. Kimodo is conditioned on it, so there
  is no pop.
- **Stream buffer for ARDY:** a small jitter buffer (2–3 source frames). Late
  frames are held; the buffer never grows.
- **Face packet:** every frame, with the head curves from 4.1. Blendshapes are
  neutral (or a procedural blink) unless the live face is mixed in.
- **Packet invariants kept:** always 421 floats; `present=1.0` while playing.

### 4.3 The services (Linux side)

- **`kimodo_service.py`:** start from `pipeline-network-osc/kimodo_service_handoff_osc_v3.py`
  and **delete the OSC streaming and all quaternion swizzling**. `/generate`
  returns the clip itself as NPZ bytes:
  - `global_rot_mats`, `root_positions`, `foot_contacts`
  - `fps`, `bone_order_names`
  - the prompt/seed metadata

  Add optional `first_frame` (for the chaining constraint) and `root_path`.
  Keep the lock, warm model, HF cache and timing log.
- **`ardy_service.py`:** a headless version of the demo's `_generate_step`
  loop. Specifically:
  - Keep a session with history.
  - On `POST /prompt` (or `/waypoints`, `/velocity`), re-encode the text
    through the text-encoder server and trigger a re-plan.
  - Push every decoded frame (global rotations + root) over UDP or ZMQ to
    the player, tagged with the frame index and fps.
  - Use Horizon8 for responsiveness, and try Horizon40 for quality. Both are
    one flag.

### 4.4 Living alongside the webcam lane — zero core changes

Three levels, cheapest first:

1. **Exclusive.** Run `conductor.py` *or* `procedural_conductor.py`; both send
   to 9001/11111. Nothing changes anywhere.
2. **Switch in Unreal.** The plugin's UDP port is already settable per source
   in the Live Link panel. Add a second `MediaPipeLiveLink` source on 9002 for
   the generator and toggle which `MediaPipePose` subject is enabled.
   (To verify: Live Link allows one enabled subject per name across sources.
   If it doesn't, this needs the subject name to become a source setting, a
   small plugin change.)
3. **Mix in Python (recommended for a performance).** `procedural/mixer.py`
   listens where the conductor sends. The conductor's pose and face target
   IP:port are already restart-required settings in the GUI and constructor,
   so pointing them at the mixer (e.g. 9101 / 11112) changes no code. The
   mixer:
   - parses the conductor's 421-float packets and face packets
   - blends them per bone with the generated pose (weights by body region:
     e.g. generated legs + root, live arms/hands, live face blendshapes)
   - forwards the result to 9001 / 11111

   This gives "AI body, webcam face and hands", or a crossfade from live to
   generated when the performer steps out of frame, without touching
   `conductor.py`.

### 4.5 What `mirroring` gains, file by file

| File | Change |
|---|---|
| `pose_solver.py`, `conductor.py`, `gui.py`, both protocol modules | **none** (read-only imports of tables, `BoneTransform` and encoders) |
| `ue_plugin` / `K:\KaiTracking\Plugins\MediaPipeLiveLink` | none (level 2 at most needs a subject-name setting, and only if the check in 4.4 fails) |
| Unreal assets | none — same Manny AnimBP, same retarget, same `ABP_Face` |
| new `procedural/` package | `source_skeletons.py`, `retarget.py`, `bvh_reader.py`, `motion_player.py`, `mixer.py`, `procedural_conductor.py` |
| new `services/` (Linux) | `kimodo_service.py`, `ardy_service.py`, `requirements-*.txt` |
| new `tests/procedural_checks.py` | see §5 |
| optional, later | a "Procedural" tab in `gui.py` (prompt box, queue, mix weights), which touches the GUI only |

---

## 5. Acceptance tests (offline, no GPU, no engine)

Same philosophy as `tests/solver_checks.py`: **ground truth independent of the
retarget's own tables**.

1. **Rest identity:** source rest pose in → Manny bind out. 22/22 body bones
   within a tight tolerance, for SOMA77 and Core27.
2. **Direction truth on real clips:** for every mapped bone, Manny-FK bone
   direction vs the source's own `posed_joints` / BVH FK direction for the
   anatomically matching joint pair, in absolute terms (not relative to the
   spine). Run on the existing `kimodo-gen/*.bvh` (80+ clips: walks, crawls,
   backflips, dances) and on ARDY-Core `.npz` samples once generated.
3. **Change-of-basis sanity:** a source character walking forward (+Z) moves
   Manny along +Y with the pelvis facing +Y; raising the source's left arm
   raises Manny's `upperarm_l`. This is the check that catches a
   transpose/inverse or a left/right swap.
4. **Head channel:** `head_rotation_to_curves()` output round-trips to the
   retargeted head on a clip with the torso turned (same as 2e).
5. **Player:** fixed packet length, slerp continuity at clip boundaries (no
   per-bone jump > N° between consecutive sent frames), re-centring keeps the
   pelvis inside the stage box.

Then the Unreal check, stage by stage, the way `tests/ue_head_probe.py` does
it: stream a handful of known poses and read the MetaHuman bones back.

---

## 6. Plan

**Phase 0 — offline retarget on existing clips (Windows only, ~no risk)** — *done*
1. `bvh_reader.py` + `source_skeletons.py` (SOMA77 from the BVH hierarchy, which
   is the same layout Kimodo exports).
2. `retarget.py` per §4.1; tests 1–3 green on `kimodo-gen/*.bvh`.
3. `procedural_conductor.py --bvh clip.bvh` plays a clip into Unreal through the
   existing encoders. **This is the first moment it is visible in the engine,
   and it needs no Kimodo install at all.**

**Phase 1 — Kimodo live-on-demand** — *working (2026-10-07); crossfade, chaining and
idle loop deferred*
4. Slim `kimodo_service.py` (returns NPZ, no OSC). Player queue, crossfade,
   chaining constraint, idle loop.
5. Head curves to the face channel; decide the blendshape default. (Done: head
   through `head_rotation_to_curves()`, blendshapes neutral.)

**Phase 2 — ARDY-Core streaming**
6. Get HF access to Llama-3-8B-Instruct; install ARDY in WSL2 (CMake build);
   confirm the demo runs.
7. `ardy_service.py` from the demo loop; Core27 table; 20 → 60 Hz slerp;
   prompt/waypoint endpoints. Measure end-to-end latency (prompt → first
   changed frame on the MetaHuman).

**Phase 3 — mixing with the webcam lane**
8. `mixer.py` proxy (§4.4 level 3), region weights, live-to-generated
   crossfade.

**Phase 4 — ARDY-SOMA when released**
9. Swap the source to the SOMA table already written for Kimodo; re-run §5.

**Hardware note.** The two text encoders (~14–17 GB each on GPU) do not fit
alongside each other and MediaPipe on one 24 GB 4090. Options: run only one
generator at a time, put the text encoder on the CPU, or run the generators on
the DGX Spark (the `HF_HOME=/opt/huggingface_cache` script already targets it)
and stream over the LAN. Only the service's target IP changes.

---

## 7. Open questions

- Does the swing-only alignment `A` leave visible bone-roll offsets on arms
  or hands at rest? Test 1 decides; if it does, derive the roll from the
  source hand/palm frame, as the solver does.
- Live Link behaviour with two sources publishing the same subject name
  (§4.4, level 2).
- Kimodo step count vs quality vs latency: 100 steps was the default here.
  Fewer steps may get on-demand clips under 1 s.
- ARDY-Core's own rest pose. `cskel27/joints.p` gives the joint positions,
  but whether identity local rotations mean a T-pose needs one FK check
  before the alignment is trusted.
- ARDY's own jitter (paper limitation) vs this pipeline's smoothing: reuse
  the conductor's pose alpha in the player, or not at all.

Sources: [Kimodo repo](https://github.com/nv-tlabs/kimodo) ·
[Kimodo docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs) ·
[ARDY repo](https://github.com/nv-tlabs/ardy) ·
[ARDY project page](https://research.nvidia.com/labs/sil/projects/ardy/) ·
[ARDY paper](https://arxiv.org/html/2607.08741v1) ·
[ARDY-Core-RP-20FPS-Horizon8 model card](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon8) ·
[ARDY-Core-RP-20FPS-Horizon40 model card](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40)
