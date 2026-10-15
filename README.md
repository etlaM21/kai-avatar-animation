# kai-avatar-animation

Drives a **MetaHuman in Unreal Engine 5.8** from two sources:

- **Live:** one ordinary webcam. MediaPipe tracks the body, fingers and face; our solver
  turns that into bone rotations for Epic's mannequin "Manny", plus the face's ARKit
  blendshapes and the head rotation. Since 2026-10-15 the character also **follows you
  around the room** (root translation).
- **Generated:** type a prompt; NVIDIA Kimodo generates the motion on a DGX Spark; the
  clip is cached and retargeted on Windows and played through the very same channels.

Both arrive in Unreal the same way: the body as OSC on UDP 9001 (our `MediaPipeLiveLink`
plugin), the face as Epic's Live Link Face packet on UDP 11111. Unreal moves Manny, and
an IK retarget carries Manny onto the MetaHuman.

```
webcam ─► MediaPipe ─► pose_solver ─┐                         ┌─► Manny ─► retarget ─► MetaHuman Body
                                    ├─► OSC 9001 / UDP 11111 ─┤
prompt ─► Kimodo (Spark) ─► retarget┘                         └─► face curves ─────────► MetaHuman Face
```

## Status (2026-10-15)

| | |
|---|---|
| Live lane (webcam) | works: body, fingers, face, head, **root translation** (new) |
| Generated lane (Kimodo) | works: prompts, cache, queue, loop; plain playback (hard cuts) |
| GUI | `mirroring\gui.py`, one tab per lane; the tab decides who sends |
| MetaHuman in Unreal | works **in Play mode (PIE)** with the setup below - not yet in the plain editor viewport |

## Run it

One venv for everything, at this folder (Python 3.12, no torch):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then, from `mirroring\` (the MediaPipe models are looked up there):

```powershell
cd mirroring
..\venv\Scripts\python.exe gui.py                           # the GUI, both lanes
..\venv\Scripts\python.exe conductor.py --debug --camera 1  # webcam lane only, no GUI
```

In the debug window `c` calibrates (stand upright and square to the camera), `Esc` quits.
`--no-root-translation` keeps the character on the spot. The generated lane also needs the
service on the Spark and an SSH tunnel: `remote_kimodo_service\README.md`.

Offline checks (no camera, Unreal or Spark needed):

```powershell
.\venv\Scripts\python.exe procedural_animation\tests\procedural_checks.py
cd mirroring
..\venv\Scripts\python.exe tests\solver_checks.py
..\venv\Scripts\python.exe tests\gui_checks.py
```

## Unreal: the setup that works, and why only in Play

**Use the MetaHuman like this** (`BP_NewMetaHumanCharacter`, project `KaiTracking`):

1. On the placed actor: **Use Live Link off, Use ARKit off, Live Link Subject = None.**
2. Leave the Blueprint's components on their own AnimBPs:
   `MannySource` → `SKM_Manny_Simple_LiveLink` (reads `MediaPipePose`),
   `Body` → `MH_LiveLink_Retarget-Manny` (our AnimBP: retargets Manny onto the MetaHuman
   and applies the face, subject `PythonConductor_Face`), `Face` → `ABP_Face`.
3. Press **Play**. Body and face are tracked, with the MetaHuman's own proportions.

**Why Play?** In the editor viewport (without Play), Unreal only animates a skeletal mesh
whose *Update Animation in Editor* flag is on. The MetaHuman Blueprint switches that flag
together with *Use Live Link* - and resets it whenever its construction script re-runs, so
ticking it by hand does not stick. Play (and the Blueprint's own preview window) always
animate, which is why they always looked right. Quick fix for the editor viewport: set the
flag from Python on the four meshes (`TODO.md` §9 has the snippet); it holds until the
actor is changed again. Permanent fix: one node in the Blueprint's construction script
(open, `TODO.md` §9).

**Do not turn on "Use Live Link" on the actor.** That is what made the head sink into the
shoulders ("a football player in full gear"). It swaps the Body to Epic's generic
`ABP_MH_LiveLink`, which applies the incoming skeleton bone by bone, *bone lengths
included*. It is made for sources that already are MetaHumans; ours is Manny. Measured:

| | MetaHuman's own | With Use Live Link (= Manny's) |
|---|---|---|
| neck base (`neck_01`) above the chest | 17.68 cm | 11.89 cm - 5.8 cm too low |
| clavicles from the spine | 11.99 cm | 5.85 cm - shoulders pulled in |

That is why it showed on webcam and Kimodo alike, even with the face receiver off, and
never in the Blueprint preview: it is a setting of the placed actor, not of the motion.

## What we found out (2026-10-15)

- **Root translation.** MediaPipe's 3D landmarks are centred on the hips, so they say how
  you are bent, never where you stand. Lining them up with the same landmarks in the
  picture (`cv2.solvePnP`, the method of another project, `UnityMediaPipePose_Findings.md`)
  gives your position: standing still it wanders 0.2 cm sideways and 1.2 cm in depth.
  Depth depends on the camera's field of view, which is a setting in the GUI (80° by
  default); an ML estimate of it (MLFocalLengths) was assessed and is not accurate enough
  on its own (`TODO.md` §8).
- **The sinking head is an Unreal setup issue, not tracking.** The solver's output puts
  the head exactly where Manny's own skeleton has it (0.07 cm); the error appeared only
  after "Use Live Link" (above). Found with `mirroring\tests\ue_head_probe.py --inspect`,
  which compares every bone's length in Unreal with the mesh's own - a check the older
  rotation-only measurement could not do.
- **The face only reaches a MetaHuman through an ARKit mapping** (`PA_MetaHuman_ARKit_Mapping`).
  `ABP_Face` has no Live Link; the face nodes now live in our Body AnimBP and `ABP_Face`
  copies the curves.
- **Dollars MONO** (a commercial webcam mocap app) was analysed for its solver ideas:
  `REA_FINDINGS.md`, summarised in CLAUDE.md. Proposals from it await a decision.

## Where to read more

| File | What |
|---|---|
| `mirroring\README.md` | Both lanes in full detail; §6 the MetaHuman setup, §9 the generated lane |
| `CLAUDE.md` | The working guide: rules that must not be broken, what is load-bearing, what is open |
| `TODO.md` | The open steps, with the exact commands |
| `remote_kimodo_service\README.md` | Running the Kimodo service on the Spark and the tunnel |
| `docs\` | Research: generated motion (`procedural-animation.md`), the network to the Spark |
| `UnityMediaPipePose_Findings.md`, `REA_FINDINGS.md` | What other projects do (root translation; Dollars MONO) |
| `_old\README.md` | What was archived and what replaced it |
