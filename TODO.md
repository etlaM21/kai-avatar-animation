# To do (Malte)

Open steps after the GUI rebuild and the restructure (2026-10-09), in order, with the
exact commands. Tick them off here; delete a section once it is done.
Windows commands are PowerShell; Spark commands run in your SSH session.

## 1. Publish the branch (Windows)

The three commits (`ae35a1d` GUI, `ca955e6` restructure, `59a1247` docs) are only on the
local branch `feat/gui-extension`. The Spark tracks `main`, so it sees nothing until
they are on `main` on GitHub.

- [ ] Push the branch and merge it (on GitHub as a PR, or locally like this):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
git push -u origin feat/gui-extension
# then merge the PR on GitHub - or locally:
git checkout main
git merge feat/gui-extension
git push origin main
```

- [ ] Optional: this repo is a submodule of `project_kaspar`. To record the new commit
  there as well:

```powershell
cd K:\Kaspar_Project\project_kaspar
git add modules/kai-avatar-animation
git commit -m "chore: bump kai-avatar-animation (GUI tabs, restructure)"
git push
```

## 2. The Spark: pull, then its own venv

Checked read-only on 2026-10-09: branch `main` at `9e47bd2`, Python 3.12.3,
`python3 -m venv` works, the patched `~/kimodo-src` is there.

- [ ] Pull (after step 1):

```bash
ssh etlam@100.83.6.8
cd ~/project_kaspar/modules/kai-avatar-animation
git pull
```

  The old `pipeline-network-osc/` folder now lives in `_old/`. Its `venv/` stays at the
  old path, because git does not move untracked files. That is expected and ignored.

- [ ] Create the service's own venv (once, ~10 min, mostly torch):

```bash
cd ~/project_kaspar/modules/kai-avatar-animation/remote_kimodo_service
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip setuptools wheel
venv/bin/python -m pip install torch==2.13.0
venv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
#   must print:  13.0 True    - if it says None / False, stop: a CPU wheel came in
venv/bin/python -m pip install ~/wheels/scenepic-1.1.2-cp312-cp312-linux_aarch64.whl
venv/bin/python -m pip install --no-build-isolation ~/kimodo-src
venv/bin/python -m pip install -r requirements.txt
```

  The scenepic line comes before Kimodo: scenepic is a Kimodo dependency with no aarch64
  wheel on PyPI, and its source package cannot build without an npm step (CMake error
  `dist/scenepic.min.js: No such file`). `~/wheels` holds the one you built in August.
  If you already ran the steps without it: run just the last three lines, the venv and
  torch are fine.

- [ ] Start the service with it (from now on, every session):

```bash
cd ~/project_kaspar/modules/kai-avatar-animation/remote_kimodo_service
venv/bin/python kimodo_service.py
#   wait for "Model loaded"; it stops when you log out (on purpose)
```

  If the new venv misbehaves, the old one still works:
  `../pipeline-network-osc/venv/bin/python kimodo_service.py`

- [ ] Once the new venv has served a few sessions, delete the old one. After the pull,
  that folder should hold nothing but `venv/` - look first:

```bash
ls -a ~/project_kaspar/modules/kai-avatar-animation/pipeline-network-osc
#   expected:  .  ..  venv     - anything else: move it out first
rm -rf ~/project_kaspar/modules/kai-avatar-animation/pipeline-network-osc
```

## 3. Run the GUI for real (Windows + Spark + Unreal)

**Prompt tab done 2026-10-13** (recording `261013_Kai_Kimodo_v2.mp4`): real Spark,
tunnel and Unreal, no calibration before playback. The Spark dot green, fresh and cached
clips, queue, history, library, loop and stay in place all worked and drove the
MetaHuman. 33 steps timed (README §9.12). What it showed about the head: step 7.
Still open below: everything that involves the **Tracking tab** and the lane switch.

- [x] Unreal open with the MetaHuman level; service running on the Spark (step 2).
- [ ] Tunnel, in its own PowerShell window, left open (prints nothing after the
  password; that is the working state):

```powershell
ssh -N -L 8765:127.0.0.1:8765 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 etlam@100.83.6.8
```

- [ ] The GUI, in another window. It must run from `mirroring\`:

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation\mirroring
..\venv\Scripts\python.exe gui.py
```

- [ ] Check, by eye in Unreal:
  - Tracking tab → Start → move. Then switch to **Prompt**: the camera stops, and the
    bottom strip says who is sending.
  - **What do Manny and the MetaHuman do on the handoff** (last webcam pose with
    present=0, before the first clip arrives)? Hold the pose, go to rest, something
    else? Write it into CLAUDE.md (gui.py section); it decides whether the handoff
    should send the bind pose instead.
  - Prompt → a clip plays; the Spark dot is green; a generation time is shown.
    **Done 2026-10-13.**
  - Switch back to **Tracking** while a clip plays AND a prompt is still generating:
    the webcam resumes, the clip stops, and the prompt shows up in History as
    "not played - cached".
  - **Stay in place** on a walking prompt (done 2026-10-13); **loop off** after the
    last clip.
- [x] Generation time at **33 steps**: 2.0–2.1 s on the GPU + 0.8–1.6 s link
  (2026-10-13, read from the GUI), now in `mirroring\README.md` §9.12.

## 4. Windows cleanup

- [ ] Once the GUI and the live lane work with the root `venv\`, delete the old venv
  (its pip is broken, and it has both OpenCV builds installed over each other):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
Remove-Item -Recurse -Force .\mirroring\venv
```

  The root venv is already created. Should it ever need rebuilding:

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
Remove-Item -Recurse -Force .\venv
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

- [ ] After any change, the three offline test suites (no camera, Spark or Unreal):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
.\venv\Scripts\python.exe procedural_animation\tests\procedural_checks.py
cd mirroring
..\venv\Scripts\python.exe tests\solver_checks.py
..\venv\Scripts\python.exe tests\gui_checks.py
```

## 5. Still open from before (unchanged)

- [ ] **Test the solver in UE** if not already done: calibrate upright once, then turn
  and nod, rotate the wrists palm-up/down, spread and curl the fingers, crouch, lean out
  of frame. Confirm the state of branch `feature/solver-fixes` (merged or not).
- [ ] **Record an occlusion clip** (now also needed to test the Dollars leg gate, see 6).
  Lean over the desk until the legs leave frame, step half out of shot, one arm behind
  the back, walk up to the camera so the legs leave frame:

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation\mirroring
..\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\occlusion.npz
```

## 6. Decide on the Dollars MONO findings

Summary in CLAUDE.md, "Lessons from Dollars MONO"; details in `REA_FINDINGS.md` (S1-S9).
Nothing is implemented. Pick what Claude should design (design first, then code):

- [ ] **One Euro landmark filter** (first candidate for the wobbly head; lowest risk).
- [ ] **Neck: pitch-only share, ~0 yaw**, plus neck counter-rotation against the spine.
- [ ] **Curved spine** (~0.26 / 0.61 / 0.13 over spine_01/02/04) instead of the rigid torso.
- [ ] **Leg gate by left/right length ratio** (0.6 / 0.8, 2 updates) with whole-leg
  smoothstep recovery. Needs the occlusion recording above.
- [ ] **Continuous visibility weights, blend to rest** (with the 20° angle override)
  instead of freezing.
- [x] **Root translation** - built with PnP and verified live 2026-10-15 (the plugin does
  apply the pelvis position). What is left: §8.
- [ ] Procedural crossfade (when it is built): smoothstep over ~0.3 s, not linear.
- [ ] Optional, to settle the leg-recovery question for good: run Dollars again with
  the UDP listener while stepping a leg out of frame (REA_FINDINGS O7).
- [ ] `dollars_capture.jsonl` (packets from a third-party app) is committed in the
  module root (`7c8c558`). Decide whether it belongs in the repo; if not:
  `git rm --cached dollars_capture.jsonl` and add it to `.gitignore`.

- [ ] **Request Llama-3-8B-Instruct access on Hugging Face** before the ARDY phase.
- [ ] Optional: **the direct-port check** (only if you want to drop the SSH tunnel). On
  the Spark: `python3 -m http.server 8765 --bind 100.83.6.8`, then on Windows:

```powershell
curl.exe -s -o NUL -w "%{http_code} %{time_total}s\n" http://100.83.6.8:8765/
```

## 7. Generated-motion head and neck (from the 2026-10-13 run)

What the recording looked like: chin up while standing (jumping jacks, high five), the
face upright while the torso hunched (break dancing), the head looking back during the
sprint's turn, a fold in the collar at the neck. What was measured (offline, on the cached
clips that played, through the real adapter, retarget and `head_rotation_to_curves`,
against Kimodo's own head pitch taken from joint positions only):

- The face channel's head pitch equals Kimodo's within 0.1–0.5° while the body faces the
  camera. The chin-up (+19…+26° in "three jumping jacks", +8° in "high fiving") is in
  Kimodo's motion. Not the T-pose reference: quiet standing over 133 clips has a median
  head pitch of +5.5° per clip.
- The upright face while hunched is the head following the torso (head −13°, torso lean
  +14°). The look back in the turn is Kimodo's too (head −80° against the chest).
- The Unreal camera sits about at chest height (the horizon crosses the character's
  chest), so it looks up under the chin: a level head already reads as chin-up.
- Head yaw at ~85° (sprint) and through ±180° (circle run) looked right in Unreal, no
  flip at the wrap - by eye only.
- What is left: the collar fold, the MetaHuman Body/Face neck seam (README §10).

Proposed fixes, details in `mirroring\README.md` §10. None implemented:

- [ ] **B - judge from eye height** (no code). Put the Unreal camera at the MetaHuman's
  eye height, then replay the clip (close the GUI first: same ports):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --clip remote_kimodo_service\cache\a-person-doing-three-jumping-jacks__s166259168__270f__33st__bad767940d7c.npz --loop
```

- [ ] **A - regression check**: approve adding "sent head pitch = Kimodo's
  position-derived head pitch" to `procedural_checks.py` (Claude writes it).
- [ ] **C - head-pitch limit layer**: after B, decide whether you want it at all (off by
  default, like stay in place).
- [ ] **D - neck seam probe**: approve a `ue_head_probe.py` stage that replays a generated
  clip and reads the Body and Face `neck_01`/`neck_02`/`head` back (Unreal open, Python
  remote execution on). Decide on the seam once it has numbers.
- [ ] The head yaw stage at 90/135/180° (CLAUDE.md open item): the run suggests Unreal
  stays linear; the probe turns that into a number.

## 8. Root translation (live lane)

**Status 2026-10-15: v1 works.** Built on `feat/root-translation` on top of `37ccf57`,
run live with the camera and Unreal by Malte: the character follows you around the room.
How it works: `pose_solver.py`, section "root translation" (CLAUDE.md, Open items).
Offline checks 2g/2h in `solver_checks.py`, all pass.

- [ ] **Publish the branch**: the code is committed (`1f3c6a6`); commit these doc updates,
  then push and merge (as a PR on GitHub, or locally):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation
git add CLAUDE.md TODO.md
git commit -m "docs: root translation status and open steps"
git push -u origin feat/root-translation
```

- [ ] **Glide back to the centre when the hips are lost** instead of freezing (`TODO(root
  glide)` in `pose_solver._root_offset_cm`; Dollars S5/S9: gain = hip visibility weight,
  0.25 s hold + 0.25 s fade).
- [ ] **Jump mode**: ground lock stays, PnP height only while airborne. Needs a recording
  first (a few jumps on the spot, one step forward and back):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation\mirroring
..\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\root_jump.npz
```

- [ ] **Watch for the off-axis turn**: when you stand well to the side, does the character
  look turned although you face the camera? (UnityMediaPipePose_Findings.md §7.7.) If so,
  it needs a recording and a fix in the solver.
- [ ] **One Euro filter** on the root, only if it looks jittery live (standing still it
  wanders 0.2 cm sideways and 1.2 cm in depth on `rest.npz`).
- [ ] **Real focal length** (the FOV field is a guess, 80° by default, and depth scales
  directly with it). Pick one when depth matters:
  - by eye: adjust the FOV in the GUI until walking a known distance looks right (works now);
  - the webcam's spec-sheet FOV;
  - a tape-grid recording: tape marks at 2.0 / 2.5 / 3.0 m straight ahead and ±0.75 m at
    2.5 m, ~3 s standing on each, then arms up and a crouch
    (`--record recordings\root_grid.npz`). Gives f, the lateral scale and the pose bias,
    and is the ground truth for any other method;
  - a checkerboard calibration (`cv2.calibrateCamera`), per camera and resolution mode;
  - an ML estimate, below.

### Option: MLFocalLengths (ML focal length from one image)

[nandometzger/MLFocalLengths](https://github.com/nandometzger/MLFocalLengths) (MIT, 2023):
an EfficientNet-B4 that predicts a photo's focal length from the picture alone, no EXIF.
Trained on ~15k personal photographs with EXIF labels; outputs the **35 mm-equivalent**
focal length; reported error **16 mm mean absolute** (35 mm-equivalent, on the author's own
held-out photos). PyTorch, 233 MB weights downloaded on first run; input centre-cropped to
256×256. Read from the README on 2026-10-15, not tried.

How it would plug in: run it once per camera and resolution mode on a saved webcam frame,
convert, type the FOV into the GUI. The 35 mm-equivalent value converts to pixels without
any crop factor (35 mm equivalence is defined over the 43.27 mm diagonal):
`f_px = f35 × image_diagonal_px / 43.27`, then `FOV_h = 2·atan(W / (2·f_px))`.

| Pros | Cons |
|---|---|
| No checkerboard, no tape, no measuring: one frame of the room is enough | **Too inaccurate for depth on its own**: a webcam is around 25-30 mm equivalent, so 16 mm average error is ±50% or more on f - and depth scales 1:1 with f. The FOV guess we have is already in that range |
| Runs locally and offline, MIT licence | Needs **PyTorch**, which the module venv must never get (CLAUDE.md): it would need its own small venv (or the Spark), run as a one-off tool, never inside the live lane |
| Can average over many frames / several rooms to cut random error | Averaging does not remove a **bias**: trained on personal photos (camera lenses, EXIF), not on webcam frames (wide, low-res, often distorted, cropped differently per mode) - accuracy on webcams is unknown |
| Per camera and resolution mode, automatically | The square centre crop throws away the sides of a 16:9 frame, the part that carries most of the wide-angle cue |
| Quick to try (`python predict.py <folder>`) | Small project (46 stars, no releases); weights come from a Google Drive link |
| | Validating it needs ground truth anyway (tape grid or checkerboard) - and with that in hand, the ML step is not needed |

**Verdict:** fine as a cheap experiment, not as the source of truth. If tried, compare its
FOV against the tape-grid recording before trusting it. Worth comparing at the same time
(not checked here): newer single-image calibration models such as GeoCalib (ECCV 2024),
which also estimate the camera's pitch and roll - the pitch would feed the deferred
camera levelling as well.
