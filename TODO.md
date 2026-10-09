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
venv/bin/python -m pip install --no-build-isolation ~/kimodo-src
venv/bin/python -m pip install -r requirements.txt
```

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

Built and checked offline only (tests + a scripted run against the fake Kimodo server).

- [ ] Unreal open with the MetaHuman level; service running on the Spark (step 2).
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
  - Switch back to **Tracking** while a clip plays AND a prompt is still generating:
    the webcam resumes, the clip stops, and the prompt shows up in History as
    "not played - cached".
  - **Stay in place** on a walking prompt; **loop off** after the last clip.
- [ ] Note the generation time at **33 steps** (the new default) from the queue or
  history, and put it into the table in `mirroring\README.md` §9.12 ("not timed yet").

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
- [ ] **Record an occlusion clip** if the gating thresholds should be tuned harder (lean
  over the desk until the legs leave frame, step half out of shot, one arm behind the
  back):

```powershell
cd K:\Kaspar_Project\project_kaspar\modules\kai-avatar-animation\mirroring
..\venv\Scripts\python.exe conductor.py --debug --camera 1 --record recordings\occlusion.npz
```

- [ ] **Request Llama-3-8B-Instruct access on Hugging Face** before the ARDY phase.
- [ ] Optional: **the direct-port check** (only if you want to drop the SSH tunnel). On
  the Spark: `python3 -m http.server 8765 --bind 100.83.6.8`, then on Windows:

```powershell
curl.exe -s -o NUL -w "%{http_code} %{time_total}s\n" http://100.83.6.8:8765/
```
