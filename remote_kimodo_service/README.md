# Remote Kimodo — prompt in the terminal, motion in Unreal

Type a prompt on the laptop; Kimodo generates it on the DGX Spark; the clip comes back
as an NPZ, is cached on disk, retargeted onto Manny with the existing tested retarget,
and played at 60 Hz into Unreal through the same OSC 9001 / UDP 11111 channels the
webcam lane uses.

**Status:** working end to end since 2026-10-07 (drives the MetaHuman). This is the
short operator's guide; every stage is explained in detail in
`mirroring/README.md` §9.

| Measured on the GB10, 9 s clip | |
|---|---|
| generation, 100 / 50 / 25 / 10 steps | 5.8 s (7.3 s first request) / 2.9 s / 1.5 s / 0.7 s |
| transfer of the ~890 KB NPZ, through the tunnel | 0.3–1.3 s |
| cache hit (no network) | ~0.07 s |

```
 Spark (spark-001)                      Windows laptop                                  Unreal
┌──────────────────────┐  HTTP / NPZ  ┌────────────────────────────────────────────┐  UDP 9001  ┌────────┐
│ kimodo_service.py    │ ◄──────────► │ procedural_conductor (REPL / --prompt)     │  ────────► │ Manny  │
│  warm model, lock    │  SSH tunnel  │  → kimodo_client → clip_cache (disk)       │  UDP 11111 │  → MH  │
│  127.0.0.1:8765      │              │  → kimodo_adapter → Retargeter → player    │  ────────► │        │
└──────────────────────┘              └────────────────────────────────────────────┘            └────────┘
```

## Files

| File | Runs on | What |
|---|---|---|
| `kimodo_service.py` | Spark | FastAPI service: `POST /generate`, `GET /health` |
| `requirements.txt` | Spark | its packages |
| `kimodo_contract.py` | both | the NPZ format, the request and its cache key (numpy only) |
| `kimodo_adapter.py` | Windows | NPZ → `Retargeter.retarget_globals()`, with an FK self-check |
| `kimodo_client.py` | Windows | HTTP client, one-line errors only |
| `clip_cache.py` | Windows | `cache/`, every returned clip (gitignored) |
| `player.py` | Windows | 60 Hz loop player on the existing encoders |
| `fake_kimodo_server.py` | Windows | stand-in service for tests and offline work |

The CLI is `procedural_animation/procedural_conductor.py`.

## 1. On the Spark: install once, start by hand

The Spark (hostname `kaspar`, Tailscale `100.83.6.8`) has a clone of this repo at
`~/project_kaspar/modules/kai-avatar-animation`; `git pull` there brings in new code.
The service has its own venv, `remote_kimodo_service/venv`; `requirements.txt` here is
the recipe to create it once (CUDA torch first, then Kimodo from the aarch64-patched
`~/kimodo-src`, then the pinned packages).

```bash
cd ~/project_kaspar/modules/kai-avatar-animation/remote_kimodo_service
venv/bin/python kimodo_service.py      # 127.0.0.1:8765; wait for "Model loaded"
```

Until that venv exists, the old OSC service's venv works the same (it already has
Kimodo, CUDA torch and FastAPI, and stayed at its old path when that folder moved to
`_old/`): `../pipeline-network-osc/venv/bin/python kimodo_service.py`.

The service runs in the foreground of your SSH session and **stops when you log out**,
on purpose. `Ctrl+C` stops it too. Loading the model takes ~25 s; until then the port
does not answer. Timings go to `timing_log.txt` next to the script.

## 2. On the laptop: the tunnel

The service only listens on the Spark's own `127.0.0.1`, so nothing on the Spark's
network can reach it. The SSH tunnel is how the laptop gets in anyway: it rides on the
SSH login you already have, and makes the Spark's port 8765 appear on the laptop as
`127.0.0.1:8765`. Open a second PowerShell window and leave it running:

```powershell
ssh -N -L 8765:127.0.0.1:8765 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 etlam@100.83.6.8
```

After the password it prints nothing: that is the working state (`-N` = forward only,
no shell). Leave the window open; if it drops, run it again. Check the whole chain:

```powershell
curl.exe -s http://127.0.0.1:8765/health
```

(Without the tunnel: start the service with `--host 100.83.6.8` and pass
`--spark-url http://100.83.6.8:8765` to the conductor. That only works if the Spark's
tailnet allows that port; the tunnel always works where SSH works.)

## 3. Play

In the GUI - from `mirroring\`, `..\venv\Scripts\python.exe gui.py`, then the **Prompt**
tab (it stops the webcam lane itself; queue, history, cache library, Spark indicator,
generation time per clip). Or from the command line, from the repo root
(`kai-avatar-animation`):

```powershell
# REPL: type prompts; the current clip loops while the next one generates
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor

# one shot: generate (or take from the cache), play once, exit
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --prompt "A person waves both arms"

# offline
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --bvh assets\kimodo\clips\kimodo-gen\wave.bvh --loop
.\venv\Scripts\python.exe -m procedural_animation.procedural_conductor --clip remote_kimodo_service\cache\<file>.npz --loop
```

Stop `conductor.py` first: both send to the same ports.

Inline options, anywhere in a prompt:

| | |
|---|---|
| `/s 25` | 25 denoising steps (default 33; fewer = faster, rougher) |
| `/seed 7`, `/seed random` | seed (default 0, so the same prompt is an instant cache hit) |
| `/t 4` | 4 seconds (default 9, max 10) |

REPL commands: `/list`, `/play N` (a cached clip, offline), `/next` (cut now instead of
at the end of the pass), `/status`, `/health`, `/help`, `/quit`.

**Playback v1 is deliberately plain.** The current clip loops; a new clip takes over
when the current pass ends, as a hard cut. Clips play exactly as generated: a new one
may start somewhere else and facing elsewhere. Re-centring, crossfade and an idle loop
come later.

## When something is off

| Message | Meaning |
|---|---|
| `connection refused at http://127.0.0.1:8765` | the tunnel is not running |
| `accepted and closed the connection` | tunnel up, service not running (or still loading the model) |
| `no answer ... within 60 s` | the Spark is busy or the link stalled; a retry of the same prompt is served from the service's memory if the generation finished meanwhile |
| `rotations don't reproduce posed_joints` | the NPZ does not mean what the retarget assumes (Kimodo version change?) - the clip is refused, never cached |

In every case the player keeps playing what it has, and cached clips stay available.

## Tests

```powershell
.\venv\Scripts\python.exe procedural_animation\tests\procedural_checks.py
```

Tests 5b–7 cover this folder without a Spark: the loop player, client resilience
against stub servers (refused, accept-and-close, timeout, HTTP 500, malformed NPZ,
cache hit without network) and the NPZ contract, including a real Kimodo clip from
the Spark (`procedural_animation/tests/fixtures/kimodo_real_turn_around_2s.npz`).
`fake_kimodo_server.py` can also stand in for the Spark for offline REPL work:

```powershell
.\venv\Scripts\python.exe -m remote_kimodo_service.fake_kimodo_server --delay 3
```
