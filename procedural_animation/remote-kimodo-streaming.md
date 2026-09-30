# Getting Kimodo motion from the Spark into Unreal

How to get clips generated on the DGX Spark (`spark-001`, Tailscale `100.83.6.8`)
into `procedural_conductor` / `conductor.py` / the GUI / Unreal. Written
2026-09-29, while the Spark is still at the university and is about to move to a
colleague's home.

**Short answer:**
- **Windows asks, the Spark answers.** A small HTTP service on the Spark takes a
  prompt and returns the **whole clip as an NPZ**. Windows starts every
  connection, so nothing ever has to reach *into* the laptop.
- **Windows buffers and plays.** Windows caches the clip, retargets it with the
  Phase 0 code, and plays it locally at 60 Hz. The network never touches the
  real-time path: Unreal only ever receives packets from `127.0.0.1`, exactly as
  it does today.
- **Transport:** plain Tailscale HTTP to `100.83.6.8`. If the tailnet ACL blocks
  the port, fall back to an SSH port forward over the Tailscale SSH you already
  use.
- **Do not stream per-frame poses from the Spark to Unreal.** The old OSC-service
  approach is ruled out by what was measured below.

---

## 1. What the link looks like today (measured)

From `malte-laptop`, 2026-09-29:

| | |
|---|---|
| `tailscale ping 100.83.6.8` | `via DERP(fra)`, 43 / 47 / 47 / 49 / 68 ms. **Direct connection not established** |
| `tailscale netcheck` | UDP works, but **`MappingVariesByDestIP: true`** (hard NAT on the laptop's side), nearest relay Frankfurt |
| Spark node | shared into your tailnet from another account (`whatphilipcodes@`) |

What this means:

- **Every packet goes through Tailscale's Frankfurt relay.** The round trip is
  about 45 ms with spikes. Relays are also rate-limited, so bulk transfers are
  slower than the raw connection would allow.
- **Relayed traffic travels over TCP, even when the app sends UDP.** A lost packet
  is retransmitted, and everything behind it waits. For a live pose stream, that
  shows up as a **stall followed by a burst**, not a clean dropped frame. Exactly
  the wrong failure for animation.
- **The Spark is shared from another account.** That account's tailnet
  controls which ports you can reach. SSH works. Whether an arbitrary port like
  `:8765` works still has to be tested (§5).
- **The move may help.** At the colleague's home, residential NAT may allow a
  direct connection: lower latency, no relay limits. Or it may not; the laptop's
  hard NAT alone can prevent it. **Design for the relay; treat direct as a bonus.**

## 2. What has to cross the network

| Data | Size | Notes |
|---|---|---|
| Request: prompt, seed, frame count, optional `first_frame` for chaining | < 1 KB (a chaining frame adds ~3 KB) | JSON |
| Response: one Kimodo clip, 270 frames (9 s at 30 fps) | `global_rot_mats` 270×77×3×3 float32 ≈ **750 KB** raw; less compressed | `np.savez_compressed` |
| Later, ARDY streaming | ~77×4 float32 ≈ 1.2 KB per frame at 20 Hz ≈ **25 KB/s** | bandwidth is trivial; jitter is the problem |

Even through a relay, a clip transfers in well under the time Kimodo takes to
generate it. Measure it once (§5) instead of trusting this estimate.

## 3. Options

### A. HTTP request → whole clip back as NPZ (recommended)

The Spark runs a trimmed `kimodo_service.py` (FastAPI, model kept warm,
`asyncio.Lock`, per plan §4.3), listening on the Tailscale address. Windows
sends `POST /generate` and receives an NPZ back.

The NPZ contains:
- `global_rot_mats` and `root_positions` (in **metres**)
- `foot_contacts`
- `fps` and `bone_order_names`
- the prompt and seed

The Phase 0 retarget takes these directly: `Retargeter.retarget_globals()`. Two
adapter details:
- `units_to_cm` must be 100 for this path.
- Kimodo's 77 joints are the BVH's joints minus the `Root` wrapper, in the same
  order. Verified in Phase 0 against Kimodo's own table.

| Pros | Cons |
|---|---|
| Survives the relay: TCP, one request per clip; retries are trivial | Latency to the first frame is generation time plus transfer. On-demand only, not reactive |
| The network is fully decoupled from playback. A slow link delays the *next* clip and never makes the current one stutter | Needs one open port on the Spark (or the SSH forward, option E) |
| Windows only makes outbound connections: no Windows firewall rule, keeps working when the laptop changes networks | Kimodo is whole-clip anyway, so nothing is lost; ARDY needs option D |
| All Manny knowledge stays on Windows, next to the tested retarget. The Spark sends plain Kimodo data | |
| Every clip can be cached on both sides and becomes the offline library | |
| Matches plan §4.3; `curl` can test it | |

### B. The Spark streams retargeted poses straight to Unreal (OSC/UDP), like `pipeline-network-osc`

| Pros | Cons |
|---|---|
| Nothing on Windows except Unreal | Over the relay this is UDP inside TCP: every lost packet becomes a stall and then a burst, **visible on the character** |
| | The Spark must reach *into* the laptop: inbound firewall rule, and a laptop address the Spark has to know |
| | The retarget moves to the Spark, away from the tests and the Manny tables. Exactly how the earlier attempts went wrong (plan §3) |
| | One network glitch = a frozen or jumping character in front of an audience |

**Not recommended.**

### C. File sync of BVH/NPZ (scp / rsync over Tailscale SSH, Taildrop, a shared folder)

| Pros | Cons |
|---|---|
| Works today, over the SSH you already have. Zero new code | Not interactive: generate, sync, then play |
| Very robust over a slow relay. The library grows by itself | Needs a polling or "sync now" step; no chaining constraint |
| `procedural_conductor --bvh` already plays the results | |

**Good as a stopgap and for building a library in bulk** (batch prompts
overnight, `rsync` in the morning). Not the live path.

### D. Persistent WebSocket / ZeroMQ stream, Windows connects out

For **ARDY** (Phase 2), which produces frames continuously.

| Pros | Cons |
|---|---|
| Windows-initiated, so no inbound firewall | Still TCP over the relay: loss stalls the stream. Needs a jitter buffer of 150–250 ms (not 2–3 frames as plan §4.2 assumes) while relayed |
| One connection carries prompts up and frames down | More code: reconnects, session state, back-pressure |
| Much better when a direct connection exists | Relay latency (45 ms + spikes) adds to every prompt-to-motion reaction |

**The right tool for ARDY later.** Overkill for Kimodo.

### E. SSH port forward over Tailscale SSH (a transport for A or D, not a protocol)

`ssh -N -L 8765:127.0.0.1:8765 <user>@100.83.6.8` makes the Spark's service appear
at `http://127.0.0.1:8765` on the laptop.

| Pros | Cons |
|---|---|
| Works whenever SSH works, so it sidesteps any port restriction in the other account's tailnet | One more process to keep alive; it dies silently when SSH drops (use `-o ServerAliveInterval=15` and a restart loop) |
| The service can stay bound to `127.0.0.1` on the Spark: nothing exposed at the university or at your colleague's home | Slightly more latency and overhead than direct Tailscale |
| The Windows code is identical either way (the URL is a setting) | |

## 4. Recommended setup

```
 Spark (spark-001)                          Windows laptop                                   Unreal
┌──────────────────────────┐  HTTP/NPZ   ┌──────────────────────────────────────────┐  localhost ┌────────┐
│ kimodo_service.py        │ ◄─────────► │ kimodo_client  → clip cache (disk)       │  UDP 9001  │ Manny  │
│  FastAPI, warm model,    │ over        │   → Retargeter (Phase 0, tested)         │  ────────► │  → MH  │
│  Lock, timing log        │ Tailscale   │   → motion_player: queue, crossfade,     │  UDP 11111 │        │
│  bound to 100.83.6.8     │ (or SSH -L) │     idle loop, 60 Hz clock               │  ────────► │        │
│  (or 127.0.0.1 + SSH -L) │             │  procedural_conductor / GUI tab / mixer  │            └────────┘
└──────────────────────────┘             └──────────────────────────────────────────┘
```

Rules that keep it working on a bad link:

1. **The network only ever feeds a queue.** The player's 60 Hz clock never waits
   on the network. While a request is in flight, the current clip finishes and
   then the idle loop plays. Never a frozen pose.
2. **Windows always initiates.** It works from the university, the colleague's
   home, the venue and hotel Wi-Fi without firewall work.
3. **Cache everything.** Write each returned clip to disk, keyed by prompt and
   seed. If the Spark is unreachable during a show, the player falls back to the
   cached library.
4. **Chaining is a request field** (`first_frame` = the last frame played). A
   frame is ~3 KB, and the request is tiny either way.
5. **Health endpoint and timing.** `GET /health` returns the model and whether it
   is loaded. Log generation time *and* transfer time per request, so a slow day
   tells you whether the GPU or the link is the problem.
6. **Security:** bind to the Tailscale IP or to `127.0.0.1` (with SSH forwarding),
   **never `0.0.0.0`**. The Spark sits on a university network now and on a
   private home network next.

How it fits the existing tools:

- **`conductor.py`:** unchanged. As in plan §4.4, either run
  `procedural_conductor` *instead of* it (level 1), or later put `mixer.py`
  between them (level 3).
- **GUI:** a later "Procedural" tab (prompt box, queue, connection status) calls
  the same client. `gui.py` is the only file touched.
- **Unreal:** nothing changes. It keeps receiving localhost UDP from Windows.

## 5. To check before building (5 minutes)

```powershell
# 1. Does the other account's tailnet let you reach a plain port on the Spark?
#    On the Spark:  python3 -m http.server 8765 --bind 100.83.6.8
curl.exe -s -o NUL -w "%{http_code} %{time_total}s\n" http://100.83.6.8:8765/
#    200 = option A works directly. Timeout = use the SSH forward (option E).

# 2. Transfer speed through the relay, with a clip-sized file.
#    On the Spark:  head -c 800000 /dev/urandom > /tmp/clip.bin   (in the served folder)
curl.exe -s -o NUL -w "%{size_download} bytes in %{time_total}s\n" http://100.83.6.8:8765/clip.bin

# 3. Path check. Repeat after the Spark moves.
tailscale ping --c 5 100.83.6.8      # "via DERP(fra)" = relayed; "via <ip>:<port>" = direct
tailscale netcheck
```

After the move, if `tailscale ping` still shows DERP, one fix helps: ask the
colleague to forward **UDP 41641** on their router to the Spark. Tailscale then
usually manages a direct connection, even with the laptop's hard NAT.

## 6. Open questions

- The Spark's Kimodo speed at 100 vs fewer denoising steps. `gen-time.txt`
  (2.5 s) was measured on a 4090; `pipeline_timing_log.txt` (3–11 s) came from a
  different machine and step count.
- Do the other tailnet's ACLs allow non-SSH ports on `spark-001` (check 1 above)?
- Does the laptop's hard NAT come from the university network or from Windows / a
  VPN? Rerun `tailscale netcheck` on home Wi-Fi. If `MappingVariesByDestIP`
  turns `false` there, direct connections become possible after the move.
- A fallback if the Spark is down on show day: the cached clip library (rule 3)
  and, if available, a local 4090 running Kimodo under WSL2 on the same
  `/generate` API.
