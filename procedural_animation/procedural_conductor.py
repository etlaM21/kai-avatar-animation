"""Play generated motion into Unreal through the existing mirroring encoders.

    .\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor
        REPL: type a prompt, it is generated on the Spark in the background while the
        current clip keeps looping; /help lists the commands.
    .\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --prompt "A person waves /s 25"
        one shot: generate (or take from the cache), play once, exit. --loop keeps looping.
    .\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --bvh clip.bvh [--loop]
    .\\venv\\Scripts\\python.exe -m procedural_animation.procedural_conductor --clip cached.npz [--loop]
        offline: an existing Kimodo BVH (either convention) or a cached NPZ.

Inline prompt options, anywhere in the prompt: `/s N` denoising steps (default 33),
`/seed N` or `/seed random`, `/t N` seconds (max 10). Seeds default to a fixed value, so
the same prompt is a cache hit; `/seed random` or `--seed random` gives variety.

Sends exactly what conductor.py sends, to the same default ports: 60 BoneTransforms
(421 floats with the present flag) over OSC to 9001, and a Live Link Face packet to
11111 carrying the head rotation (neutral blendshapes). Run it INSTEAD of conductor.py,
not alongside it. Clips play exactly as retargeted: no re-centring, no crossfade -
a new clip may start elsewhere, facing elsewhere (v1, see remote_kimodo_service/player.py).

Settings with defaults, overridable by flag or environment: --spark-url (KIMODO_URL,
http://127.0.0.1:8765 = through the SSH tunnel), --timeout (KIMODO_TIMEOUT, 60 s),
--cache-dir (KIMODO_CACHE_DIR, remote_kimodo_service/cache).

The frozen Phase 0 BVH-only version of this script is procedural_conductor_bvh.py.
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import MIRRORING_DIR  # noqa: F401  (puts mirroring on sys.path)
from .bvh_reader import read_bvh
from .retarget import STREAMED_BONES, Retargeter, RetargetedMotion
from .source_skeletons import load_soma77, to_standard_convention

from remote_kimodo_service import kimodo_contract as contract
from remote_kimodo_service.clip_cache import DEFAULT_CACHE_DIR, ClipCache
from remote_kimodo_service.kimodo_adapter import retarget as retarget_npz
from remote_kimodo_service.kimodo_client import (
    DEFAULT_TIMEOUT_S, DEFAULT_URL, GenerationFailed, GenerationResult, KimodoClient, SparkUnavailable,
)
from remote_kimodo_service.kimodo_contract import ContractError, GenerationRequest
from remote_kimodo_service.player import LIVE_LINK_FACE_PORT, POSE_OSC_PORT, LoopPlayer, Sender

DEFAULT_SEED = 0
REPL_HELP = """commands:
  <prompt> [/s steps] [/seed N|random] [/t seconds]   generate (or take from cache) and queue
  /list            cached clips          /play N     queue cached clip N (offline)
  /next            cut to the queued clip now instead of at the end of the pass
  /health          ask the Spark service /status   what is playing and queued
  /help            this text             /quit       stop (Ctrl+C works too)"""

_OPTION = re.compile(r"(?:^|\s)/(s|seed|t)\s+(\S+)")
_STRAY = re.compile(r"(?:^|\s)(/\S+)")


def parse_seed(text: str | int) -> int:
    if isinstance(text, int):
        return text
    if str(text).lower() == "random":
        return random.randrange(2**31)
    return int(text)


def parse_prompt(line: str, seconds: float, steps: int, seed: int | str,
                 model: str) -> GenerationRequest:
    """'A person walks /s 25 /seed random' -> GenerationRequest. Raises ValueError with a
    readable message."""
    opts = {k: v for k, v in _OPTION.findall(line)}
    text = _OPTION.sub(" ", line)
    try:
        if "s" in opts:
            steps = int(opts["s"])
        if "t" in opts:
            seconds = float(opts["t"])
        seed = parse_seed(opts.get("seed", seed))
    except ValueError:
        raise ValueError(f"bad option value in {line!r} (expected /s N, /t N, /seed N|random)") from None
    stray = _STRAY.findall(text)       # standalone /word tokens only, so "and/or" is fine
    if stray:
        raise ValueError(f"unknown option {stray[0]!r} (known: /s, /seed, /t)")
    return GenerationRequest.build(text, seed=seed, seconds=seconds, steps=steps, model=model)


@dataclass
class Fetched:
    """Everything Lane.fetch_detailed() learned about one clip, for callers that show it
    (gui.py's Prompt tab shows the generation time of every clip, cached ones included).

    meta is the NPZ's own meta_json. The Spark writes `generation_s` (GPU time) and
    `created` into it, so a clip from the disk cache still knows how long it took when it
    was generated. result is None for a disk-cache hit: there was no request, so there is
    no total / queue / transfer time to report."""
    motion: RetargetedMotion
    label: str
    meta: dict
    result: GenerationResult | None   # None = answered from the disk cache
    convention: str                   # "standard" or "native" (kimodo_adapter)
    path: Path | None                 # the cache file written; None for a cache hit


class Lane:
    """Request -> cache or Spark -> validated clip -> retarget. Every failure comes back
    as a one-line message; nothing here ever touches the player thread's timing."""

    def __init__(self, client: KimodoClient, cache: ClipCache, retargeter: Retargeter) -> None:
        self.client = client
        self.cache = cache
        self.retargeter = retargeter

    def motion_from_npz(self, data: bytes) -> tuple[RetargetedMotion, str]:
        motion, conv, _meta = self._motion_conv_meta(data)
        return motion, conv

    def _motion_conv_meta(self, data: bytes) -> tuple[RetargetedMotion, str, dict]:
        clip = contract.unpack(data)
        motion, adapted = retarget_npz(clip, self.retargeter)
        return motion, adapted.convention, clip.meta

    def fetch_detailed(self, req: GenerationRequest) -> Fetched:
        """fetch() with everything it learns along the way, instead of only printing it.

        Added for the GUI (2026-10-09) as a hook, not a second code path: fetch() is now a
        thin wrapper around this, so the order that matters - validate (retarget) BEFORE
        caching, CLAUDE.md procedural invariant 8 - exists exactly once. The CLI's output
        is unchanged; the "generated ..." line it printed is printed by fetch().
        Raises SparkUnavailable / GenerationFailed / ContractError."""
        data = self.cache.get(req)
        if data is not None:
            try:
                motion, conv, meta = self._motion_conv_meta(data)
                return Fetched(motion, f"{req.prompt!r} seed {req.seed} (cache)", meta, None, conv, None)
            except ContractError as e:
                print(f"  cached file for {req.prompt!r} is unusable ({e}); asking the Spark again")
        result = self.client.generate(req)
        # Validate BEFORE caching: a clip the retarget refuses must not become library.
        motion, conv, meta = self._motion_conv_meta(result.data)
        path = self.cache.put(req, result.data)
        return Fetched(motion, f"{req.prompt!r} seed {req.seed}", meta, result, conv, path)

    def fetch(self, req: GenerationRequest) -> tuple[RetargetedMotion, str]:
        """(motion, label). Raises SparkUnavailable / GenerationFailed / ContractError."""
        f = self.fetch_detailed(req)
        result = f.result
        if result is not None:
            timing = f"{result.total_s:.1f} s total"
            if result.generation_s is not None:
                timing += (f" = {result.generation_s:.1f} s generating + {result.queue_s or 0:.1f} s queued"
                           f" + {result.transfer_s:.1f} s transfer")
            print(f"  generated {req.prompt!r} seed {req.seed}, {req.steps} steps: {timing}, "
                  f"{len(result.data) / 1e3:.0f} KB{' (server cache)' if result.server_cache else ''}"
                  f"{'' if f.convention == 'standard' else f', {f.convention} convention'} -> {f.path.name}")
        return f.motion, f.label


def load_bvh_motion(path: Path, retargeter: Retargeter) -> RetargetedMotion:
    clip, convention = to_standard_convention(read_bvh(path), retargeter.source)
    motion = retargeter.retarget_clip(clip)
    print(f"{path.name} ({convention} convention): {motion.num_frames} frames @ {motion.fps:.1f} fps "
          f"({motion.duration_s:.2f} s)")
    return motion


def repl(args, lane: Lane, player: LoopPlayer) -> int:
    jobs: queue.Queue[GenerationRequest | None] = queue.Queue()

    def worker() -> None:
        while True:
            req = jobs.get()
            if req is None:
                return
            try:
                motion, label = lane.fetch(req)
            except SparkUnavailable as e:
                print(f"\n  Spark service not reachable: {e}\n  playback continues; cached clips: /list, /play N")
                continue
            except (GenerationFailed, ContractError) as e:
                print(f"\n  generation of {req.prompt!r} failed: {e}")
                continue
            player.enqueue(motion, label)
            if player.current is not None:
                print(f"  queued {label}: plays when the current pass ends (/next to cut now)")

    gen_thread = threading.Thread(target=worker, name="generation", daemon=True)
    gen_thread.start()

    try:
        h = lane.client.health()
        print(f"Spark: {h.get('model')} on {h.get('device')}, {'busy' if h.get('busy') else 'idle'}")
    except (SparkUnavailable, GenerationFailed) as e:
        print(f"Spark not reachable right now ({e}). Cached clips still play: /list")
    print("Type a prompt, or /help.")

    entries = []
    while True:
        try:
            line = input("kimodo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/help":
            print(REPL_HELP)
        elif cmd == "/list":
            entries = lane.cache.entries()
            if not entries:
                print(f"  cache is empty ({lane.cache.root})")
            for i, e in enumerate(entries, 1):
                print(f"  {i:3d}  {e.label}")
        elif cmd == "/play":
            entries = entries or lane.cache.entries()
            try:
                entry = entries[int(rest) - 1]
                motion, _ = lane.motion_from_npz(entry.path.read_bytes())
            except (ValueError, IndexError):
                print("  usage: /play N  (numbers from /list)")
                continue
            except (ContractError, OSError) as e:
                print(f"  cannot play {entry.path.name}: {e}")
                continue
            player.enqueue(motion, f"{entry.meta.get('prompt')!r} seed {entry.meta.get('seed')} (cache)")
        elif cmd == "/next":
            player.skip()
        elif cmd == "/status":
            cur = player.current.label if player.current else "nothing"
            print(f"  playing {cur} (pass {player.passes + 1}); {player.queued} queued, "
                  f"{jobs.qsize()} waiting for generation")
        elif cmd == "/health":
            try:
                print(f"  {lane.client.health()}")
            except (SparkUnavailable, GenerationFailed) as e:
                print(f"  {e}")
        elif cmd.startswith("/"):
            print(f"  unknown command {cmd!r}; /help")
        else:
            try:
                req = parse_prompt(line, args.seconds, args.steps, args.seed, args.model)
            except ValueError as e:
                print(f"  {e}")
                continue
            jobs.put(req)
            print(f"  -> {req.prompt!r} seed {req.seed}, {req.num_frames} frames, {req.steps} steps")
    jobs.put(None)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="one shot: generate, play, exit (inline /s /seed /t work here too)")
    src.add_argument("--bvh", type=Path, help="Kimodo SOMA77 BVH clip (either convention), offline")
    src.add_argument("--clip", type=Path, help="a cached NPZ clip, offline")
    ap.add_argument("--seconds", type=float, default=contract.DEFAULT_SECONDS)
    ap.add_argument("--steps", type=int, default=contract.DEFAULT_STEPS)
    ap.add_argument("--seed", default=DEFAULT_SEED, type=parse_seed, help="an integer or 'random'")
    ap.add_argument("--model", default=contract.DEFAULT_MODEL)
    ap.add_argument("--spark-url", default=os.environ.get("KIMODO_URL", DEFAULT_URL))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("KIMODO_TIMEOUT", DEFAULT_TIMEOUT_S)))
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("KIMODO_CACHE_DIR", DEFAULT_CACHE_DIR)))
    ap.add_argument("--loop", action="store_true", help="one shot / offline: loop until Ctrl+C (the REPL always loops)")
    ap.add_argument("--rate", type=float, default=60.0, help="send rate in Hz (source frames are slerped)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--pose-ip", default="127.0.0.1")
    ap.add_argument("--pose-port", type=int, default=POSE_OSC_PORT)
    ap.add_argument("--face-ip", default="127.0.0.1")
    ap.add_argument("--face-port", type=int, default=LIVE_LINK_FACE_PORT)
    ap.add_argument("--no-face", action="store_true", help="don't send the Live Link Face head channel")
    ap.add_argument("--dry-run", action="store_true", help="retarget and run the clock, but send nothing")
    args = ap.parse_args(argv)

    retargeter = Retargeter(load_soma77())
    lane = Lane(KimodoClient(args.spark_url, timeout=args.timeout), ClipCache(args.cache_dir), retargeter)
    interactive = not (args.prompt or args.bvh or args.clip)

    # Load what plays BEFORE opening sockets, so a failure exits without sending anything.
    first: tuple[RetargetedMotion, str] | None = None
    try:
        if args.bvh:
            first = load_bvh_motion(args.bvh, retargeter), args.bvh.name
        elif args.clip:
            first = lane.motion_from_npz(args.clip.read_bytes())[0], args.clip.name
        elif args.prompt:
            req = parse_prompt(args.prompt, args.seconds, args.steps, args.seed, args.model)
            print(f"-> {req.prompt!r} seed {req.seed}, {req.num_frames} frames, {req.steps} steps", flush=True)
            first = lane.fetch(req)
    except SparkUnavailable as e:
        print(f"Spark service not reachable: {e}\nNo cached clip for this prompt/seed/length/steps.", file=sys.stderr)
        return 3
    except (GenerationFailed, ContractError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    sender = Sender(args.pose_ip, args.pose_port, args.face_ip, args.face_port,
                    face=not args.no_face, dry_run=args.dry_run, rate=args.rate)
    player = LoopPlayer(sender, rate=args.rate, speed=args.speed, loop=interactive or args.loop,
                        on_start=lambda c: print(f"\n  now playing {c.label} "
                                                 f"({c.motion.duration_s:.1f} s)"))
    if not args.dry_run:
        print(f"streaming pose -> {args.pose_ip}:{args.pose_port}"
              + ("" if args.no_face else f", head -> {args.face_ip}:{args.face_port}"))
    if first is not None:
        player.enqueue(*first)
    player.start()
    try:
        if interactive:
            repl(args, lane, player)
        else:
            while not player.finished.wait(0.2):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
    print(f"{'would have sent' if args.dry_run else 'sent'} {sender.sent} frames "
          f"({1 + 7 * len(STREAMED_BONES)} floats each); worst tick lateness {player.max_late_s * 1e3:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
