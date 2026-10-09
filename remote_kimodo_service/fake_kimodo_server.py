"""A stand-in for the Spark's kimodo_service.py, stdlib only, for tests and offline work.

    .\\venv\\Scripts\\python.exe -m remote_kimodo_service.fake_kimodo_server
    .\\venv\\Scripts\\python.exe -m remote_kimodo_service.fake_kimodo_server --delay 3

Same endpoints and headers as the real service. "Generation" picks one of the existing
kimodo-gen/*.bvh clips (by prompt hash) and returns it as the NPZ the real service
would send: Root dropped, metres, FK positions as posed_joints. Clips are sent in the
convention their file uses, so the adapter's convention detection is exercised too.

Failure modes for the client-resilience tests: "close" accepts and closes (what the
SSH tunnel does when the service behind it is down), "slow" never answers within the
client's timeout, "error" answers 500, "garbage" sends bytes that are not an NPZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from procedural_animation import MODULE_ROOT
from procedural_animation.bvh_reader import BvhClip, read_bvh

from . import kimodo_contract as contract

# Order matters: a prompt picks its clip by hash index into the combined list, so
# reordering changes which clip a given prompt gets.
CLIP_DIRS = [MODULE_ROOT / "assets" / "kimodo" / "clips" / "kimodo-gen",
             MODULE_ROOT / "assets" / "kimodo" / "clips" / "editor-gen"]


def clip_from_bvh(bvh: BvhClip, meta: dict | None = None) -> contract.KimodoClip:
    """What Kimodo would have returned for this BVH, in the BVH's own rest convention."""
    glob, pos = bvh.forward_kinematics()
    T = bvh.num_frames
    rots = np.stack([g.as_matrix() for g in glob[1:]], axis=1)       # drop Root
    posed_m = pos[:, 1:] / 100.0                                      # cm -> m
    return contract.KimodoClip(global_rot_mats=rots, root_positions=posed_m[:, 0].copy(),
                               posed_joints=posed_m, foot_contacts=np.zeros((T, 4)),
                               fps=bvh.fps, bone_order_names=bvh.names[1:], meta=dict(meta or {}))


class FakeKimodo:
    def __init__(self, mode: str = "ok", delay_s: float = 0.0, port: int = 0,
                 library: list[Path] | None = None) -> None:
        self.mode = mode
        self.delay_s = delay_s
        self.library = library if library is not None else sorted(p for d in CLIP_DIRS for p in d.glob("*.bvh"))
        self.requests = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a) -> None:
                pass

            def _json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                fake.requests += 1
                if fake.mode == "close":
                    self.close_connection = True
                    return
                if self.path == "/health":
                    self._json(200, {"status": "ok", "model": contract.DEFAULT_MODEL, "loaded": True,
                                     "busy": False, "device": "fake", "contract_version": contract.CONTRACT_VERSION})
                else:
                    self._json(404, {"detail": "not found"})

            def do_POST(self) -> None:
                fake.requests += 1
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if fake.mode == "close":
                    self.close_connection = True
                    return
                if fake.mode == "slow":
                    time.sleep(fake.delay_s or 30.0)
                if fake.mode == "error":
                    self._json(500, {"detail": "generation failed: RuntimeError: CUDA out of memory"})
                    return
                if body.get("first_frame") is not None:
                    self._json(501, {"detail": "first_frame (chaining) is reserved and not implemented in v1"})
                    return
                if fake.mode == "garbage":
                    data = b"PK\x03\x04 this is not an npz"
                else:
                    t0 = time.time()
                    time.sleep(fake.delay_s)
                    data = fake.make(body)
                    gen_s = time.time() - t0
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                if fake.mode != "garbage":
                    self.send_header("X-Generation-Seconds", f"{gen_s:.3f}")
                    self.send_header("X-Queue-Seconds", "0.000")
                    self.send_header("X-Seed", str(body.get("seed")))
                    self.send_header("X-Cache", "miss")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def make(self, body: dict) -> bytes:
        h = int(hashlib.sha256(body["prompt"].encode()).hexdigest(), 16)
        path = self.library[h % len(self.library)]
        bvh = read_bvh(path)
        frames = min(int(body.get("num_frames", bvh.num_frames)), bvh.num_frames)
        meta = {k: body.get(k) for k in ("prompt", "seed", "steps", "model")}
        meta.update(num_frames=frames, fake_source=path.name)
        clip = clip_from_bvh(bvh, meta)
        clip.global_rot_mats = clip.global_rot_mats[:frames]
        clip.root_positions = clip.root_positions[:frames]
        clip.posed_joints = clip.posed_joints[:frames]
        clip.foot_contacts = clip.foot_contacts[:frames]
        return contract.pack(clip)

    def __enter__(self) -> "FakeKimodo":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0, help="pretend generation takes this long (s)")
    ap.add_argument("--mode", choices=("ok", "close", "slow", "error", "garbage"), default="ok")
    args = ap.parse_args()
    with FakeKimodo(mode=args.mode, delay_s=args.delay, port=args.port) as fake:
        print(f"fake Kimodo service on {fake.url} ({len(fake.library)} BVH clips, mode {args.mode}); Ctrl+C stops")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
