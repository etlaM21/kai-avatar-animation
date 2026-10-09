"""Kimodo generation service. Runs on the DGX Spark, started by hand in your SSH session:

    python kimodo_service.py                        # 127.0.0.1:8765, through the SSH tunnel
    python kimodo_service.py --host 100.83.6.8      # on the Tailscale IP, no tunnel needed

It ends when you log out, on purpose (CLAUDE.md, "Service lifetime"). No tmux/screen/
nohup/systemd, no auto-restart.

    POST /generate   {"prompt", "seed", "num_frames", "steps", "model", "first_frame"?}
                     -> the clip as NPZ bytes (kimodo_contract.py), headers
                        X-Generation-Seconds, X-Queue-Seconds, X-Seed, X-Cache
    GET  /health     -> {"model", "loaded", "busy", ...}

Rewritten from _old/pipeline-network-osc/kimodo_service_handoff_osc_v3.py (left untouched).
Kept: FastAPI, the model loaded once and kept warm, an asyncio.Lock so a generation is
never interrupted or overlapped, the shared HF cache, a timing log. Dropped: all OSC
streaming, all quaternion swizzling, the WSL gateway lookup. This service returns raw
Kimodo data and knows nothing about Manny - every retarget step lives on Windows.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kimodo_contract as contract  # noqa: E402  (same file the Windows client uses)
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

TIMING_LOG = HERE / "timing_log.txt"
SERVER_CACHE_SIZE = 32
FORBIDDEN_HOSTS = {"0.0.0.0", "::", ""}


# Module level, not inside build_app: with postponed annotations FastAPI resolves the
# endpoint's `body: GenerateBody` against module globals.
class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seed: int | None = None
    num_frames: int = Field(default=round(contract.DEFAULT_SECONDS * contract.KIMODO_FPS), ge=1,
                            le=round(contract.MAX_SECONDS * contract.KIMODO_FPS))
    steps: int = Field(default=contract.DEFAULT_STEPS, ge=1, le=1000)
    model: str = contract.DEFAULT_MODEL
    first_frame: list | None = None   # reserved for chaining; not implemented in v1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (with the SSH tunnel) or the Tailscale IP. Never 0.0.0.0.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=contract.DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/opt/huggingface_cache"),
                    help="shared Hugging Face cache, so the weights are not downloaded per user")
    args = ap.parse_args(argv)
    if args.host in FORBIDDEN_HOSTS:
        ap.error(f"refusing to bind {args.host!r}: use 127.0.0.1 or the Tailscale IP (CLAUDE.md invariant 10)")
    return args


def log_timing(**fields) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print("   " + line)
    try:
        with TIMING_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"   (timing log not written: {e})")


def _np(x):
    """Kimodo returns torch tensors (or numpy with return_numpy=True); drop the batch dim."""
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    return x[0] if x.ndim >= 1 and x.shape[0] == 1 else x


class Generator:
    """The warm model. Only ever called with the lock held, from a worker thread."""

    def __init__(self, model_name: str, device: str | None) -> None:
        self.model_name = model_name
        self.device_arg = device
        self.model = None
        self.skeleton = None
        self.device = None
        self.kimodo_version = "unknown"

    def load(self) -> None:
        import torch
        from kimodo.model.load_model import load_model
        try:
            from importlib.metadata import version
            self.kimodo_version = version("kimodo")
        except Exception:
            pass
        self.device = self.device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("WARNING: no CUDA device - generation will be very slow")
        t0 = time.time()
        print(f"Loading {self.model_name} on {self.device} ...")
        self.model = load_model(self.model_name, device=self.device)
        self.model.eval()
        self.skeleton = self.model.output_skeleton
        print(f"Model loaded in {time.time() - t0:.1f} s (kimodo {self.kimodo_version})")

    def generate(self, req: contract.GenerationRequest) -> tuple[bytes, float]:
        import numpy as np
        import torch
        random.seed(req.seed)
        np.random.seed(req.seed % (2**32))
        torch.manual_seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

        t0 = time.time()
        with torch.no_grad():
            out = self.model(prompts=[req.prompt], num_frames=req.num_frames,
                             num_denoising_steps=req.steps,
                             post_processing=True)   # foot-skate cleanup; stays on (invariant 11)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_s = time.time() - t0

        clip = contract.KimodoClip(
            global_rot_mats=_np(out["global_rot_mats"]),
            root_positions=_np(out["root_positions"]),
            posed_joints=_np(out["posed_joints"]),
            foot_contacts=_np(out["foot_contacts"]),
            fps=float(getattr(self.model, "fps", contract.KIMODO_FPS)),
            bone_order_names=list(self.skeleton.bone_order_names),
            meta={**req.to_json(), "generation_s": round(gen_s, 3), "kimodo_version": self.kimodo_version,
                  "created": datetime.now().isoformat(timespec="seconds")},
        )
        data = contract.pack(clip)
        contract.unpack(data)   # never send what the client would refuse
        return data, gen_s


def build_app(gen: Generator):
    state = {"lock": None, "started": time.time()}
    results: OrderedDict[str, bytes] = OrderedDict()

    @asynccontextmanager
    async def lifespan(_app):
        state["lock"] = asyncio.Lock()
        gen.load()            # before the socket accepts anything: no half-loaded requests
        yield

    app = FastAPI(title="Kimodo service", version=str(contract.CONTRACT_VERSION), lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": gen.model_name, "loaded": gen.model is not None,
                "busy": state["lock"].locked(), "device": gen.device, "kimodo_version": gen.kimodo_version,
                "contract_version": contract.CONTRACT_VERSION, "fps": contract.KIMODO_FPS,
                "defaults": {"steps": contract.DEFAULT_STEPS, "seconds": contract.DEFAULT_SECONDS},
                "uptime_s": round(time.time() - state["started"], 1)}

    @app.post("/generate")
    async def generate(body: GenerateBody):
        if body.first_frame is not None:
            raise HTTPException(501, "first_frame (chaining) is reserved and not implemented in v1")
        if body.model != gen.model_name:
            raise HTTPException(409, f"this service runs {gen.model_name}, request asked for {body.model}")
        seed = body.seed if body.seed is not None else random.randrange(2**31)
        req = contract.GenerationRequest(prompt=contract.GenerationRequest.normalize_prompt(body.prompt),
                                         seed=seed, num_frames=body.num_frames, steps=body.steps,
                                         model=body.model)
        key = req.cache_key()
        print(f"\n[request] {req.prompt!r} seed={seed} frames={req.num_frames} steps={req.steps}")

        t_wait = time.time()
        async with state["lock"]:
            queue_s = time.time() - t_wait
            if key in results:
                # The client timed out on an earlier identical request while the GPU kept
                # going; the retry gets that result instead of a second generation.
                results.move_to_end(key)
                data, gen_s, cache = results[key], 0.0, "hit"
            else:
                try:
                    data, gen_s = await asyncio.to_thread(gen.generate, req)
                except Exception as e:
                    print(f"[failure] {type(e).__name__}: {e}")
                    raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}"[:300]) from None
                results[key] = data
                while len(results) > SERVER_CACHE_SIZE:
                    results.popitem(last=False)
                cache = "miss"

        log_timing(prompt=repr(req.prompt)[:60], seed=seed, frames=req.num_frames, steps=req.steps,
                   queue_s=f"{queue_s:.3f}", generation_s=f"{gen_s:.3f}", bytes=len(data), cache=cache)
        return Response(content=data, media_type="application/octet-stream",
                        headers={"X-Generation-Seconds": f"{gen_s:.3f}", "X-Queue-Seconds": f"{queue_s:.3f}",
                                 "X-Seed": str(seed), "X-Cache": cache})

    return app


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HOME"] = args.hf_home      # before kimodo/transformers are imported
    import uvicorn
    app = build_app(Generator(args.model, args.device))
    print(f"Kimodo service on http://{args.host}:{args.port} (Ctrl+C or logging out stops it)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
