import time
import os
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Direct imports based on Kimodo SOMA API documentation
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

# --- Configuration ---
MODEL_NAME = "Kimodo-SOMA-RP-v1.1"
OUTPUT_DIR = "./kimodo-gen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Motion Settings ---
FPS = 30
DURATION_SEC = 9.0
NUM_FRAMES = int(DURATION_SEC * FPS)  # 270 frames
STEPS = 100                           # High-quality DDIM steps

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global placeholders to maintain hot status in memory
model = None
skeleton = None

class GenerationRequest(BaseModel):
    prompt: str
    filename_prefix: str

def load_kimodo_model():
    """Executes once when the server boots. Ensures the heavy neural network model stays in VRAM."""
    global model, skeleton
    print("\n" + "="*60)
    print(f"[@Startup] Pinned Initialization: Loading {MODEL_NAME} on {DEVICE}...")
    init_start = time.time()
    
    # Core asset footprint instantiation
    model = load_model(MODEL_NAME, device=DEVICE)
    model.eval()
    skeleton = model.output_skeleton
    
    init_duration = time.time() - init_start
    print(f"[@Startup] Initialization Complete: Model loaded in {init_duration:.2f}s.")
    print("="*60 + "\n")

def blocking_inference(long_p: str, output_path: str):
    """
    Executes heavy tensor compilation on a sub-thread to keep logs 
    synchronous and keep the master FastAPI application highly responsive.
    """
    # Step 1: Core Neural Network Computation
    inf_start = time.time()
    with torch.no_grad():
        motion_dict = model(
            prompts=[long_p], 
            num_frames=NUM_FRAMES,
            num_denoising_steps=STEPS,
            post_processing=True  
        )
    inf_duration = time.time() - inf_start
    print(f"   [Time Step 1/2] Core Tensor Inference: {inf_duration:.2f}s")

    # Step 2: Extract data arrays and serialize file to local I/O storage
    export_start = time.time()
    rot_mats = motion_dict['local_rot_mats']  # Shape: [1, Frames, Joints, 3, 3]
    root_pos = motion_dict['root_positions']   # Shape: [1, Frames, 3]

    save_motion_bvh(
        output_path, 
        root_positions=root_pos, 
        local_rot_mats=rot_mats, 
        skeleton=skeleton, 
        fps=FPS,
        standard_tpose=False 
    )
    export_duration = time.time() - export_start
    print(f"   [Time Step 2/2] BVH File Serialization & I/O Export: {export_duration:.2f}s")
    
    return inf_duration, export_duration

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_kimodo_model()  # startup
    yield

app = FastAPI(
    title="Kimodo Motion Generation Service",
    description="Maintains a warm instance of Kimodo-SOMA to rapidly execute animation prompts via HTTP endpoints.",
    version="1.1.0",
    lifespan=lifespan
)

_inference_lock: asyncio.Lock | None = None

@app.post("/generate")
async def generate_motion(payload: GenerationRequest):

    global _inference_lock
    if _inference_lock is None:
        _inference_lock = asyncio.Lock()

    req_start = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n[Incoming Request] Active cycle triggered at {timestamp}")
    print(f"   Prompt payload: \"{payload.prompt}\"")
    print(f"   Target storage identifier: {payload.filename_prefix}.bvh")

    # Clear filename strings to protect against basic directory-traversal vulnerabilities
    clean_prefix = "".join([c for c in payload.filename_prefix if c.isalnum() or c in ('_', '-')]).strip()
    if not clean_prefix:
        clean_prefix = f"motion_{int(time.time())}"
        
    output_path = os.path.join(OUTPUT_DIR, f"{clean_prefix}.bvh")

    try:
        # Relinquish the main event loop worker; process inference safely inside a companion thread
        async with _inference_lock:
            inf_time, export_time = await asyncio.to_thread(blocking_inference, payload.prompt, output_path)
        
        total_time = time.time() - req_start
        print(f"[Success] Request lifecycle finished cleanly. Total Processing Time: {total_time:.2f}s")
        
        return {
            "status": "success",
            "prompt": payload.prompt,
            "output_file_abs_path": os.path.abspath(output_path),
            "performance_metrics": {
                "inference_duration_seconds": round(inf_time, 2),
                "export_duration_seconds": round(export_time, 2),
                "total_request_handling_seconds": round(total_time, 2)
            }
        }
        
    except Exception as e:
        print(f"[Failure] Operational processing execution broke down: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal generation failure occurred: {str(e)}")

if __name__ == "__main__":
    # Explicit port routing config assignment
    uvicorn.run("kimodo_service:app", host="127.0.0.1", port=42069, reload=False)