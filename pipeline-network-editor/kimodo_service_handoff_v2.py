import time
import os
import torch
import asyncio
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

# Direct imports based on Kimodo SOMA API documentation
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

# --- LOGGER UTILITY ---
METRICS_FILE = "pipeline_timing_log.txt"

def log_pipeline_step(filename, stage, duration_seconds):
    """Logs precise stage durations to a shared file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] | File: {filename} | Stage: {stage} | Duration: {duration_seconds:.3f}s\n"
    
    with open(METRICS_FILE, "a") as f:
        f.write(log_line)

# --- Configuration Fallbacks ---
MODEL_NAME = "Kimodo-SOMA-RP-v1.1"
OUTPUT_DIR = "./kimodo-gen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLENDER_ZMQ_PORT = 42070

# --- Default Fallback Values ---
FPS = 30
DURATION_SEC = 9.0
NUM_FRAMES = int(DURATION_SEC * FPS)  # 270 frames
STEPS = 100                           # High-quality DDIM steps

# --- ZeroMQ Setup ---
zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.PUSH)
zmq_socket.bind(f"tcp://0.0.0.0:{BLENDER_ZMQ_PORT}")
print(f"ZMQ Push Server listening on 0.0.0.0:{BLENDER_ZMQ_PORT}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global placeholders to maintain hot status in memory
model = None
skeleton = None

# --- Updated Pydantic Model with Optional Fields ---
class GenerationRequest(BaseModel):
    prompt: str
    filename_prefix: str
    fps: int | None = None
    duration: float | None = None
    steps: int | None = None
    seed: int | None = None

def load_kimodo_model():
    """Executes once when the server boots. Ensures the heavy neural network model stays in VRAM."""
    global model, skeleton
    print("\n" + "="*60)
    print(f"[@Startup] Pinned Initialization: Loading {MODEL_NAME} on {DEVICE}...")
    init_start = time.time()
    
    model = load_model(MODEL_NAME, device=DEVICE)
    model.eval()
    skeleton = model.output_skeleton
    
    init_duration = time.time() - init_start
    print(f"[@Startup] Initialization Complete: Model loaded in {init_duration:.2f}s.")
    print("="*60 + "\n")

def blocking_inference(long_p: str, output_path: str, req_fps: int, req_frames: int, req_steps: int, req_seed: int | None):
    """
    Executes heavy tensor compilation on a sub-thread.
    Dynamically configures parameters and injects PyTorch seeds for generation reproducibility.
    """
    # Inject seed if provided by the incoming API request
    if req_seed is not None:
        torch.manual_seed(req_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req_seed)
        print(f"   [Seed Engine] PyTorch manual seed locked to: {req_seed}")

    # Step 1: Core Neural Network Computation
    inf_start = time.time()
    with torch.no_grad():
        motion_dict = model(
            prompts=[long_p], 
            num_frames=req_frames,
            num_denoising_steps=req_steps,
            post_processing=True  
        )
    inf_duration = time.time() - inf_start
    print(f"   [Time Step 1/2] Core Tensor Inference ({req_steps} steps, {req_frames} frames): {inf_duration:.2f}s")
    log_pipeline_step(output_path, "Kimodo_Inference", inf_duration)

    # Step 2: Extract data arrays and serialize file to local I/O storage
    export_start = time.time()
    rot_mats = motion_dict['local_rot_mats']  
    root_pos = motion_dict['root_positions']   

    save_motion_bvh(
        output_path, 
        root_positions=root_pos, 
        local_rot_mats=rot_mats, 
        skeleton=skeleton, 
        fps=req_fps,
        standard_tpose=True 
    )
    export_duration = time.time() - export_start
    print(f"   [Time Step 2/2] BVH File Serialization & I/O Export ({req_fps} FPS): {export_duration:.2f}s")
    log_pipeline_step(output_path, "Kimodo_BVH_Export", export_duration)  

    return inf_duration, export_duration

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_kimodo_model()
    yield

app = FastAPI(
    title="Kimodo Motion Generation Service",
    description="Maintains a warm instance of Kimodo-SOMA to rapidly execute animation prompts via HTTP endpoints.",
    version="1.2.0",
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
    
    # Resolve parameters dynamically with global fallbacks
    runtime_fps = payload.fps if payload.fps is not None else FPS
    runtime_duration = payload.duration if payload.duration is not None else DURATION_SEC
    runtime_steps = payload.steps if payload.steps is not None else STEPS
    runtime_seed = payload.seed  # Defaults to None (stochastic generation)
    
    # Calculate exact frames needed for runtime duration execution
    runtime_frames = int(runtime_duration * runtime_fps)

    # Clear filename strings to protect against basic directory-traversal vulnerabilities
    clean_prefix = "".join([c for c in payload.filename_prefix if c.isalnum() or c in ('_', '-')]).strip()
    if not clean_prefix:
        clean_prefix = f"motion_{int(time.time())}"
        
    output_path = os.path.join(OUTPUT_DIR, f"{clean_prefix}.bvh")

    try:
        # Process inference safely inside a companion thread
        async with _inference_lock:
            inf_time, export_time = await asyncio.to_thread(
                blocking_inference, 
                payload.prompt, 
                output_path,
                runtime_fps,
                runtime_frames,
                runtime_steps,
                runtime_seed
            )
        
        total_time = time.time() - req_start
        print(f"[Success] Request lifecycle finished cleanly. Total Processing Time: {total_time:.2f}s")

        # --- THE BLENDER HANDOFF ---
        abs_output_path = os.path.abspath(output_path)
        windows_path = abs_output_path
        if windows_path.startswith("/mnt/c/"):
            windows_path = windows_path.replace("/mnt/c/", "C:\\", 1).replace("/", "\\")
            
        zmq_socket.send_string(windows_path)
        print(f"   [Handoff] Pinged Blender via ZMQ with Windows path: {windows_path}")
        
        return {
            "status": "success",
            "prompt": payload.prompt,
            "output_file_abs_path": os.path.abspath(output_path),
            "executed_parameters": {
                "fps": runtime_fps,
                "duration_seconds": runtime_duration,
                "frames": runtime_frames,
                "steps": runtime_steps,
                "seed": runtime_seed
            },
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
    uvicorn.run(app, host="127.0.0.1", port=42069)