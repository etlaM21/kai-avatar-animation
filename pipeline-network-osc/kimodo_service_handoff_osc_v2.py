import time
import os
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

# Imports for OSC and math
from pythonosc import udp_client
from scipy.spatial.transform import Rotation as R
import numpy as np
import subprocess

# Direct imports based on Kimodo SOMA API documentation
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

SCRIPT_VERSION = 0.71

# --- OSC SETUP ---
# Sending to local machine on port 8000

def get_windows_host_ip():
    """Gets the real Windows host IP via the WSL2 default gateway.
    (/etc/resolv.conf is unreliable now — under DNS tunneling it returns a 
    synthetic loopback-bound address, 10.255.255.254, that only services DNS.)"""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, check=True
        )
        # Output looks like: "default via 172.29.16.1 dev eth0 ..."
        return result.stdout.split()[2]
    except Exception:
        pass
    return "127.0.0.1"

TARGET_IP = get_windows_host_ip()
osc_client = udp_client.SimpleUDPClient(TARGET_IP, 8000)
print(f"OSC Client initialized. Sending to Windows Host at {TARGET_IP}:8000")

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

# --- Default Fallback Values ---
FPS = 30
DURATION_SEC = 9.0
NUM_FRAMES = int(DURATION_SEC * FPS)  # 270 frames
STEPS = 100                           # High-quality DDIM steps

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
    print("="*25 + f"SCRIPT V{SCRIPT_VERSION}" + "="*25 + "\n")

# Helper list of the exact joint names from Kimodo
KIMODO_JOINTS = [
    "Root", "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", 
    "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", 
    "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb1", 
    "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd", "RightHandIndex1", 
    "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"
]

# Dictionary of MetaHuman joints we map to

METAHUMAN_BONE_MAP = {
    "Root": "root",
    "Hips": "pelvis",
    "Spine1": "spine_01",
    "Spine2": "spine_03", 
    "Chest": "spine_05",
    "Neck1": "neck_01",
    "Neck2": "neck_02",
    "Head": "head",# "HeadEnd", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l", 
    "LeftHand": "hand_l",
    "LeftHandThumb1": "thumb_01_l",
    "LeftHandThumb2": "thumb_02_l",
    "LeftHandThumb3": "thumb_03_l", # "LeftHandThumbEnd",
    "LeftHandIndex1": "index_01_l", 
    "LeftHandIndex2": "index_02_l", 
    "LeftHandIndex3": "index_03_l", # "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1": "middle_01_l",
    "LeftHandMiddle2": "middle_02_l",
    "LeftHandMiddle3": "middle_03_l", # "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1": "ring_01_l",
    "LeftHandRing2": "ring_02_l",
    "LeftHandRing3": "ring_03_l", #"LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1": "pinky_01_l",
    "LeftHandPinky2": "pinky_02_l",
    "LeftHandPinky3": "pinky_03_l", #"LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r", 
    "RightHand": "hand_r",
    "RightHandThumb1": "thumb_01_r",
    "RightHandThumb2": "thumb_02_r",
    "RightHandThumb3": "thumb_03_r", # "RightHandThumbEnd",
    "RightHandIndex1": "index_01_r", 
    "RightHandIndex2": "index_02_r", 
    "RightHandIndex3": "index_03_r", # "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1": "middle_01_r",
    "RightHandMiddle2": "middle_02_r",
    "RightHandMiddle3": "middle_03_r", # "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1": "ring_01_r",
    "RightHandRing2": "ring_02_r",
    "RightHandRing3": "ring_03_r", #"RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1": "pinky_01_r",
    "RightHandPinky2": "pinky_02_r",
    "RightHandPinky3": "pinky_03_r", #"RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg": "thigh_l",
    "LeftShin": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l", #"LeftToeEnd",
    "RightLeg": "thigh_r",
    "RightShin": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r", #"RightToeEnd",
}

def stream_motion_data(root_positions, local_rot_mats, fps):
    print(f"[@Stream] Beginning Local Space OSC broadcast at {fps} FPS...")
    
    # 1. Slice [0] to remove the AI Batch Dimension
    pos_np = root_positions.cpu().numpy()[0]  
    rot_np = local_rot_mats.cpu().numpy()[0]  

    num_frames = pos_np.shape[0] 
    frame_time = 1.0 / fps

    if hasattr(skeleton, 'bone_order_names'):
        soma_joint_names = skeleton.bone_order_names
    else:
        print("[Error] Cannot extract 'bone_order_names' from skeleton object.")
        return

    # SOMA (Y-Up RH) to UE5 (Z-Up LH) World Matrix for the Root ONLY
    M = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0,  1,  0]
    ])

    for frame_idx in range(num_frames):
        start_time = time.perf_counter()
        payload = []
        
        # --- 1. Root Position (Translation) ---
        raw_pos = pos_np[frame_idx].flatten()[0:3]
        ue_pos = M @ raw_pos
        payload.extend([float(ue_pos[0]), float(ue_pos[1]), float(ue_pos[2])])
        
        ARM_CHAIN = {"LeftShoulder", "LeftArm", "LeftForeArm", "RightShoulder", "RightArm", "RightForeArm"}

        for joint_idx, kimodo_name in enumerate(soma_joint_names):
            if joint_idx >= rot_np.shape[1]: 
                continue
                
            if kimodo_name.endswith("End") or kimodo_name in ["Jaw", "LeftEye", "RightEye"]:
                continue

            rot_matrix = rot_np[frame_idx, joint_idx]
            u, s, vh = np.linalg.svd(rot_matrix)
            clean_rot_matrix = np.dot(u, vh)
            quat = R.from_matrix(clean_rot_matrix).as_quat()

            if kimodo_name in ARM_CHAIN:
                final_quat = [quat[0], quat[2], -quat[1], quat[3]]
                if kimodo_name == "RightForeArm":
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: {kimodo_name} raw bend angle = "
                            f"{np.degrees(2 * np.arccos(np.clip(abs(quat[3]), -1.0, 1.0))):.1f}°, "
                            f"sending {final_quat}")
            else:
                final_quat = [quat[0], quat[1], -quat[2], -quat[3]]
                if kimodo_name == "RightLeg":
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: {kimodo_name} raw bend angle = "
                            f"{np.degrees(2 * np.arccos(np.clip(abs(quat[3]), -1.0, 1.0))):.1f}°, "
                            f"sending {final_quat}")
            payload.append(kimodo_name)
            payload.extend([float(final_quat[0]), float(final_quat[1]), float(final_quat[2]), float(final_quat[3])])
    
        # Send OSC message
        osc_client.send_message("/kaspar/frame", payload)
        
        # Pace loop
        elapsed = time.perf_counter() - start_time
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("[@Stream] Broadcast complete.")

def blocking_inference(long_p: str, output_path: str, req_fps: int, req_frames: int, req_steps: int, req_seed: int | None):
    """
    Executes heavy tensor compilation on a sub-thread and streams the result.
    """
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
    log_pipeline_step("Live_Stream", "Kimodo_Inference", inf_duration)

    # Extract data arrays
    rot_mats = motion_dict['local_rot_mats']  
    root_pos = motion_dict['root_positions']   

    # Step 2: Trigger the OSC Stream
    stream_start = time.time()
    stream_motion_data(root_pos, rot_mats, req_fps)
    stream_duration = time.time() - stream_start
    print(f"   [Time Step 2/2] OSC Network Stream: {stream_duration:.2f}s")
    log_pipeline_step("Live_Stream", "Kimodo_OSC_Stream", stream_duration)

    # Return both metrics so the endpoint can unpack them
    return inf_duration, stream_duration

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("============================================================")
    print(f"[Version {SCRIPT_VERSION}] Running Kimodo generation service with handoff to UE via OSC")
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
    
    # Resolve parameters dynamically
    runtime_fps = payload.fps if payload.fps is not None else FPS
    runtime_duration = payload.duration if payload.duration is not None else DURATION_SEC
    runtime_steps = payload.steps if payload.steps is not None else STEPS
    runtime_seed = payload.seed 
    
    runtime_frames = int(runtime_duration * runtime_fps)

    # We keep output_path just in case you ever want to write a debug file again, 
    # but it is largely unused in the pure streaming pipeline.
    clean_prefix = "".join([c for c in payload.filename_prefix if c.isalnum() or c in ('_', '-')]).strip()
    if not clean_prefix:
        clean_prefix = f"motion_{int(time.time())}"
    output_path = os.path.join(OUTPUT_DIR, f"{clean_prefix}.bvh")

    try:
        # Process inference and stream safely inside a companion thread
        async with _inference_lock:
            inf_time, stream_time = await asyncio.to_thread(
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

        # Clean JSON response (ZMQ Handoff completely removed)
        return {
            "status": "success",
            "prompt": payload.prompt,
            "executed_parameters": {
                "fps": runtime_fps,
                "duration_seconds": runtime_duration,
                "frames": runtime_frames,
                "steps": runtime_steps,
                "seed": runtime_seed
            },
            "performance_metrics": {
                "inference_duration_seconds": round(inf_time, 2),
                "stream_duration_seconds": round(stream_time, 2),
                "total_request_handling_seconds": round(total_time, 2)
            }
        }
        
    except Exception as e:
        print(f"[Failure] Operational processing execution broke down: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal generation failure occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=42069)