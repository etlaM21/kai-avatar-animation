# import subprocess
# import time
# import os

# # Configuration
# MODEL = "Kimodo-SOMA-RP-v1.1"
# OUTPUT_DIR = "./kimodo-gen"
# TIME_LOG = os.path.join(OUTPUT_DIR, "gen-time.txt")

# # Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 10 Prompts for motion generation
# prompts = [
#     ("A person performing a roundhouse kick", "kick"),
#     ("A person sitting down in a chair and crossing their legs", "sit"),
#     ("A person celebrating a goal with a backflip", "backflip"),
#     ("A person walking cautiously while carrying a heavy box", "carry"),
#     ("A person waving both arms to get someone's attention", "wave"),
#     ("A person bending down to tie their shoelaces", "tie"),
#     ("A person throwing a baseball", "throw"),
#     ("A person stretching their arms over their head in the morning", "stretch"),
#     ("A person sneaking forward in a crouched position", "sneak"),
#     ("A person dancing a rhythmic salsa sequence", "dance")
# ]

# def run_generation():
#     total_start_time = time.time()
#     timing_results = []

#     print(f"Starting Kimodo-SOMA generation for {len(prompts)} prompts...")

#     for long_p, short_p in prompts:
#         print(f"Generating: {short_p}...")
        
#         # Format the command
#         cmd = [
#             "kimodo_gen",
#             long_p, 
#             "--model", MODEL,
#             "--duration", "9.0",
#             "--bvh",
#             "--output", os.path.join(OUTPUT_DIR, short_p)
#         ]

#         start_inst = time.time()
        
#         try:
#             # Execute the command
#             subprocess.run(cmd, check=True)
#             duration = time.time() - start_inst
#             timing_results.append(f"{short_p}: {duration:.2f} seconds")
#             print(f"Success! Took {duration:.2f}s")
            
#         except subprocess.CalledProcessError as e:
#             print(f"Error generating motion for {short_p}: {e}")
#             timing_results.append(f"{short_p}: FAILED")

#     total_duration = time.time() - total_start_time
    
#     # Save results to gen-time.txt
#     with open(TIME_LOG, "w") as f:
#         f.write("--- Kimodo Generation Timing Report ---\n")
#         for entry in timing_results:
#             f.write(entry + "\n")
#         f.write("-" * 40 + "\n")
#         f.write(f"Total Execution Time: {total_duration:.2f} seconds\n")

#     print(f"Processing complete. Logs saved to {TIME_LOG}")

# if __name__ == "__main__":
#     run_generation()

import time
import os
import torch
from tqdm import tqdm

# Direct imports based on Kimodo SOMA API documentation
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

# --- Configuration ---
MODEL_NAME = "Kimodo-SOMA-RP-v1.1"
OUTPUT_DIR = "./kimodo-gen"
TIME_LOG = os.path.join(OUTPUT_DIR, "gen-time.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Motion Settings ---
FPS = 30
DURATION_SEC = 9.0
NUM_FRAMES = int(DURATION_SEC * FPS)  # 270 frames
STEPS = 100                           # High-quality DDIM steps

os.makedirs(OUTPUT_DIR, exist_ok=True)

prompts = [
    ("A person performing a roundhouse kick", "kick"),
    ("A person sitting down in a chair and crossing their legs", "sit"),
    ("A person celebrating a goal with a backflip", "backflip"),
    ("A person walking cautiously while carrying a heavy box", "carry"),
    ("A person waving both arms to get someone's attention", "wave"),
    ("A person bending down to tie their shoelaces", "tie"),
    ("A person throwing a baseball", "throw"),
    ("A person stretching their arms over their head in the morning", "stretch"),
    ("A person sneaking forward in a crouched position", "sneak"),
    ("A person dancing a rhythmic salsa sequence", "dance")
]

def run_inference_session():
    print(f"Initializing {MODEL_NAME} on {DEVICE}...")
    init_start = time.time()
    
    # load_model returns the Kimodo object directly (no unpacking)
    model = load_model(MODEL_NAME, device=DEVICE)
    model.eval()
    
    # Use output_skeleton (SOMA77) as specified in the API property list
    skeleton = model.output_skeleton
    
    init_duration = time.time() - init_start
    print(f"Model loaded in {init_duration:.2f}s. Starting batch generation...")

    timing_results = []
    total_gen_start = time.time()

    for long_p, short_p in prompts:
        print(f"\n[Processing] {long_p}...")
        start_inst = time.time()
        
        try:
            with torch.no_grad():
                # Trigger __call__ via model() instance
                # Pass the prompt as a list
                motion_dict = model(
                    prompts=[long_p], # <--- Wrapped in a list
                    num_frames=NUM_FRAMES,
                    num_denoising_steps=STEPS,
                    post_processing=True  
                )

                # Extract data (These will now inherently include the batch dimension)
                rot_mats = motion_dict['local_rot_mats'] # Shape: [1, Frames, Joints, 3, 3]
                root_pos = motion_dict['root_positions'] # Shape: [1, Frames, 3]

                output_path = os.path.join(OUTPUT_DIR, f"{short_p}.bvh")

                # Export
                save_motion_bvh(
                    output_path, 
                    root_positions=root_pos, 
                    local_rot_mats=rot_mats, 
                    skeleton=skeleton, 
                    fps=FPS,
                    standard_tpose=False 
                )
            
            duration = time.time() - start_inst
            timing_results.append(f"{short_p}: {duration:.2f} seconds")
            print(f"Success: Saved to {OUTPUT_DIR} ({duration:.2f}s)")

        except Exception as e:
            print(f"Error generating {short_p}: {e}")
            timing_results.append(f"{short_p}: FAILED")

    total_duration = time.time() - total_gen_start
    
    # Save professional timing report
    with open(TIME_LOG, "w") as f:
        f.write(f"--- Kimodo SOMA Generation Report ---\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}\n")
        f.write(f"Model Load Time: {init_duration:.2f}s\n")
        f.write("-" * 40 + "\n")
        for entry in timing_results:
            f.write(entry + "\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Batch Processing Time: {total_duration:.2f}s\n")

    print(f"\nBatch complete. View results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference_session()