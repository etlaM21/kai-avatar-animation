import requests
import time

# =============================================================================
# KIMODO BEAR ACTOR BATCH TEST — 2026-06-08
# 10 prompt styles × 3 repetitions = 30 generations
#
# Prompt strategy spectrum:
#   P01 — Minimal/literal
#   P02 — Simple descriptive
#   P03 — Action-sequence
#   P04 — Biomechanical/rational
#   P05 — Character-driven
#   P06 — Theatrical/artsy
#   P07 — Emotion-led
#   P08 — Instructional/stage-direction
#   P09 — Metaphorical
#   P10 — Verbose/cinematic
# =============================================================================

animation_batch = [

    # --- P01: Minimal/literal ---
    {"prompt": "A person crawls on all fours like a bear",                   "identifier": "test260608_p01_r1"},
    {"prompt": "A person crawls on all fours like a bear",                   "identifier": "test260608_p01_r2"},
    {"prompt": "A person crawls on all fours like a bear",                   "identifier": "test260608_p01_r3"},

    # --- P02: Simple descriptive ---
    {"prompt": "A person gets down on hands and knees and walks forward",    "identifier": "test260608_p02_r1"},
    {"prompt": "A person gets down on hands and knees and walks forward",    "identifier": "test260608_p02_r2"},
    {"prompt": "A person gets down on hands and knees and walks forward",    "identifier": "test260608_p02_r3"},

    # --- P03: Action-sequence ---
    {"prompt": "A person bends down, places both hands on the ground, and walks on all fours with a heavy lumbering gait", "identifier": "test260608_p03_r1"},
    {"prompt": "A person bends down, places both hands on the ground, and walks on all fours with a heavy lumbering gait", "identifier": "test260608_p03_r2"},
    {"prompt": "A person bends down, places both hands on the ground, and walks on all fours with a heavy lumbering gait", "identifier": "test260608_p03_r3"},

    # --- P04: Biomechanical/rational ---
    {"prompt": "A person shifts their weight forward onto extended arms, drops their knees to the ground, and moves in a quadrupedal walking pattern with alternating limb steps", "identifier": "test260608_p04_r1"},
    {"prompt": "A person shifts their weight forward onto extended arms, drops their knees to the ground, and moves in a quadrupedal walking pattern with alternating limb steps", "identifier": "test260608_p04_r2"},
    {"prompt": "A person shifts their weight forward onto extended arms, drops their knees to the ground, and moves in a quadrupedal walking pattern with alternating limb steps", "identifier": "test260608_p04_r3"},

    # --- P05: Character-driven ---
    {"prompt": "A person plays the role of a bear, getting down on all fours and walking with slow powerful strides", "identifier": "test260608_p05_r1"},
    {"prompt": "A person plays the role of a bear, getting down on all fours and walking with slow powerful strides", "identifier": "test260608_p05_r2"},
    {"prompt": "A person plays the role of a bear, getting down on all fours and walking with slow powerful strides", "identifier": "test260608_p05_r3"},

    # --- P06: Theatrical/artsy ---
    {"prompt": "A person inhabits the body of a great bear, surrendering upright posture, sinking hands to earth and prowling forward with primal deliberate weight", "identifier": "test260608_p06_r1"},
    {"prompt": "A person inhabits the body of a great bear, surrendering upright posture, sinking hands to earth and prowling forward with primal deliberate weight", "identifier": "test260608_p06_r2"},
    {"prompt": "A person inhabits the body of a great bear, surrendering upright posture, sinking hands to earth and prowling forward with primal deliberate weight", "identifier": "test260608_p06_r3"},

    # --- P07: Emotion-led ---
    {"prompt": "A person slowly and heavily crouches down onto all four limbs, moving with the calm unhurried confidence of a large animal",  "identifier": "test260608_p07_r1"},
    {"prompt": "A person slowly and heavily crouches down onto all four limbs, moving with the calm unhurried confidence of a large animal",  "identifier": "test260608_p07_r2"},
    {"prompt": "A person slowly and heavily crouches down onto all four limbs, moving with the calm unhurried confidence of a large animal",  "identifier": "test260608_p07_r3"},

    # --- P08: Instructional/stage-direction ---
    {"prompt": "A person performs a bear walk: feet flat, hips high, arms straight, moving opposite hand and foot together across the ground", "identifier": "test260608_p08_r1"},
    {"prompt": "A person performs a bear walk: feet flat, hips high, arms straight, moving opposite hand and foot together across the ground", "identifier": "test260608_p08_r2"},
    {"prompt": "A person performs a bear walk: feet flat, hips high, arms straight, moving opposite hand and foot together across the ground", "identifier": "test260608_p08_r3"},

    # --- P09: Metaphorical ---
    {"prompt": "A person becomes four-legged, their spine horizontal, knuckles grazing the floor as they lumber forward like something wild and ancient", "identifier": "test260608_p09_r1"},
    {"prompt": "A person becomes four-legged, their spine horizontal, knuckles grazing the floor as they lumber forward like something wild and ancient", "identifier": "test260608_p09_r2"},
    {"prompt": "A person becomes four-legged, their spine horizontal, knuckles grazing the floor as they lumber forward like something wild and ancient", "identifier": "test260608_p09_r3"},

    # --- P10: Verbose/cinematic ---
    {"prompt": "A person portraying a bear in a theatrical performance lowers themselves deliberately from a standing position onto both hands and feet, their torso parallel to the ground, then walks forward in a slow rhythmic quadrupedal motion, shoulders rolling with each step", "identifier": "test260608_p10_r1"},
    {"prompt": "A person portraying a bear in a theatrical performance lowers themselves deliberately from a standing position onto both hands and feet, their torso parallel to the ground, then walks forward in a slow rhythmic quadrupedal motion, shoulders rolling with each step", "identifier": "test260608_p10_r2"},
    {"prompt": "A person portraying a bear in a theatrical performance lowers themselves deliberately from a standing position onto both hands and feet, their torso parallel to the ground, then walks forward in a slow rhythmic quadrupedal motion, shoulders rolling with each step", "identifier": "test260608_p10_r3"},
]

KIMODO_URL = "http://127.0.0.1:42069/generate"

print(f"Starting batch execution of {len(animation_batch)} animations...\n")
print("Prompt spectrum: P01=Minimal → P10=Cinematic\n")
print("-" * 60)

for idx, item in enumerate(animation_batch):
    print(f"[{idx+1:02d}/{len(animation_batch)}] {item['identifier']}")
    print(f"         \"{item['prompt'][:80]}{'...' if len(item['prompt']) > 80 else ''}\"")

    payload = {
        "prompt": item["prompt"],
        "filename_prefix": item["identifier"] 
    }

    try:
        t_start = time.time()
        response = requests.post(KIMODO_URL, json=payload)
        elapsed = time.time() - t_start

        if response.status_code == 200:
            print(f"         ✓ Done in {elapsed:.1f}s")
        else:
            print(f"         ✗ Status {response.status_code} after {elapsed:.1f}s")

    except Exception as e:
        print(f"         ✗ Connection error: {e}")

    time.sleep(0.5)

print("-" * 60)
print(f"\nBatch complete. Check 'pipeline_timing_log.txt' for full metrics.")