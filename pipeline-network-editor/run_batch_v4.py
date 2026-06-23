import requests
import time

# =============================================================================
# KIMODO BENCHMARK — ADVERB TEST 2 — 2026-06-23
#
# 3 base prompts × 3 categories × 3 words = 27 generations
# Same seed across all (single-seed comparison, no rep variance)
#
# Categories & words (as specified):
#   emotion/expression: joyful, depressed, angry
#   physicality:         heavy, light, broken
#   speed:               slow, quick, hectic
#
# Concurrent physical detail per learnings so far:
#   verb + immediate physical state, no transitions, no preconditions
#
# Filenames: test260623_x_y_z
#   x = base prompt number (1, 2, 3)
#   y = category number (1=emotion, 2=physicality, 3=speed)
#   z = the adverb word
# =============================================================================

SEED = 420

animation_batch = [

    # =========================================================================
    # BASE 1 — JUMPING JACKS
    # =========================================================================

    # --- y1: emotion ---
    {"prompt": "A person does jumping jacks, arms and legs snapping outward with bright joyful energy",        "filename_prefix": "test260623_1_1_joyful"},
    {"prompt": "A person does jumping jacks, arms and legs moving with low heavy depressed effort",            "filename_prefix": "test260623_1_1_depressed"},
    {"prompt": "A person does jumping jacks, arms and legs snapping outward with sharp angry tension",         "filename_prefix": "test260623_1_1_angry"},

    # --- y2: physicality ---
    {"prompt": "A person does jumping jacks, body heavy and landing hard with each jump",                      "filename_prefix": "test260623_1_2_heavy"},
    {"prompt": "A person does jumping jacks, body light and barely touching the ground with each jump",       "filename_prefix": "test260623_1_2_light"},
    {"prompt": "A person does jumping jacks, limbs jerking in broken uneven uncoordinated motion",             "filename_prefix": "test260623_1_2_broken"},

    # --- y3: speed ---
    {"prompt": "A person does jumping jacks slowly, arms and legs moving with unhurried deliberate pace",      "filename_prefix": "test260623_1_3_slow"},
    {"prompt": "A person does jumping jacks quickly, arms and legs snapping out at a brisk rapid pace",        "filename_prefix": "test260623_1_3_quick"},
    {"prompt": "A person does jumping jacks in a hectic frenzy, arms and legs flailing rapidly out of control", "filename_prefix": "test260623_1_3_hectic"},

    # =========================================================================
    # BASE 2 — WALKING LIKE A BEAR
    # =========================================================================

    # --- y1: emotion ---
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, body loose with joyful energy",      "filename_prefix": "test260623_2_1_joyful"},
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, body low and heavy with depressed weight", "filename_prefix": "test260623_2_1_depressed"},
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, limbs pressing down with angry force",     "filename_prefix": "test260623_2_1_angry"},

    # --- y2: physicality ---
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, body sinking with heavy full weight into each step", "filename_prefix": "test260623_2_2_heavy"},
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, body barely skimming the ground with light steps",   "filename_prefix": "test260623_2_2_light"},
    {"prompt": "A person walks on all fours like a bear, hands flat on the ground, limbs moving in broken uneven jerky steps",           "filename_prefix": "test260623_2_2_broken"},

    # --- y3: speed ---
    {"prompt": "A person walks slowly on all fours like a bear, hands flat on the ground, limbs moving at an unhurried pace",   "filename_prefix": "test260623_2_3_slow"},
    {"prompt": "A person walks quickly on all fours like a bear, hands flat on the ground, limbs moving at a brisk pace",       "filename_prefix": "test260623_2_3_quick"},
    {"prompt": "A person walks on all fours like a bear in a hectic rush, limbs moving rapidly and erratically",                "filename_prefix": "test260623_2_3_hectic"},

    # =========================================================================
    # BASE 3 — SITTING DOWN
    # =========================================================================

    # --- y1: emotion ---
    {"prompt": "A person sits down, body settling with light joyful ease",                          "filename_prefix": "test260623_3_1_joyful"},
    {"prompt": "A person sits down, body sinking with depressed heavy resignation",                 "filename_prefix": "test260623_3_1_depressed"},
    {"prompt": "A person sits down, body dropping into the seat with sharp angry force",            "filename_prefix": "test260623_3_1_angry"},

    # --- y2: physicality ---
    {"prompt": "A person sits down, body heavy and dropping hard onto the seat",                    "filename_prefix": "test260623_3_2_heavy"},
    {"prompt": "A person sits down, body light and floating gently down onto the seat",             "filename_prefix": "test260623_3_2_light"},
    {"prompt": "A person sits down, limbs moving in broken uneven uncoordinated stages",             "filename_prefix": "test260623_3_2_broken"},

    # --- y3: speed ---
    {"prompt": "A person sits down slowly, body lowering at an unhurried deliberate pace",           "filename_prefix": "test260623_3_3_slow"},
    {"prompt": "A person sits down quickly, body dropping onto the seat at a brisk rapid pace",      "filename_prefix": "test260623_3_3_quick"},
    {"prompt": "A person sits down in a hectic rush, body dropping onto the seat rapidly and erratically", "filename_prefix": "test260623_3_3_hectic"},

]

# -----------------------------------------------------------------------------
KIMODO_URL = "http://127.0.0.1:42069/generate"

print(f"Starting batch: {len(animation_batch)} animations")
print(f"  3 base prompts × 3 categories × 3 adverbs = 27 generations")
print(f"  Single seed ({SEED}) across all generations")
print("-" * 64)

for idx, item in enumerate(animation_batch):
    if   idx ==  0: print("BASE 1 — JUMPING JACKS\n" + "-" * 64)
    elif idx ==  9: print("-" * 64 + "\nBASE 2 — WALKING LIKE A BEAR\n" + "-" * 64)
    elif idx == 18: print("-" * 64 + "\nBASE 3 — SITTING DOWN\n" + "-" * 64)

    print(f"[{idx+1:02d}/{len(animation_batch)}] {item['filename_prefix']}")
    print(f"         \"{item['prompt']}\"")

    payload = {
        "prompt":          item["prompt"],
        "filename_prefix": item["filename_prefix"],
        "fps":             30,
        "duration":        9.0,
        "steps":           100,
        "seed":            SEED
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

print("-" * 64)
print(f"\nBatch complete.")