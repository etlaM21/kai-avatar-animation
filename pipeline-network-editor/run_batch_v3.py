import requests
import time

# =============================================================================
# KIMODO BENCHMARK — ADVERB/MANNER TEST 1 — 2026-06-11
#
# Structure: verb + concurrent physical detail (not adverb on verb)
# 4 base verbs × 8 manner descriptors × 3 seeds = 96 generations
#
# Seeds: r1=420, r2=69, r3=67 (fixed for comparability)
#
# Manner descriptor categories:
#   Emotional:      sadly, joyfully
#   Arousal/energy: frantically, languidly
#   Physical weight: heavily, lightly
#   Intention:      hesitantly, deliberately
#
# Base verbs (all confirmed 10-scorers):
#   V1 — quadrupedal:    "A person walks on all fours, hands flat on the ground"
#   V2 — gesture:        "A person reaches both arms outward toward somebody in front of them"
#   V3 — collapse:       "A person collapses to the floor and lies still"
#   V4 — performative:   "A person walks slowly across the stage with arms held out to the sides"
# =============================================================================

SEEDS = [420, 69, 67]

animation_batch = [

    # =========================================================================
    # V1 — QUADRUPEDAL
    # =========================================================================

    # --- Emotional: sadly ---
    {"prompt": "A person walks on all fours, hands flat on the ground, body low and heavy with grief",         "filename_prefix": "test260611_v1_sadly_r1",        "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body low and heavy with grief",         "filename_prefix": "test260611_v1_sadly_r2",        "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body low and heavy with grief",         "filename_prefix": "test260611_v1_sadly_r3",        "seed": SEEDS[2]},

    # --- Emotional: joyfully ---
    {"prompt": "A person walks on all fours, hands flat on the ground, body light and buoyant with excitement", "filename_prefix": "test260611_v1_joyfully_r1",     "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body light and buoyant with excitement", "filename_prefix": "test260611_v1_joyfully_r2",     "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body light and buoyant with excitement", "filename_prefix": "test260611_v1_joyfully_r3",     "seed": SEEDS[2]},

    # --- Arousal: frantically ---
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in rapid urgent bursts",   "filename_prefix": "test260611_v1_frantically_r1",  "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in rapid urgent bursts",   "filename_prefix": "test260611_v1_frantically_r2",  "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in rapid urgent bursts",   "filename_prefix": "test260611_v1_frantically_r3",  "seed": SEEDS[2]},

    # --- Arousal: languidly ---
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in slow unhurried ease",   "filename_prefix": "test260611_v1_languidly_r1",    "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in slow unhurried ease",   "filename_prefix": "test260611_v1_languidly_r2",    "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, limbs moving in slow unhurried ease",   "filename_prefix": "test260611_v1_languidly_r3",    "seed": SEEDS[2]},

    # --- Physical weight: heavily ---
    {"prompt": "A person walks on all fours, hands flat on the ground, body sinking with full weight into each step", "filename_prefix": "test260611_v1_heavily_r1",      "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body sinking with full weight into each step", "filename_prefix": "test260611_v1_heavily_r2",      "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body sinking with full weight into each step", "filename_prefix": "test260611_v1_heavily_r3",      "seed": SEEDS[2]},

    # --- Physical weight: lightly ---
    {"prompt": "A person walks on all fours, hands flat on the ground, body barely skimming the surface with each step", "filename_prefix": "test260611_v1_lightly_r1",      "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body barely skimming the surface with each step", "filename_prefix": "test260611_v1_lightly_r2",      "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, body barely skimming the surface with each step", "filename_prefix": "test260611_v1_lightly_r3",      "seed": SEEDS[2]},

    # --- Intention: hesitantly ---
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with uncertain pausing hesitation", "filename_prefix": "test260611_v1_hesitantly_r1",  "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with uncertain pausing hesitation", "filename_prefix": "test260611_v1_hesitantly_r2",  "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with uncertain pausing hesitation", "filename_prefix": "test260611_v1_hesitantly_r3",  "seed": SEEDS[2]},

    # --- Intention: deliberately ---
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with precise controlled intention", "filename_prefix": "test260611_v1_deliberately_r1", "seed": SEEDS[0]},
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with precise controlled intention", "filename_prefix": "test260611_v1_deliberately_r2", "seed": SEEDS[1]},
    {"prompt": "A person walks on all fours, hands flat on the ground, each limb placed with precise controlled intention", "filename_prefix": "test260611_v1_deliberately_r3", "seed": SEEDS[2]},

    # =========================================================================
    # V2 — GESTURE
    # =========================================================================

    # --- Emotional: sadly ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest collapsed and shoulders drawn inward with sorrow", "filename_prefix": "test260611_v2_sadly_r1",        "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest collapsed and shoulders drawn inward with sorrow", "filename_prefix": "test260611_v2_sadly_r2",        "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest collapsed and shoulders drawn inward with sorrow", "filename_prefix": "test260611_v2_sadly_r3",        "seed": SEEDS[2]},

    # --- Emotional: joyfully ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest open and lifted with warmth and delight",          "filename_prefix": "test260611_v2_joyfully_r1",     "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest open and lifted with warmth and delight",          "filename_prefix": "test260611_v2_joyfully_r2",     "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, chest open and lifted with warmth and delight",          "filename_prefix": "test260611_v2_joyfully_r3",     "seed": SEEDS[2]},

    # --- Arousal: frantically ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms moving in rapid tense urgent extension",            "filename_prefix": "test260611_v2_frantically_r1",  "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms moving in rapid tense urgent extension",            "filename_prefix": "test260611_v2_frantically_r2",  "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms moving in rapid tense urgent extension",            "filename_prefix": "test260611_v2_frantically_r3",  "seed": SEEDS[2]},

    # --- Arousal: languidly ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms drifting in slow fluid unhurried extension",        "filename_prefix": "test260611_v2_languidly_r1",    "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms drifting in slow fluid unhurried extension",        "filename_prefix": "test260611_v2_languidly_r2",    "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms drifting in slow fluid unhurried extension",        "filename_prefix": "test260611_v2_languidly_r3",    "seed": SEEDS[2]},

    # --- Physical weight: heavily ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms heavy and dropping under their own full weight",    "filename_prefix": "test260611_v2_heavily_r1",      "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms heavy and dropping under their own full weight",    "filename_prefix": "test260611_v2_heavily_r2",      "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms heavy and dropping under their own full weight",    "filename_prefix": "test260611_v2_heavily_r3",      "seed": SEEDS[2]},

    # --- Physical weight: lightly ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms weightless and floating in gentle extension",       "filename_prefix": "test260611_v2_lightly_r1",      "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms weightless and floating in gentle extension",       "filename_prefix": "test260611_v2_lightly_r2",      "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms weightless and floating in gentle extension",       "filename_prefix": "test260611_v2_lightly_r3",      "seed": SEEDS[2]},

    # --- Intention: hesitantly ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with trembling uncertain hesitation",     "filename_prefix": "test260611_v2_hesitantly_r1",  "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with trembling uncertain hesitation",     "filename_prefix": "test260611_v2_hesitantly_r2",  "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with trembling uncertain hesitation",     "filename_prefix": "test260611_v2_hesitantly_r3",  "seed": SEEDS[2]},

    # --- Intention: deliberately ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with slow precise controlled intention",  "filename_prefix": "test260611_v2_deliberately_r1", "seed": SEEDS[0]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with slow precise controlled intention",  "filename_prefix": "test260611_v2_deliberately_r2", "seed": SEEDS[1]},
    {"prompt": "A person reaches both arms outward toward somebody in front of them, arms extending with slow precise controlled intention",  "filename_prefix": "test260611_v2_deliberately_r3", "seed": SEEDS[2]},

    # =========================================================================
    # V3 — COLLAPSE
    # =========================================================================

    # --- Emotional: sadly ---
    {"prompt": "A person collapses to the floor and lies still, body folding inward with exhausted grief",                                   "filename_prefix": "test260611_v3_sadly_r1",        "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body folding inward with exhausted grief",                                   "filename_prefix": "test260611_v3_sadly_r2",        "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body folding inward with exhausted grief",                                   "filename_prefix": "test260611_v3_sadly_r3",        "seed": SEEDS[2]},

    # --- Emotional: joyfully ---
    {"prompt": "A person collapses to the floor and lies still, body loose and open with relieved elation",                                  "filename_prefix": "test260611_v3_joyfully_r1",     "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body loose and open with relieved elation",                                  "filename_prefix": "test260611_v3_joyfully_r2",     "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body loose and open with relieved elation",                                  "filename_prefix": "test260611_v3_joyfully_r3",     "seed": SEEDS[2]},

    # --- Arousal: frantically ---
    {"prompt": "A person collapses to the floor and lies still, limbs flailing in sudden uncontrolled rapid descent",                        "filename_prefix": "test260611_v3_frantically_r1",  "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, limbs flailing in sudden uncontrolled rapid descent",                        "filename_prefix": "test260611_v3_frantically_r2",  "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, limbs flailing in sudden uncontrolled rapid descent",                        "filename_prefix": "test260611_v3_frantically_r3",  "seed": SEEDS[2]},

    # --- Arousal: languidly ---
    {"prompt": "A person collapses to the floor and lies still, body slowly melting downward in gradual boneless descent",                   "filename_prefix": "test260611_v3_languidly_r1",    "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body slowly melting downward in gradual boneless descent",                   "filename_prefix": "test260611_v3_languidly_r2",    "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body slowly melting downward in gradual boneless descent",                   "filename_prefix": "test260611_v3_languidly_r3",    "seed": SEEDS[2]},

    # --- Physical weight: heavily ---
    {"prompt": "A person collapses to the floor and lies still, full body weight dropping hard and fast into the ground",                    "filename_prefix": "test260611_v3_heavily_r1",      "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, full body weight dropping hard and fast into the ground",                    "filename_prefix": "test260611_v3_heavily_r2",      "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, full body weight dropping hard and fast into the ground",                    "filename_prefix": "test260611_v3_heavily_r3",      "seed": SEEDS[2]},

    # --- Physical weight: lightly ---
    {"prompt": "A person collapses to the floor and lies still, body floating downward and settling with barely any impact",                 "filename_prefix": "test260611_v3_lightly_r1",      "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body floating downward and settling with barely any impact",                 "filename_prefix": "test260611_v3_lightly_r2",      "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body floating downward and settling with barely any impact",                 "filename_prefix": "test260611_v3_lightly_r3",      "seed": SEEDS[2]},

    # --- Intention: hesitantly ---
    {"prompt": "A person collapses to the floor and lies still, body lowering in reluctant halting uncertain stages",                        "filename_prefix": "test260611_v3_hesitantly_r1",  "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body lowering in reluctant halting uncertain stages",                        "filename_prefix": "test260611_v3_hesitantly_r2",  "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body lowering in reluctant halting uncertain stages",                        "filename_prefix": "test260611_v3_hesitantly_r3",  "seed": SEEDS[2]},

    # --- Intention: deliberately ---
    {"prompt": "A person collapses to the floor and lies still, body lowering with slow measured controlled intention",                      "filename_prefix": "test260611_v3_deliberately_r1", "seed": SEEDS[0]},
    {"prompt": "A person collapses to the floor and lies still, body lowering with slow measured controlled intention",                      "filename_prefix": "test260611_v3_deliberately_r2", "seed": SEEDS[1]},
    {"prompt": "A person collapses to the floor and lies still, body lowering with slow measured controlled intention",                      "filename_prefix": "test260611_v3_deliberately_r3", "seed": SEEDS[2]},

    # =========================================================================
    # V4 — PERFORMATIVE WALK
    # =========================================================================

    # --- Emotional: sadly ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest sunken and head bowed with sorrow",            "filename_prefix": "test260611_v4_sadly_r1",        "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest sunken and head bowed with sorrow",            "filename_prefix": "test260611_v4_sadly_r2",        "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest sunken and head bowed with sorrow",            "filename_prefix": "test260611_v4_sadly_r3",        "seed": SEEDS[2]},

    # --- Emotional: joyfully ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest lifted and face raised with open joy",         "filename_prefix": "test260611_v4_joyfully_r1",     "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest lifted and face raised with open joy",         "filename_prefix": "test260611_v4_joyfully_r2",     "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, chest lifted and face raised with open joy",         "filename_prefix": "test260611_v4_joyfully_r3",     "seed": SEEDS[2]},

    # --- Arousal: frantically ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs tense and trembling with frantic agitation",   "filename_prefix": "test260611_v4_frantically_r1",  "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs tense and trembling with frantic agitation",   "filename_prefix": "test260611_v4_frantically_r2",  "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs tense and trembling with frantic agitation",   "filename_prefix": "test260611_v4_frantically_r3",  "seed": SEEDS[2]},

    # --- Arousal: languidly ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs loose and drifting in total relaxed ease",     "filename_prefix": "test260611_v4_languidly_r1",    "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs loose and drifting in total relaxed ease",     "filename_prefix": "test260611_v4_languidly_r2",    "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, limbs loose and drifting in total relaxed ease",     "filename_prefix": "test260611_v4_languidly_r3",    "seed": SEEDS[2]},

    # --- Physical weight: heavily ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step pressing hard and full into the ground",   "filename_prefix": "test260611_v4_heavily_r1",      "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step pressing hard and full into the ground",   "filename_prefix": "test260611_v4_heavily_r2",      "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step pressing hard and full into the ground",   "filename_prefix": "test260611_v4_heavily_r3",      "seed": SEEDS[2]},

    # --- Physical weight: lightly ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step barely grazing the ground with light feet", "filename_prefix": "test260611_v4_lightly_r1",      "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step barely grazing the ground with light feet", "filename_prefix": "test260611_v4_lightly_r2",      "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step barely grazing the ground with light feet", "filename_prefix": "test260611_v4_lightly_r3",      "seed": SEEDS[2]},

    # --- Intention: hesitantly ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with faltering uncertain hesitation", "filename_prefix": "test260611_v4_hesitantly_r1",  "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with faltering uncertain hesitation", "filename_prefix": "test260611_v4_hesitantly_r2",  "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with faltering uncertain hesitation", "filename_prefix": "test260611_v4_hesitantly_r3",  "seed": SEEDS[2]},

    # --- Intention: deliberately ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with slow precise controlled intention", "filename_prefix": "test260611_v4_deliberately_r1", "seed": SEEDS[0]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with slow precise controlled intention", "filename_prefix": "test260611_v4_deliberately_r2", "seed": SEEDS[1]},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides, each step placed with slow precise controlled intention", "filename_prefix": "test260611_v4_deliberately_r3", "seed": SEEDS[2]},

]

# -----------------------------------------------------------------------------
KIMODO_URL = "http://127.0.0.1:42069/generate"

print(f"Starting batch: {len(animation_batch)} animations")
print(f"  4 verbs × 8 descriptors × 3 seeds = 96 generations")
print("-" * 64)

for idx, item in enumerate(animation_batch):
    # Section headers
    if   idx ==  0: print("V1 — QUADRUPEDAL\n" + "-" * 64)
    elif idx == 24: print("-" * 64 + "\nV2 — GESTURE\n" + "-" * 64)
    elif idx == 48: print("-" * 64 + "\nV3 — COLLAPSE\n" + "-" * 64)
    elif idx == 72: print("-" * 64 + "\nV4 — PERFORMATIVE WALK\n" + "-" * 64)

    print(f"[{idx+1:02d}/{len(animation_batch)}] {item['filename_prefix']}  (seed {item['seed']})")
    print(f"         \"{item['prompt'][:90]}{'...' if len(item['prompt']) > 90 else ''}\"")

    payload = {
        "prompt":          item["prompt"],
        "filename_prefix": item["filename_prefix"],
        "fps":             30,
        "duration":        9.0,
        "steps":           100,
        "seed":            item["seed"]
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