import requests
import time

# =============================================================================
# KIMODO BENCHMARK BATCH 2 — 2026-06-09
# Experiments: Verb Swap (A) + Word Order (B)
# Contexts spread across theatrical action families
#
# EXPERIMENT A — VERB SWAP (30 files)
#   Fixed skeleton per context, only the action verb changes
#   10 verbs × 3 reps = 30 files
#   Contexts: quadrupedal locomotion, upper-body gesture,
#             theatrical collapse, performative walk
#
# EXPERIMENT B — WORD ORDER (27 files)
#   Same vocabulary, 3 sentence structures × 3 reps × 3 contexts = 27 files
#   Structures: posture-first, verb-first, end-placed
#   Contexts: quadrupedal locomotion, upper-body gesture, theatrical transition
# =============================================================================

animation_batch = [

    # =========================================================================
    # EXPERIMENT A — VERB SWAP
    # =========================================================================

    # --- A01: walks — quadrupedal locomotion ---
    {"prompt": "A person walks on all fours, hands flat on the ground",                                   "identifier": "test260609_A_v01_walks_r1.bvh"},
    {"prompt": "A person walks on all fours, hands flat on the ground",                                   "identifier": "test260609_A_v01_walks_r2.bvh"},
    {"prompt": "A person walks on all fours, hands flat on the ground",                                   "identifier": "test260609_A_v01_walks_r3.bvh"},

    # --- A02: crawls — quadrupedal locomotion ---
    {"prompt": "A person crawls on all fours, hands flat on the ground",                                  "identifier": "test260609_A_v02_crawls_r1.bvh"},
    {"prompt": "A person crawls on all fours, hands flat on the ground",                                  "identifier": "test260609_A_v02_crawls_r2.bvh"},
    {"prompt": "A person crawls on all fours, hands flat on the ground",                                  "identifier": "test260609_A_v02_crawls_r3.bvh"},

    # --- A03: lumbers — quadrupedal locomotion ---
    {"prompt": "A person lumbers on all fours, hands flat on the ground",                                 "identifier": "test260609_A_v03_lumbers_r1.bvh"},
    {"prompt": "A person lumbers on all fours, hands flat on the ground",                                 "identifier": "test260609_A_v03_lumbers_r2.bvh"},
    {"prompt": "A person lumbers on all fours, hands flat on the ground",                                 "identifier": "test260609_A_v03_lumbers_r3.bvh"},

    # --- A04: reaches — upper-body gesture ---
    {"prompt": "A person reaches both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v04_reaches_r1.bvh"},
    {"prompt": "A person reaches both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v04_reaches_r2.bvh"},
    {"prompt": "A person reaches both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v04_reaches_r3.bvh"},

    # --- A05: extends — upper-body gesture ---
    {"prompt": "A person extends both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v05_extends_r1.bvh"},
    {"prompt": "A person extends both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v05_extends_r2.bvh"},
    {"prompt": "A person extends both arms outward toward somebody in front of them",                     "identifier": "test260609_A_v05_extends_r3.bvh"},

    # --- A06: stretches — upper-body gesture ---
    {"prompt": "A person stretches both arms outward toward somebody in front of them",                   "identifier": "test260609_A_v06_stretches_r1.bvh"},
    {"prompt": "A person stretches both arms outward toward somebody in front of them",                   "identifier": "test260609_A_v06_stretches_r2.bvh"},
    {"prompt": "A person stretches both arms outward toward somebody in front of them",                   "identifier": "test260609_A_v06_stretches_r3.bvh"},

    # --- A07: collapses — theatrical transition ---
    {"prompt": "A person collapses to the floor and lies still",                                          "identifier": "test260609_A_v07_collapses_r1.bvh"},
    {"prompt": "A person collapses to the floor and lies still",                                          "identifier": "test260609_A_v07_collapses_r2.bvh"},
    {"prompt": "A person collapses to the floor and lies still",                                          "identifier": "test260609_A_v07_collapses_r3.bvh"},

    # --- A08: sinks — theatrical transition ---
    {"prompt": "A person sinks to the floor and lies still",                                              "identifier": "test260609_A_v08_sinks_r1.bvh"},
    {"prompt": "A person sinks to the floor and lies still",                                              "identifier": "test260609_A_v08_sinks_r2.bvh"},
    {"prompt": "A person sinks to the floor and lies still",                                              "identifier": "test260609_A_v08_sinks_r3.bvh"},

    # --- A09: strides — performative walk ---
    {"prompt": "A person strides slowly across the stage with arms held out to the sides",                "identifier": "test260609_A_v09_strides_r1.bvh"},
    {"prompt": "A person strides slowly across the stage with arms held out to the sides",                "identifier": "test260609_A_v09_strides_r2.bvh"},
    {"prompt": "A person strides slowly across the stage with arms held out to the sides",                "identifier": "test260609_A_v09_strides_r3.bvh"},

    # --- A10: walks — same verb as A01, different context (performative walk) ---
    {"prompt": "A person walks slowly across the stage with arms held out to the sides",                  "identifier": "test260609_A_v10_walks_ctx2_r1.bvh"},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides",                  "identifier": "test260609_A_v10_walks_ctx2_r2.bvh"},
    {"prompt": "A person walks slowly across the stage with arms held out to the sides",                  "identifier": "test260609_A_v10_walks_ctx2_r3.bvh"},

    # =========================================================================
    # EXPERIMENT B — WORD ORDER
    # =========================================================================

    # --- Context X: quadrupedal locomotion ---

    # WO-A posture-first
    {"prompt": "A person, on all fours with hands flat on the ground, walks forward",                     "identifier": "test260609_B_X_posture_first_r1.bvh"},
    {"prompt": "A person, on all fours with hands flat on the ground, walks forward",                     "identifier": "test260609_B_X_posture_first_r2.bvh"},
    {"prompt": "A person, on all fours with hands flat on the ground, walks forward",                     "identifier": "test260609_B_X_posture_first_r3.bvh"},

    # WO-B verb-first
    {"prompt": "A person walks forward, dropping onto all fours with hands flat on the ground",           "identifier": "test260609_B_X_verb_first_r1.bvh"},
    {"prompt": "A person walks forward, dropping onto all fours with hands flat on the ground",           "identifier": "test260609_B_X_verb_first_r2.bvh"},
    {"prompt": "A person walks forward, dropping onto all fours with hands flat on the ground",           "identifier": "test260609_B_X_verb_first_r3.bvh"},

    # WO-C end-placed
    {"prompt": "A person walks forward and lowers into a quadrupedal position, hands flat on the ground", "identifier": "test260609_B_X_end_placed_r1.bvh"},
    {"prompt": "A person walks forward and lowers into a quadrupedal position, hands flat on the ground", "identifier": "test260609_B_X_end_placed_r2.bvh"},
    {"prompt": "A person walks forward and lowers into a quadrupedal position, hands flat on the ground", "identifier": "test260609_B_X_end_placed_r3.bvh"},

    # --- Context Y: upper-body gesture ---

    # WO-A posture-first
    {"prompt": "A person, standing upright with arms at their sides, slowly raises both arms above their head",            "identifier": "test260609_B_Y_posture_first_r1.bvh"},
    {"prompt": "A person, standing upright with arms at their sides, slowly raises both arms above their head",            "identifier": "test260609_B_Y_posture_first_r2.bvh"},
    {"prompt": "A person, standing upright with arms at their sides, slowly raises both arms above their head",            "identifier": "test260609_B_Y_posture_first_r3.bvh"},

    # WO-B verb-first
    {"prompt": "A person slowly raises both arms above their head, starting from a standing position with arms at their sides", "identifier": "test260609_B_Y_verb_first_r1.bvh"},
    {"prompt": "A person slowly raises both arms above their head, starting from a standing position with arms at their sides", "identifier": "test260609_B_Y_verb_first_r2.bvh"},
    {"prompt": "A person slowly raises both arms above their head, starting from a standing position with arms at their sides", "identifier": "test260609_B_Y_verb_first_r3.bvh"},

    # WO-C end-placed
    {"prompt": "A person stands upright and brings both arms upward until they are raised fully above their head",         "identifier": "test260609_B_Y_end_placed_r1.bvh"},
    {"prompt": "A person stands upright and brings both arms upward until they are raised fully above their head",         "identifier": "test260609_B_Y_end_placed_r2.bvh"},
    {"prompt": "A person stands upright and brings both arms upward until they are raised fully above their head",         "identifier": "test260609_B_Y_end_placed_r3.bvh"},

    # --- Context Z: theatrical transition ---

    # WO-A posture-first
    {"prompt": "A person, standing with arms at their sides, slowly kneels down and bows their head",     "identifier": "test260609_B_Z_posture_first_r1.bvh"},
    {"prompt": "A person, standing with arms at their sides, slowly kneels down and bows their head",     "identifier": "test260609_B_Z_posture_first_r2.bvh"},
    {"prompt": "A person, standing with arms at their sides, slowly kneels down and bows their head",     "identifier": "test260609_B_Z_posture_first_r3.bvh"},

    # WO-B verb-first
    {"prompt": "A person slowly kneels down and bows their head, lowering from a standing position with arms at their sides", "identifier": "test260609_B_Z_verb_first_r1.bvh"},
    {"prompt": "A person slowly kneels down and bows their head, lowering from a standing position with arms at their sides", "identifier": "test260609_B_Z_verb_first_r2.bvh"},
    {"prompt": "A person slowly kneels down and bows their head, lowering from a standing position with arms at their sides", "identifier": "test260609_B_Z_verb_first_r3.bvh"},

    # WO-C end-placed
    {"prompt": "A person stands upright, then lowers themselves until kneeling with their head bowed",    "identifier": "test260609_B_Z_end_placed_r1.bvh"},
    {"prompt": "A person stands upright, then lowers themselves until kneeling with their head bowed",    "identifier": "test260609_B_Z_end_placed_r2.bvh"},
    {"prompt": "A person stands upright, then lowers themselves until kneeling with their head bowed",    "identifier": "test260609_B_Z_end_placed_r3.bvh"},

]

# -----------------------------------------------------------------------------
KIMODO_URL = "http://127.0.0.1:42069/generate"

print(f"Starting batch execution: {len(animation_batch)} animations total")
print(f"  Experiment A (Verb Swap):  30 files")
print(f"  Experiment B (Word Order): 27 files")
print("-" * 64)

for idx, item in enumerate(animation_batch):
    if idx == 0:
        print("EXPERIMENT A — VERB SWAP")
        print("-" * 64)
    elif idx == 30:
        print("-" * 64)
        print("EXPERIMENT B — WORD ORDER")
        print("-" * 64)

    print(f"[{idx+1:02d}/{len(animation_batch)}] {item['identifier']}")
    print(f"         \"{item['prompt']}\"")

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

print("-" * 64)
print(f"\nBatch complete. Check 'pipeline_timing_log.txt' for full metrics.")