# _old/ - archived, not live

Moved here with `git mv` on 2026-10-09, so `git log --follow` still shows each file's
history. Nothing here is imported, loaded or run by the live code; nothing here is
edited (CLAUDE.md: "never edit the old Kimodo services"). It is kept as reference.

| Path | What it was | Replaced by |
|---|---|---|
| `kimodo/` | First Kimodo playground: batch generation script (all commented out), uv project files | `remote_kimodo_service/` (generation on the Spark) |
| `pipeline-network-editor/` | Kimodo service → BVH → Blender → FBX → Unreal import route, batch runners, timing logs | `remote_kimodo_service/` + `procedural_animation/` (retarget in Python, no FBX) |
| `pipeline-network-osc/` | Kimodo services that streamed swizzled quaternions over OSC to a dummy skeleton; `kimodo_service_handoff_osc_v3.py` is what `remote_kimodo_service/kimodo_service.py` was rewritten from | `remote_kimodo_service/kimodo_service.py` |
| `mirroring/ue_plugin/` | The plugin's original home, stale since the build moved | `K:\KaiTracking\Plugins\MediaPipeLiveLink` (edit the C++ there) |
| `mirroring/live_link_pose_json_protocol.py` | JSON pose wire format, never wired in; imported nowhere | `mirroring/live_link_pose_osc_protocol.py` |
| `mirroring/pose_landmarker_full.task` | Pose-only MediaPipe model from before the Holistic migration; loaded by nothing | `holistic_landmarker.task` |

What did NOT move, although it looks old:

- **The data the tests and the retarget need** moved to `assets/kimodo/` instead:
  the SOMA77 T-pose BVH (`soma_skeleton/`, the retarget skeleton) and the 243 library
  clips (`clips/kimodo-gen`, `clips/editor-gen`) that `procedural_checks.py` and the
  fake Kimodo server read.
- `mirroring/mediapipe_pose_osc_protocol.py` and `mediapipe_pose_capture.py`:
  `conductor.py` imports them at module load, and `conductor.py` is not edited.
- `mirroring/head_pose_capture.py` and `face_landmarker.task`: disabled on purpose, kept
  for the face-crop experiment (CLAUDE.md).
- `procedural_animation/procedural_conductor_bvh.py`: the frozen Phase 0 fallback; it
  uses package-relative imports and only works inside `procedural_animation/`.

On the Spark, the service's old interpreter stays at `pipeline-network-osc/venv`: it is
untracked, so `git pull` leaves it where it was (ignored by the root `.gitignore`). Its
replacement is `remote_kimodo_service/venv` (see `remote_kimodo_service/requirements.txt`).
