"""Procedural animation lane: generated motion (Kimodo, later ARDY) retargeted onto
the 60 Manny bones that `mirroring` streams. See mirroring/procedural-animation.md.

Lives beside `mirroring`, not inside it, but runs on the mirroring venv and imports
pose_solver and the two protocol modules from there read-only. Nothing in
`mirroring` is edited; this shim only makes it importable.
"""

from __future__ import annotations
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
MIRRORING_DIR = MODULE_ROOT / "mirroring"

if str(MIRRORING_DIR) not in sys.path:
    sys.path.insert(0, str(MIRRORING_DIR))
