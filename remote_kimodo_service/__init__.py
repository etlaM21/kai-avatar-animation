"""Remote Kimodo lane: generation on the DGX Spark, everything Manny-related on Windows.

    kimodo_service.py   runs on the Spark (FastAPI, warm model) - see README.md
    requirements.txt    the Spark side's packages
    kimodo_contract.py  the NPZ both sides agree on (numpy only, imported by both)
    kimodo_adapter.py   NPZ -> the existing, tested Retargeter (Windows)
    kimodo_client.py    HTTP client, never raises anything but its own one-line errors
    clip_cache.py       every returned clip on disk; the offline library
    player.py           60 Hz loop player feeding the existing encoders

The Windows modules run on the mirroring venv via procedural_animation.procedural_conductor.
"""
