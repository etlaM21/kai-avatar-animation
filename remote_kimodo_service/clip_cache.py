"""Every clip the Spark returns, on disk. If the Spark is unreachable, this is the library.

Keyed by GenerationRequest.cache_key() (model, prompt, seed, frame count, steps), so a
repeated request is answered from disk and never touches the network. File names keep
the prompt readable; the metadata inside each NPZ is what /list shows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .kimodo_contract import ContractError, GenerationRequest, KimodoClip, unpack

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_HASH_LEN = 12


@dataclass
class CacheEntry:
    path: Path
    meta: dict

    @property
    def label(self) -> str:
        m = self.meta
        return (f"{m.get('prompt', '?')!r}  seed {m.get('seed', '?')}, {m.get('num_frames', '?')} frames, "
                f"{m.get('steps', '?')} steps")


def _slug(prompt: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:limit] or "clip"


class ClipCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)

    def _matches(self, request: GenerationRequest) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob(f"*__{request.cache_key()[:_HASH_LEN]}.npz"))

    def get(self, request: GenerationRequest) -> bytes | None:
        for path in self._matches(request):
            try:
                return path.read_bytes()
            except OSError:
                continue
        return None

    def put(self, request: GenerationRequest, data: bytes) -> Path:
        """Atomic: a crash mid-write leaves a .tmp file, never a truncated .npz."""
        self.root.mkdir(parents=True, exist_ok=True)
        name = (f"{_slug(request.prompt)}__s{request.seed}__{request.num_frames}f__{request.steps}st"
                f"__{request.cache_key()[:_HASH_LEN]}.npz")
        path = self.root / name
        tmp = path.with_suffix(".npz.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path

    def entries(self) -> list[CacheEntry]:
        """Readable clips, oldest first. Unreadable files are skipped, not fatal."""
        if not self.root.is_dir():
            return []
        out: list[CacheEntry] = []
        for path in sorted(self.root.glob("*.npz"), key=lambda p: p.stat().st_mtime):
            try:
                out.append(CacheEntry(path=path, meta=unpack(path.read_bytes()).meta))
            except (ContractError, OSError):
                continue
        return out

    @staticmethod
    def load(path: Path) -> KimodoClip:
        return unpack(Path(path).read_bytes())
