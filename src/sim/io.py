"""Atomic file writes for cache entries and dialogue logs.

A reader (``sim eval``, rsync, ``--resume``) must never see half a file: write
to a unique temporary name in the same directory and ``rename`` onto the
destination, which is atomic on one filesystem.
"""

from pathlib import Path
from uuid import uuid4


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, replacing it whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
