from __future__ import annotations

from pathlib import Path
import shutil


def prune_outputs(root: Path, keep: int = 40) -> int:
    if keep < 1 or not root.is_dir():
        return 0
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    extra = dirs[:-keep]
    for path in extra:
        shutil.rmtree(path, ignore_errors=True)
    return len(extra)
