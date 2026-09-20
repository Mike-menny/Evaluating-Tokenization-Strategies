from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from dafx26_demo.schema import GenerationResult


def cache_path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.json"


def load_cached(root: Path, key: str) -> GenerationResult | None:
    path = cache_path(root, key)
    if not path.is_file():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    result = GenerationResult(**payload)
    result.cached = True
    return result


def store_cached(root: Path, key: str, result: GenerationResult) -> None:
    path = cache_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["cached"] = False
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
