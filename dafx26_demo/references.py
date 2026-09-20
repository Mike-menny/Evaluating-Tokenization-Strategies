from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from dafx26_demo.paths import DATA_ROOT

MANIFEST_PATH = DATA_ROOT / "reference_performances.json"


def load_reference_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads((path or MANIFEST_PATH).read_text(encoding="utf-8"))
    items = list(payload["items"])
    for item in items:
        item.setdefault("kind", "training_reference")
    return items
