from __future__ import annotations

import json
from pathlib import Path

from dafx26_demo.references import load_reference_manifest


def test_reference_manifest_has_provenance_fields() -> None:
    rows = load_reference_manifest()
    assert len(rows) >= 6
    required = {
        "id",
        "dataset",
        "composer",
        "genre",
        "split",
        "work",
        "excerpt_start_sec",
        "excerpt_end_sec",
        "license",
        "local_relpath",
    }
    for row in rows:
        assert required <= set(row)
        assert row["dataset"] in {"ASAP", "MAESTRO"}
        assert row["kind"] == "training_reference"


def test_manifest_file_is_json(tmp_path: Path) -> None:
    path = Path("dafx26_demo/data/reference_performances.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["description"]
    assert payload["items"]
