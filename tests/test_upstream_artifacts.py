from __future__ import annotations

from dafx26_demo.artifacts import MIN_MIDI_BYTES, index_generated_artifacts
from dafx26_demo.paths import UPSTREAM_ROOT


def test_upstream_artifacts_are_labeled_generated_and_skip_tiny_files() -> None:
    items = index_generated_artifacts(UPSTREAM_ROOT / "artifacts")
    assert items
    assert all(item.kind == "generated" for item in items)
    assert all(item.midi_path.stat().st_size >= MIN_MIDI_BYTES for item in items)
    names = {item.midi_path.name for item in items}
    assert "r6_cond_004_11.mid" not in names  # known 212-byte failed Chopin ballade
