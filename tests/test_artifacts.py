from __future__ import annotations

from pathlib import Path

from dafx26_demo.artifacts import MIN_MIDI_BYTES, index_generated_artifacts, is_featured_artifact


def test_tiny_midi_files_are_not_featured(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.mid"
    tiny.write_bytes(b"MThd" + b"\x00" * 20)
    assert tiny.stat().st_size < MIN_MIDI_BYTES
    assert is_featured_artifact(tiny, tmp_path / "tiny.wav") is False


def test_index_skips_missing_or_tiny_pairs(tmp_path: Path) -> None:
    root = tmp_path / "asap_note" / "Chopin" / "etude"
    root.mkdir(parents=True)
    good_mid = root / "ok.mid"
    good_wav = root / "oksample.wav"
    good_mid.write_bytes(b"M" * (MIN_MIDI_BYTES + 10))
    good_wav.write_bytes(b"R" * 1024)
    tiny_mid = root / "tiny.mid"
    tiny_wav = root / "tinysample.wav"
    tiny_mid.write_bytes(b"MThd")
    tiny_wav.write_bytes(b"R" * 1024)
    items = index_generated_artifacts(tmp_path)
    names = {item.midi_path.name for item in items}
    assert "ok.mid" in names
    assert "tiny.mid" not in names
    assert all(item.kind == "generated" for item in items)
    assert all(item.composer == "Chopin" for item in items)
