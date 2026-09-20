from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MIN_MIDI_BYTES = 500
MIN_WAV_BYTES = 512


@dataclass(frozen=True)
class GeneratedArtifact:
    mode: str
    composer: str
    genre: str
    midi_path: Path
    wav_path: Path | None
    kind: str = "generated"


def is_featured_artifact(midi_path: Path, wav_path: Path | None) -> bool:
    if not midi_path.is_file() or midi_path.stat().st_size < MIN_MIDI_BYTES:
        return False
    if wav_path is None or not wav_path.is_file() or wav_path.stat().st_size < MIN_WAV_BYTES:
        return False
    return True


def index_generated_artifacts(root: Path) -> list[GeneratedArtifact]:
    items: list[GeneratedArtifact] = []
    if not root.is_dir():
        return items
    for midi_path in sorted(root.glob("asap_*/*/*/*.mid")):
        rel = midi_path.relative_to(root)
        mode = rel.parts[0].removeprefix("asap_")
        composer = rel.parts[1]
        genre = rel.parts[2]
        wav_path = midi_path.with_name(midi_path.stem + "sample.wav")
        if not is_featured_artifact(midi_path, wav_path if wav_path.is_file() else None):
            continue
        items.append(
            GeneratedArtifact(
                mode=mode,
                composer=composer,
                genre=genre,
                midi_path=midi_path,
                wav_path=wav_path,
                kind="generated",
            )
        )
    return items
