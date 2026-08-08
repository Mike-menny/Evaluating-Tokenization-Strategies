#!/usr/bin/env python3
"""
Augment ASAP MIDI files (pitch / time / velocity) like augment_dataset.py.
Writes under <asap-repo>/augmented/ mirroring paths from asap-dataset-master/.
Time-stretch scales both MIDI deltas and matching *_annotations.txt timestamps.
For ``midi_score*.mid`` (score), tempo factors are drawn from a wide range (arbitrary tempo).
For performance MIDIs, tempo factors are restricted to 0.9–1.1.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import mido

# Performance time-stretch: keep near original tempo
PERF_TIME_FACTOR_MIN = 0.9
PERF_TIME_FACTOR_MAX = 1.1

# Score (midi_score) time-stretch: no narrow band — wide random tempo
SCORE_TIME_FACTOR_MIN = 0.25
SCORE_TIME_FACTOR_MAX = 4.0


def clamp(val, min_val=0, max_val=127):
    return max(min_val, min(max_val, int(val)))


def augment_pitch(midi_path: Path, output_path: Path, semitones: int) -> bool:
    try:
        mid = mido.MidiFile(str(midi_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type in ("note_on", "note_off"):
                    msg.note = clamp(msg.note + semitones)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_path))
        return True
    except Exception as e:
        print(f"Error augmenting pitch for {midi_path}: {e}")
        return False


def augment_time(midi_path: Path, output_path: Path, factor: float) -> bool:
    try:
        mid = mido.MidiFile(str(midi_path))
        for track in mid.tracks:
            for msg in track:
                msg.time = int(round(msg.time * factor))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_path))
        return True
    except Exception as e:
        print(f"Error augmenting time for {midi_path}: {e}")
        return False


def augment_velocity(midi_path: Path, output_path: Path, factor: float) -> bool:
    try:
        mid = mido.MidiFile(str(midi_path))
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on":
                    msg.velocity = clamp(msg.velocity * factor)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_path))
        return True
    except Exception as e:
        print(f"Error augmenting velocity for {midi_path}: {e}")
        return False


def copy_or_scale_annotation(src_ann: Path, dst_ann: Path, time_factor: float | None) -> None:
    dst_ann.parent.mkdir(parents=True, exist_ok=True)
    if time_factor is None or abs(time_factor - 1.0) < 1e-9:
        shutil.copy2(src_ann, dst_ann)
        return
    with open(src_ann, encoding="utf-8") as f, open(dst_ann, "w", encoding="utf-8") as w:
        for line in f:
            raw = line.rstrip("\n")
            if not raw:
                w.write("\n")
                continue
            parts = raw.split("\t")
            if len(parts) >= 2:
                try:
                    parts[0] = str(float(parts[0]) * time_factor)
                    parts[1] = str(float(parts[1]) * time_factor)
                except ValueError:
                    pass
            w.write("\t".join(parts) + "\n")


def _iter_midis(root: Path):
    for pat in ("*.mid", "*.midi"):
        yield from root.rglob(pat)


def main():
    project_root = Path(__file__).resolve().parents[2]
    repo_asap = project_root / "asap-dataset-master"
    default_data = repo_asap / "asap-dataset-master"
    default_aug = repo_asap / "augmented"

    parser = argparse.ArgumentParser(description="Augment ASAP MIDIs + annotations into asap-dataset-master/augmented/")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="ASAP inner tree (default: .../asap-dataset-master/asap-dataset-master)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_aug,
        help="Output root (default: .../asap-dataset-master/augmented)",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=None,
        help="Limit number of source MIDIs (for testing).",
    )
    parser.add_argument(
        "--score-time-min",
        type=float,
        default=SCORE_TIME_FACTOR_MIN,
        help="midi_score time-stretch factor lower bound (default: wide range).",
    )
    parser.add_argument(
        "--score-time-max",
        type=float,
        default=SCORE_TIME_FACTOR_MAX,
        help="midi_score time-stretch factor upper bound.",
    )
    parser.add_argument(
        "--perf-time-min",
        type=float,
        default=PERF_TIME_FACTOR_MIN,
        help="Performance MIDI time-stretch lower bound (default 0.9).",
    )
    parser.add_argument(
        "--perf-time-max",
        type=float,
        default=PERF_TIME_FACTOR_MAX,
        help="Performance MIDI time-stretch upper bound (default 1.1).",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    aug_root: Path = args.output_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ASAP data dir not found: {data_dir}")

    aug_root.mkdir(parents=True, exist_ok=True)

    midis = sorted({p for p in _iter_midis(data_dir) if p.is_file()})
    if args.max_sample is not None:
        midis = random.sample(midis, min(len(midis), args.max_sample))

    semitones_options = [-3, -2, -1, 1, 2, 3]
    print(f"ASAP augment: {len(midis)} source MIDIs -> {aug_root}")

    for midi_path in midis:
        ann = midi_path.parent / f"{midi_path.stem}_annotations.txt"
        if not ann.is_file():
            continue
        try:
            rel = midi_path.relative_to(data_dir)
        except ValueError:
            continue
        rel_parent = rel.parent
        stem = midi_path.stem
        ext = midi_path.suffix
        is_midi_score = stem == "midi_score" or stem.startswith("midi_score_")

        # Pitch
        for semitone in random.sample(semitones_options, min(6, len(semitones_options))):
            out_midi = aug_root / rel_parent / f"{stem}_pitch_{semitone:+d}{ext}"
            out_ann = aug_root / rel_parent / f"{stem}_pitch_{semitone:+d}_annotations.txt"
            if augment_pitch(midi_path, out_midi, semitone):
                copy_or_scale_annotation(ann, out_ann, None)

        # Time (tempo): score = wide factor; performances = 0.9–1.1
        t_lo = args.score_time_min if is_midi_score else args.perf_time_min
        t_hi = args.score_time_max if is_midi_score else args.perf_time_max
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
        for _ in range(5):
            factor = round(random.uniform(t_lo, t_hi), 5)
            out_midi = aug_root / rel_parent / f"{stem}_time_{factor:.2f}{ext}"
            out_ann = aug_root / rel_parent / f"{stem}_time_{factor:.2f}_annotations.txt"
            if augment_time(midi_path, out_midi, factor):
                copy_or_scale_annotation(ann, out_ann, factor)

        # Velocity
        for _ in range(5):
            factor = round(random.uniform(0.9, 1.1), 5)
            out_midi = aug_root / rel_parent / f"{stem}_vel_{factor:.2f}{ext}"
            out_ann = aug_root / rel_parent / f"{stem}_vel_{factor:.2f}_annotations.txt"
            if augment_velocity(midi_path, out_midi, factor):
                copy_or_scale_annotation(ann, out_ann, None)

    print("Done.")


if __name__ == "__main__":
    main()
