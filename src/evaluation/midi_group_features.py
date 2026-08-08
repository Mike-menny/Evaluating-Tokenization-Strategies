#!/usr/bin/env python3
"""Extract per-MIDI and per-group statistical features from reference test-set MIDI files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import mido
from mido import merge_tracks, tick2second

from src.tokenization.conditional_vocab import composer_genre_from_manifest_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_tempo_us() -> int:
    return mido.bpm2tempo(120)


def extract_midi_features(midi_path: Path) -> dict[str, float]:
    """Per-file MIDI statistics used in group_midi_stats_vs_fad.csv."""
    mid = mido.MidiFile(str(midi_path))
    tempo = _default_tempo_us()
    t_sec = 0.0

    onsets: list[float] = []
    velocities: list[int] = []
    pitches: list[int] = []
    note_durs: list[float] = []
    cc64_events: list[float] = []
    open_notes: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)

    for msg in merge_tracks(mid.tracks):
        t_sec += tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 64:
            cc64_events.append(t_sec)
        elif msg.type in ("note_on", "note_off"):
            key = (msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                onsets.append(t_sec)
                velocities.append(int(msg.velocity))
                pitches.append(int(msg.note))
                open_notes[key].append((t_sec, int(msg.velocity)))
            else:
                if not open_notes[key]:
                    continue
                onset, _vel = open_notes[key].pop(0)
                note_durs.append(max(t_sec - onset, 1e-6))

    duration_s = max(float(mid.length), t_sec, 1e-6)
    note_count = len(onsets)

    u, counts = np.unique(np.asarray(onsets, dtype=np.float64), return_counts=True)
    simul_onset_ratio = float(np.mean(counts > 1)) if counts.size else 0.0

    if u.size >= 2:
        iois = np.diff(u)
        ioi_cv = float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else 0.0
    else:
        ioi_cv = 0.0

    if pitches:
        pitch_range = float(max(pitches) - min(pitches))
        hist = np.bincount(pitches, minlength=128).astype(np.float64)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        pitch_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        pitch_range = 0.0
        pitch_entropy = 0.0

    return {
        "duration_s": duration_s,
        "note_count": float(note_count),
        "notes_per_s": note_count / duration_s,
        "mean_note_dur": float(np.mean(note_durs)) if note_durs else 0.0,
        "velocity_std": float(np.std(velocities)) if len(velocities) > 1 else 0.0,
        "pitch_range": pitch_range,
        "pitch_entropy": pitch_entropy,
        "ioi_cv": ioi_cv,
        "simul_onset_ratio": simul_onset_ratio,
        "cc64_per_min": len(cc64_events) / duration_s * 60.0,
    }


def _summarize_group(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": float(arr.size),
    }


def run_midi_group_features(test_set_root: Path, out_csv: Path) -> None:
    rows: list[dict] = []
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for mid in sorted(test_set_root.rglob("*.mid")):
        if mid.stem == "midi_score":
            continue
        rel = mid.relative_to(test_set_root)
        composer, genre = composer_genre_from_manifest_path(str(rel))
        try:
            feats = extract_midi_features(mid)
        except Exception as e:
            print(f"skip {rel}: {e}")
            continue
        row = {"composer": composer, "genre": genre, "midi_path": str(rel), **feats}
        rows.append(row)
        by_group[(composer, genre)].append(feats)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary_csv = out_csv.with_name("midi_group_features_summary.csv")
    summary_rows: list[dict] = []
    feature_keys = [
        "duration_s", "note_count", "notes_per_s", "mean_note_dur", "velocity_std",
        "pitch_range", "pitch_entropy", "ioi_cv", "simul_onset_ratio", "cc64_per_min",
    ]
    for (composer, genre), feats_list in sorted(by_group.items()):
        summary: dict = {"composer": composer, "genre": genre}
        for key in feature_keys:
            vals = [f[key] for f in feats_list]
            stats = _summarize_group(vals)
            summary[f"{key}_mean"] = stats["mean"]
            summary[f"{key}_std"] = stats["std"]
            summary[f"{key}_count"] = stats["count"]
        summary_rows.append(summary)

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote {out_csv} ({len(rows)} MIDI files)")
    print(f"Wrote {summary_csv} ({len(summary_rows)} groups)")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI statistical features per composer/genre group.")
    parser.add_argument(
        "--test_set_root",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "asap_test_set"),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/inference/midi_group_features_per_file.csv"),
    )
    args = parser.parse_args()
    run_midi_group_features(Path(args.test_set_root), Path(args.out_csv))


if __name__ == "__main__":
    main()
