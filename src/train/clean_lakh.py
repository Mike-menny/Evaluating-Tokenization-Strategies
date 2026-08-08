#!/usr/bin/env python3
"""Clean Lakh MIDI for piano-only pretraining.

Strategy:
- Scan every MIDI file under --lakh_root (lmd_full / clean_midi / lmd_matched).
- Keep a file only if it has at least one program_change with program in 0..7 (Acoustic Grand
  .. Clavinet) OR has note_on events on a channel without a drum channel (9).
- For kept files, write a NEW single-track MIDI that contains ONLY the piano-channel notes
  (control_change #64 sustain pedal preserved). Drum channels (9) and non-piano pitched
  channels are dropped.
- Output flat structure: lakh_midi/<sha8>.mid plus a manifest lakh_midi/manifest.csv with
  composer / genre inferred from path when possible.

The cleaning uses mido's absolute-time semantics, so the output MIDI preserves the original
absolute timing of every note. We force tempo=500000 (120 BPM) and tpb=480 on the output
so downstream tokenizers that rely on `message.time` (seconds) see a consistent file; mido
recomputes seconds from tempo*tpb, so we rescale ticks accordingly.

Usage:
    python3 -m src.train.clean_lakh \
        --lakh_root /path/to/Lakh/raw \
        --output_dir lakh_midi \
        --num_workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mido
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIANO_PROGRAMS = set(range(8))  # 0..7
DRUM_CHANNEL = 9


def _is_piano_program(program: int) -> bool:
    return program in PIANO_PROGRAMS


def _scan_channels(midi_path: str) -> tuple[set[int], dict[int, int]]:
    """Return (piano_channels, channel_to_program). Returns ({}, {}) on error."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return set(), {}
    channel_programs: dict[int, int] = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                # In MIDI, channel 9 is reserved for drums regardless of program; skip it.
                ch = msg.channel if hasattr(msg, "channel") else 0
                if ch != DRUM_CHANNEL:
                    channel_programs[ch] = msg.program
    piano_channels = {ch for ch, prog in channel_programs.items() if _is_piano_program(prog)}
    # If a channel has no program_change but is not drum, treat as piano (program 0 default).
    # We detect such channels during note extraction below.
    return piano_channels, channel_programs


def _extract_piano_events(midi_path: str, piano_channels: set[int]) -> list[dict]:
    """Extract note_on/note_off/control_change(64) events from piano channels only.

    Returns a list of events with absolute_time in seconds.
    Uses mido's merged iterator (`for msg in mid`) to get a single global timeline
    across all tracks. Falls back to per-track iteration with a SHARED accumulator
    for MIDI type 2 files where mido refuses to merge.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []
    events: list[dict] = []
    open_notes: dict[tuple[int, int], list[int]] = {}
    sustain_on = False
    inferred_piano: set[int] = set()

    def _handle(msg, abs_sec):
        """Process one message at given absolute time. Returns nothing; appends to events."""
        nonlocal sustain_on
        if msg.type == "note_on" and msg.velocity > 0:
            ch = msg.channel if hasattr(msg, "channel") else 0
            if ch == DRUM_CHANNEL:
                return
            if ch in piano_channels or ch in inferred_piano or not piano_channels:
                inferred_piano.add(ch)
                events.append({
                    "type": "note_on",
                    "abs_sec": abs_sec,
                    "channel": 0,
                    "note": int(msg.note),
                    "velocity": int(msg.velocity),
                })
                key = (ch, msg.note)
                open_notes.setdefault(key, []).append(len(events) - 1)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            ch = msg.channel if hasattr(msg, "channel") else 0
            key = (ch, msg.note)
            if key in open_notes and open_notes[key]:
                idx = open_notes[key].pop(0)
                events[idx]["duration_sec"] = abs_sec - events[idx]["abs_sec"]
        elif msg.type == "control_change" and msg.control == 64:
            ch = msg.channel if hasattr(msg, "channel") else 0
            if ch in piano_channels or ch in inferred_piano or not piano_channels:
                new_on = msg.value >= 64
                if new_on != sustain_on:
                    sustain_on = new_on
                    events.append({
                        "type": "pedal",
                        "abs_sec": abs_sec,
                        "channel": 0,
                        "value": 127 if new_on else 0,
                    })

    # Primary path: mido merges tracks into one timeline (correct for type 0/1).
    use_merged = True
    try:
        abs_sec = 0.0
        for msg in mid:
            abs_sec += msg.time
            _handle(msg, abs_sec)
    except TypeError:
        # type 2 (asynchronous) can't merge tracks
        use_merged = False

    if not use_merged:
        # Fallback for type 2: iterate each track with its OWN absolute time.
        # Take the track with the most piano notes.
        best_events: list[dict] = []
        best_open: dict[tuple[int, int], list[int]] = {}
        for track in mid.tracks:
            track_events: list[dict] = []
            track_open: dict[tuple[int, int], list[int]] = {}
            abs_sec = 0.0
            for msg in track:
                abs_sec += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    ch = msg.channel if hasattr(msg, "channel") else 0
                    if ch == DRUM_CHANNEL:
                        continue
                    if ch in piano_channels or ch in inferred_piano or not piano_channels:
                        inferred_piano.add(ch)
                        track_events.append({
                            "type": "note_on",
                            "abs_sec": abs_sec,
                            "channel": 0,
                            "note": int(msg.note),
                            "velocity": int(msg.velocity),
                        })
                        key = (ch, msg.note)
                        track_open.setdefault(key, []).append(len(track_events) - 1)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    ch = msg.channel if hasattr(msg, "channel") else 0
                    key = (ch, msg.note)
                    if key in track_open and track_open[key]:
                        idx = track_open[key].pop(0)
                        track_events[idx]["duration_sec"] = abs_sec - track_events[idx]["abs_sec"]
            if len(track_events) > len(best_events):
                best_events = track_events
                best_open = track_open
        events = best_events
        open_notes = best_open

    # Close any still-open notes with a tiny duration
    for key, idxs in open_notes.items():
        for idx in idxs:
            if "duration_sec" not in events[idx]:
                events[idx]["duration_sec"] = 0.05
    events.sort(key=lambda e: e["abs_sec"])
    return events


def _events_to_midi(events: list[dict], tpb: int = 480) -> mido.MidiFile:
    """Build a single-track piano MIDI from absolute-second events.

    Uses tempo=500000 (120 BPM) so 1 beat = 0.5s = 480 ticks. This makes
    `message.time` (seconds) preserved exactly when mido re-parses.
    """
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))

    # Convert abs_sec to abs_ticks: 1 beat = 0.5s = tpb ticks -> 1 sec = tpb/0.5 = 2*tpb ticks
    ticks_per_sec = tpb * 2.0  # at 120 BPM, 1 sec = 960 ticks (for tpb=480)

    abs_messages: list[tuple[int, int, mido.Message]] = []
    for ev in events:
        tick = max(0, int(round(ev["abs_sec"] * ticks_per_sec)))
        if ev["type"] == "note_on":
            dur_ticks = max(1, int(round(ev.get("duration_sec", 0.1) * ticks_per_sec)))
            abs_messages.append((tick, 1, mido.Message("note_on", channel=0, note=ev["note"], velocity=ev["velocity"], time=0)))
            abs_messages.append((tick + dur_ticks, 0, mido.Message("note_off", channel=0, note=ev["note"], velocity=0, time=0)))
        elif ev["type"] == "pedal":
            abs_messages.append((tick, 2, mido.Message("control_change", channel=0, control=64, value=ev["value"], time=0)))

    abs_messages.sort(key=lambda x: (x[0], x[1]))
    prev_tick = 0
    for tick, _prio, msg in abs_messages:
        msg.time = max(0, tick - prev_tick)
        prev_tick = tick
        track.append(msg)
    return mid


def _infer_composer_genre(path: str) -> tuple[str, str]:
    """Best-effort composer / genre from Lakh path.

    Lakh clean_midi layout: clean_midi/Artist/Title.mid  -> artist is parent dir
    Lakh lmd_full layout:   lmd_full/<hex>/<hash>.mid   -> no metadata, "Unknown"
    """
    parts = Path(path).parts
    name = Path(path).stem
    parent = Path(path).parent.name
    # lmd_full / lmd_matched: parent is a hex bucket, no artist info
    if parent and len(parent) == 1 and parent in "0123456789abcdef":
        return "Unknown", _guess_genre(name)
    # lmd_full with deeper hash dir
    if parent in ("lmd_full", "lmd_matched", "lmd_aligned", "raw"):
        return "Unknown", _guess_genre(name)
    if parent and parent not in ("clean_midi", "lmd_full", "lmd_matched", "lmd_aligned", "raw"):
        return parent, _guess_genre(name)
    return "Unknown", _guess_genre(name)


def _guess_genre(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ("sonata",)):
        return "sonata"
    if any(k in t for k in ("etude", "study",)):
        return "etude"
    if any(k in t for k in ("prelude", "preludes")):
        return "prelude"
    if any(k in t for k in ("concerto",)):
        return "concerto"
    if any(k in t for k in ("waltz",)):
        return "waltz"
    if any(k in t for k in ("nocturne",)):
        return "nocturne"
    if any(k in t for k in ("fugue",)):
        return "fugue"
    if any(k in t for k in ("suite",)):
        return "suite"
    if any(k in t for k in ("variation",)):
        return "variation"
    if any(k in t for k in ("rhapsody",)):
        return "rhapsody"
    if any(k in t for k in ("impromptu",)):
        return "impromptu"
    if any(k in t for k in ("mazurka",)):
        return "mazurka"
    if any(k in t for k in ("ballade",)):
        return "ballade"
    if any(k in t for k in ("toccata",)):
        return "toccata"
    return "piece"


def _process_one(payload: dict) -> dict | None:
    try:
        midi_path = payload["midi_path"]
        rel_path = payload["rel_path"]
        output_dir = Path(payload["output_dir"])

        piano_channels, _ = _scan_channels(midi_path)
        events = _extract_piano_events(midi_path, piano_channels)
        if len(events) < 10:
            return None
        note_count = sum(1 for e in events if e["type"] == "note_on")
        if note_count < 5:
            return None

        duration_sec = max(e["abs_sec"] for e in events)
        # Sanity: reject files shorter than 10s or longer than 30 min (likely parse error)
        if duration_sec < 10 or duration_sec > 1800:
            return None

        new_mid = _events_to_midi(events)
        sha = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8]
        out_name = f"{sha}.mid"
        out_path = output_dir / out_name
        try:
            new_mid.save(str(out_path))
        except Exception:
            return None

        composer, genre = _infer_composer_genre(rel_path)
        return {
            "rel_path": rel_path,
            "out_name": out_name,
            "composer": composer,
            "genre": genre,
            "duration_sec": duration_sec,
            "note_count": note_count,
        }
    except Exception:
        return None


def _iter_midi_files(lakh_root: Path, subsets: list[str]) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for subset in subsets:
        root = lakh_root / subset
        if not root.exists():
            continue
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".mid", ".midi")):
                    full = os.path.join(r, f)
                    rel = os.path.relpath(full, lakh_root)
                    files.append((full, rel))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Lakh MIDI to piano-only for pretraining")
    parser.add_argument(
        "--lakh_root",
        type=str,
        required=True,
        help="Root of raw Lakh MIDI (contains clean_midi/ and/or lmd_full/)",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "lakh_midi"))
    parser.add_argument("--subsets", type=str, default="clean_midi,lmd_full",
                        help="Comma-separated Lakh subsets to scan (clean_midi, lmd_full, lmd_matched)")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_files", type=int, default=0, help="Cap files processed (0 = all)")
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    lakh_root = Path(args.lakh_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    print(f"Scanning Lakh subsets {subsets} under {lakh_root} ...")
    files = _iter_midi_files(lakh_root, subsets)
    print(f"Found {len(files):,} MIDI files total")
    if args.max_files > 0:
        files = files[: args.max_files]
        print(f"Capped to {len(files)} files")

    payloads = [
        {"midi_path": full, "rel_path": rel, "output_dir": str(output_dir)}
        for full, rel in files
    ]

    manifest_rows: list[dict] = []
    failed = 0
    done = 0
    num_workers = max(1, int(args.num_workers))
    total = len(payloads)

    if num_workers == 1:
        for p in payloads:
            r = _process_one(p)
            done += 1
            if r is None:
                failed += 1
            else:
                manifest_rows.append(r)
            if args.log_every > 0 and done % args.log_every == 0:
                print(f"  done {done}/{total}, kept={len(manifest_rows)}, failed={failed}", flush=True)
    else:
        # Use multiprocessing.Pool.imap_unordered for streaming over 195k tasks.
        # ProcessPoolExecutor.submit on all tasks upfront OOMs / stalls.
        import multiprocessing as mp

        def _gen():
            for p in payloads:
                yield p

        with mp.Pool(processes=num_workers, maxtasksperchild=50) as pool:
            for r in pool.imap_unordered(_process_one, _gen(), chunksize=32):
                done += 1
                if r is None:
                    failed += 1
                else:
                    manifest_rows.append(r)
                if args.log_every > 0 and done % args.log_every == 0:
                    print(f"  done {done}/{total}, kept={len(manifest_rows)}, failed={failed}", flush=True)

    print(f"\nDone. kept={len(manifest_rows)}, failed/dropped={failed}")

    df = pd.DataFrame(manifest_rows)
    df.to_csv(output_dir / "manifest.csv", index=False)
    total_hours = df["duration_sec"].sum() / 3600 if len(df) > 0 else 0.0
    print(f"Total kept duration: {total_hours:,.1f} hours, {len(df)} files")
    print(f"Manifest -> {output_dir / 'manifest.csv'}")

    summary = {
        "lakh_root": str(lakh_root),
        "subsets": subsets,
        "total_scanned": len(files),
        "kept": len(manifest_rows),
        "failed": failed,
        "total_duration_hours": total_hours,
    }
    (output_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
