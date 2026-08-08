#!/usr/bin/env python3
"""Offline-tokenize MAESTRO (augmented) dataset windows for the compact conditional MIDI model.

Deduplicates against ASAP base MIDIs so that pretraining data does not overlap
with the downstream ASAP finetuning set.  Runs in note_velocity_pedal mode by
default (MAESTRO has no beat annotations).

Usage:
    python3 -m src.train.pretokenize_maestro \
        --maestro_csv maestro_dataset/final.csv \
        --maestro_root maestro_dataset \
        --asap_manifest asap-dataset-master/asap_v2_manifest.csv \
        --asap_root asap-dataset-master \
        --cache_dir cache/conditional_maestro_nvp \
        --tokenization_mode note_velocity_pedal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from src.dataset.conditional_asap import condition_metadata
from src.tokenization.conditional_convert import _midi_notes_pedals_raw, events_window_to_tokens
from src.tokenization.conditional_vocab import (
    BOS_ID,
    EOS_ID,
    TOKENIZATION_MODE_CHOICES,
    WINDOW_SEC_DEFAULT,
    composer_id,
    genre_id,
    get_tokenization_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# ASAP dedup helpers
# ---------------------------------------------------------------------------

def _collect_asap_base_hashes(asap_manifest: Path, asap_root: Path) -> set[str]:
    """Return SHA256 hashes of all base (non-augmented) ASAP MIDI files."""
    a = pd.read_csv(asap_manifest)
    base = a[a["midi_path"].astype(str).str.startswith("asap-dataset-master/")]
    hashes: set[str] = set()
    for rel in base["midi_path"].drop_duplicates():
        p = asap_root / rel
        if p.is_file():
            hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    return hashes


def _collect_asap_overlap_stems(
    asap_manifest: Path,
    asap_root: Path,
    maestro_csv: Path,
    maestro_root: Path,
) -> set[tuple[str, str]]:
    """Return (parent_dir, stem) pairs for MAESTRO base files that are exact duplicates of ASAP base."""
    asap_hashes = _collect_asap_base_hashes(asap_manifest, asap_root)

    h = pd.read_csv(maestro_csv)
    h_base = h[~h["midi_filename"].astype(str).str.startswith("augmented/")]

    overlap_stems: set[tuple[str, str]] = set()
    for rel in h_base["midi_filename"].drop_duplicates():
        p = maestro_root / "grouped" / rel
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() in asap_hashes:
            overlap_stems.add((str(Path(rel).parent), Path(rel).stem))
    return overlap_stems


_AUG_SUFFIX_RE = re.compile(r"_(pitch_[-\d]+|time_\d+\.\d+|vel_\d+\.\d+)$")


def _source_key(rel_path: str) -> tuple[str, str]:
    """Map an (possibly augmented) MAESTRO midi_filename back to its (parent_dir, stem) key."""
    path = Path(rel_path)
    parts = path.parts
    if parts and parts[0] == "augmented":
        parent = str(Path(*parts[1:-1])) if len(parts) > 2 else "."
    else:
        parent = str(path.parent)
    stem = _AUG_SUFFIX_RE.sub("", path.stem)
    return (parent, stem)


# ---------------------------------------------------------------------------
# Tokenization for MAESTRO (no annotation file)
# ---------------------------------------------------------------------------

def _midi_to_events(
    midi_path: str | Path,
    mode: str = "note_velocity_pedal",
) -> list[tuple[float, int, str, list[int] | int]]:
    """Parse a MIDI once into merged events (no beat annotations)."""
    from src.tokenization.conditional_vocab import DUR_OFFSET

    spec = get_tokenization_spec(mode)
    notes, pedals = _midi_notes_pedals_raw(midi_path, mode=mode)

    events: list[tuple[float, int, str, list[int] | int]] = []
    for row in notes:
        payload: list[int | float] = [row["onset"], DUR_OFFSET + row["dur_ticks"], row["note_id"]]
        if spec.use_velocity:
            payload.append(row["vel_id"])
        events.append((row["onset"], 1, "note", payload))
    for t, pedal_id in pedals:
        events.append((t, 2, "pedal", pedal_id))
    events.sort(key=lambda x: (x[0], x[1]))
    return events


def _composer_genre_from_rel(midi_rel: str) -> tuple[str, str]:
    parts = Path(midi_rel).parts
    if parts and parts[0] == "augmented" and len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _process_row(payload: dict) -> tuple[list[dict], int]:
    """Worker: tokenize one CSV row into all its windows. Returns (samples, failed_windows)."""
    midi_rel = payload["midi_rel"]
    midi_path = Path(payload["midi_path"])
    if not midi_path.is_file():
        return [], 1

    composer_name, genre_name = _composer_genre_from_rel(midi_rel)
    c_id = composer_id(composer_name)
    g_id = genre_id(genre_name)
    mode = payload["tokenization_mode"]
    window_duration_sec = float(payload["window_duration_sec"])

    try:
        events = _midi_to_events(midi_path, mode=mode)
    except Exception:
        return [], 1
    if not events:
        return [], 1

    duration = max(t for t, *_rest in events)
    n_windows = max(1, int(math.ceil(max(duration, 0.001) / window_duration_sec)))

    samples: list[dict] = []
    failed = 0
    for k in range(n_windows):
        start = k * window_duration_sec
        try:
            window_tokens = events_window_to_tokens(events, start, start + window_duration_sec)
        except Exception:
            failed += 1
            continue
        if not window_tokens:
            failed += 1
            continue
        # Return plain list[int] from workers to avoid torch shared-memory mmap OOM.
        samples.append(
            {
                "composer_id": c_id,
                "genre_id": g_id,
                "tokens": [BOS_ID, *window_tokens, EOS_ID],
                "composer": composer_name,
                "genre": genre_name,
                "window_start_sec": float(start),
            }
        )
    return samples, failed


def _finalize_samples(samples: list[dict]) -> list[dict]:
    """Convert token lists to int16 tensors once, right before saving."""
    for sample in samples:
        tok = sample["tokens"]
        if not isinstance(tok, torch.Tensor):
            sample["tokens"] = torch.tensor(tok, dtype=torch.int16)
    return samples


# ---------------------------------------------------------------------------
# Build split
# ---------------------------------------------------------------------------

def build_split(
    args: argparse.Namespace,
    split: str,
    overlap_stems: set[tuple[str, str]],
) -> dict:
    maestro_root = Path(args.maestro_root)
    df = pd.read_csv(args.maestro_csv)
    df = df[df["split"] == split].reset_index(drop=True)
    if args.max_rows is not None and args.max_rows > 0:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    # Filter out rows whose source MIDI overlaps with ASAP
    before = len(df)
    df = df[~df["midi_filename"].astype(str).map(lambda p: _source_key(p) in overlap_stems)].reset_index(drop=True)
    if _is_main_process():
        print(f"{split}: filtered {before - len(df)} rows overlapping with ASAP base ({before} -> {len(df)})")

    row_payloads = [
        {
            "midi_rel": str(row["midi_filename"]),
            "midi_path": str(maestro_root / "grouped" / row["midi_filename"]),
            "tokenization_mode": args.tokenization_mode,
            "window_duration_sec": args.window_duration_sec,
        }
        for _, row in df.iterrows()
    ]

    samples: list[dict] = []
    failed = 0
    num_workers = max(1, int(args.num_workers))
    chunk_size = max(num_workers, int(args.chunk_size))
    done = 0

    def _consume(row_samples: list[dict], row_failed: int) -> None:
        nonlocal done, failed
        samples.extend(row_samples)
        failed += row_failed
        done += 1
        if args.log_every > 0 and done % args.log_every == 0:
            print(f"{split}: rows {done}/{len(row_payloads)}, kept={len(samples)}, failed={failed}")

    if num_workers == 1:
        for payload in row_payloads:
            _consume(*_process_row(payload))
    else:
        print(f"{split}: tokenizing with {num_workers} workers, chunk_size={chunk_size} ...")
        with ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=50) as pool:
            for chunk_start in range(0, len(row_payloads), chunk_size):
                chunk = row_payloads[chunk_start : chunk_start + chunk_size]
                futures = [pool.submit(_process_row, p) for p in chunk]
                for fut in as_completed(futures):
                    _consume(*fut.result())

    samples = _finalize_samples(samples)
    metadata = {
        **condition_metadata(),
        "split": split,
        "tokenization_mode": args.tokenization_mode,
        "window_duration_sec": args.window_duration_sec,
        "num_samples": len(samples),
        "failed_samples": failed,
    }
    return {"metadata": metadata, "samples": samples}


def _is_main_process() -> bool:
    import os
    return int(os.environ.get("RANK", "0")) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize MAESTRO windows into .pt cache files (deduped vs ASAP)")
    parser.add_argument("--maestro_csv", type=str, default=str(PROJECT_ROOT / "maestro_dataset" / "final.csv"))
    parser.add_argument("--maestro_root", type=str, default=str(PROJECT_ROOT / "maestro_dataset"))
    parser.add_argument("--asap_manifest", type=str,
                        default=str(PROJECT_ROOT / "asap-dataset-master" / "asap_v2_manifest.csv"))
    parser.add_argument("--asap_root", type=str, default=str(PROJECT_ROOT / "asap-dataset-master"))
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "cache" / "conditional_maestro_nvp"))
    parser.add_argument("--tokenization_mode", type=str, default="note_velocity_pedal", choices=TOKENIZATION_MODE_CHOICES)
    parser.add_argument("--window_duration_sec", type=float, default=WINDOW_SEC_DEFAULT)
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Parallel workers for MIDI tokenization (default: 4; lower if OOM)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Rows per multiprocessing chunk (limits in-flight tasks; default: 256)",
    )
    args = parser.parse_args()

    print("Computing ASAP overlap stems ...")
    overlap_stems = _collect_asap_overlap_stems(
        Path(args.asap_manifest), Path(args.asap_root),
        Path(args.maestro_csv), Path(args.maestro_root),
    )
    print(f"Found {len(overlap_stems)} MAESTRO base files overlapping with ASAP base")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["asap_overlap_base_count"] = len(overlap_stems)
    (cache_dir / "pretokenize_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        payload = build_split(args, split, overlap_stems)
        out = cache_dir / f"{split}.pt"
        torch.save(payload, out)
        print(f"saved {split}: {payload['metadata']['num_samples']} samples -> {out}")


if __name__ == "__main__":
    main()
