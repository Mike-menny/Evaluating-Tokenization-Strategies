#!/usr/bin/env python3
"""Offline-tokenize cleaned Lakh piano MIDI windows for pretraining.

Reads the manifest written by clean_lakh.py (lakh_midi/manifest.csv) and produces
cache/conditional_lakh_nvp/{train,validation,test}.pt in the same format as the MAESTRO
cache, so the two can be merged by merge_pretrain_cache.py.

Uses note_velocity_pedal mode by default (Lakh has no beat annotations).

Usage:
    uv run python3 -m src.train.pretokenize_lakh \
        --lakh_dir lakh_midi \
        --cache_dir cache/conditional_lakh_nvp \
        --tokenization_mode note_velocity_pedal \
        --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


def _midi_to_events(midi_path: str | Path, mode: str = "note_velocity_pedal"):
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


def _process_row(payload: dict) -> tuple[list[dict], int]:
    midi_path = Path(payload["midi_path"])
    if not midi_path.is_file():
        return [], 1
    composer_name = payload["composer"]
    genre_name = payload["genre"]
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
    for sample in samples:
        tok = sample["tokens"]
        if not isinstance(tok, torch.Tensor):
            sample["tokens"] = torch.tensor(tok, dtype=torch.int16)
    return samples


def build_split(args: argparse.Namespace, df: pd.DataFrame, split: str) -> dict:
    samples: list[dict] = []
    failed = 0
    num_workers = max(1, int(args.num_workers))
    chunk_size = max(num_workers, int(args.chunk_size))
    done = 0

    payloads = [
        {
            "midi_path": str(Path(args.lakh_dir) / row["out_name"]),
            "composer": row["composer"],
            "genre": row["genre"],
            "tokenization_mode": args.tokenization_mode,
            "window_duration_sec": args.window_duration_sec,
        }
        for _, row in df.iterrows()
    ]

    def _consume(row_samples: list[dict], row_failed: int) -> None:
        nonlocal done, failed
        samples.extend(row_samples)
        failed += row_failed
        done += 1
        if args.log_every > 0 and done % args.log_every == 0:
            print(f"{split}: rows {done}/{len(payloads)}, kept={len(samples)}, failed={failed}")

    if num_workers == 1:
        for p in payloads:
            _consume(*_process_row(p))
    else:
        print(f"{split}: tokenizing with {num_workers} workers, chunk_size={chunk_size} ...")
        with ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=50) as pool:
            for chunk_start in range(0, len(payloads), chunk_size):
                chunk = payloads[chunk_start : chunk_start + chunk_size]
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
        "source": "lakh_piano_clean",
    }
    return {"metadata": metadata, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize cleaned Lakh piano MIDI into .pt cache files")
    parser.add_argument("--lakh_dir", type=str, default=str(PROJECT_ROOT / "lakh_midi"))
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "cache" / "conditional_lakh_nvp"))
    parser.add_argument("--tokenization_mode", type=str, default="note_velocity_pedal", choices=TOKENIZATION_MODE_CHOICES)
    parser.add_argument("--window_duration_sec", type=float, default=WINDOW_SEC_DEFAULT)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    lakh_dir = Path(args.lakh_dir)
    manifest = lakh_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}. Run clean_lakh.py first.")

    df = pd.read_csv(manifest)
    print(f"Loaded manifest: {len(df)} files")
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].reset_index(drop=True)
        print(f"Capped to {len(df)} files")

    # Split by file (not window) to avoid leakage
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * args.val_fraction))
    n_test = max(1, int(n * args.test_fraction))
    splits = {
        "train": df.iloc[: n - n_val - n_test].reset_index(drop=True),
        "validation": df.iloc[n - n_val - n_test : n - n_test].reset_index(drop=True),
        "test": df.iloc[n - n_test :].reset_index(drop=True),
    }
    for name, sub in splits.items():
        print(f"  {name}: {len(sub)} files")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    (cache_dir / "pretokenize_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    for split_name, sub_df in splits.items():
        payload = build_split(args, sub_df, split_name)
        out = cache_dir / f"{split_name}.pt"
        torch.save(payload, out)
        print(f"saved {split_name}: {payload['metadata']['num_samples']} samples -> {out}")


if __name__ == "__main__":
    main()
