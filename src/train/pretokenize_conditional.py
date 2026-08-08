#!/usr/bin/env python3
"""Offline-tokenize ASAP windows for the compact conditional MIDI model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from src.dataset.conditional_asap import condition_metadata
from src.tokenization.conditional_convert import events_window_to_tokens, merge_midi_annotation_events
from src.tokenization.conditional_vocab import (
    BOS_ID,
    EOS_ID,
    TOKENIZATION_MODE_CHOICES,
    WINDOW_SEC_DEFAULT,
    composer_genre_from_manifest_path,
    composer_id,
    genre_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(data_root: Path, rel_path: str) -> Path:
    path = data_root / rel_path
    if path.exists():
        return path
    parts = Path(rel_path).parts
    if parts and parts[0] == data_root.name:
        alt = data_root / Path(*parts[1:])
        if alt.exists():
            return alt
    return path


def build_split(args: argparse.Namespace, split: str) -> dict:
    data_root = Path(args.data_root)
    df = pd.read_csv(args.manifest)
    df = df[df["split"] == split].reset_index(drop=True)
    if args.max_rows is not None and args.max_rows > 0:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    samples: list[dict] = []
    failed = 0
    total_windows = 0
    for row_idx, row in df.iterrows():
        composer, genre = composer_genre_from_manifest_path(str(row["midi_path"]))
        midi_path = _resolve(data_root, str(row["midi_path"]))
        ann_path = _resolve(data_root, str(row["annotation_path"]))
        try:
            events = merge_midi_annotation_events(midi_path, ann_path, mode=args.tokenization_mode)
        except Exception as e:
            failed += 1
            print(f"{split}: failed row {row_idx} {midi_path}: {e}")
            continue
        if not events:
            failed += 1
            continue
        duration = max(t for t, *_rest in events)
        n_windows = max(1, int(math.ceil(max(duration, 0.001) / args.window_duration_sec)))
        for k in range(n_windows):
            start = k * args.window_duration_sec
            window_tokens = events_window_to_tokens(events, start, start + args.window_duration_sec)
            if not window_tokens:
                failed += 1
                continue
            tokens = torch.tensor([BOS_ID, *window_tokens, EOS_ID], dtype=torch.int16)
            samples.append(
                {
                    "composer_id": composer_id(composer),
                    "genre_id": genre_id(genre),
                    "tokens": tokens,
                    "composer": composer,
                    "genre": genre,
                    "window_start_sec": float(start),
                }
            )
        total_windows += n_windows
        if args.log_every > 0 and (row_idx + 1) % args.log_every == 0:
            print(f"{split}: rows {row_idx + 1}/{len(df)}, windows~{total_windows}, kept={len(samples)}, failed={failed}")

    metadata = {
        **condition_metadata(),
        "split": split,
        "tokenization_mode": args.tokenization_mode,
        "window_duration_sec": args.window_duration_sec,
        "num_samples": len(samples),
        "failed_samples": failed,
    }
    return {"metadata": metadata, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize conditional ASAP windows into .pt cache files")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "asap-dataset-master" / "asap_v2_manifest.csv"))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "asap-dataset-master"))
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "cache" / "conditional_asap_full"))
    parser.add_argument("--tokenization_mode", type=str, default="full", choices=TOKENIZATION_MODE_CHOICES)
    parser.add_argument("--window_duration_sec", type=float, default=WINDOW_SEC_DEFAULT)
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    (cache_dir / "pretokenize_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        payload = build_split(args, split)
        out = cache_dir / f"{split}.pt"
        torch.save(payload, out)
        print(f"saved {split}: {payload['metadata']['num_samples']} samples -> {out}")


if __name__ == "__main__":
    main()
