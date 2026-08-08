#!/usr/bin/env python3
"""Merge multiple pretraining caches (MAESTRO + Lakh) into one.

Reads train.pt / validation.pt / test.pt from each --input cache and concatenates
the sample lists, writing a combined cache to --output_dir. Metadata from the first
input is kept and augmented with source counts.

Usage:
    uv run python3 -m src.train.merge_pretrain_cache \
        --inputs cache/conditional_maestro_nvp cache/conditional_lakh_nvp \
        --output_dir cache/conditional_pretrain_combined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(cache_dir: Path, split: str) -> dict:
    p = cache_dir / f"{split}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple pretraining caches into one")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input cache directories")
    parser.add_argument("--output_dir", required=True, help="Output cache directory")
    parser.add_argument("--splits", default="train,validation,test")
    args = parser.parse_args()

    inputs = [Path(d) for d in args.inputs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    source_counts: dict[str, dict[str, int]] = {}

    for split in splits:
        merged_samples = []
        base_meta = None
        for cache_dir in inputs:
            payload = _load(cache_dir, split)
            if payload is None:
                print(f"  {cache_dir}/{split}.pt not found, skipping")
                continue
            if base_meta is None:
                base_meta = payload["metadata"]
            n = len(payload["samples"])
            merged_samples.extend(payload["samples"])
            source_counts.setdefault(cache_dir.name, {})[split] = n
            print(f"  {split} <- {cache_dir.name}: {n} samples")

        if base_meta is None:
            print(f"No data for split {split}, skipping")
            continue

        meta = dict(base_meta)
        meta["num_samples"] = len(merged_samples)
        meta["sources"] = source_counts
        meta["merged"] = True
        out = {"metadata": meta, "samples": merged_samples}
        out_path = output_dir / f"{split}.pt"
        torch.save(out, out_path)
        print(f"saved {split}: {len(merged_samples)} samples -> {out_path}")

    summary = {
        "inputs": [str(p) for p in inputs],
        "output_dir": str(output_dir),
        "source_counts": source_counts,
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nMerge summary -> {output_dir / 'merge_summary.json'}")
    print(json.dumps(source_counts, indent=2))


if __name__ == "__main__":
    main()
