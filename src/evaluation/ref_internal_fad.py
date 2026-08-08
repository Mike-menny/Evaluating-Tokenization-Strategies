#!/usr/bin/env python3
"""Reference-only per-file FAD within each composer/genre group."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.evaluation.per_piece_fad import (
    _cache_key,
    _fad_one_vs_pool,
    _file_level_embeddings,
    _summarize_scores,
    build_ref_embedding_cache,
)
from src.evaluation.fad import _load_frechet_audio_distance, _maybe_debug_torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _discover_groups(ref_root: Path) -> list[Path]:
    groups: list[Path] = []
    for comp_dir in sorted(ref_root.iterdir()):
        if not comp_dir.is_dir():
            continue
        for genre_dir in sorted(comp_dir.iterdir()):
            if not genre_dir.is_dir():
                continue
            if list(genre_dir.glob("*.wav")):
                groups.append(Path(comp_dir.name) / genre_dir.name)
    return groups


def run_ref_internal_fad(
    reference_path: str,
    out_dir: str,
    cache_dir: str | None = None,
) -> None:
    ref_root = Path(reference_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    emb_cache = Path(cache_dir) if cache_dir else out_path / "ref_emb_cache"

    groups = _discover_groups(ref_root)
    if not groups:
        print(f"No composer/genre groups under {ref_root}")
        return

    _maybe_debug_torch()
    fad = _load_frechet_audio_distance()(model_name="vggish", use_pca=False)
    ref_cache = build_ref_embedding_cache(fad, ref_root, groups, emb_cache)

    piece_rows: list[dict] = []
    summary_rows: list[dict] = []
    skipped: list[tuple[str, str, str]] = []

    for rel in groups:
        composer, genre = rel.parent.name, rel.name
        key = _cache_key(composer, genre)
        ref_files = sorted((ref_root / rel).glob("*.wav"))

        if key not in ref_cache:
            reason = "only 1 reference file" if len(ref_files) < 2 else "embedding failed"
            skipped.append((composer, genre, reason))
            if len(ref_files) == 1:
                piece_rows.append({
                    "composer": composer,
                    "genre": genre,
                    "wav_name": ref_files[0].name,
                    "per_piece_fad": "",
                    "note": "n<2, skipped",
                })
            continue

        entry = ref_cache[key]
        scores = entry["ref_scores"]
        summary = entry["ref_summary"]
        summary_row = {
            "composer": composer,
            "genre": genre,
            **summary,
            "cv": summary["std"] / summary["mean"] if summary.get("mean", 0) > 0 else 0.0,
        }
        summary_rows.append(summary_row)

        for wav, score in zip(entry["ref_names"], scores):
            piece_rows.append({
                "composer": composer,
                "genre": genre,
                "wav_name": wav,
                "per_piece_fad": score,
                "note": "",
            })

    piece_csv = out_path / "ref_internal_fad_per_piece.csv"
    with piece_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["composer", "genre", "wav_name", "per_piece_fad", "note"],
        )
        writer.writeheader()
        writer.writerows(piece_rows)

    summary_csv = out_path / "ref_internal_fad_summary.csv"
    if summary_rows:
        fieldnames = ["composer", "genre", "n", "mean", "std", "cv", "min", "p25", "median", "p75", "p90", "max"]
        summary_rows = sorted(summary_rows, key=lambda r: r["std"], reverse=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Groups analyzed: {len(summary_rows)}")
    print(f"Skipped: {len(skipped)}")
    for c, g, why in skipped:
        print(f"  {c}/{g}: {why}")
    print(f"Wrote {piece_csv} ({len(piece_rows)} rows)")
    print(f"Wrote {summary_csv} ({len(summary_rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-only internal FAD per composer/genre.")
    parser.add_argument(
        "--reference_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "asap_test_set_wav"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/inference/ref_internal_fad"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/inference/per_piece_fad/ref_emb_cache"),
        help="Reuse precomputed reference embeddings if available.",
    )
    args = parser.parse_args()

    run_ref_internal_fad(
        reference_path=args.reference_path,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
