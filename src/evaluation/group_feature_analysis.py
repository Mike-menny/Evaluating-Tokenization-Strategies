#!/usr/bin/env python3
"""
Merge FAD scores, MIDI features, token counts, and per-piece summaries;
compute Pearson/Spearman correlations vs mean_fad.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.evaluation.midi_group_features import run_midi_group_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_OUT = PROJECT_ROOT / "outputs/inference"
FAD_COMPARISON = INFERENCE_OUT / "fad_comparison.csv"
MIDI_SUMMARY = INFERENCE_OUT / "midi_group_features_summary.csv"


def load_token_stats_by_group(cache_dir: Path) -> pd.DataFrame:
    """Aggregate token counts from pretokenized cache/{split}.pt files."""
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        pt = cache_dir / f"{split}.pt"
        if not pt.is_file():
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        by_group: dict[tuple[str, str], list[int]] = {}
        for sample in payload["samples"]:
            key = (sample["composer"], sample["genre"])
            n_tok = int(sample["tokens"].numel())
            by_group.setdefault(key, []).append(n_tok)
        for (composer, genre), lengths in by_group.items():
            arr = np.asarray(lengths, dtype=np.float64)
            rows.append({
                "split": split,
                "composer": composer,
                "genre": genre,
                "token_total": float(arr.sum()),
                "token_mean": float(arr.mean()),
                "token_median": float(np.median(arr)),
                "n_windows": int(arr.size),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide_rows: list[dict] = []
    for (composer, genre), g in df.groupby(["composer", "genre"]):
        row: dict = {"composer": composer, "genre": genre}
        for split, sg in g.groupby("split"):
            r = sg.iloc[0]
            row[f"token_total_{split}"] = r["token_total"]
            row[f"token_mean_{split}"] = r["token_mean"]
            row[f"token_median_{split}"] = r["token_median"]
            row[f"n_windows_{split}"] = r["n_windows"]
        train = g[g["split"] == "train"]
        if not train.empty:
            r = train.iloc[0]
            row["token_total_train"] = r["token_total"]
            row["token_mean_train"] = r["token_mean"]
            row["token_median_train"] = r["token_median"]
            row["n_windows_train"] = r["n_windows"]
        row["token_total_all"] = float(g["token_total"].sum())
        row["token_mean_all"] = float(g["token_total"].sum() / g["n_windows"].sum())
        row["n_windows_all"] = int(g["n_windows"].sum())
        wide_rows.append(row)
    return pd.DataFrame(wide_rows)


def build_group_midi_stats_vs_fad(
    fad_csv: Path,
    midi_summary_csv: Path,
    token_df: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    fad = pd.read_csv(fad_csv)
    fad = fad[fad["composer"] != "MEAN"].copy()
    model_cols = [c for c in fad.columns if c not in ("composer", "genre")]
    fad["mean_fad"] = fad[model_cols].mean(axis=1)

    midi = pd.read_csv(midi_summary_csv)
    df = fad.merge(midi, on=["composer", "genre"], how="left")
    if not token_df.empty:
        df = df.merge(token_df, on=["composer", "genre"], how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def compute_correlations(
    df: pd.DataFrame,
    target: str = "mean_fad",
    out_csv: Path | None = None,
) -> pd.DataFrame:
    skip = {"composer", "genre", target, *[
        c for c in df.columns if c.startswith("asap_")
    ]}
    rows = []
    y = df[target]
    for col in df.columns:
        if col in skip:
            continue
        mask = df[col].notna() & y.notna()
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(df.loc[mask, col], y.loc[mask])
        rho, _ = stats.spearmanr(df.loc[mask, col], y.loc[mask])
        rows.append({
            "feature": col,
            "pearson_r": r,
            "spearman_r": rho,
            "p": p,
            "n": int(mask.sum()),
        })
    corr = pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    if out_csv:
        corr.to_csv(out_csv, index=False)
    return corr


def build_per_piece_fad_analysis(
    group_summary_csv: Path,
    fad_csv: Path,
    ref_summary_csv: Path,
    run_keyword: str = "note_pedal",
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Merge per-piece note_pedal stats with mean_fad and ref internal stats."""
    gs = pd.read_csv(group_summary_csv)
    gs = gs[gs["run"].str.contains(run_keyword, na=False)].copy()
    fad = pd.read_csv(fad_csv)
    fad = fad[fad["composer"] != "MEAN"]
    model_cols = [c for c in fad.columns if c not in ("composer", "genre")]
    fad["mean_fad"] = fad[model_cols].mean(axis=1)

    ref = pd.read_csv(ref_summary_csv)
    ref = ref.rename(columns={"mean": "ref_internal_mean", "std": "ref_internal_std"})

    fad_col = "asap_note_pedal" if "note_pedal" in run_keyword else run_keyword
    merged = gs.merge(
        fad[["composer", "genre", "mean_fad", fad_col]],
        on=["composer", "genre"],
        how="left",
    )
    merged = merged.merge(
        ref[["composer", "genre", "ref_internal_mean", "ref_internal_std"]],
        on=["composer", "genre"],
        how="left",
    )
    merged["max_share"] = merged["gen_max"] / merged["gen_group_pooled_fad"]
    merged = merged.rename(columns={
        "ref_internal_std": "ref_std",
        "ref_internal_mean": "ref_mean",
        "gen_std": "gen_std",
        "gen_mean": "gen_mean",
    })
    cols = [
        "composer", "genre", "asap_note_pedal", "gen_mean", "gen_std", "gen_max", "gen_min",
        "gen_median", "gen_group_pooled_fad", "ref_std", "ref_mean", "mean_fad", "max_share",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged = merged.sort_values("mean_fad", ascending=False)
    if out_csv:
        merged.to_csv(out_csv, index=False)
    return merged


def build_ref_vs_gen_variance(
    ref_summary_csv: Path,
    per_piece_analysis_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    ref = pd.read_csv(ref_summary_csv)
    gen = pd.read_csv(per_piece_analysis_csv)
    df = gen.merge(ref, on=["composer", "genre"], suffixes=("_gen", "_ref"))
    if "std" in df.columns:
        df = df.rename(columns={"std": "ref_internal_std", "mean": "ref_internal_mean", "n": "ref_n", "cv": "ref_cv"})
    if "ref_internal_std" in df.columns and "gen_std" in df.columns:
        df["var_ratio"] = df["gen_std"] / df["ref_internal_std"].replace(0, np.nan)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_all(
    cache_dir: Path,
    test_set_root: Path,
    skip_midi: bool = False,
) -> None:
    out = INFERENCE_OUT
    if not skip_midi:
        run_midi_group_features(test_set_root, out / "midi_group_features_per_file.csv")

    token_df = load_token_stats_by_group(cache_dir)
    if not token_df.empty:
        token_df.to_csv(out / "token_stats_by_split_group.csv", index=False)

    df = build_group_midi_stats_vs_fad(
        FAD_COMPARISON,
        MIDI_SUMMARY,
        token_df,
        out / "group_midi_stats_vs_fad.csv",
    )
    compute_correlations(df, out_csv=out / "mean_fad_feature_correlations_full.csv")

    summary_cols = [
        "composer", "genre", "mean_fad",
        "notes_per_s_mean", "mean_note_dur_mean", "velocity_std_mean",
        "pitch_range_mean", "pitch_entropy_mean", "ioi_cv_mean",
        "simul_onset_ratio_mean", "cc64_per_min_mean",
        "duration_s_mean", "note_count_mean",
        "token_total_train", "n_windows_train", "token_mean_train",
    ]
    df[[c for c in summary_cols if c in df.columns]].to_csv(
        out / "fad_feature_summary_19groups.csv", index=False
    )

    ref_summary = out / "ref_internal_fad/ref_internal_fad_summary.csv"
    per_piece_summary = out / "per_piece_fad/per_piece_fad_group_summary.csv"
    if per_piece_summary.is_file():
        build_per_piece_fad_analysis(
            per_piece_summary, FAD_COMPARISON, ref_summary,
            out_csv=out / "per_piece_fad/per_piece_fad_analysis.csv",
        )
    if ref_summary.is_file() and (out / "per_piece_fad/per_piece_fad_analysis.csv").is_file():
        build_ref_vs_gen_variance(
            ref_summary,
            out / "per_piece_fad/per_piece_fad_analysis.csv",
            out / "ref_internal_fad/ref_vs_gen_variance.csv",
        )

    print(f"Done. Wrote tables under {INFERENCE_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged feature tables and correlation CSVs.")
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "cache/conditional_asap_full"))
    parser.add_argument(
        "--test_set_root",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "asap_test_set"),
    )
    parser.add_argument("--skip_midi", action="store_true", help="Skip MIDI re-extraction if summary exists.")
    args = parser.parse_args()
    run_all(Path(args.cache_dir), Path(args.test_set_root), skip_midi=args.skip_midi)


if __name__ == "__main__":
    main()
