#!/usr/bin/env python3
"""Plot per-category FAD scores for multiple inference runs on one chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNS = [
    ("asap_full", "outputs/inference/asap_full/2026-06-13_165541/fad.csv"),
    ("asap_note", "outputs/inference/asap_note/2026-06-13_173127/fad.csv"),
    ("asap_note_pedal", "outputs/inference/asap_note_pedal/2026-06-17_102100/fad.csv"),
    ("asap_note_velocity", "outputs/inference/asap_note_velocity/2026-06-16_104550/fad.csv"),
    ("asap_note_velocity_beat", "outputs/inference/asap_note_velocity_beat/2026-06-16_105456/fad.csv"),
    ("asap_note_velocity_pedal", "outputs/inference/asap_note_velocity_pedal/2026-06-13_180400/fad.csv"),
]


def _category_label(composer: str, genre: str) -> str:
    return f"{composer}\n{genre}"


def load_run_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~((df["composer"] == "OVERALL") & (df["genre"] == "OVERALL"))]
    return df[["composer", "genre", "fad_score"]].astype({"fad_score": float})


def build_fad_summary_table(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    """Wide table: composer, genre, one column per tokenization, plus MEAN row."""
    merged: pd.DataFrame | None = None
    model_names: list[str] = []

    for name, csv_path in runs:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing FAD CSV: {csv_path}")
        part = load_run_table(csv_path).rename(columns={"fad_score": name})
        model_names.append(name)
        merged = part if merged is None else merged.merge(part, on=["composer", "genre"], how="outer")

    assert merged is not None
    merged = merged.sort_values(["composer", "genre"]).reset_index(drop=True)

    mean_row = {name: merged[name].mean() for name in model_names}
    mean_row["composer"] = "MEAN"
    mean_row["genre"] = "MEAN"
    return pd.concat([merged, pd.DataFrame([mean_row])], ignore_index=True)


def export_fad_summary_csv(
    runs: list[tuple[str, Path]],
    output_path: Path,
) -> pd.DataFrame:
    table = build_fad_summary_table(runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved CSV -> {output_path}")
    return table


def load_run_scores(csv_path: Path) -> pd.Series:
    df = load_run_table(csv_path)
    df["category"] = df.apply(lambda r: _category_label(r["composer"], r["genre"]), axis=1)
    return df.set_index("category")["fad_score"]


def plot_fad_comparison(
    runs: list[tuple[str, Path]],
    output_path: Path,
    title: str = "FAD by composer / genre",
) -> None:
    series_by_run: dict[str, pd.Series] = {}
    categories: list[str] = []

    for name, csv_path in runs:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing FAD CSV: {csv_path}")
        series_by_run[name] = load_run_scores(csv_path)
        if not categories:
            categories = list(series_by_run[name].index)

    # Keep a stable x order; append any extras from other runs.
    seen = set(categories)
    for s in series_by_run.values():
        for cat in s.index:
            if cat not in seen:
                categories.append(cat)
                seen.add(cat)

    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(16, 6))

    markers = ["o", "s", "^", "D", "v"]
    for i, (name, series) in enumerate(series_by_run.items()):
        y = [series.get(cat, float("nan")) for cat in categories]
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=name,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("FAD")
    ax.set_xlabel("Composer / genre")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FAD comparison across inference runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/inference/fad_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/inference/fad_comparison.csv",
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    runs = [(name, PROJECT_ROOT / rel) for name, rel in DEFAULT_RUNS]
    plot_fad_comparison(runs, args.output.resolve())
    export_fad_summary_csv(runs, args.csv_output.resolve())


if __name__ == "__main__":
    main()
