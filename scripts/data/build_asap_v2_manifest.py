#!/usr/bin/env python3
"""
Build CSV manifest for train_v2: midi_path, annotation_path, system_prompt, user_prompt, split.
Scans <repo>/asap-dataset-master/asap-dataset-master and optional augmented/.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

PROMPT_SCORE_SYS = (
    "You are a world-class piano composer. Please compose some music according to the user's description: "
)
PROMPT_PERF_SYS = (
    "You are a world-class piano composer. Please compose some expressive and human-like "
    "musical performance according to the user's description: "
)


def _piece_key(rel_midi: Path) -> str:
    return str(rel_midi.parent)


def _split_from_key(piece_key: str) -> str:
    h = int(hashlib.md5(piece_key.encode("utf-8")).hexdigest(), 16) % 10
    if h < 8:
        return "train"
    if h == 8:
        return "validation"
    return "test"


def _prompts_for(rel_midi: Path) -> tuple[str, str]:
    parts = rel_midi.parts
    composer = parts[0] if parts else "Unknown"
    genre = parts[1].lower() if len(parts) > 1 else "piano piece"
    user = f"please generate a {genre} in the style of {composer}."
    stem = rel_midi.stem
    if stem == "midi_score":
        return PROMPT_SCORE_SYS, user
    return PROMPT_PERF_SYS, user


def _collect_rows(scan_root: Path, path_prefix: str) -> list[dict]:
    rows: list[dict] = []
    for ext in (".mid", ".midi"):
        for midi_path in sorted(scan_root.rglob(f"*{ext}")):
            if not midi_path.is_file():
                continue
            ann = midi_path.parent / f"{midi_path.stem}_annotations.txt"
            if not ann.is_file():
                continue
            try:
                rel_m = midi_path.relative_to(scan_root)
            except ValueError:
                continue
            rel_a = ann.relative_to(scan_root)
            midi_rel = f"{path_prefix}/{rel_m.as_posix()}"
            ann_rel = f"{path_prefix}/{rel_a.as_posix()}"
            piece_key = _piece_key(Path(path_prefix) / rel_m)
            sys_p, user_p = _prompts_for(rel_m)
            rows.append(
                {
                    "midi_path": midi_rel,
                    "annotation_path": ann_rel,
                    "system_prompt": sys_p,
                    "user_prompt": user_p,
                    "split": _split_from_key(piece_key),
                }
            )
    return rows


def main():
    project_root = Path(__file__).resolve().parents[2]
    repo_asap = project_root / "asap-dataset-master"
    default_data = repo_asap / "asap-dataset-master"
    default_aug = repo_asap / "augmented"
    default_out = repo_asap / "asap_v2_manifest.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--augmented-dir", type=Path, default=default_aug)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--skip-augmented", action="store_true")
    args = parser.parse_args()

    all_rows: list[dict] = []
    if args.data_dir.is_dir():
        all_rows.extend(_collect_rows(args.data_dir, "asap-dataset-master"))
    if not args.skip_augmented and args.augmented_dir.is_dir():
        all_rows.extend(_collect_rows(args.augmented_dir, "augmented"))

    df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
