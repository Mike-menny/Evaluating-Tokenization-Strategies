#!/usr/bin/env python3
"""
Group dataset by composer and genre: copy each MIDI to
  output_root / [augmented/] / composer / genre / filename.midi
- If midi_filename contains "augmented", add "augmented" under output_root.
- Genre is extracted from user_prompt ("please generate a X in the style of ...")
  or from prompt ("It should be a/an X in ...") if user_prompt is missing.
- If no genre found, use "no_genre".
- Writes final.csv under output_root: same rows as copied, midi_filename = path under grouped, audio_filename dropped.
"""

import argparse
import re
import shutil
import pandas as pd
from pathlib import Path

# Default roots (project_root / maestro_dataset)
def _default_root():
    return Path(__file__).resolve().parents[2] / "maestro_dataset"

# user_prompt: "please generate a sonata in the style of ..." or "please generate a piano piece in ..."
GENRE_IN_USER_PROMPT_RE = re.compile(
    r"please generate a ([^.]+?) in the style of",
    re.IGNORECASE
)
# Legacy prompt column: "It should be a Sonata in ..." or "It should be an Etude in ..."
GENRE_IN_PROMPT_RE = re.compile(
    r"It should be (?:a|an) ([^\s,]+) (?:in|-)",
    re.IGNORECASE
)


def sanitize_dirname(s: str) -> str:
    """Make string safe for use as directory name."""
    if not s or not str(s).strip():
        return "unknown"
    s = str(s).strip()
    for c in r'\/:*?"<>|':
        s = s.replace(c, "_")
    return s.strip("._") or "unknown"


def get_genre_from_row(row: pd.Series) -> str | None:
    """Extract genre from user_prompt (preferred) or prompt. Returns None if not found."""
    # Prefer user_prompt: "please generate a sonata in the style of ..."
    s = row.get("user_prompt")
    if s and str(s).strip():
        m = GENRE_IN_USER_PROMPT_RE.search(str(s))
        if m:
            return m.group(1).strip()
    # Fallback: prompt column "It should be a/an X in ..."
    s = row.get("prompt")
    if s and str(s).strip():
        m = GENRE_IN_PROMPT_RE.search(str(s))
        if m:
            return m.group(1).strip()
    return None


def main():
    default_dataset = _default_root()

    parser = argparse.ArgumentParser(
        description="Group dataset: copy MIDIs to output_root/[augmented/]composer/genre/."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=default_dataset / "augmented.csv",
        help="Input CSV with canonical_composer, midi_filename, prompt.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset,
        help="Root where midi_filename paths are resolved.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_dataset / "grouped",
        help="Output root (default: <dataset>/grouped).",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    for col in ["canonical_composer", "midi_filename"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    has_genre_source = "user_prompt" in df.columns or "prompt" in df.columns
    if not has_genre_source:
        print("Warning: no 'user_prompt' or 'prompt' column; all rows will use genre 'no_genre'.")

    copied = 0
    skipped = 0
    final_rows = []

    for idx, row in df.iterrows():
        midi_rel = str(row["midi_filename"]).strip()
        if not midi_rel:
            skipped += 1
            continue

        src = dataset_root / midi_rel
        if not src.exists():
            print(f"Warning: missing {src}, skip.")
            skipped += 1
            continue

        is_augmented = "augmented" in midi_rel
        composer = sanitize_dirname(row["canonical_composer"])
        genre = get_genre_from_row(row)
        genre_dir = sanitize_dirname(genre) if genre else "no_genre"
        # Output as .mid (source may be .midi)
        filename = Path(midi_rel).stem + ".mid"

        # Path under grouped (for final.csv midi_filename)
        rel_parts = [composer, genre_dir, filename]
        if is_augmented:
            rel_parts.insert(0, "augmented")
        grouped_rel = "/".join(rel_parts)

        # output_root / [augmented/] / composer / genre / filename
        out_dir = output_root / "augmented" / composer / genre_dir if is_augmented else output_root / composer / genre_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        shutil.copy2(src, out_path)
        copied += 1

        # Row for final.csv: same as input but midi_filename -> grouped path, drop audio_filename
        new_row = row.drop(labels=["audio_filename"], errors="ignore").to_dict()
        new_row["midi_filename"] = grouped_rel
        final_rows.append(new_row)

    # Write final.csv under output_root (no audio_filename, midi_filename = path under grouped)
    if final_rows:
        out_cols = [c for c in df.columns if c != "audio_filename"]
        final_df = pd.DataFrame(final_rows, columns=out_cols)
        final_csv = Path(args.dataset_root) / "final.csv"
        final_df.to_csv(final_csv, index=False, encoding="utf-8")
        print(f"Wrote {final_csv} ({len(final_df)} rows).")

    print(f"Done: copied {copied} files to {output_root}, skipped {skipped}.")


if __name__ == "__main__":
    main()
