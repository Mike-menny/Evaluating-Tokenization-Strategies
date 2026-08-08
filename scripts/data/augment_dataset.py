#!/usr/bin/env python3
"""
Augment the piano dataset: for each original MIDI, create pitch/time/velocity variants.
Output CSV is compatible with generate_prompt.py (includes system_prompt, user_prompt).
If the input CSV lacks system_prompt/user_prompt, they are generated from canonical_composer/canonical_title.
Augmentations are applied to ALL original rows (no random subset).
"""

import argparse
import csv
import random
import mido
import pandas as pd
from pathlib import Path

try:
    from generate_prompt import SYSTEM_PROMPT, get_genre_from_title
except ImportError:
    SYSTEM_PROMPT = "You are a world-class piano composer. Please compose some music according to the user's description: "

    def get_genre_from_title(title: str):
        t = title.lower()
        for keyword, label in _GENRE_KEYWORDS:
            if keyword.lower() in t:
                return label
        return None

    _GENRE_KEYWORDS = [
        ("sonata", "sonata"), ("sonatas", "sonata"), ("sonate", "sonata"), ("son.", "sonata"), ("sonatina", "sonatina"),
        ("prelude", "prelude"), ("preludes", "prelude"), ("prélude", "prelude"), ("préludes", "prelude"), ("prel.", "prelude"),
        ("etude", "etude"), ("etudes", "etude"), ("étude", "etude"), ("études", "etude"), ("ètudes", "etude"),
        ("nocturne", "nocturne"), ("nocturnes", "nocturne"), ("waltz", "waltz"), ("waltzes", "waltz"), ("valse", "waltz"),
        ("mazurka", "mazurka"), ("ballade", "ballade"), ("scherzo", "scherzo"), ("impromptu", "impromptu"),
        ("fantasy", "fantasy"), ("fantasie", "fantasy"), ("fantasia", "fantasy"), ("suite", "suite"), ("concerto", "concerto"),
        ("variation", "variation"), ("variations", "variation"), ("var.", "variation"), ("fugue", "fugue"), ("chorale", "chorale"),
        ("barcarolle", "barcarolle"), ("rhapsody", "rhapsody"), ("rhapsodie", "rhapsody"), ("caprice", "caprice"), ("capriccio", "capriccio"),
        ("intermezzo", "intermezzo"), ("romance", "romance"), ("march", "march"), ("polonaise", "polonaise"),
        ("toccata", "toccata"), ("bagatelle", "bagatelle"), ("rondo", "rondo"), ("minuet", "minuet"), ("gavotte", "gavotte"),
        ("sarabande", "sarabande"), ("invention", "invention"), ("partita", "partita"), ("berceuse", "berceuse"), ("chaconne", "chaconne"),
        ("moment musical", "moment musical"), ("moments musicaux", "moment musical"), ("dumka", "dumka"), ("elegie", "elegy"), ("elegy", "elegy"),
        ("humoresque", "humoresque"), ("tarantella", "tarantella"), ("paraphrase", "paraphrase"), ("overture", "overture"),
        ("arabesque", "arabesque"), ("carnaval", "carnaval"), ("kreisleriana", "kreisleriana"), ("pavan", "pavan"), ("duet", "duet"),
        ("image", "image"), ("images", "image"), ("estampes", "estampes"), ("poème", "poème"), ("morceau", "morceau"), ("piece", "piece"), ("tango", "tango"),
    ]


def _ensure_prompt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure system_prompt and user_prompt exist; fill from canonical_composer/canonical_title if missing."""
    if "system_prompt" not in df.columns:
        df["system_prompt"] = ""
    if "user_prompt" not in df.columns:
        df["user_prompt"] = ""

    need_prompt = df["user_prompt"].isna() | (df["user_prompt"].astype(str).str.strip() == "")
    if not need_prompt.any():
        return df

    for idx in df.index[need_prompt]:
        composer = str(df.at[idx, "canonical_composer"]).strip()
        title = str(df.at[idx, "canonical_title"]).strip()
        genre = get_genre_from_title(title)
        df.at[idx, "system_prompt"] = SYSTEM_PROMPT
        if genre is not None:
            df.at[idx, "user_prompt"] = f"please generate a {genre} in the style of {composer}."
        else:
            df.at[idx, "user_prompt"] = f"please generate a piano piece in the style of {composer}."
    return df


def clamp(val, min_val=0, max_val=127):
    return max(min_val, min(max_val, int(val)))


def augment_pitch(midi_path, output_path, semitones):
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type in ["note_on", "note_off"]:
                    msg.note = clamp(msg.note + semitones)
        mid.save(output_path)
        return True
    except Exception as e:
        print(f"Error augmenting pitch for {midi_path}: {e}")
        return False


def augment_time(midi_path, output_path, factor):
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                msg.time = int(msg.time * factor)
        mid.save(output_path)
        return True
    except Exception as e:
        print(f"Error augmenting time for {midi_path}: {e}")
        return False


def augment_velocity(midi_path, output_path, factor):
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on":
                    msg.velocity = clamp(msg.velocity * factor)
        mid.save(output_path)
        return True
    except Exception as e:
        print(f"Error augmenting velocity for {midi_path}: {e}")
        return False


def main():
    script_dir = Path(__file__).resolve().parent
    default_root = Path(__file__).resolve().parents[2] / "maestro_dataset"

    parser = argparse.ArgumentParser(
        description="Augment the dataset: create pitch/time/velocity variants for ALL originals. CSV compatible with generate_prompt (system_prompt, user_prompt)."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=default_root / "with_prompt.csv",
        help="Input CSV (default: maestro_dataset/with_prompt.csv). Must have canonical_composer, canonical_title.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        default=default_root / "augmented.csv",
        help="Output CSV path (default: maestro_dataset/augmented.csv).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help="Dataset root where midi_filename paths are resolved (default: maestro_dataset/).",
    )
    parser.add_argument(
        "--augmented-dir",
        type=Path,
        default=None,
        help="Directory for augmented MIDI files (default: <dataset_root>/augmented).",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=None,
        help="Limit number of original files to process (for testing). By default process all.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    augmented_dir = Path(args.augmented_dir) if args.augmented_dir else dataset_root / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Loading CSV from {input_csv}...")
    df = pd.read_csv(input_csv)

    if "canonical_composer" not in df.columns or "canonical_title" not in df.columns:
        raise ValueError("Input CSV must contain canonical_composer and canonical_title.")

    df = _ensure_prompt_columns(df)

    # Only process rows whose midi is not already under augmented/
    original_mask = ~df["midi_filename"].astype(str).str.startswith("augmented/")
    df_originals = df[original_mask]
    all_indices = df_originals.index.tolist()

    if args.max_sample is not None:
        indices = random.sample(all_indices, min(len(all_indices), args.max_sample))
        print(f"Testing: processing subset of {len(indices)} originals (--max-sample={args.max_sample}).")
    else:
        indices = all_indices
        print(f"Processing ALL {len(indices)} original rows (full augmentation).")

    new_rows = []

    def out_rel_dir(original_midi_rel_path):
        d = str(original_midi_rel_path)
        if d.startswith("augmented/"):
            d = d.replace("augmented/", "", 1)
        return str(Path(d).parent) if Path(d).parent != Path(".") else ""

    # Semitone options for pitch; we apply ALL of them to EVERY original
    semitones_options = [-3, -2, -1, 1, 2, 3]
    # 1. Pitch shift — ALL originals, 6 semitones each (-3,-2,-1,1,2,3)
    print(f"Applying Pitch Shift to all {len(indices)} files (6 semitones each)...")
    for idx in indices:
        row = df.loc[idx].copy()
        original_midi_rel_path = row["midi_filename"]
        original_midi_path = dataset_root / original_midi_rel_path
        if not original_midi_path.exists():
            print(f"Warning: File not found {original_midi_path}, skipping.")
            continue
        selected_semitones = random.sample(semitones_options, 6)
        rel_dir = out_rel_dir(original_midi_rel_path)
        for semitone in selected_semitones:
            suffix = f"_pitch_{semitone}"
            filename = Path(original_midi_rel_path).stem + suffix + ".midi"
            save_dir = augmented_dir / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / filename
            rel_output_path = str(Path("augmented") / rel_dir / filename).replace("\\", "/")
            if augment_pitch(str(original_midi_path), str(output_path), semitone):
                new_row = row.copy()
                new_row["midi_filename"] = rel_output_path
                new_row["audio_filename"] = ""
                new_rows.append(new_row)

    # 2. Time stretch — ALL originals, 5 factors each (0.8–1.2)
    print(f"Applying Time Stretch to all {len(indices)} files (5 factors each)...")
    for idx in indices:
        row = df.loc[idx].copy()
        original_midi_rel_path = row["midi_filename"]
        original_midi_path = dataset_root / original_midi_rel_path
        if not original_midi_path.exists():
            continue
        rel_dir = out_rel_dir(original_midi_rel_path)
        factors = [round(random.uniform(0.8, 1.2), 5) for _ in range(5)]
        for factor in factors:
            suffix = f"_time_{factor:.2f}"
            filename = Path(original_midi_rel_path).stem + suffix + ".midi"
            save_dir = augmented_dir / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / filename
            rel_output_path = str(Path("augmented") / rel_dir / filename).replace("\\", "/")
            if augment_time(str(original_midi_path), str(output_path), factor):
                new_row = row.copy()
                new_row["midi_filename"] = rel_output_path
                new_row["audio_filename"] = ""
                try:
                    new_row["duration"] = float(row["duration"]) * factor
                except (TypeError, ValueError):
                    pass
                new_rows.append(new_row)

    # 3. Velocity scale — ALL originals, 5 factors each (0.9–1.1)
    print(f"Applying Velocity Scale to all {len(indices)} files (5 factors each)...")
    for idx in indices:
        row = df.loc[idx].copy()
        original_midi_rel_path = row["midi_filename"]
        original_midi_path = dataset_root / original_midi_rel_path
        if not original_midi_path.exists():
            continue
        rel_dir = out_rel_dir(original_midi_rel_path)
        factors = [round(random.uniform(0.9, 1.1), 5) for _ in range(5)]
        for factor in factors:
            suffix = f"_vel_{factor:.2f}"
            filename = Path(original_midi_rel_path).stem + suffix + ".midi"
            save_dir = augmented_dir / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / filename
            rel_output_path = str(Path("augmented") / rel_dir / filename).replace("\\", "/")
            if augment_velocity(str(original_midi_path), str(output_path), factor):
                new_row = row.copy()
                new_row["midi_filename"] = rel_output_path
                new_row["audio_filename"] = ""
                new_rows.append(new_row)

    # Original rows + augmented rows
    new_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, new_df], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Added {len(new_rows)} augmented rows. Saved {len(final_df)} total rows to {output_csv}")


if __name__ == "__main__":
    main()
