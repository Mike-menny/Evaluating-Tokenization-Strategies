#!/usr/bin/env python3
"""
Add system_prompt and user_prompt columns to a CSV.
- system_prompt: fixed instruction for the model.
- user_prompt: "please generate a <genre> in the style of <composer>." or
  "please generate a piano piece in the style of <composer>." when no genre is detected.
Genre is inferred from canonical_title via keyword matching.
"""

import argparse
import csv
import os
from pathlib import Path

# Keywords to detect genre from canonical_title (case-insensitive substring).
# First match wins; value is the label used in user_prompt (e.g. "sonata", "prelude").
GENRE_KEYWORDS: list[tuple[str, str]] = [
    ("sonata", "sonata"),
    ("sonatas", "sonata"),
    ("sonate", "sonata"),
    ("son.", "sonata"),
    ("sonatina", "sonatina"),
    ("prelude", "prelude"),
    ("preludes", "prelude"),
    ("prélude", "prelude"),
    ("préludes", "prelude"),
    ("prel.", "prelude"),
    ("etude", "etude"),
    ("etudes", "etude"),
    ("étude", "etude"),
    ("études", "etude"),
    ("ètudes", "etude"),
    ("nocturne", "nocturne"),
    ("nocturnes", "nocturne"),
    ("waltz", "waltz"),
    ("waltzes", "waltz"),
    ("valse", "waltz"),
    ("mazurka", "mazurka"),
    ("mazurkas", "mazurka"),
    ("ballade", "ballade"),
    ("ballades", "ballade"),
    ("scherzo", "scherzo"),
    ("scherzi", "scherzo"),
    ("impromptu", "impromptu"),
    ("impromptus", "impromptu"),
    ("fantasy", "fantasy"),
    ("fantasie", "fantasy"),
    ("fantaisie", "fantasy"),
    ("fantasias", "fantasy"),
    ("fantasia", "fantasy"),
    ("suite", "suite"),
    ("suites", "suite"),
    ("concerto", "concerto"),
    ("concertos", "concerto"),
    ("variation", "variation"),
    ("variations", "variation"),
    ("var.", "variation"),
    ("fugue", "fugue"),
    ("fugues", "fugue"),
    ("chorale", "chorale"),
    ("choral", "chorale"),
    ("barcarolle", "barcarolle"),
    ("barcarolles", "barcarolle"),
    ("rhapsody", "rhapsody"),
    ("rhapsodies", "rhapsody"),
    ("rhapsodie", "rhapsody"),
    ("caprice", "caprice"),
    ("caprices", "caprice"),
    ("capriccio", "capriccio"),
    ("intermezzo", "intermezzo"),
    ("intermezzi", "intermezzo"),
    ("romance", "romance"),
    ("romances", "romance"),
    ("march", "march"),
    ("marches", "march"),
    ("polonaise", "polonaise"),
    ("polonaises", "polonaise"),
    ("toccata", "toccata"),
    ("toccatas", "toccata"),
    ("bagatelle", "bagatelle"),
    ("bagatelles", "bagatelle"),
    ("rondo", "rondo"),
    ("rondos", "rondo"),
    ("minuet", "minuet"),
    ("minuets", "minuet"),
    ("gavotte", "gavotte"),
    ("sarabande", "sarabande"),
    ("invention", "invention"),
    ("inventions", "invention"),
    ("partita", "partita"),
    ("partitas", "partita"),
    ("berceuse", "berceuse"),
    ("chaconne", "chaconne"),
    ("moment musical", "moment musical"),
    ("moments musicaux", "moment musical"),
    ("dumka", "dumka"),
    ("elegie", "elegy"),
    ("elegy", "elegy"),
    ("humoresque", "humoresque"),
    ("tarantella", "tarantella"),
    ("paraphrase", "paraphrase"),
    ("overture", "overture"),
    ("arabesque", "arabesque"),
    ("carnaval", "carnaval"),
    ("kreisleriana", "kreisleriana"),
    ("pavan", "pavan"),
    ("duet", "duet"),
    ("image", "image"),
    ("images", "image"),
    ("estampes", "estampes"),
    ("poème", "poème"),
    ("morceau", "morceau"),
    ("piece", "piece"),
    ("tango", "tango"),
]

SYSTEM_PROMPT = "You are a world-class piano composer. Please compose some music according to the user's description: "


def get_genre_from_title(title: str) -> str | None:
    """Return the first matching genre label for title, or None."""
    t = title.lower()
    for keyword, label in GENRE_KEYWORDS:
        if keyword.lower() in t:
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add system_prompt and user_prompt columns to a CSV and write to an output folder."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "maestro_dataset" / "maestro-v3.0.0.csv",
        help="Path to the input CSV (default: maestro_dataset/maestro-v3.0.0.csv).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "maestro_dataset",
        help="Output folder where the new CSV will be written (default: maestro_dataset/).",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        type=str,
        default="with_prompt.csv",
        help="Output CSV filename (default: same as input_csv basename).",
    )
    args = parser.parse_args()

    input_path = args.input_csv
    output_dir = args.output_dir
    output_name = args.output_name or input_path.name
    output_path = output_dir / output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "canonical_composer" not in fieldnames or "canonical_title" not in fieldnames:
            raise ValueError(
                "Input CSV must contain columns: canonical_composer, canonical_title"
            )
        for row in reader:
            rows.append(dict(row))

    if "system_prompt" not in fieldnames:
        fieldnames.append("system_prompt")
    if "user_prompt" not in fieldnames:
        fieldnames.append("user_prompt")

    for row in rows:
        row["system_prompt"] = SYSTEM_PROMPT
        composer = row.get("canonical_composer", "").strip()
        title = row.get("canonical_title", "").strip()
        genre = get_genre_from_title(title)
        if genre is not None:
            row["user_prompt"] = f"please generate a {genre} in the style of {composer}."
        else:
            row["user_prompt"] = f"please generate a piano piece in the style of {composer}."

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
