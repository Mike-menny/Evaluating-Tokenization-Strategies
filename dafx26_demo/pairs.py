from __future__ import annotations

from dataclasses import dataclass

OBSERVED_PAIRS: tuple[tuple[str, str], ...] = (
    ("Bach", "fugue"),
    ("Bach", "prelude"),
    ("Beethoven", "sonata"),
    ("Chopin", "ballade"),
    ("Chopin", "etude"),
    ("Chopin", "sonata"),
    ("Debussy", "images"),
    ("Debussy", "suite"),
    ("Haydn", "sonata"),
    ("Liszt", "etude"),
    ("Liszt", "rhapsody"),
    ("Liszt", "sonata"),
    ("Ravel", "miroirs"),
    ("Schubert", "fantasie"),
    ("Schubert", "impromptu"),
    ("Schubert", "sonata"),
    ("Schumann", "arabeske"),
    ("Schumann", "kreisleriana"),
    ("Scriabin", "sonata"),
)

DEFAULT_PAIR = ("Chopin", "etude")


class ObservedPairError(ValueError):
    pass


def get_observed_pairs() -> tuple[tuple[str, str], ...]:
    return OBSERVED_PAIRS


def validate_pair(composer: str, genre: str) -> tuple[str, str]:
    pair = (composer.strip(), genre.strip().lower())
    # Preserve canonical composer capitalization from the table.
    for composer_name, genre_name in OBSERVED_PAIRS:
        if composer_name.lower() == pair[0].lower() and genre_name == pair[1]:
            return composer_name, genre_name
    raise ObservedPairError(
        f"Unsupported composer/genre pair {composer!r}/{genre!r}. "
        "The demo only exposes observed ASAP pairs from paper 119 Table 2."
    )
