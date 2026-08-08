"""Small MIDI vocabulary and ASAP condition labels for composer/genre conditioning.

This module is independent of the existing Llama-appended MIDI vocabulary.  It
uses compact token IDs because the conditional model has no text tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

TIME_RESOLUTION = 100
MAX_TIME = 10000
MAX_DUR = 1000
MAX_PITCH = 128
MAX_VELOCITY = 128

TIME_OFFSET = 3
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
VELOCITY_OFFSET = NOTE_OFFSET + MAX_PITCH
BEAT_OFFSET = VELOCITY_OFFSET + MAX_VELOCITY
BEAT_B_ID = BEAT_OFFSET
BEAT_DB_ID = BEAT_OFFSET + 1
BEAT_BR_ID = BEAT_OFFSET + 2
PEDAL_OFFSET = BEAT_OFFSET + 3
PEDAL_OFF_ID = PEDAL_OFFSET
PEDAL_ON_ID = PEDAL_OFFSET + 1
FULL_VOCAB_SIZE = PEDAL_ON_ID + 1

WINDOW_SEC_DEFAULT = MAX_TIME / TIME_RESOLUTION

TokenizationMode = Literal[
    "note",
    "note_pedal",
    "note_velocity",
    "note_velocity_beat",
    "note_velocity_pedal",
    "full",
]


@dataclass(frozen=True)
class TokenizationSpec:
    name: TokenizationMode
    use_velocity: bool
    use_beat: bool
    use_pedal: bool
    vocab_size: int

    @property
    def note_event_size(self) -> int:
        return 4 if self.use_velocity else 3

    @property
    def min_event_size(self) -> int:
        return 2 if (self.use_beat or self.use_pedal) else self.note_event_size


def get_tokenization_spec(mode: str | None) -> TokenizationSpec:
    name = (mode or "full").strip().lower()
    specs: dict[str, TokenizationSpec] = {
        "note": TokenizationSpec("note", False, False, False, NOTE_OFFSET + MAX_PITCH),
        "note_pedal": TokenizationSpec("note_pedal", False, False, True, PEDAL_ON_ID + 1),
        "note_velocity": TokenizationSpec("note_velocity", True, False, False, VELOCITY_OFFSET + MAX_VELOCITY),
        "note_velocity_beat": TokenizationSpec("note_velocity_beat", True, True, False, BEAT_BR_ID + 1),
        "note_velocity_pedal": TokenizationSpec("note_velocity_pedal", True, False, True, PEDAL_ON_ID + 1),
        "full": TokenizationSpec("full", True, True, True, FULL_VOCAB_SIZE),
    }
    try:
        return specs[name]
    except KeyError as e:
        raise ValueError(f"Unknown tokenization mode {mode!r}. Valid modes: {', '.join(specs)}") from e


TOKENIZATION_MODE_CHOICES = tuple(
    get_tokenization_spec(k).name
    for k in ("note", "note_pedal", "note_velocity", "note_velocity_beat", "note_velocity_pedal", "full")
)

_VOCAB_REGION_BOUNDS: tuple[tuple[int, str], ...] = (
    (NOTE_OFFSET + MAX_PITCH, "note"),
    (VELOCITY_OFFSET + MAX_VELOCITY, "velocity"),
    (BEAT_BR_ID + 1, "beat"),
    (PEDAL_ON_ID + 1, "pedal"),
)


def describe_dropped_vocab_regions(ckpt_vocab_size: int, target_vocab_size: int) -> str:
    """Summarise which token regions are dropped when shrinking vocab."""
    if ckpt_vocab_size <= target_vocab_size:
        return ""
    dropped: list[str] = []
    for bound, label in _VOCAB_REGION_BOUNDS[1:]:
        if target_vocab_size < bound <= ckpt_vocab_size:
            dropped.append(label)
    if not dropped:
        return f"{ckpt_vocab_size - target_vocab_size} token rows"
    return ", ".join(dropped)


_COMPOSERS_ASAP = (
    "Bach",
    "Balakirev",
    "Beethoven",
    "Brahms",
    "Chopin",
    "Debussy",
    "Glinka",
    "Haydn",
    "Liszt",
    "Mozart",
    "Prokofiev",
    "Rachmaninoff",
    "Ravel",
    "Schubert",
    "Schumann",
    "Scriabin",
)

_COMPOSERS_MAESTRO_EXTRA = (
    "Alban Berg",
    "Antonio Soler",
    "Carl Maria von Weber",
    "César Franck",
    "Charles Gounod / Franz Liszt",
    "Domenico Scarlatti",
    "Edvard Grieg",
    "Felix Mendelssohn",
    "Felix Mendelssohn / Sergei Rachmaninoff",
    "Franz Liszt / Camille Saint-Saëns",
    "Franz Liszt / Vladimir Horowitz",
    "Franz Schubert / Franz Liszt",
    "Franz Schubert / Leopold Godowsky",
    "Fritz Kreisler / Sergei Rachmaninoff",
    "George Enescu",
    "George Frideric Handel",
    "Georges Bizet / Ferruccio Busoni",
    "Georges Bizet / Moritz Moszkowski",
    "Georges Bizet / Vladimir Horowitz",
    "Giuseppe Verdi / Franz Liszt",
    "Henry Purcell",
    "Isaac Albéniz",
    "Isaac Albéniz / Leopold Godowsky",
    "Jean-Philippe Rameau",
    "Johann Christian Fischer / Wolfgang Amadeus Mozart",
    "Johann Pachelbel",
    "Johann Sebastian Bach / Egon Petri",
    "Johann Sebastian Bach / Ferruccio Busoni",
    "Johann Sebastian Bach / Franz Liszt",
    "Johann Sebastian Bach / Myra Hess",
    "Johann Strauss / Alfred Grünfeld",
    "Leoš Janáček",
    "Mikhail Glinka / Mily Balakirev",
    "Modest Mussorgsky",
    "Muzio Clementi",
    "Niccolò Paganini / Franz Liszt",
    "Nikolai Medtner",
    "Nikolai Rimsky-Korsakov / Sergei Rachmaninoff",
    "Orlando Gibbons",
    "Percy Grainger",
    "Pyotr Ilyich Tchaikovsky",
    "Pyotr Ilyich Tchaikovsky / Mikhail Pletnev",
    "Pyotr Ilyich Tchaikovsky / Sergei Rachmaninoff",
    "Richard Wagner / Franz Liszt",
    "Robert Schumann / Franz Liszt",
    "Sergei Rachmaninoff / György Cziffra",
    "Sergei Rachmaninoff / Vyacheslav Gryaznov",
)

# Keep ASAP composers at the front so their IDs remain stable for existing checkpoints.
COMPOSERS = _COMPOSERS_ASAP + _COMPOSERS_MAESTRO_EXTRA

COMPOSER_TO_ID = {name: i for i, name in enumerate(COMPOSERS)}

# Special padding / unknown IDs for composer and genre conditioning.
COMPOSER_PAD_ID = len(COMPOSERS)       # index after all real composers
COMPOSER_UNKNOWN_ID = COMPOSER_PAD_ID + 1
NUM_COMPOSER_IDS = COMPOSER_UNKNOWN_ID + 1   # total embedding rows

_GENRE_ALIASES = {
    "annees_de_pelerinage_2": "annees_de_pelerinage",
    "arabeske": "arabeske",
    "arabesque": "arabesque",
    "bagatelle": "bagatelle",
    "ballade_2": "ballade",
    "ballades": "ballade",
    "barcarolle": "barcarolle",
    "berceuse_op_57": "berceuse",
    "capriccio": "capriccio",
    "carnaval": "carnaval",
    "chaconne": "chaconne",
    "concert_etude_s145": "etude",
    "concerto": "concerto",
    "duet": "duet",
    "dumka": "dumka",
    "elegy": "elegy",
    "estampes": "estampes",
    "etudes_op_10": "etude",
    "etudes_op_25": "etude",
    "etudes_op_8": "etude",
    "fantasie_475": "fantasie",
    "fantasy": "fantasy",
    "fugue": "fugue",
    "gaspard_de_la_nuit": "gaspard_de_la_nuit",
    "gran_etudes_de_paganini": "etude",
    "hungarian_rhapsodies": "rhapsody",
    "humoresque": "humoresque",
    "image": "image",
    "images_book_1": "images",
    "impromptu_op.90_d.899": "impromptu",
    "impromptu_op142": "impromptu",
    "intermezzo": "intermezzo",
    "islamey": "islamey",
    "italian_concerto": "concerto",
    "keyboard_sonatas": "sonata",
    "kreisleriana": "kreisleriana",
    "mazurka": "mazurka",
    "mephisto_waltz": "waltz",
    "miroirs": "miroirs",
    "moment_musical": "moment_musical",
    "moment_musical_no_1": "moment_musical",
    "moment_musical_no_3": "moment_musical",
    "nocturne": "nocturne",
    "overture": "overture",
    "paraphrase": "paraphrase",
    "partita": "partita",
    "pavan": "pavan",
    "piano_piece": "piano_piece",
    "piano_sonatas": "sonata",
    "piece": "piece",
    "polonaises": "polonaise",
    "pour_le_piano": "suite",
    "prelude": "prelude",
    "preludes_op_23": "prelude",
    "preludes_op_32": "prelude",
    "romance": "romance",
    "rondo": "rondo",
    "scherzos": "scherzo",
    "six_pieces_op_118": "piece",
    "sonata": "sonata",
    "sonata_2": "sonata",
    "sonata_3": "sonata",
    "sonatas": "sonata",
    "sonatina": "sonatina",
    "suite": "suite",
    "tango": "tango",
    "tarantella": "tarantella",
    "the_lark": "song_transcription",
    "toccata": "toccata",
    "toccata_repeat": "toccata",
    "transcendental_etudes": "etude",
    "variation": "variation",
    "waltz": "waltz",
    "wanderer_fantasie": "fantasie",
}

GENRES = tuple(sorted(set(_GENRE_ALIASES.values())))
GENRE_TO_ID = {name: i for i, name in enumerate(GENRES)}

# Special padding / unknown IDs for genre conditioning.
GENRE_PAD_ID = len(GENRES)
GENRE_UNKNOWN_ID = GENRE_PAD_ID + 1
NUM_GENRE_IDS = GENRE_UNKNOWN_ID + 1   # total embedding rows


def canonicalize_genre(raw_genre: str) -> str:
    key = str(raw_genre).strip().lower()
    return _GENRE_ALIASES.get(key, key)


def composer_genre_from_manifest_path(midi_path: str) -> tuple[str, str]:
    parts = str(midi_path).split("/")
    if len(parts) < 3:
        raise ValueError(f"Expected manifest midi_path like root/Composer/Genre/..., got {midi_path!r}")
    return parts[1], canonicalize_genre(parts[2])


def composer_id(name: str) -> int:
    try:
        return COMPOSER_TO_ID[str(name)]
    except KeyError:
        return COMPOSER_UNKNOWN_ID


def genre_id(name: str) -> int:
    genre = canonicalize_genre(name)
    try:
        return GENRE_TO_ID[genre]
    except KeyError:
        return GENRE_UNKNOWN_ID


def is_time_token(tid: int) -> bool:
    return TIME_OFFSET <= tid < DUR_OFFSET


def is_duration_token(tid: int) -> bool:
    return DUR_OFFSET <= tid < NOTE_OFFSET


def is_note_token(tid: int) -> bool:
    return NOTE_OFFSET <= tid < VELOCITY_OFFSET


def is_velocity_token(tid: int) -> bool:
    return VELOCITY_OFFSET <= tid < BEAT_OFFSET


def is_beat_token(tid: int) -> bool:
    return tid in (BEAT_B_ID, BEAT_DB_ID, BEAT_BR_ID)


def is_pedal_token(tid: int) -> bool:
    return tid in (PEDAL_OFF_ID, PEDAL_ON_ID)


def iter_event_slices(tokens: list[int], mode: str | None = "full"):
    spec = get_tokenization_spec(mode)
    i = 0
    n = len(tokens)
    while i < n:
        if (
            i + spec.note_event_size <= n
            and is_time_token(tokens[i])
            and is_duration_token(tokens[i + 1])
            and is_note_token(tokens[i + 2])
        ):
            if spec.use_velocity and not is_velocity_token(tokens[i + 3]):
                i += 1
                continue
            yield i, i + spec.note_event_size
            i += spec.note_event_size
        elif i + 2 <= n and spec.use_beat and is_time_token(tokens[i]) and is_beat_token(tokens[i + 1]):
            yield i, i + 2
            i += 2
        elif i + 2 <= n and spec.use_pedal and is_time_token(tokens[i]) and is_pedal_token(tokens[i + 1]):
            yield i, i + 2
            i += 2
        else:
            i += 1


def truncate_to_complete_events(tokens: list[int], mode: str | None = "full") -> list[int]:
    end = 0
    for _s, e in iter_event_slices(tokens, mode):
        end = e
    return tokens[:end]
