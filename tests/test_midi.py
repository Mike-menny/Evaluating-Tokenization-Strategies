from __future__ import annotations

from dafx26_demo.beats import extract_beat_markers
from dafx26_demo.midi_convert import midi_is_valid, tokens_to_midi_bytes
from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import (  # noqa: E402
    BEAT_DB_ID,
    BOS_ID,
    DUR_OFFSET,
    EOS_ID,
    NOTE_OFFSET,
    PEDAL_ON_ID,
    TIME_OFFSET,
    VELOCITY_OFFSET,
)


def _note(mode_has_velocity: bool) -> list[int]:
    tokens = [BOS_ID, TIME_OFFSET + 10, DUR_OFFSET + 20, NOTE_OFFSET + 60]
    if mode_has_velocity:
        tokens.append(VELOCITY_OFFSET + 64)
    tokens.append(EOS_ID)
    return tokens


def test_all_six_modes_convert_to_valid_midi() -> None:
    modes = {
        "note": False,
        "note_pedal": False,
        "note_velocity": True,
        "note_velocity_beat": True,
        "note_velocity_pedal": True,
        "full": True,
    }
    for mode, has_vel in modes.items():
        data = tokens_to_midi_bytes(_note(has_vel), mode=mode)
        assert midi_is_valid(data), mode


def test_beat_markers_are_extracted_even_though_midi_drops_them() -> None:
    tokens = [
        BOS_ID,
        TIME_OFFSET + 0,
        BEAT_DB_ID,
        TIME_OFFSET + 10,
        DUR_OFFSET + 20,
        NOTE_OFFSET + 60,
        VELOCITY_OFFSET + 64,
        EOS_ID,
    ]
    markers = extract_beat_markers(tokens, mode="full")
    assert markers
    assert markers[0]["kind"] == "downbeat"
    data = tokens_to_midi_bytes(tokens, mode="full")
    assert midi_is_valid(data)
