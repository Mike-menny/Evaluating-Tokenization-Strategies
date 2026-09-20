from __future__ import annotations

import json

import mido
import pytest

from dafx26_demo.mlx_backend.model import TokenStep
from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import (  # noqa: E402
    BEAT_B_ID,
    DUR_OFFSET,
    EOS_ID,
    NOTE_OFFSET,
    PEDAL_OFF_ID,
    PEDAL_ON_ID,
    TIME_OFFSET,
    VELOCITY_OFFSET,
)


def note_tokens(*, time: int, duration: int, pitch: int, velocity: int) -> list[TokenStep]:
    return [
        TokenStep(token=TIME_OFFSET + time, n_tokens=1, event_complete=False),
        TokenStep(token=DUR_OFFSET + duration, n_tokens=2, event_complete=False),
        TokenStep(token=NOTE_OFFSET + pitch, n_tokens=3, event_complete=False),
        TokenStep(token=VELOCITY_OFFSET + velocity, n_tokens=4, event_complete=True),
    ]


def beat_tokens(*, time: int) -> list[TokenStep]:
    return [
        TokenStep(token=TIME_OFFSET + time, n_tokens=1, event_complete=False),
        TokenStep(token=BEAT_B_ID, n_tokens=2, event_complete=True),
    ]


def pedal_tokens(*, time: int, value: int) -> list[TokenStep]:
    token = PEDAL_ON_ID if value >= 64 else PEDAL_OFF_ID
    return [
        TokenStep(token=TIME_OFFSET + time, n_tokens=1, event_complete=False),
        TokenStep(token=token, n_tokens=2, event_complete=True),
    ]


def incomplete_step() -> TokenStep:
    return TokenStep(token=TIME_OFFSET + 1, n_tokens=1, event_complete=False)


def push_tokens(converter, steps: list[TokenStep]):
    last = None
    for step in steps:
        result = converter.push(step)
        if result is not None:
            last = result
    return last


def test_converter_emits_each_complete_event_once() -> None:
    from dafx26_demo.mlx_backend.live import EventConverter, LiveNote

    converter = EventConverter("note_velocity_pedal", segment=0, offset_sec=0.0, next_event_id=0)
    note = push_tokens(converter, note_tokens(time=10, duration=20, pitch=60, velocity=64))
    assert note == LiveNote("note", 0, 0, 0.10, 0.30, 60, 64)
    assert converter.push(incomplete_step()) is None
    assert note.to_dict()["start_sec"] == pytest.approx(0.10)


def test_pedal_and_beat_behavior() -> None:
    from dafx26_demo.mlx_backend.live import EventConverter, LiveBeat, LivePedal

    converter = EventConverter("full", segment=0, offset_sec=2.0, next_event_id=0)
    beat = push_tokens(converter, beat_tokens(time=10))
    assert beat == LiveBeat("beat", 0, 0, 2.10, "beat")
    pedal = push_tokens(converter, pedal_tokens(time=20, value=127))
    assert pedal == LivePedal("pedal", 1, 0, 2.20, 127)
    assert pedal.time_sec == pytest.approx(2.20)
    assert pedal.to_dict()["value"] == 127
    assert converter.music_sec == pytest.approx(2.20)
    off = push_tokens(converter, pedal_tokens(time=30, value=0))
    assert off.value == 0
    summary = converter.finish_segment()
    assert summary.complete_events == 3
    assert summary.max_source_sec == pytest.approx(0.30)
    assert summary.next_event_id == 3


def test_converter_sends_only_completed_event_slice(monkeypatch) -> None:
    from dafx26_demo.mlx_backend import live as live_mod
    from dafx26_demo.mlx_backend.live import EventConverter

    seen: list[list[int]] = []
    real = live_mod.tokens_to_midi

    def wrapped(tokens, mode=None):
        seen.append(list(tokens))
        return real(tokens, mode=mode)

    monkeypatch.setattr(live_mod, "tokens_to_midi", wrapped)
    converter = EventConverter("note_velocity_pedal", segment=0, offset_sec=0.0, next_event_id=0)
    first = note_tokens(time=10, duration=20, pitch=60, velocity=64)
    second = note_tokens(time=40, duration=10, pitch=64, velocity=70)
    push_tokens(converter, first)
    push_tokens(converter, second)
    prefix = [step.token for step in first + second]
    assert seen[0] == [step.token for step in first]
    assert seen[1] == [step.token for step in second]
    assert all(slice_ != prefix for slice_ in seen)


def test_incomplete_eos_is_discarded() -> None:
    from dafx26_demo.mlx_backend.live import EventConverter

    converter = EventConverter("note_velocity_pedal", segment=1, offset_sec=1.0, next_event_id=4)
    converter.push(TokenStep(token=TIME_OFFSET + 10, n_tokens=1, event_complete=False))
    converter.push(TokenStep(token=DUR_OFFSET + 20, n_tokens=2, event_complete=False))
    converter.push(TokenStep(token=EOS_ID, n_tokens=3, event_complete=False))
    summary = converter.finish_segment()
    assert summary.complete_events == 0
    assert summary.next_event_id == 4
    assert summary.max_source_sec == 0.0
    assert summary.max_note_end_sec == 0.0


def test_out_of_order_times_keep_model_time_and_duration() -> None:
    from dafx26_demo.mlx_backend.live import EventConverter

    converter = EventConverter("note_velocity_pedal", segment=0, offset_sec=0.0, next_event_id=0)
    later = push_tokens(converter, note_tokens(time=50, duration=10, pitch=60, velocity=64))
    earlier = push_tokens(converter, note_tokens(time=10, duration=20, pitch=61, velocity=65))
    assert later.start_sec == pytest.approx(0.50)
    assert later.end_sec == pytest.approx(0.60)
    assert earlier.start_sec == pytest.approx(0.10)
    assert earlier.end_sec == pytest.approx(0.30)
    summary = converter.finish_segment()
    assert summary.max_source_sec == pytest.approx(0.50)
    assert summary.max_note_end_sec == pytest.approx(0.60)


def test_extract_uses_mido_seconds_not_ticks(monkeypatch) -> None:
    from dafx26_demo.mlx_backend import live as live_mod
    from dafx26_demo.mlx_backend.live import EventConverter, LiveNote

    mid = mido.MidiFile()
    mid.ticks_per_beat = 480
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=64, time=480))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))

    monkeypatch.setattr(live_mod, "tokens_to_midi", lambda tokens, mode=None: mid)
    converter = EventConverter("note", segment=2, offset_sec=1.0, next_event_id=7)
    note = push_tokens(
        converter,
        [
            TokenStep(token=TIME_OFFSET + 10, n_tokens=1, event_complete=False),
            TokenStep(token=DUR_OFFSET + 20, n_tokens=2, event_complete=False),
            TokenStep(token=NOTE_OFFSET + 60, n_tokens=3, event_complete=True),
        ],
    )
    assert note == LiveNote("note", 7, 2, 1.5, 2.0, 60, 64)


def test_format_sse_is_event_stream_framed() -> None:
    from dafx26_demo.mlx_backend.live import format_sse

    payload = {"kind": "note", "event_id": 1}
    text = format_sse("midi", payload)
    assert text.startswith("event: midi\n")
    assert text.endswith("\n\n")
    _, data_line, blank, extra = text.split("\n")
    assert data_line.startswith("data: ")
    assert extra == ""
    assert json.loads(data_line.removeprefix("data: ")) == payload
    assert blank == ""


def test_segment_summary_preserves_event_ids_for_rollover() -> None:
    from dafx26_demo.mlx_backend.live import EventConverter

    first = EventConverter("note_velocity_pedal", segment=0, offset_sec=0.0, next_event_id=0)
    push_tokens(first, note_tokens(time=0, duration=100, pitch=60, velocity=64))
    summary = first.finish_segment()
    second = EventConverter(
        "note_velocity_pedal",
        segment=1,
        offset_sec=summary.max_note_end_sec + 0.05,
        next_event_id=summary.next_event_id,
    )
    note = push_tokens(second, note_tokens(time=0, duration=50, pitch=61, velocity=65))
    assert note.event_id == 1
    assert note.start_sec == pytest.approx(1.05)
