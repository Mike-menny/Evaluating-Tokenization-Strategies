"""ASAP MIDI/annotation conversion for the compact conditional vocabulary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import mido

from src.tokenization.conditional_vocab import (
    BEAT_B_ID,
    BEAT_BR_ID,
    BEAT_DB_ID,
    BOS_ID,
    DUR_OFFSET,
    EOS_ID,
    MAX_DUR,
    MAX_PITCH,
    MAX_TIME,
    NOTE_OFFSET,
    PEDAL_OFF_ID,
    PEDAL_ON_ID,
    TIME_OFFSET,
    TIME_RESOLUTION,
    VELOCITY_OFFSET,
    get_tokenization_spec,
    is_pedal_token,
    iter_event_slices,
    truncate_to_complete_events,
)

MergedEvent = tuple[float, int, Literal["beat", "note", "pedal"], list[int] | int]


def _rel_time_token(rel_sec: float) -> int:
    ticks = round(TIME_RESOLUTION * rel_sec)
    ticks = max(0, min(ticks, MAX_TIME - 1))
    return TIME_OFFSET + ticks


def parse_annotation_beats(annotation_path: str | Path) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    with open(annotation_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label = parts[2].split(",")[0].strip()
            if label == "b":
                out.append((t, BEAT_B_ID))
            elif label == "db":
                out.append((t, BEAT_DB_ID))
            elif label == "bR":
                out.append((t, BEAT_BR_ID))
    return out


def estimate_performance_duration_sec(
    midi_path: str | Path,
    annotation_path: str | Path | None = None,
    mode: str | None = "full",
) -> float:
    """Cheap duration estimate used to build deterministic 100 s window indices."""
    duration = 0.0
    try:
        duration = max(duration, float(mido.MidiFile(str(midi_path)).length))
    except Exception:
        duration = 0.0
    if annotation_path is not None and get_tokenization_spec(mode).use_beat:
        try:
            beats = parse_annotation_beats(annotation_path)
            if beats:
                duration = max(duration, max(t for t, _ in beats))
        except Exception:
            pass
    return max(duration, 0.0)


def _midi_notes_pedals_raw(
    midi_path: str | Path,
    mode: str | None = "full",
) -> tuple[list[dict], list[tuple[float, int]]]:
    spec = get_tokenization_spec(mode)
    midi = mido.MidiFile(str(midi_path))
    note_rows: list[dict] = []
    open_notes = defaultdict(list)
    time_sec = 0.0
    sustain_on = False
    pedals: list[tuple[float, int]] = []

    for message in midi:
        time_sec += message.time
        if spec.use_pedal and message.type == "control_change" and message.control == 64:
            new_on = message.value >= 64
            if new_on != sustain_on:
                sustain_on = new_on
                pedals.append((time_sec, PEDAL_ON_ID if new_on else PEDAL_OFF_ID))
        elif message.type in ("note_on", "note_off"):
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                idx = len(note_rows)
                note_rows.append(
                    {
                        "onset": time_sec,
                        "note_id": NOTE_OFFSET + max(0, min(int(message.note), MAX_PITCH - 1)),
                        "vel_id": VELOCITY_OFFSET + max(0, min(int(message.velocity), 127)),
                        "dur_ticks": 0,
                    }
                )
                open_notes[key].append(idx)
            else:
                try:
                    open_idx = open_notes[key].pop(0)
                except (KeyError, IndexError):
                    continue
                duration_ticks = round(TIME_RESOLUTION * (time_sec - note_rows[open_idx]["onset"]))
                note_rows[open_idx]["dur_ticks"] = max(1, min(duration_ticks, MAX_DUR - 1))

    for row in note_rows:
        if row["dur_ticks"] <= 0:
            row["dur_ticks"] = 1
    return note_rows, pedals


def merge_midi_annotation_events(
    midi_path: str | Path,
    annotation_path: str | Path,
    mode: str | None = "full",
) -> list[MergedEvent]:
    spec = get_tokenization_spec(mode)
    beats = parse_annotation_beats(annotation_path) if spec.use_beat else []
    notes, pedals = _midi_notes_pedals_raw(midi_path, mode=mode)

    merged: list[MergedEvent] = []
    for t, beat_id in beats:
        merged.append((t, 0, "beat", beat_id))
    for row in notes:
        payload = [row["onset"], DUR_OFFSET + row["dur_ticks"], row["note_id"]]
        if spec.use_velocity:
            payload.append(row["vel_id"])
        merged.append((row["onset"], 1, "note", payload))
    for t, pedal_id in pedals:
        merged.append((t, 2, "pedal", pedal_id))
    merged.sort(key=lambda x: (x[0], x[1]))
    return merged


def events_window_to_tokens(
    events: list[MergedEvent],
    window_start_sec: float,
    window_end_sec: float,
) -> list[int]:
    tokens: list[int] = []
    for t_sec, _prio, kind, payload in events:
        if not (window_start_sec <= t_sec < window_end_sec):
            continue
        if kind in ("beat", "pedal"):
            assert isinstance(payload, int)
            tokens.extend([_rel_time_token(t_sec - window_start_sec), payload])
        else:
            assert isinstance(payload, list)
            row = payload.copy()
            row[0] = _rel_time_token(t_sec - window_start_sec)
            tokens.extend(row)
    return tokens


def midi_and_annotations_to_tokens(
    midi_path: str | Path,
    annotation_path: str | Path,
    window_start_sec: float,
    window_duration_sec: float,
    mode: str | None = "full",
    add_special_tokens: bool = True,
) -> list[int]:
    events = merge_midi_annotation_events(midi_path, annotation_path, mode=mode)
    tokens = events_window_to_tokens(events, window_start_sec, window_start_sec + window_duration_sec)
    if add_special_tokens:
        return [BOS_ID] + tokens + [EOS_ID]
    return tokens


def generation_phase(tokens: list[int], mode: str | None = "full") -> int:
    """Return next-token phase: 0=time, 1=dur/beat/pedal, 2=note, 3=velocity."""
    spec = get_tokenization_spec(mode)
    end = 0
    for _s, e in iter_event_slices(tokens, mode=mode):
        end = e
    rem = tokens[end:]
    if not rem:
        return 0
    if len(rem) == 1:
        return 1
    if len(rem) == 2:
        if DUR_OFFSET <= rem[1] < NOTE_OFFSET:
            return 2
        return 0
    if len(rem) == 3:
        if spec.use_velocity:
            return 3
        return 0
    return 0


def allowed_token_ids(tokens: list[int], mode: str | None = "full", allow_eos: bool = True) -> list[int]:
    spec = get_tokenization_spec(mode)
    phase = generation_phase(tokens, mode=mode)
    if phase == 0:
        ids = list(range(TIME_OFFSET, DUR_OFFSET))
        if allow_eos:
            ids.append(EOS_ID)
        return ids
    if phase == 1:
        ids = list(range(DUR_OFFSET, NOTE_OFFSET))
        if spec.use_beat:
            ids.extend([BEAT_B_ID, BEAT_DB_ID, BEAT_BR_ID])
        if spec.use_pedal:
            ids.extend([PEDAL_OFF_ID, PEDAL_ON_ID])
        return ids
    if phase == 2:
        return list(range(NOTE_OFFSET, VELOCITY_OFFSET))
    return list(range(VELOCITY_OFFSET, VELOCITY_OFFSET + 128))


def tokens_to_midi(tokens: list[int], mode: str | None = "full") -> mido.MidiFile:
    clean = [t for t in tokens if t not in (BOS_ID, EOS_ID)]
    clean = truncate_to_complete_events(clean, mode=mode)

    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))

    abs_messages: list[tuple[int, int, mido.Message]] = []
    spec = get_tokenization_spec(mode)
    default_velocity = 80
    for s, e in iter_event_slices(clean, mode=mode):
        seg = clean[s:e]
        tick = max(0, seg[0] - TIME_OFFSET)
        if len(seg) == spec.note_event_size:
            dur = max(1, seg[1] - DUR_OFFSET)
            pitch = max(0, min(seg[2] - NOTE_OFFSET, 127))
            velocity = seg[3] - VELOCITY_OFFSET if spec.use_velocity else default_velocity
            velocity = max(1, min(velocity, 127))
            abs_messages.append((tick, 1, mido.Message("note_on", channel=0, note=pitch, velocity=velocity, time=0)))
            abs_messages.append((tick + dur, 0, mido.Message("note_off", channel=0, note=pitch, velocity=0, time=0)))
        elif len(seg) == 2 and is_pedal_token(seg[1]):
            value = 127 if seg[1] == PEDAL_ON_ID else 0
            abs_messages.append((tick, 2, mido.Message("control_change", channel=0, control=64, value=value, time=0)))

    abs_messages.sort(key=lambda x: (x[0], x[1]))
    prev = 0
    for tick, _prio, msg in abs_messages:
        msg.time = max(0, int(tick - prev))
        prev = int(tick)
        track.append(msg)
    return mid
