from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from dafx26_demo.metrics import rtf
from dafx26_demo.mlx_backend.model import TokenStep
from dafx26_demo.paths import ensure_upstream_on_path
from dafx26_demo.schema import GenerationRequest

MAX_LEAD_SEC = 5.0


def generated_lead_sec(*, music_sec: float, wall_sec: float, live_buffer_sec: float) -> float:
    return float(music_sec) - float(wall_sec) + float(live_buffer_sec)


def wait_if_buffer_full(
    *,
    music_sec: float,
    live_buffer_sec: float,
    clock,
    t0: float,
    sleeper,
    cancel_event,
    max_lead_sec: float = MAX_LEAD_SEC,
) -> None:
    if cancel_event.is_set():
        return
    wall_sec = float(clock() - t0)
    excess = generated_lead_sec(music_sec=music_sec, wall_sec=wall_sec, live_buffer_sec=live_buffer_sec) - max_lead_sec
    if excess > 0:
        sleeper(excess)


ensure_upstream_on_path()

from src.tokenization.conditional_convert import tokens_to_midi  # noqa: E402
from src.tokenization.conditional_vocab import (  # noqa: E402
    BEAT_B_ID,
    BEAT_BR_ID,
    BEAT_DB_ID,
    EOS_ID,
    TIME_OFFSET,
    TIME_RESOLUTION,
    is_time_token,
)

_BEAT_KINDS = {
    BEAT_B_ID: "beat",
    BEAT_DB_ID: "downbeat",
    BEAT_BR_ID: "rubato",
}


@dataclass(frozen=True)
class LiveNote:
    kind: str
    event_id: int
    segment: int
    start_sec: float
    end_sec: float
    pitch: int
    velocity: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LivePedal:
    kind: str
    event_id: int
    segment: int
    time_sec: float
    value: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveBeat:
    kind: str
    event_id: int
    segment: int
    time_sec: float
    beat_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentSummary:
    complete_events: int
    max_source_sec: float
    max_note_end_sec: float
    next_event_id: int
    music_sec: float


def format_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _source_sec(tokens: list[int]) -> float:
    if tokens and is_time_token(tokens[0]):
        return (int(tokens[0]) - TIME_OFFSET) / TIME_RESOLUTION
    return 0.0


def _channel_event(midi) -> tuple[str, float, float, int, int] | tuple[str, float, int] | None:
    abs_time = 0.0
    onset: tuple[float, int, int] | None = None
    for message in midi:
        abs_time += float(message.time)
        abs_time = round(abs_time, 10)
        if getattr(message, "is_meta", False):
            continue
        msg_type = getattr(message, "type", None)
        if msg_type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
            onset = (abs_time, int(message.note), int(message.velocity))
            continue
        if onset is not None and msg_type in {"note_off", "note_on"}:
            start, pitch, velocity = onset
            return ("note", start, abs_time, pitch, velocity)
        if msg_type == "control_change" and int(getattr(message, "control", -1)) == 64:
            return ("pedal", abs_time, int(message.value))
    return None


class EventConverter:
    def __init__(self, mode: str, *, segment: int, offset_sec: float, next_event_id: int) -> None:
        self.mode = mode
        self.segment = int(segment)
        self.offset_sec = float(offset_sec)
        self._next_event_id = int(next_event_id)
        self._current: list[int] = []
        self.complete_events = 0
        self.max_source_sec = 0.0
        self.max_note_end_sec = 0.0
        self.music_sec = float(offset_sec)

    def push(self, step: TokenStep) -> LiveNote | LivePedal | LiveBeat | None:
        self._current.append(int(step.token))
        if not step.event_complete:
            return None
        event_tokens, self._current = self._current, []
        self.complete_events += 1
        source_sec = _source_sec(event_tokens)
        if len(event_tokens) == 2 and event_tokens[1] in _BEAT_KINDS:
            event_id = self._next_event_id
            self._next_event_id += 1
            beat = LiveBeat(
                kind="beat",
                event_id=event_id,
                segment=self.segment,
                time_sec=self.offset_sec + source_sec,
                beat_kind=_BEAT_KINDS[event_tokens[1]],
            )
            self.max_source_sec = max(self.max_source_sec, source_sec)
            self.music_sec = max(self.music_sec, beat.time_sec)
            return beat
        parsed = _channel_event(tokens_to_midi(event_tokens, mode=self.mode))
        if parsed is None:
            self.max_source_sec = max(self.max_source_sec, source_sec)
            self.music_sec = max(self.music_sec, self.offset_sec + source_sec)
            return None
        event_id = self._next_event_id
        self._next_event_id += 1
        if parsed[0] == "note":
            _, start, end, pitch, velocity = parsed
            self.max_source_sec = max(self.max_source_sec, float(start), source_sec)
            self.max_note_end_sec = max(self.max_note_end_sec, float(end))
            note = LiveNote(
                kind="note",
                event_id=event_id,
                segment=self.segment,
                start_sec=float(start) + self.offset_sec,
                end_sec=float(end) + self.offset_sec,
                pitch=int(pitch),
                velocity=int(velocity),
            )
            self.music_sec = max(self.music_sec, note.end_sec)
            return note
        _, time_sec, value = parsed
        self.max_source_sec = max(self.max_source_sec, float(time_sec), source_sec)
        pedal = LivePedal(
            kind="pedal",
            event_id=event_id,
            segment=self.segment,
            time_sec=float(time_sec) + self.offset_sec,
            value=int(value),
        )
        self.music_sec = max(self.music_sec, pedal.time_sec)
        return pedal

    def finish_segment(self) -> SegmentSummary:
        self._current = []
        return SegmentSummary(
            complete_events=self.complete_events,
            max_source_sec=self.max_source_sec,
            max_note_end_sec=self.max_note_end_sec,
            next_event_id=self._next_event_id,
            music_sec=self.music_sec,
        )


def _status_payload(*, tokens: int, music_sec: float, wall_sec: float) -> dict[str, Any]:
    return {
        "tokens": int(tokens),
        "music_sec": round(float(music_sec), 6),
        "wall_sec": round(float(wall_sec), 6),
        "rtf": rtf(wall_sec=float(wall_sec), music_sec=float(music_sec)),
    }


def _iter_segment(model, request: GenerationRequest, converter: EventConverter, cancel_event) -> Iterator[TokenStep]:
    import numpy as np

    from dafx26_demo.mlx_backend.model import iter_generate_tokens
    from dafx26_demo.mlx_backend.sampling import MidiConstraint
    from src.tokenization.conditional_vocab import composer_id, genre_id

    constraint = MidiConstraint(
        batch_size=1,
        vocab_size=int(model.config["vocab_size"]),
        mode=request.mode,
        min_tokens=request.min_tokens,
    )
    yield from iter_generate_tokens(
        model,
        np.array([composer_id(request.composer)], dtype=np.int32),
        np.array([genre_id(request.genre)], dtype=np.int32),
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        constraint=constraint,
        seed=request.seed,
        cache_strategy="block",
        cancel_event=cancel_event,
    )


def _stream_segment(
    *,
    model,
    request: GenerationRequest,
    converter: EventConverter,
    cancel_event,
    segment_fn: Callable,
    monotonic,
    t0: float,
    heartbeat_sec: float,
    counters: dict[str, float | int],
    live_buffer_sec: float,
    sleeper,
) -> Iterator[str]:
    last_step: TokenStep | None = None
    generated = 0
    for step in segment_fn(model, request, converter, cancel_event):
        last_step = step
        generated += 1
        counters["tokens"] = int(counters["token_base"]) + generated
        event = converter.push(step)
        if event is not None:
            yield format_sse("midi", event.to_dict())
        wait_if_buffer_full(
            music_sec=converter.music_sec,
            live_buffer_sec=live_buffer_sec,
            clock=monotonic,
            t0=t0,
            sleeper=sleeper,
            cancel_event=cancel_event,
        )
        wall = float(monotonic() - t0)
        if wall - float(counters["last_heartbeat"]) >= heartbeat_sec:
            counters["last_heartbeat"] = wall
            yield format_sse(
                "status",
                _status_payload(tokens=int(counters["tokens"]), music_sec=converter.music_sec, wall_sec=wall),
            )
    counters["token_base"] = int(counters["token_base"]) + generated
    if last_step is not None and last_step.token == EOS_ID:
        return "eos"
    if last_step is None:
        return "stop"
    return "max_seq_len"


def iter_live_sse(
    hub: object,
    *,
    mode: str,
    composer: str,
    genre: str,
    seed: int,
    live_buffer_sec: float,
    job: Any,
    temperature: float = 0.95,
    top_p: float = 0.98,
    min_tokens: int = 64,
    segment_fn: Callable | None = None,
    load_fn: Callable | None = None,
    monotonic: Callable[[], float] | None = None,
    heartbeat_sec: float = 15.0,
    sleeper: Callable[[float], None] | None = None,
) -> Iterator[str]:
    clock = time.monotonic if monotonic is None else monotonic
    pause = time.sleep if sleeper is None else sleeper
    step_fn = _iter_segment if segment_fn is None else segment_fn
    max_seq_len = int(hub.max_seq_len(mode))
    yield format_sse(
        "meta",
        {
            "job_id": job.job_id,
            "mode": mode,
            "composer": composer,
            "genre": genre,
            "seed": int(seed),
            "live_buffer_sec": float(live_buffer_sec),
            "max_seq_len": max_seq_len,
        },
    )
    yield format_sse("status", _status_payload(tokens=0, music_sec=0.0, wall_sec=0.0))
    try:
        model = load_fn() if load_fn is not None else hub.load(mode)[0]
    except Exception as exc:
        yield format_sse("end", {"reason": "error", "message": str(exc)})
        return

    t0 = clock()
    counters: dict[str, float | int] = {"tokens": 0, "token_base": 0, "last_heartbeat": 0.0}
    offset = 0.0
    next_event_id = 0
    empty_segments = 0
    index = 0
    while not job.cancel_event.is_set():
        converter = EventConverter(mode, segment=index, offset_sec=offset, next_event_id=next_event_id)
        request = GenerationRequest(
            mode=mode,
            composer=composer,
            genre=genre,
            seed=int(seed) + index,
            max_tokens=max_seq_len - 1,
            temperature=temperature,
            top_p=top_p,
            device="mlx",
            min_tokens=min_tokens,
        )
        try:
            reason = yield from _stream_segment(
                model=model,
                request=request,
                converter=converter,
                cancel_event=job.cancel_event,
                segment_fn=step_fn,
                monotonic=clock,
                t0=t0,
                heartbeat_sec=heartbeat_sec,
                counters=counters,
                live_buffer_sec=float(live_buffer_sec),
                sleeper=pause,
            )
        except Exception as exc:
            yield format_sse("end", {"reason": "error", "message": str(exc)})
            return
        summary = converter.finish_segment()
        next_event_id = summary.next_event_id
        next_offset = offset
        if summary.complete_events == 0:
            empty_segments += 1
        else:
            empty_segments = 0
            next_offset = offset + max(summary.max_source_sec, summary.max_note_end_sec) + 0.05
        if reason != "stop":
            yield format_sse(
                "segment",
                {
                    "segment": index,
                    "reason": reason,
                    "duration_sec": max(summary.max_source_sec, summary.max_note_end_sec),
                    "next_offset_sec": next_offset,
                    "complete_events": summary.complete_events,
                },
            )
        offset = next_offset
        index += 1
        if job.cancel_event.is_set():
            yield format_sse("end", {"reason": "stop"})
            return
        if empty_segments >= 3:
            yield format_sse(
                "end",
                {"reason": "error", "message": "three consecutive empty segments"},
            )
            return
    yield format_sse("end", {"reason": "stop"})
