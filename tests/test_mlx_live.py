from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import threading

import numpy as np
import pytest
import torch

from dafx26_demo.mlx_backend import mlx_available
from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_vocab import EOS_ID  # noqa: E402

requires_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX is not installed")

TINY_CONFIG = ConditionalMidiConfig(
    vocab_size=64,
    num_composers=4,
    num_genres=4,
    d_model=32,
    n_heads=4,
    num_layers=2,
    ffn_dim=64,
    max_seq_len=64,
    cond_dim=16,
    tokenization_mode="note_velocity",
    num_composer_ids=6,
    num_genre_ids=6,
)

SHORT_CONFIG = ConditionalMidiConfig(
    vocab_size=64,
    num_composers=4,
    num_genres=4,
    d_model=32,
    n_heads=4,
    num_layers=2,
    ffn_dim=64,
    max_seq_len=8,
    cond_dim=16,
    tokenization_mode="note_velocity",
    num_composer_ids=6,
    num_genre_ids=6,
)


def _mlx_model(config: ConditionalMidiConfig, seed: int):
    from dafx26_demo.mlx_backend.model import model_from_numpy

    torch.manual_seed(seed)
    pt_model = ConditionalMidiTransformer(config)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    return model_from_numpy(asdict(config), arrays, dtype="float32")


@pytest.fixture(scope="module")
def tiny_model():
    return _mlx_model(TINY_CONFIG, seed=2)


@pytest.fixture(scope="module")
def short_model():
    return _mlx_model(SHORT_CONFIG, seed=5)


def composer() -> np.ndarray:
    return np.array([1], dtype=np.int32)


def genre() -> np.ndarray:
    return np.array([2], dtype=np.int32)


def composers(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32) % 4


def genres(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.int32) + 1) % 4


def _wrap_forward(model, wrapper):
    original = model.forward

    def wrapped(*args, **kwargs):
        return wrapper(original, *args, **kwargs)

    model.forward = wrapped
    return original


@requires_mlx
def test_iterator_stops_on_eos_cancel_and_exact_position_limit(tiny_model) -> None:
    from dafx26_demo.mlx_backend.model import iter_generate_tokens

    cancel = threading.Event()
    steps = iter_generate_tokens(
        tiny_model,
        composer(),
        genre(),
        max_new_tokens=100,
        temperature=0.0,
        seed=0,
        cache_strategy="block",
        cancel_event=cancel,
    )
    first = next(steps)
    cancel.set()
    assert list(steps) == []
    assert first.n_tokens == 1


@requires_mlx
def test_collector_keeps_existing_output_contract(tiny_model) -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens

    tokens = generate_tokens(tiny_model, composer(), genre(), max_new_tokens=8, seed=0)
    assert tokens.shape[1] <= 9
    assert tokens[0, 0] == tiny_model.config["bos_token_id"]


@requires_mlx
def test_existing_collector_preserves_two_item_batch(tiny_model) -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens

    tokens = generate_tokens(tiny_model, composers(2), genres(2), max_new_tokens=8, seed=0)
    assert tokens.shape[0] == 2
    assert tokens.shape[1] <= 9
    assert tokens[0, 0] == tiny_model.config["bos_token_id"]
    assert tokens[1, 0] == tiny_model.config["bos_token_id"]


@requires_mlx
def test_live_iterator_rejects_batch(tiny_model) -> None:
    from dafx26_demo.mlx_backend.model import iter_generate_tokens

    with pytest.raises(ValueError, match="batch size 1"):
        next(iter_generate_tokens(tiny_model, composers(2), genres(2), max_new_tokens=1, seed=0))


@requires_mlx
def test_forced_eos_yields_once_and_never_completes_event(tiny_model) -> None:
    from dafx26_demo.mlx_backend.model import iter_generate_tokens

    original = _wrap_forward(
        tiny_model,
        lambda orig, *args, **kwargs: _force_token_logits(orig(*args, **kwargs), EOS_ID),
    )
    try:
        steps = list(
            iter_generate_tokens(
                tiny_model,
                composer(),
                genre(),
                max_new_tokens=8,
                temperature=0.0,
                seed=0,
            )
        )
    finally:
        tiny_model.forward = original

    assert len(steps) == 1
    assert steps[0].token == EOS_ID
    assert steps[0].event_complete is False
    assert steps[0].n_tokens == 1


@requires_mlx
def test_max_seq_len_uses_exact_forwards_and_last_position(short_model) -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens, iter_generate_tokens

    offsets: list[int] = []
    original = _wrap_forward(
        short_model,
        lambda orig, *args, **kwargs: _record_offset(offsets, orig, *args, **kwargs),
    )
    try:
        steps = list(
            iter_generate_tokens(
                short_model,
                composer(),
                genre(),
                max_new_tokens=100,
                temperature=0.0,
                seed=0,
            )
        )
        collected = generate_tokens(short_model, composer(), genre(), max_new_tokens=100, seed=0)
    finally:
        short_model.forward = original

    max_seq_len = int(short_model.config["max_seq_len"])
    assert len(steps) == max_seq_len - 1
    assert steps[-1].n_tokens == max_seq_len - 1
    assert offsets[: max_seq_len - 1] == list(range(max_seq_len - 1))
    assert max(offsets) == max_seq_len - 2
    assert collected.shape[1] == max_seq_len


def test_eos_never_completes_a_partial_event() -> None:
    from dafx26_demo.mlx_backend.sampling import event_completed

    for phase in range(4):
        assert event_completed(pre_update_phase=phase, token=EOS_ID, use_velocity=True) is False
        assert event_completed(pre_update_phase=phase, token=EOS_ID, use_velocity=False) is False


def _force_token_logits(result, token_id: int):
    logits, cache = result
    forced = np.full_like(np.asarray(logits), -1.0e6)
    forced[..., int(token_id)] = 0.0
    return forced, cache


def _record_offset(offsets: list[int], orig, *args, **kwargs):
    offsets.append(int(kwargs.get("position_offset", 0)))
    return orig(*args, **kwargs)


@dataclass
class SseEvent:
    name: str
    data: dict


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class StubLiveHub:
    backend = "mlx"
    device = "mlx"

    def max_seq_len(self, mode: str) -> int:
        return 64

    def load(self, mode: str):
        return object(), 0.0


class FakeSegmentSource:
    def __init__(self, segments: list[list], *, cancel_after: int | None = None, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        self._segments = segments
        self.seeds: list[int] = []
        self._index = 0
        self.cancel_after = cancel_after
        self.clock = clock
        self.advance = advance

    def __call__(self, model, request, converter, cancel_event):
        self.seeds.append(int(request.seed))
        steps = self._segments[self._index]
        self._index += 1
        for step in steps:
            if self.clock is not None and self.advance:
                self.clock.advance(self.advance)
            yield step
        if self.cancel_after is not None and self._index >= self.cancel_after:
            cancel_event.set()


def decode_sse(chunks) -> list[SseEvent]:
    text = "".join(chunks)
    events: list[SseEvent] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = "message"
        data: dict = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        events.append(SseEvent(name, data))
    return events


def _eos_step() -> TokenStep:
    from dafx26_demo.mlx_backend.model import TokenStep

    return TokenStep(token=EOS_ID, n_tokens=1, event_complete=False)


def _live_events(
    *,
    segment_fn,
    seed: int = 7,
    job=None,
    load_fn=None,
    monotonic=None,
    heartbeat_sec: float = 15.0,
    sleeper=None,
):
    from dafx26_demo.live_jobs import LiveJob
    from dafx26_demo.mlx_backend.live import iter_live_sse

    job = job or LiveJob("job1", threading.Event())
    return (
        decode_sse(
            iter_live_sse(
                StubLiveHub(),
                mode="note_velocity_pedal",
                composer="Chopin",
                genre="etude",
                seed=seed,
                live_buffer_sec=2.0,
                job=job,
                segment_fn=segment_fn,
                load_fn=load_fn,
                monotonic=monotonic,
                heartbeat_sec=heartbeat_sec,
                sleeper=sleeper,
            )
        ),
        job,
    )


def test_eos_rolls_to_offset_deterministic_segment() -> None:
    from tests.test_mlx_live_events import note_tokens

    source = FakeSegmentSource(
        [
            [*note_tokens(time=0, duration=100, pitch=60, velocity=64), _eos_step()],
            [*note_tokens(time=0, duration=50, pitch=61, velocity=65)],
        ],
        cancel_after=2,
    )
    events, _job = _live_events(segment_fn=source)
    notes = [event for event in events if event.name == "midi"]
    assert notes[0].data["start_sec"] == 0.0
    assert notes[1].data["start_sec"] == pytest.approx(1.05)
    assert source.seeds == [7, 8]
    assert notes[1].data["event_id"] > notes[0].data["event_id"]
    assert events[0].name == "meta"
    assert events[-1].name == "end"
    assert events[-1].data["reason"] == "stop"


def test_three_empty_segments_end_with_error() -> None:
    source = FakeSegmentSource([[_eos_step()], [_eos_step()], [_eos_step()]])
    events, _job = _live_events(segment_fn=source)
    assert events[-1].name == "end"
    assert events[-1].data["reason"] == "error"
    assert len([event for event in events if event.name == "segment"]) == 3


def test_beat_only_segment_is_not_empty() -> None:
    from tests.test_mlx_live_events import beat_tokens

    source = FakeSegmentSource(
        [[*beat_tokens(time=10), _eos_step()]],
        cancel_after=1,
    )
    events, _job = _live_events(segment_fn=source)
    segments = [event for event in events if event.name == "segment"]
    assert segments[0].data["complete_events"] == 1
    assert events[-1].data["reason"] == "stop"
    midi = [event for event in events if event.name == "midi"]
    assert len(midi) == 1
    assert midi[0].data["kind"] == "beat"
    assert midi[0].data["beat_kind"] == "beat"


def test_heartbeat_and_rtf_use_wall_over_music() -> None:
    from dafx26_demo.metrics import rtf
    from tests.test_mlx_live_events import note_tokens

    clock = FakeClock()
    source = FakeSegmentSource(
        [[*note_tokens(time=0, duration=100, pitch=60, velocity=64), _eos_step()]],
        cancel_after=1,
        clock=clock,
        advance=4.0,
    )
    events, _job = _live_events(segment_fn=source, monotonic=clock, heartbeat_sec=15.0)
    statuses = [event for event in events if event.name == "status"]
    assert statuses[0].data["tokens"] == 0
    heartbeats = [row for row in statuses[1:] if row.data["wall_sec"] >= 15]
    assert heartbeats
    last = statuses[-1].data
    assert last["music_sec"] == pytest.approx(1.0)
    assert last["rtf"] == pytest.approx(rtf(wall_sec=last["wall_sec"], music_sec=last["music_sec"]))
    assert last["rtf"] == pytest.approx(last["wall_sec"] / last["music_sec"])


def test_generated_lead_includes_two_second_preroll() -> None:
    from dafx26_demo.mlx_backend.live import generated_lead_sec

    assert generated_lead_sec(music_sec=0.0, wall_sec=0.0, live_buffer_sec=2.0) == 2.0
    assert generated_lead_sec(music_sec=4.0, wall_sec=0.0, live_buffer_sec=2.0) == 6.0
    assert generated_lead_sec(music_sec=4.0, wall_sec=1.1, live_buffer_sec=2.0) == pytest.approx(4.9)


def test_generation_pauses_when_lead_reaches_five_seconds() -> None:
    from tests.test_mlx_live_events import note_tokens

    clock = FakeClock()
    sleeps: list[float] = []

    def sleeper(dt: float) -> None:
        sleeps.append(dt)
        clock.advance(dt)

    source = FakeSegmentSource(
        [[*note_tokens(time=300, duration=100, pitch=60, velocity=64), _eos_step()]],
        cancel_after=1,
        clock=clock,
    )
    events, _job = _live_events(segment_fn=source, monotonic=clock, sleeper=sleeper)
    notes = [event for event in events if event.name == "midi"]
    assert notes[0].data["end_sec"] == pytest.approx(4.0)
    assert sleeps == [pytest.approx(1.0)]
    assert clock.t == pytest.approx(1.0)
    assert events[-1].data["reason"] == "stop"


def test_load_failure_emits_end_error() -> None:
    def boom():
        raise RuntimeError("converted MLX weights are unavailable")

    events, _job = _live_events(segment_fn=FakeSegmentSource([]), load_fn=boom)
    assert events[0].name == "meta"
    assert events[1].name == "status"
    assert events[-1].data["reason"] == "error"
    assert "unavailable" in events[-1].data["message"]
