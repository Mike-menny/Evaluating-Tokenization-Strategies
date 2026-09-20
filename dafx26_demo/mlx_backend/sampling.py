from __future__ import annotations

import numpy as np

from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import (  # noqa: E402
    BEAT_B_ID,
    BEAT_BR_ID,
    BEAT_DB_ID,
    DUR_OFFSET,
    EOS_ID,
    NOTE_OFFSET,
    PEDAL_OFF_ID,
    PEDAL_ON_ID,
    VELOCITY_OFFSET,
    get_tokenization_spec,
)

_BEAT_IDS = np.array([BEAT_B_ID, BEAT_DB_ID, BEAT_BR_ID], dtype=np.int64)
_PEDAL_IDS = np.array([PEDAL_OFF_ID, PEDAL_ON_ID], dtype=np.int64)

NEG_INF = np.float32("-inf")


def greedy_tokens(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=-1).astype(np.int64)


def nucleus_logits(logits: np.ndarray, *, temperature: float, top_p: float) -> np.ndarray:
    scaled = logits.astype(np.float32, copy=False) / max(float(temperature), 1e-6)
    if top_p >= 1.0:
        return scaled
    sorted_indices = np.argsort(-scaled, axis=-1)
    sorted_logits = np.take_along_axis(scaled, sorted_indices, axis=-1)
    shifted = sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    cumulative = np.cumsum(probs, axis=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1]
    remove[..., 0] = False
    sorted_logits = np.where(remove, NEG_INF, sorted_logits)
    filtered = np.full_like(scaled, NEG_INF)
    np.put_along_axis(filtered, sorted_indices, sorted_logits, axis=-1)
    return filtered


def sample_tokens(
    logits: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if temperature <= 0:
        return greedy_tokens(logits)
    filtered = nucleus_logits(logits, temperature=temperature, top_p=top_p)
    shifted = filtered - np.nanmax(np.where(np.isfinite(filtered), filtered, NEG_INF), axis=-1, keepdims=True)
    exp = np.exp(np.where(np.isfinite(filtered), shifted, NEG_INF))
    denom = np.sum(exp, axis=-1, keepdims=True)
    probs = exp / np.clip(denom, 1e-12, None)
    batch, vocab = probs.shape
    out = np.empty(batch, dtype=np.int64)
    for index in range(batch):
        out[index] = int(rng.choice(vocab, p=probs[index]))
    return out


class MidiConstraint:
    """NumPy port of upstream FastMidiConstraint phase masks."""

    def __init__(self, batch_size: int, vocab_size: int, mode: str, min_tokens: int) -> None:
        spec = get_tokenization_spec(mode)
        if spec.vocab_size > vocab_size:
            raise ValueError(
                f"Tokenization mode {spec.name!r} requires vocab_size >= {spec.vocab_size}, "
                f"got {vocab_size}"
            )
        self.phase = np.zeros(batch_size, dtype=np.int64)
        self.length = np.zeros(batch_size, dtype=np.int64)
        self.min_tokens = int(min_tokens)
        self.eos_id = EOS_ID
        self.spec = spec
        self.vocab_size = vocab_size
        masks = [np.full((vocab_size,), NEG_INF, dtype=np.float32) for _ in range(4)]
        masks[0][3:DUR_OFFSET] = 0.0
        masks[1][DUR_OFFSET:NOTE_OFFSET] = 0.0
        if spec.use_beat:
            masks[1][[BEAT_B_ID, BEAT_DB_ID, BEAT_BR_ID]] = 0.0
        if spec.use_pedal:
            masks[1][[PEDAL_OFF_ID, PEDAL_ON_ID]] = 0.0
        masks[2][NOTE_OFFSET:VELOCITY_OFFSET] = 0.0
        masks[3][VELOCITY_OFFSET : VELOCITY_OFFSET + 128] = 0.0
        self.masks = np.stack(masks, axis=0)

    def mask(self, logits: np.ndarray) -> np.ndarray:
        phase_mask = self.masks[self.phase].astype(logits.dtype, copy=False)
        masked = logits + phase_mask
        allow_eos = self.length >= self.min_tokens
        if np.any(allow_eos):
            masked[allow_eos, self.eos_id] = logits[allow_eos, self.eos_id]
        return masked

    def update(self, next_token: np.ndarray) -> None:
        tok = np.asarray(next_token)
        phase = self.phase
        is_eos = tok == self.eos_id
        is_dur = (DUR_OFFSET <= tok) & (tok < NOTE_OFFSET)
        is_short_event = (phase == 1) & ~is_dur
        new_phase = phase.copy()
        new_phase = np.where(phase == 0, 1, new_phase)
        new_phase = np.where(is_short_event, 0, new_phase)
        new_phase = np.where((phase == 1) & is_dur, 2, new_phase)
        if self.spec.use_velocity:
            new_phase = np.where(phase == 2, 3, new_phase)
            new_phase = np.where(phase == 3, 0, new_phase)
        else:
            new_phase = np.where(phase == 2, 0, new_phase)
        new_phase = np.where(is_eos, 0, new_phase)
        self.phase = new_phase
        self.length += (~is_eos).astype(np.int64)


def events_completed(
    pre_update_phase: np.ndarray,
    tokens: np.ndarray,
    *,
    use_velocity: bool,
) -> np.ndarray:
    phase = np.asarray(pre_update_phase, dtype=np.int64)
    tok = np.asarray(tokens, dtype=np.int64)
    is_eos = tok == EOS_ID
    is_beat = np.isin(tok, _BEAT_IDS)
    is_pedal = np.isin(tok, _PEDAL_IDS)
    is_note = (NOTE_OFFSET <= tok) & (tok < VELOCITY_OFFSET)
    is_velocity = (VELOCITY_OFFSET <= tok) & (tok < VELOCITY_OFFSET + 128)
    complete = (phase == 1) & (is_beat | is_pedal)
    if use_velocity:
        complete = complete | ((phase == 3) & is_velocity)
    else:
        complete = complete | ((phase == 2) & is_note)
    return complete & ~is_eos


def event_completed(*, pre_update_phase: int, token: int, use_velocity: bool) -> bool:
    return bool(
        events_completed(
            np.asarray([pre_update_phase]),
            np.asarray([token]),
            use_velocity=use_velocity,
        )[0]
    )
