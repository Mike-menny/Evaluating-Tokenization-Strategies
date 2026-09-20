from __future__ import annotations

import numpy as np
import pytest
import torch

from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.inference.inference_conditional import FastMidiConstraint  # noqa: E402
from src.model.conditional_transformer import _sample_top_p  # noqa: E402
from src.tokenization.conditional_vocab import get_tokenization_spec  # noqa: E402


def test_top_p_one_keeps_scaled_logits() -> None:
    from dafx26_demo.mlx_backend.sampling import MidiConstraint, nucleus_logits

    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    filtered = nucleus_logits(logits, temperature=2.0, top_p=1.0)
    np.testing.assert_allclose(filtered, logits / 2.0)
    with pytest.raises(ValueError, match="requires vocab_size"):
        MidiConstraint(batch_size=1, vocab_size=8, mode="note", min_tokens=1)


def test_beat_mode_constraint_allows_beat_tokens() -> None:
    from dafx26_demo.mlx_backend.sampling import MidiConstraint
    from src.tokenization.conditional_vocab import BEAT_B_ID, DUR_OFFSET

    spec = get_tokenization_spec("note_velocity_beat")
    constraint = MidiConstraint(batch_size=1, vocab_size=spec.vocab_size, mode=spec.name, min_tokens=1)
    constraint.phase[0] = 1
    logits = np.zeros((1, spec.vocab_size), dtype=np.float32)
    masked = constraint.mask(logits)
    assert np.isfinite(masked[0, BEAT_B_ID])
    assert np.isfinite(masked[0, DUR_OFFSET])
    from dafx26_demo.mlx_backend.sampling import greedy_tokens

    logits = np.array([[0.1, 2.0, 0.3], [4.0, -1.0, 0.2]], dtype=np.float32)
    assert greedy_tokens(logits).tolist() == [1, 0]


def test_midi_constraint_matches_pytorch() -> None:
    from dafx26_demo.mlx_backend.sampling import MidiConstraint

    vocab_size = get_tokenization_spec("note_velocity_pedal").vocab_size
    mode = "note_velocity_pedal"
    torch_constraint = FastMidiConstraint(
        batch_size=2,
        vocab_size=vocab_size,
        mode=mode,
        min_tokens=4,
        device=torch.device("cpu"),
    )
    mlx_constraint = MidiConstraint(batch_size=2, vocab_size=vocab_size, mode=mode, min_tokens=4)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, vocab_size)).astype(np.float32)
    torch_masked = torch_constraint.mask(torch.from_numpy(logits)).numpy()
    mlx_masked = mlx_constraint.mask(logits)
    np.testing.assert_allclose(mlx_masked, torch_masked, equal_nan=False)

    tokens = np.array([10, 11], dtype=np.int64)
    torch_constraint.update(torch.from_numpy(tokens))
    mlx_constraint.update(tokens)
    logits2 = rng.normal(size=(2, vocab_size)).astype(np.float32)
    torch_masked2 = torch_constraint.mask(torch.from_numpy(logits2)).numpy()
    mlx_masked2 = mlx_constraint.mask(logits2)
    np.testing.assert_allclose(mlx_masked2, torch_masked2)


def test_temperature_zero_uses_argmax() -> None:
    from dafx26_demo.mlx_backend.sampling import sample_tokens

    logits = np.array([[0.2, 5.0, 0.1]], dtype=np.float32)
    sampled = sample_tokens(logits, temperature=0.0, top_p=0.98, rng=np.random.default_rng(1))
    assert sampled.tolist() == [1]


def test_top_p_is_seeded_reproducible() -> None:
    from dafx26_demo.mlx_backend.sampling import sample_tokens

    logits = np.linspace(-1.0, 2.0, 64, dtype=np.float32).reshape(1, 64)
    a = sample_tokens(logits, temperature=0.95, top_p=0.9, rng=np.random.default_rng(7))
    b = sample_tokens(logits, temperature=0.95, top_p=0.9, rng=np.random.default_rng(7))
    c = sample_tokens(logits, temperature=0.95, top_p=0.9, rng=np.random.default_rng(8))
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()


def test_event_completed_matches_grammar_phases() -> None:
    from dafx26_demo.mlx_backend.sampling import event_completed
    from src.tokenization.conditional_vocab import (
        BEAT_B_ID,
        DUR_OFFSET,
        NOTE_OFFSET,
        PEDAL_ON_ID,
        VELOCITY_OFFSET,
    )

    assert event_completed(pre_update_phase=1, token=BEAT_B_ID, use_velocity=True) is True
    assert event_completed(pre_update_phase=1, token=PEDAL_ON_ID, use_velocity=False) is True
    assert event_completed(pre_update_phase=1, token=DUR_OFFSET + 3, use_velocity=False) is False
    assert event_completed(pre_update_phase=2, token=NOTE_OFFSET + 60, use_velocity=False) is True
    assert event_completed(pre_update_phase=2, token=NOTE_OFFSET + 60, use_velocity=True) is False
    assert event_completed(pre_update_phase=3, token=VELOCITY_OFFSET + 64, use_velocity=True) is True
    assert event_completed(pre_update_phase=0, token=NOTE_OFFSET + 60, use_velocity=False) is False


def test_top_p_mask_matches_pytorch_nucleus() -> None:
    from dafx26_demo.mlx_backend.sampling import nucleus_logits

    logits = torch.linspace(-2.0, 3.0, 32).unsqueeze(0)
    filtered = nucleus_logits(logits.numpy(), temperature=0.8, top_p=0.9)
    torch.manual_seed(0)
    # Reconstruct the same filtered logits by running the PyTorch sort/mask path.
    scaled = logits / 0.8
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > 0.9
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    expected = torch.full_like(scaled, float("-inf")).scatter(1, sorted_indices, sorted_logits)
    np.testing.assert_allclose(filtered, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert _sample_top_p is not None
