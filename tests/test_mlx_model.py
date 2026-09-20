from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from dafx26_demo.mlx_backend import mlx_available
from dafx26_demo.midi_convert import clean_tokens, midi_is_valid, tokens_to_midi_bytes
from dafx26_demo.paths import MODELS_ROOT, ensure_upstream_on_path

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_vocab import BOS_ID, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402

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

requires_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX is not installed")

# Pinned grammar-valid `note` prefix: BOS + 10 complete notes + one extra time token.
REAL_PARITY_PREFIX_TOKENS = [
    BOS_ID,
    TIME_OFFSET + 0, DUR_OFFSET + 20, NOTE_OFFSET + 60,
    TIME_OFFSET + 30, DUR_OFFSET + 18, NOTE_OFFSET + 64,
    TIME_OFFSET + 60, DUR_OFFSET + 24, NOTE_OFFSET + 67,
    TIME_OFFSET + 90, DUR_OFFSET + 16, NOTE_OFFSET + 72,
    TIME_OFFSET + 120, DUR_OFFSET + 20, NOTE_OFFSET + 67,
    TIME_OFFSET + 150, DUR_OFFSET + 22, NOTE_OFFSET + 64,
    TIME_OFFSET + 180, DUR_OFFSET + 20, NOTE_OFFSET + 60,
    TIME_OFFSET + 210, DUR_OFFSET + 30, NOTE_OFFSET + 55,
    TIME_OFFSET + 250, DUR_OFFSET + 20, NOTE_OFFSET + 62,
    TIME_OFFSET + 280, DUR_OFFSET + 40, NOTE_OFFSET + 67,
    TIME_OFFSET + 330,
]


def test_parameter_names_cover_pytorch_state_dict() -> None:
    from dafx26_demo.mlx_backend.model import parameter_names

    model = ConditionalMidiTransformer(TINY_CONFIG)
    assert set(parameter_names(asdict(TINY_CONFIG))) == set(model.state_dict())


@requires_mlx
def test_tiny_fp32_logits_match_pytorch() -> None:
    from dafx26_demo.mlx_backend.model import logits_numpy, model_from_numpy

    torch.manual_seed(0)
    pt_model = ConditionalMidiTransformer(TINY_CONFIG)
    pt_model.eval()
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    mlx_model = model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="float32")
    input_ids = torch.tensor([[1, 7, 9, 11, 13]], dtype=torch.long)
    composer_ids = torch.tensor([1], dtype=torch.long)
    genre_ids = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        pt_logits = pt_model(input_ids, composer_ids, genre_ids)["logits"].numpy()
    mx_logits = logits_numpy(mlx_model, input_ids.numpy(), composer_ids.numpy(), genre_ids.numpy())
    np.testing.assert_allclose(mx_logits, pt_logits, rtol=1e-4, atol=1e-4)


@requires_mlx
def test_tiny_cached_decode_matches_full_forward() -> None:
    from dafx26_demo.mlx_backend.model import logits_numpy, model_from_numpy

    torch.manual_seed(1)
    pt_model = ConditionalMidiTransformer(TINY_CONFIG)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    mlx_model = model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="float32")
    tokens = np.array([[1, 5, 8, 12]], dtype=np.int32)
    composer = np.array([0], dtype=np.int32)
    genre = np.array([1], dtype=np.int32)
    full = logits_numpy(mlx_model, tokens, composer, genre)
    cache = None
    step_logits = []
    for index in range(tokens.shape[1]):
        step, cache = mlx_model.forward(
            tokens[:, index : index + 1],
            composer,
            genre,
            past_key_values=cache,
            use_cache=True,
            position_offset=index,
            return_numpy=True,
        )
        step_logits.append(step[:, -1])
    stacked = np.stack(step_logits, axis=1)
    np.testing.assert_allclose(stacked, full, rtol=1e-4, atol=1e-4)


@requires_mlx
def test_tiny_greedy_sequence_matches_pytorch() -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens, model_from_numpy

    torch.manual_seed(2)
    pt_model = ConditionalMidiTransformer(TINY_CONFIG)
    pt_model.eval()
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    mlx_model = model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="float32")
    composer = torch.tensor([1], dtype=torch.long)
    genre = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        pt_tokens = pt_model.generate(composer, genre, max_new_tokens=32, temperature=0.0, top_p=1.0)[0].tolist()
    mx_tokens = generate_tokens(
        mlx_model,
        composer.numpy(),
        genre.numpy(),
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
    )[0].tolist()
    assert mx_tokens == pt_tokens


@requires_mlx
def test_real_fp16_logits_within_tolerance() -> None:
    from dafx26_demo.mlx_backend.model import logits_numpy, load_converted_model

    src = MODELS_ROOT / "note"
    if not (src / "pytorch_model.bin").is_file():
        pytest.skip("pinned note checkpoint missing")
    pt_model, _payload = ConditionalMidiTransformer.from_pretrained(src, map_location="cpu")
    pt_model.eval()
    pt_model = pt_model.to(dtype=torch.float16)
    mlx_model = load_converted_model(src / "mlx", dtype="float16")
    input_ids = torch.tensor([[1, 20, 40, 80, 120, 200, 300, 400]], dtype=torch.long)
    composer_ids = torch.tensor([4], dtype=torch.long)
    genre_ids = torch.tensor([15], dtype=torch.long)
    with torch.no_grad():
        pt_logits = pt_model(input_ids, composer_ids, genre_ids)["logits"].float().numpy()
    mx_logits = logits_numpy(mlx_model, input_ids.numpy(), composer_ids.numpy(), genre_ids.numpy()).astype(np.float32)
    assert np.isfinite(mx_logits).all()
    np.testing.assert_allclose(mx_logits, pt_logits, rtol=5e-2, atol=5e-2)


@requires_mlx
def test_real_fp16_fixed_prefix_decode_parity() -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens, logits_numpy, load_converted_model

    assert len(REAL_PARITY_PREFIX_TOKENS) == 32
    src = MODELS_ROOT / "note"
    if not (src / "pytorch_model.bin").is_file() or not (src / "mlx" / "mlx_model.safetensors").is_file():
        pytest.skip("pinned note checkpoint or converted MLX weights missing")
    pt_model, _payload = ConditionalMidiTransformer.from_pretrained(src, map_location="cpu")
    pt_model.eval()
    pt_model = pt_model.to(dtype=torch.float16)
    mlx_model = load_converted_model(src / "mlx", dtype="float16")
    composer = np.array([4], dtype=np.int32)
    genre = np.array([15], dtype=np.int32)
    composer_t = torch.tensor(composer, dtype=torch.long)
    genre_t = torch.tensor(genre, dtype=torch.long)
    low_margin = 0
    for length in range(1, 33):
        prefix = REAL_PARITY_PREFIX_TOKENS[:length]
        ids = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            pt_logits = pt_model(ids, composer_t, genre_t)["logits"][0, -1].float().numpy()
        mx_logits = logits_numpy(
            mlx_model,
            np.asarray([prefix], dtype=np.int32),
            composer,
            genre,
        )[0, -1].astype(np.float32)
        pt = np.asarray(pt_logits, dtype=np.float32).reshape(-1)
        mx = np.asarray(mx_logits, dtype=np.float32).reshape(-1)
        assert pt.shape == mx.shape
        assert np.isfinite(pt).all() and np.isfinite(mx).all()
        np.testing.assert_allclose(mx, pt, rtol=5e-2, atol=5e-2)
        top2 = np.partition(pt, -2)[-2:]
        pt_argmax = int(np.argmax(pt))
        if float(top2[-1] - top2[-2]) > 0.1:
            assert int(np.argmax(mx)) == pt_argmax
        elif int(np.argmax(mx)) != pt_argmax:
            low_margin += 1
    mlx_ar = generate_tokens(mlx_model, composer, genre, max_new_tokens=32, temperature=0.0, top_p=1.0)[0].tolist()
    with torch.no_grad():
        torch_ar = pt_model.generate(composer_t, genre_t, max_new_tokens=32, temperature=0.0, top_p=1.0)[0].tolist()
    print(
        f"fp16 AR diagnostic: mlx={mlx_ar[:12]} pytorch={torch_ar[:12]} "
        f"match={mlx_ar == torch_ar} low_margin_argmax_diffs={low_margin}",
        flush=True,
    )


@requires_mlx
def test_real_greedy_tokens_are_grammar_and_midi_valid() -> None:
    from dafx26_demo.mlx_backend.model import generate_tokens, load_converted_model
    from dafx26_demo.mlx_backend.sampling import MidiConstraint

    src = MODELS_ROOT / "note"
    if not (src / "pytorch_model.bin").is_file():
        pytest.skip("pinned note checkpoint missing")
    mlx_model = load_converted_model(src / "mlx", dtype="float16")
    composer = np.array([4], dtype=np.int32)
    genre = np.array([15], dtype=np.int32)
    constraint = MidiConstraint(
        batch_size=1,
        vocab_size=int(mlx_model.config["vocab_size"]),
        mode="note",
        min_tokens=8,
    )
    tokens = generate_tokens(
        mlx_model,
        composer,
        genre,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        constraint=constraint,
    )[0].tolist()
    cleaned = clean_tokens(tokens)
    assert cleaned
    midi = tokens_to_midi_bytes(cleaned, mode="note")
    assert midi_is_valid(midi)


@requires_mlx
def test_model_rejects_bad_inputs_and_is_seeded() -> None:
    from dataclasses import asdict

    from dafx26_demo.mlx_backend.model import generate_tokens, model_from_numpy

    torch.manual_seed(3)
    pt_model = ConditionalMidiTransformer(TINY_CONFIG)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    mlx_model = model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="float32")
    with pytest.raises(ValueError, match="unsupported model dtype"):
        model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="int8")
    missing = dict(arrays)
    del missing["final_norm.weight"]
    with pytest.raises(ValueError, match="missing tensors"):
        model_from_numpy(asdict(TINY_CONFIG), missing, dtype="float32")
    extra = dict(arrays)
    extra["bonus.weight"] = extra["final_norm.weight"]
    with pytest.raises(ValueError, match="unexpected tensors"):
        model_from_numpy(asdict(TINY_CONFIG), extra, dtype="float32")
    composer = np.array([1], dtype=np.int32)
    genre = np.array([2], dtype=np.int32)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        mlx_model.forward(np.zeros((1, 8), dtype=np.int32), composer, genre, position_offset=60)
    a = generate_tokens(mlx_model, composer, genre, max_new_tokens=8, temperature=0.8, top_p=0.9, seed=11)
    b = generate_tokens(mlx_model, composer, genre, max_new_tokens=8, temperature=0.8, top_p=0.9, seed=11)
    c = generate_tokens(mlx_model, composer, genre, max_new_tokens=8, temperature=0.8, top_p=0.9, seed=12)
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()
