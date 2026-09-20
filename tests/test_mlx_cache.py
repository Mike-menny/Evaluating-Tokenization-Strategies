from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from dafx26_demo.mlx_backend import mlx_available
from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402

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
STRATEGIES = ("dynamic", "block", "bucket")


def _tiny_mlx_model():
    from dafx26_demo.mlx_backend.model import model_from_numpy

    torch.manual_seed(4)
    pt_model = ConditionalMidiTransformer(TINY_CONFIG)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt_model.state_dict().items()}
    return model_from_numpy(asdict(TINY_CONFIG), arrays, dtype="float32")


@requires_mlx
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("prefix", (1, 2, 17))
def test_cache_strategy_matches_full_forward(strategy: str, prefix: int) -> None:
    from dafx26_demo.mlx_backend.cache import make_kv_cache
    from dafx26_demo.mlx_backend.model import logits_numpy

    model = _tiny_mlx_model()
    tokens = np.arange(1, prefix + 1, dtype=np.int32).reshape(1, -1)
    composer = np.array([1], dtype=np.int32)
    genre = np.array([2], dtype=np.int32)
    full = logits_numpy(model, tokens, composer, genre)
    cache = make_kv_cache(strategy, model, max_len=32)
    steps = []
    for index in range(prefix):
        logits, cache = model.forward(
            tokens[:, index : index + 1],
            composer,
            genre,
            kv_cache=cache,
            use_cache=True,
            position_offset=index,
            return_numpy=True,
        )
        steps.append(logits[:, -1])
    stacked = np.stack(steps, axis=1)
    np.testing.assert_allclose(stacked, full, rtol=1e-4, atol=1e-4)


@requires_mlx
def test_block_cache_grows_at_256_and_512_boundaries() -> None:
    import mlx.core as mx

    from dafx26_demo.mlx_backend.cache import make_kv_cache

    cache = make_kv_cache(
        "block",
        n_layers=1,
        batch=1,
        n_heads=2,
        head_dim=4,
        dtype=mx.float32,
        block_size=256,
    )
    k = mx.ones((1, 2, 1, 4))
    v = mx.ones((1, 2, 1, 4))
    for _ in range(255):
        cache.update(0, k, v)
    assert cache.capacity == 256
    assert cache.offset == 255
    cache.update(0, k, v)
    assert cache.capacity == 256
    assert cache.offset == 256
    cache.update(0, k, v)
    assert cache.capacity == 512
    assert cache.offset == 257
    for _ in range(511 - 257):
        cache.update(0, k, v)
    assert cache.offset == 511
    assert cache.capacity == 512
    cache.update(0, k, v)
    assert cache.offset == 512
    assert cache.capacity == 512
    cache.update(0, k, v)
    assert cache.capacity == 768
    assert cache.offset == 513


@requires_mlx
def test_bucket_mask_hides_unused_positions() -> None:
    import mlx.core as mx

    from dafx26_demo.mlx_backend.cache import make_kv_cache

    cache = make_kv_cache(
        "bucket",
        n_layers=1,
        batch=1,
        n_heads=1,
        head_dim=2,
        dtype=mx.float32,
        bucket_len=8,
    )
    k = mx.ones((1, 1, 1, 2))
    v = mx.ones((1, 1, 1, 2))
    keys, values = cache.update(0, k, v)
    mask = cache.attention_mask(q_len=1)
    mx.eval(keys, values, mask)
    mask_np = np.array(mask)
    assert mask_np.shape[-1] == 8
    assert np.isfinite(mask_np[..., 0]).all()
    assert not np.isfinite(mask_np[..., 1:]).any()


@requires_mlx
def test_compiled_bucket_matches_eager_greedy() -> None:
    from dafx26_demo.mlx_backend.cache import CompiledDecode
    from dafx26_demo.mlx_backend.model import generate_tokens

    model = _tiny_mlx_model()
    composer = np.array([1], dtype=np.int32)
    genre = np.array([2], dtype=np.int32)
    eager = generate_tokens(
        model,
        composer,
        genre,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        cache_strategy="bucket",
        bucket_len=32,
    )
    compiled = CompiledDecode(model, bucket_len=32)
    compiled.warmup(composer, genre)
    out = compiled.generate(composer, genre, max_new_tokens=8, temperature=0.0)
    assert out.tolist() == eager.tolist()
    assert compiled.stats["recompilations"] == 0
    assert compiled.stats["warmup_sec"] >= 0
    assert compiled.stats["failures"] == []


@requires_mlx
def test_hoisted_condition_memory_matches_recomputed() -> None:
    from dafx26_demo.mlx_backend.model import logits_numpy

    model = _tiny_mlx_model()
    tokens = np.array([[1, 4, 7, 9]], dtype=np.int32)
    composer = np.array([1], dtype=np.int32)
    genre = np.array([2], dtype=np.int32)
    plain = logits_numpy(model, tokens, composer, genre)
    memory = model.condition_memory(composer, genre)
    hoisted, _ = model.forward(
        tokens,
        composer,
        genre,
        memory=memory,
        return_numpy=True,
    )
    np.testing.assert_allclose(hoisted, plain, rtol=1e-4, atol=1e-4)


def test_block_smoke_expectations_match_8192_plan() -> None:
    from dafx26_demo.mlx_backend.benchmark import analytical_block_capacity, block_smoke_expectations

    expected = block_smoke_expectations(8192)
    assert expected == {
        "forwards": 8191,
        "final_offset": 8191,
        "capacity": 8192,
        "max_position": 8190,
    }
    assert analytical_block_capacity(8191) == 8192


@requires_mlx
def test_cache_helpers_reject_invalid_inputs() -> None:
    import mlx.core as mx

    from dafx26_demo.mlx_backend.cache import make_kv_cache

    with pytest.raises(ValueError, match="unknown cache"):
        make_kv_cache("nope", n_layers=1, n_heads=1, head_dim=2, dtype=mx.float32)
    with pytest.raises(ValueError, match="dimensions are required"):
        make_kv_cache("dynamic")
    cache = make_kv_cache("bucket", n_layers=1, batch=1, n_heads=1, head_dim=2, dtype=mx.float32, bucket_len=2)
    k = mx.ones((1, 1, 1, 2))
    v = mx.ones((1, 1, 1, 2))
    cache.update(0, k, v)
    cache.update(0, k, v)
    with pytest.raises(ValueError, match="exceeded"):
        cache.update(0, k, v)
