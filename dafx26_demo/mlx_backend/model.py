from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dafx26_demo.mlx_backend.attention import merge_heads, scaled_dot_product_attention, split_heads
from dafx26_demo.mlx_backend.availability import require_mlx
from dafx26_demo.mlx_backend.sampling import MidiConstraint, events_completed, sample_tokens
from dafx26_demo.mlx_backend.weights import expected_tensors, load_config, load_converted


def parameter_names(config: dict[str, Any]) -> list[str]:
    return list(expected_tensors(config))


def _mx_dtype(name: str):
    import mlx.core as mx

    if name == "float16":
        return mx.float16
    if name == "float32":
        return mx.float32
    raise ValueError(f"unsupported model dtype {name!r}")


def _to_mx(value: Any, *, dtype=None):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return value if dtype is None else value.astype(dtype)
    array = mx.array(np.asarray(value))
    return array if dtype is None else array.astype(dtype)


def _to_int_mx(value: Any):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return value
    return mx.array(np.asarray(value, dtype=np.int32))


class _Linear:
    def __init__(self, weight, bias) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class _LayerNorm:
    def __init__(self, weight, bias, eps: float = 1e-5) -> None:
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        import mlx.core as mx

        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) / mx.sqrt(var + mx.array(self.eps, dtype=x.dtype))
        return self.weight * x_hat + self.bias


class _SelfAttention:
    def __init__(self, n_heads: int, qkv: _Linear, out: _Linear) -> None:
        self.n_heads = n_heads
        self.qkv = qkv
        self.out = out

    def __call__(self, x, past_key_value=None, kv_cache=None, layer_index: int | None = None):
        import mlx.core as mx

        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        mask = None
        if kv_cache is not None:
            k, v = kv_cache.update(layer_index, k, v)
            mask = kv_cache.attention_mask(q_len=int(q.shape[2]))
        elif past_key_value is not None:
            pk, pv = past_key_value
            k = mx.concatenate([pk, k], axis=2)
            v = mx.concatenate([pv, v], axis=2)
        else:
            q_len = int(q.shape[2])
            mask = "causal" if q_len > 1 else None
        y = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out(merge_heads(y)), (k, v)


class _CrossAttention:
    def __init__(self, n_heads: int, q: _Linear, k: _Linear, v: _Linear, out: _Linear) -> None:
        self.n_heads = n_heads
        self.q = q
        self.k = k
        self.v = v
        self.out = out

    def __call__(self, x, memory):
        q = split_heads(self.q(x), self.n_heads)
        k = split_heads(self.k(memory), self.n_heads)
        v = split_heads(self.v(memory), self.n_heads)
        y = scaled_dot_product_attention(q, k, v, mask=None)
        return self.out(merge_heads(y))


class _DecoderLayer:
    def __init__(self, n_heads: int, arrays: dict[str, Any], prefix: str) -> None:
        self.self_norm = _LayerNorm(arrays[f"{prefix}.self_norm.weight"], arrays[f"{prefix}.self_norm.bias"])
        self.cross_norm = _LayerNorm(arrays[f"{prefix}.cross_norm.weight"], arrays[f"{prefix}.cross_norm.bias"])
        self.ffn_norm = _LayerNorm(arrays[f"{prefix}.ffn_norm.weight"], arrays[f"{prefix}.ffn_norm.bias"])
        self.self_attn = _SelfAttention(
            n_heads,
            _Linear(arrays[f"{prefix}.self_attn.qkv.weight"], arrays[f"{prefix}.self_attn.qkv.bias"]),
            _Linear(arrays[f"{prefix}.self_attn.out.weight"], arrays[f"{prefix}.self_attn.out.bias"]),
        )
        self.cross_attn = _CrossAttention(
            n_heads,
            _Linear(arrays[f"{prefix}.cross_attn.q.weight"], arrays[f"{prefix}.cross_attn.q.bias"]),
            _Linear(arrays[f"{prefix}.cross_attn.k.weight"], arrays[f"{prefix}.cross_attn.k.bias"]),
            _Linear(arrays[f"{prefix}.cross_attn.v.weight"], arrays[f"{prefix}.cross_attn.v.bias"]),
            _Linear(arrays[f"{prefix}.cross_attn.out.weight"], arrays[f"{prefix}.cross_attn.out.bias"]),
        )
        self.ffn_in = _Linear(arrays[f"{prefix}.ffn.0.weight"], arrays[f"{prefix}.ffn.0.bias"])
        self.ffn_out = _Linear(arrays[f"{prefix}.ffn.3.weight"], arrays[f"{prefix}.ffn.3.bias"])

    def __call__(self, x, memory, past_key_value=None, kv_cache=None, layer_index: int | None = None):
        import mlx.nn as nn

        y, present = self.self_attn(
            self.self_norm(x),
            past_key_value=past_key_value,
            kv_cache=kv_cache,
            layer_index=layer_index,
        )
        x = x + y
        x = x + self.cross_attn(self.cross_norm(x), memory)
        hidden = nn.gelu(self.ffn_in(self.ffn_norm(x)))
        x = x + self.ffn_out(hidden)
        return x, present


class MlxConditionalMidiTransformer:
    def __init__(self, config: dict[str, Any], arrays: dict[str, Any]) -> None:
        self.config = config
        self.token_embedding = arrays["token_embedding.weight"]
        self.position_embedding = arrays["position_embedding.weight"]
        self.condition_embedding = arrays["condition_embedding.weight"]
        self.condition_in = _Linear(arrays["condition_projection.0.weight"], arrays["condition_projection.0.bias"])
        self.condition_out = _Linear(arrays["condition_projection.2.weight"], arrays["condition_projection.2.bias"])
        self.final_norm = _LayerNorm(arrays["final_norm.weight"], arrays["final_norm.bias"])
        n_heads = int(config["n_heads"])
        self.layers = [
            _DecoderLayer(n_heads, arrays, f"layers.{index}")
            for index in range(int(config["num_layers"]))
        ]

    def _memory(self, composer_ids, genre_ids):
        import mlx.core as mx
        import mlx.nn as nn

        genre_ids = genre_ids + int(self.config["num_composers"])
        cond_ids = mx.stack([composer_ids, genre_ids], axis=1)
        hidden = nn.gelu(self.condition_in(self.condition_embedding[cond_ids]))
        return self.condition_out(hidden)

    def condition_memory(self, composer_ids, genre_ids):
        import mlx.core as mx

        memory = self._memory(_to_int_mx(composer_ids), _to_int_mx(genre_ids))
        mx.eval(memory)
        return memory

    def forward(
        self,
        input_ids,
        composer_ids,
        genre_ids,
        *,
        past_key_values=None,
        use_cache: bool = False,
        position_offset: int = 0,
        return_numpy: bool = False,
        kv_cache=None,
        memory=None,
    ):
        require_mlx()
        import mlx.core as mx

        input_ids = _to_int_mx(input_ids)
        composer_ids = _to_int_mx(composer_ids)
        genre_ids = _to_int_mx(genre_ids)
        seq_len = int(input_ids.shape[1])
        max_seq_len = int(self.config["max_seq_len"])
        if position_offset + seq_len > max_seq_len:
            raise ValueError(f"Sequence length {position_offset + seq_len} exceeds max_seq_len={max_seq_len}")
        positions = mx.arange(position_offset, position_offset + seq_len)
        hidden = self.token_embedding[input_ids] + self.position_embedding[positions]
        if memory is None:
            memory = self._memory(composer_ids, genre_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        presents = []
        for index, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            hidden, present = layer(
                hidden,
                memory,
                past_key_value=past,
                kv_cache=kv_cache,
                layer_index=index,
            )
            if use_cache:
                presents.append(present)
        if kv_cache is not None:
            kv_cache.finish_step()
        logits = self.final_norm(hidden) @ self.token_embedding.T
        mx.eval(logits, *([item for pair in presents for item in pair] if presents else []))
        out_logits = np.array(logits) if return_numpy else logits
        cache = kv_cache if kv_cache is not None else (presents if use_cache else None)
        return out_logits, cache


def model_from_numpy(config: dict[str, Any], arrays: dict[str, np.ndarray], *, dtype: str = "float32") -> MlxConditionalMidiTransformer:
    require_mlx()
    mx_dtype = _mx_dtype(dtype)
    missing = sorted(set(parameter_names(config)) - set(arrays))
    unexpected = sorted(set(arrays) - set(parameter_names(config)))
    if missing:
        raise ValueError(f"missing tensors: {missing}")
    if unexpected:
        raise ValueError(f"unexpected tensors: {unexpected}")
    converted = {name: _to_mx(array, dtype=mx_dtype) for name, array in arrays.items()}
    converted["lm_head.weight"] = converted["token_embedding.weight"]
    return MlxConditionalMidiTransformer(config, converted)


def load_converted_model(dest_dir: str | Path, *, dtype: str = "float16") -> MlxConditionalMidiTransformer:
    dest = Path(dest_dir)
    try:
        arrays, _meta = load_converted(dest)
        config = load_config(dest)
        return model_from_numpy(config, arrays, dtype=dtype)
    except Exception as exc:
        raise RuntimeError(
            f"converted MLX weights are unavailable or invalid at {dest}. "
            "Run: uv run python scripts/convert_models_mlx.py"
        ) from exc


def logits_numpy(
    model: MlxConditionalMidiTransformer,
    input_ids: np.ndarray,
    composer_ids: np.ndarray,
    genre_ids: np.ndarray,
    *,
    position_offset: int = 0,
) -> np.ndarray:
    logits, _cache = model.forward(
        input_ids,
        composer_ids,
        genre_ids,
        position_offset=position_offset,
        return_numpy=True,
    )
    return logits


@dataclass(frozen=True)
class TokenStep:
    token: int
    n_tokens: int
    event_complete: bool


@dataclass(frozen=True)
class BatchTokenStep:
    tokens: np.ndarray
    n_tokens: int
    events_complete: np.ndarray


def _iter_generate_batches(
    model: MlxConditionalMidiTransformer,
    composer_ids: np.ndarray,
    genre_ids: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    constraint: MidiConstraint | None = None,
    seed: int | None = None,
    cache_strategy: str | None = None,
    bucket_len: int | None = None,
    cancel_event: Any | None = None,
) -> Iterator[BatchTokenStep]:
    composer_ids = np.asarray(composer_ids)
    genre_ids = np.asarray(genre_ids)
    batch = int(composer_ids.shape[0])
    bos = int(model.config["bos_token_id"])
    eos = int(model.config["eos_token_id"])
    last_token = np.full((batch, 1), bos, dtype=np.int64)
    finished = np.zeros(batch, dtype=bool)
    cache = None
    kv_cache = None
    max_seq_len = int(model.config["max_seq_len"])
    if cache_strategy:
        from dafx26_demo.mlx_backend.cache import make_kv_cache

        kv_cache = make_kv_cache(
            cache_strategy,
            model,
            max_len=bucket_len or max_seq_len,
            bucket_len=bucket_len,
        )
    rng = np.random.default_rng(seed)
    seq_len = 1
    n_tokens = 0
    use_velocity = bool(constraint.spec.use_velocity) if constraint is not None else False
    while n_tokens < max_new_tokens and seq_len < max_seq_len:
        if cancel_event is not None and cancel_event.is_set():
            return
        logits, cache = model.forward(
            last_token,
            composer_ids,
            genre_ids,
            past_key_values=None if kv_cache is not None else cache,
            kv_cache=kv_cache,
            use_cache=True,
            position_offset=seq_len - 1,
            return_numpy=True,
        )
        step_logits = logits[:, -1, :]
        if constraint is not None:
            step_logits = constraint.mask(step_logits)
        next_token = sample_tokens(step_logits, temperature=temperature, top_p=top_p, rng=rng)
        next_token = np.where(finished, eos, next_token)
        if constraint is not None:
            events = events_completed(constraint.phase, next_token, use_velocity=use_velocity)
            constraint.update(next_token)
        else:
            events = np.zeros(batch, dtype=bool)
        n_tokens += 1
        seq_len += 1
        last_token = next_token[:, None]
        finished |= next_token == eos
        yield BatchTokenStep(tokens=next_token, n_tokens=n_tokens, events_complete=events)
        if bool(finished.all()):
            return


def generate_tokens(
    model: MlxConditionalMidiTransformer,
    composer_ids: np.ndarray,
    genre_ids: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    constraint: MidiConstraint | None = None,
    seed: int | None = None,
    cache_strategy: str | None = None,
    bucket_len: int | None = None,
) -> np.ndarray:
    composer_ids = np.asarray(composer_ids)
    rows = [[int(model.config["bos_token_id"])] for _ in range(int(composer_ids.shape[0]))]
    for step in _iter_generate_batches(
        model,
        composer_ids,
        genre_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        constraint=constraint,
        seed=seed,
        cache_strategy=cache_strategy,
        bucket_len=bucket_len,
    ):
        for row, token in zip(rows, step.tokens, strict=True):
            row.append(int(token))
    return np.asarray(rows, dtype=np.int64)


def iter_generate_tokens(
    model: MlxConditionalMidiTransformer,
    composer_ids: np.ndarray,
    genre_ids: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    constraint: MidiConstraint | None = None,
    seed: int | None = None,
    cache_strategy: str | None = None,
    bucket_len: int | None = None,
    cancel_event: Any | None = None,
) -> Iterator[TokenStep]:
    composer_ids = np.asarray(composer_ids)
    if composer_ids.shape[0] != 1:
        raise ValueError("live token iteration requires batch size 1")
    for step in _iter_generate_batches(
        model,
        composer_ids,
        genre_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        constraint=constraint,
        seed=seed,
        cache_strategy=cache_strategy,
        bucket_len=bucket_len,
        cancel_event=cancel_event,
    ):
        yield TokenStep(
            token=int(step.tokens[0]),
            n_tokens=step.n_tokens,
            event_complete=bool(step.events_complete[0]),
        )
