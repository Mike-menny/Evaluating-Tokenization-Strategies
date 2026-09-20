from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from dafx26_demo.mlx_backend.availability import require_mlx
from dafx26_demo.mlx_backend.sampling import greedy_tokens


class BaseKVCache:
    offset: int = 0
    capacity: int = 0
    _pending_t: int = 0

    def update(self, layer: int, key, value):
        raise NotImplementedError

    def attention_mask(self, q_len: int = 1):
        return None

    def finish_step(self) -> None:
        if self._pending_t:
            self.offset += int(self._pending_t)
            self._pending_t = 0

    def _after_write(self, layer: int, width: int) -> None:
        self._pending_t = int(width)
        if layer == len(self.keys) - 1:
            self.finish_step()


class DynamicKVCache(BaseKVCache):
    def __init__(self, n_layers: int) -> None:
        self.keys: list[Any] = [None] * n_layers
        self.values: list[Any] = [None] * n_layers
        self.offset = 0
        self.capacity = 0
        self._pending_t = 0

    def update(self, layer: int, key, value):
        import mlx.core as mx

        if self.keys[layer] is None:
            self.keys[layer] = key
            self.values[layer] = value
        else:
            self.keys[layer] = mx.concatenate([self.keys[layer], key], axis=2)
            self.values[layer] = mx.concatenate([self.values[layer], value], axis=2)
        self.capacity = int(self.keys[layer].shape[2])
        self._after_write(layer, int(key.shape[2]))
        return self.keys[layer], self.values[layer]


class BlockKVCache(BaseKVCache):
    def __init__(self, n_layers: int, batch: int, n_heads: int, head_dim: int, dtype, block_size: int = 256) -> None:
        import mlx.core as mx

        self.block_size = int(block_size)
        self.capacity = self.block_size
        self.offset = 0
        self._pending_t = 0
        self.dtype = dtype
        zeros = mx.zeros((batch, n_heads, self.capacity, head_dim), dtype=dtype)
        self.keys = [mx.zeros_like(zeros) for _ in range(n_layers)]
        self.values = [mx.zeros_like(zeros) for _ in range(n_layers)]

    def _grow(self, needed: int) -> None:
        import mlx.core as mx

        if needed <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < needed:
            new_cap += self.block_size
        batch, n_heads, _, head_dim = self.keys[0].shape
        for index, key in enumerate(self.keys):
            grown_k = mx.zeros((batch, n_heads, new_cap, head_dim), dtype=self.dtype)
            grown_v = mx.zeros((batch, n_heads, new_cap, head_dim), dtype=self.dtype)
            if self.offset:
                grown_k[:, :, : self.offset, :] = key[:, :, : self.offset, :]
                grown_v[:, :, : self.offset, :] = self.values[index][:, :, : self.offset, :]
            self.keys[index] = grown_k
            self.values[index] = grown_v
        self.capacity = new_cap

    def update(self, layer: int, key, value):
        t = int(key.shape[2])
        self._grow(self.offset + t)
        end = self.offset + t
        self.keys[layer][:, :, self.offset : end, :] = key
        self.values[layer][:, :, self.offset : end, :] = value
        used = self.offset + t
        keys, values = self.keys[layer][:, :, :used, :], self.values[layer][:, :, :used, :]
        self._after_write(layer, t)
        return keys, values


class BucketKVCache(BaseKVCache):
    def __init__(self, n_layers: int, batch: int, n_heads: int, head_dim: int, dtype, bucket_len: int) -> None:
        import mlx.core as mx

        self.bucket_len = int(bucket_len)
        self.capacity = self.bucket_len
        self.offset = 0
        self._pending_t = 0
        self.dtype = dtype
        zeros = mx.zeros((batch, n_heads, self.bucket_len, head_dim), dtype=dtype)
        self.keys = [mx.zeros_like(zeros) for _ in range(n_layers)]
        self.values = [mx.zeros_like(zeros) for _ in range(n_layers)]

    def update(self, layer: int, key, value):
        t = int(key.shape[2])
        if self.offset + t > self.bucket_len:
            raise ValueError(f"bucket length {self.bucket_len} exceeded at offset {self.offset}")
        end = self.offset + t
        self.keys[layer][:, :, self.offset : end, :] = key
        self.values[layer][:, :, self.offset : end, :] = value
        keys, values = self.keys[layer], self.values[layer]
        self._after_write(layer, t)
        return keys, values

    def attention_mask(self, q_len: int = 1):
        import mlx.core as mx

        used = self.offset + self._pending_t
        neg = mx.array(-np.inf, dtype=self.dtype)
        zero = mx.array(0, dtype=self.dtype)
        mask = mx.where(mx.arange(self.bucket_len) < used, zero, neg)
        return mask.reshape(1, 1, 1, self.bucket_len)


def make_kv_cache(
    strategy: str,
    model: Any | None = None,
    *,
    max_len: int | None = None,
    n_layers: int | None = None,
    batch: int = 1,
    n_heads: int | None = None,
    head_dim: int | None = None,
    dtype=None,
    block_size: int = 256,
    bucket_len: int | None = None,
) -> BaseKVCache:
    require_mlx()
    if model is not None:
        n_layers = int(model.config["num_layers"])
        n_heads = int(model.config["n_heads"])
        head_dim = int(model.config["d_model"]) // n_heads
        dtype = model.token_embedding.dtype
        batch = 1
    if n_layers is None or n_heads is None or head_dim is None or dtype is None:
        raise ValueError("cache dimensions are required")
    name = strategy.strip().lower()
    if name == "dynamic":
        return DynamicKVCache(n_layers)
    if name == "block":
        return BlockKVCache(n_layers, batch, n_heads, head_dim, dtype, block_size=block_size)
    if name == "bucket":
        length = bucket_len or max_len or 256
        return BucketKVCache(n_layers, batch, n_heads, head_dim, dtype, bucket_len=length)
    raise ValueError(f"unknown cache strategy {strategy!r}")


class CompiledDecode:
    """Eager-parity compiled one-token decode using a fixed-bucket cache."""

    def __init__(self, model, *, bucket_len: int = 256) -> None:
        require_mlx()
        self.model = model
        self.bucket_len = int(bucket_len)
        self.stats = {"warmup_sec": 0.0, "recompilations": 0, "failures": []}
        self._compiled = None
        self._builds = 0

    def _build(self):
        import mlx.core as mx

        self._builds += 1
        if self._builds > 1:
            self.stats["recompilations"] += 1
        model = self.model
        n_heads = int(model.config["n_heads"])
        bucket_len = self.bucket_len

        def step(input_ids, composer_ids, genre_ids, offset, *kv_flat):
            from dafx26_demo.mlx_backend.attention import merge_heads, scaled_dot_product_attention, split_heads
            import mlx.nn as nn

            hidden = model.token_embedding[input_ids] + model.position_embedding[offset]
            memory = model._memory(composer_ids, genre_ids)
            used = offset + 1
            neg = mx.array(-np.inf, dtype=hidden.dtype)
            zero = mx.array(0, dtype=hidden.dtype)
            mask = mx.where(mx.arange(bucket_len) < used, zero, neg).reshape(1, 1, 1, bucket_len)
            new_kv = []
            for index, layer in enumerate(model.layers):
                keys = kv_flat[2 * index]
                values = kv_flat[2 * index + 1]
                y = layer.self_norm(hidden)
                q, k, v = mx.split(layer.self_attn.qkv(y), 3, axis=-1)
                q = split_heads(q, n_heads)
                k = split_heads(k, n_heads)
                v = split_heads(v, n_heads)
                keys = keys.at[:, :, offset, :].add(k.squeeze(2))
                values = values.at[:, :, offset, :].add(v.squeeze(2))
                attn = scaled_dot_product_attention(q, keys, values, mask=mask)
                hidden = hidden + layer.self_attn.out(merge_heads(attn))
                hidden = hidden + layer.cross_attn(layer.cross_norm(hidden), memory)
                hidden = hidden + layer.ffn_out(nn.gelu(layer.ffn_in(layer.ffn_norm(hidden))))
                new_kv.extend([keys, values])
            logits = layer_final_logits(model, hidden)
            return logits, offset + 1, *new_kv

        return mx.compile(step)

    def _empty_kv(self, batch: int):
        import mlx.core as mx

        n_layers = len(self.model.layers)
        n_heads = int(self.model.config["n_heads"])
        head_dim = int(self.model.config["d_model"]) // n_heads
        dtype = self.model.token_embedding.dtype
        kv = []
        for _ in range(n_layers):
            kv.append(mx.zeros((batch, n_heads, self.bucket_len, head_dim), dtype=dtype))
            kv.append(mx.zeros((batch, n_heads, self.bucket_len, head_dim), dtype=dtype))
        return kv

    def _run(self, input_ids, composer_ids, genre_ids, offset, kv_flat):
        import mlx.core as mx

        if self._compiled is None:
            self._compiled = self._build()
        logits, next_offset, *kv_flat = self._compiled(input_ids, composer_ids, genre_ids, offset, *kv_flat)
        mx.eval(logits, next_offset, *kv_flat)
        return logits, next_offset, kv_flat

    def warmup(self, composer_ids, genre_ids) -> None:
        import mlx.core as mx

        t0 = perf_counter()
        try:
            composer = mx.array(np.asarray(composer_ids, dtype=np.int32))
            genre = mx.array(np.asarray(genre_ids, dtype=np.int32))
            bos = mx.array([[int(self.model.config["bos_token_id"])]], dtype=mx.int32)
            offset = mx.array(0)
            kv = self._empty_kv(int(composer.shape[0]))
            self._run(bos, composer, genre, offset, kv)
            self.stats["warmup_sec"] = perf_counter() - t0
        except Exception as exc:  # pragma: no cover - compile failure path
            self.stats["failures"].append(str(exc))
            self.stats["warmup_sec"] = perf_counter() - t0
            raise

    def generate(
        self,
        composer_ids,
        genre_ids,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
    ):
        import mlx.core as mx

        composer = mx.array(np.asarray(composer_ids, dtype=np.int32))
        genre = mx.array(np.asarray(genre_ids, dtype=np.int32))
        bos_id = int(self.model.config["bos_token_id"])
        generated = np.array([[bos_id]], dtype=np.int64)
        offset = mx.array(0)
        kv = self._empty_kv(int(composer.shape[0]))
        token = mx.array([[bos_id]], dtype=mx.int32)
        for _ in range(max_new_tokens):
            logits, offset, kv = self._run(token, composer, genre, offset, kv)
            next_id = int(greedy_tokens(np.array(logits)[:, -1, :])[0])
            generated = np.concatenate([generated, np.array([[next_id]], dtype=np.int64)], axis=1)
            token = mx.array([[next_id]], dtype=mx.int32)
        return generated


def layer_final_logits(model, hidden):
    return model.final_norm(hidden) @ model.token_embedding.T
