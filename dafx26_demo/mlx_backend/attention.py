from __future__ import annotations

from typing import Any


def split_heads(x: Any, n_heads: int) -> Any:
    batch, seq_len, dim = x.shape
    head_dim = dim // n_heads
    return x.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)


def merge_heads(x: Any) -> Any:
    batch, _n_heads, seq_len, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, _n_heads * head_dim)


def scaled_dot_product_attention(q: Any, k: Any, v: Any, *, mask: Any | None = None) -> Any:
    import mlx.core as mx

    scale = float(q.shape[-1] ** -0.5)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
