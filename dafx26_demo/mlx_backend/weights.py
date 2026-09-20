from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from dafx26_demo.mlx_backend import CONVERTER_VERSION
from dafx26_demo.paths import MODEL_REVISION, UPSTREAM_REVISION

WEIGHTS_NAME = "mlx_model.safetensors"
METADATA_NAME = "mlx_metadata.json"
CONFIG_NAME = "config.json"


class ConversionError(ValueError):
    """Raised when a checkpoint cannot be converted or a converted cache is invalid."""


def _dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ConversionError(f"unsupported conversion dtype {name!r}")


def expected_tensors(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    vocab = int(config["vocab_size"])
    d_model = int(config["d_model"])
    n_heads = int(config["n_heads"])
    num_layers = int(config["num_layers"])
    ffn_dim = int(config["ffn_dim"])
    max_seq_len = int(config["max_seq_len"])
    cond_dim = int(config["cond_dim"])
    num_composer_ids = int(config.get("num_composer_ids") or config["num_composers"])
    num_genre_ids = int(config.get("num_genre_ids") or config["num_genres"])
    cond_vocab = num_composer_ids + num_genre_ids
    if d_model % n_heads != 0:
        raise ConversionError("d_model must be divisible by n_heads")

    tensors: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (vocab, d_model),
        "position_embedding.weight": (max_seq_len, d_model),
        "condition_embedding.weight": (cond_vocab, cond_dim),
        "condition_projection.0.weight": (d_model, cond_dim),
        "condition_projection.0.bias": (d_model,),
        "condition_projection.2.weight": (d_model, d_model),
        "condition_projection.2.bias": (d_model,),
        "final_norm.weight": (d_model,),
        "final_norm.bias": (d_model,),
        "lm_head.weight": (vocab, d_model),
    }
    for index in range(num_layers):
        prefix = f"layers.{index}"
        tensors.update(
            {
                f"{prefix}.self_norm.weight": (d_model,),
                f"{prefix}.self_norm.bias": (d_model,),
                f"{prefix}.cross_norm.weight": (d_model,),
                f"{prefix}.cross_norm.bias": (d_model,),
                f"{prefix}.ffn_norm.weight": (d_model,),
                f"{prefix}.ffn_norm.bias": (d_model,),
                f"{prefix}.self_attn.qkv.weight": (3 * d_model, d_model),
                f"{prefix}.self_attn.qkv.bias": (3 * d_model,),
                f"{prefix}.self_attn.out.weight": (d_model, d_model),
                f"{prefix}.self_attn.out.bias": (d_model,),
                f"{prefix}.cross_attn.q.weight": (d_model, d_model),
                f"{prefix}.cross_attn.q.bias": (d_model,),
                f"{prefix}.cross_attn.k.weight": (d_model, d_model),
                f"{prefix}.cross_attn.k.bias": (d_model,),
                f"{prefix}.cross_attn.v.weight": (d_model, d_model),
                f"{prefix}.cross_attn.v.bias": (d_model,),
                f"{prefix}.cross_attn.out.weight": (d_model, d_model),
                f"{prefix}.cross_attn.out.bias": (d_model,),
                f"{prefix}.ffn.0.weight": (ffn_dim, d_model),
                f"{prefix}.ffn.0.bias": (ffn_dim,),
                f"{prefix}.ffn.3.weight": (d_model, ffn_dim),
                f"{prefix}.ffn.3.bias": (d_model,),
            }
        )
    return tensors


def load_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / CONFIG_NAME
    if not path.is_file():
        raise ConversionError(f"missing config at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_state(state: dict[str, torch.Tensor], expected: dict[str, tuple[int, ...]]) -> None:
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing:
        raise ConversionError(f"missing tensors: {missing}")
    if unexpected:
        raise ConversionError(f"unexpected tensors: {unexpected}")
    for name, shape in expected.items():
        actual = tuple(state[name].shape)
        if actual != shape:
            raise ConversionError(f"shape mismatch for {name}: expected {shape}, got {actual}")


def _tensor_records(arrays: dict[str, np.ndarray], dtype: str) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(array.shape), "dtype": dtype}
        for name, array in arrays.items()
    }


def build_metadata(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "converter_version": CONVERTER_VERSION,
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "mode": mode,
        "dtype": dtype,
        "tensors": _tensor_records(arrays, dtype),
    }


def validate_metadata(meta: dict[str, Any], *, mode: str | None = None) -> None:
    if meta.get("converter_version") != CONVERTER_VERSION:
        raise ConversionError(f"stale converter version {meta.get('converter_version')!r}")
    if meta.get("model_revision") != MODEL_REVISION:
        raise ConversionError(f"stale model revision {meta.get('model_revision')!r}")
    if meta.get("upstream_revision") != UPSTREAM_REVISION:
        raise ConversionError(f"stale upstream revision {meta.get('upstream_revision')!r}")
    if mode is not None and meta.get("mode") != mode:
        raise ConversionError(f"stale mode {meta.get('mode')!r}, expected {mode!r}")


def convert_state_dict(
    state: dict[str, torch.Tensor],
    *,
    dtype: str = "float16",
) -> dict[str, np.ndarray]:
    np_dtype = _dtype(dtype)
    converted: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False)
        converted[name] = np.ascontiguousarray(array)
    return converted


def convert_checkpoint(
    src_dir: str | Path,
    dest_dir: str | Path,
    *,
    mode: str,
    dtype: str = "float16",
    force: bool = False,
) -> Path:
    src = Path(src_dir)
    dest = Path(dest_dir)
    weights_path = src / "pytorch_model.bin"
    if not weights_path.is_file():
        raise ConversionError(f"missing PyTorch weights at {weights_path}")
    config = load_config(src)
    configured_mode = str(config.get("tokenization_mode", mode))
    if configured_mode != mode:
        raise ConversionError(f"checkpoint mode {configured_mode!r} does not match {mode!r}")
    expected = expected_tensors(config)
    if not force:
        existing = dest / METADATA_NAME
        existing_weights = dest / WEIGHTS_NAME
        if existing.is_file() and existing_weights.is_file():
            try:
                meta = json.loads(existing.read_text(encoding="utf-8"))
                validate_metadata(meta, mode=mode)
                if meta.get("dtype") == dtype and set(meta.get("tensors", {})) == set(expected):
                    return dest
            except (ConversionError, json.JSONDecodeError, OSError):
                pass

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    _validate_state(state, expected)
    arrays = convert_state_dict(state, dtype=dtype)
    metadata = build_metadata(arrays, mode=mode, dtype=dtype)
    dest.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(dest / WEIGHTS_NAME))
    (dest / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (dest / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return dest


def load_converted(
    dest_dir: str | Path,
    *,
    mode: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dest = Path(dest_dir)
    meta_path = dest / METADATA_NAME
    weights_path = dest / WEIGHTS_NAME
    if not meta_path.is_file() or not weights_path.is_file():
        raise ConversionError(f"converted MLX weights missing at {dest}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_metadata(meta, mode=mode)
    arrays = load_file(str(weights_path))
    recorded = meta.get("tensors") or {}
    if set(arrays) != set(recorded):
        raise ConversionError("converted tensor keys do not match metadata")
    return arrays, meta
