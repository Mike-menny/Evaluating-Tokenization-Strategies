from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dafx26_demo.paths import MODELS_ROOT, MODEL_REVISION, UPSTREAM_REVISION, ensure_upstream_on_path

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402

TINY_CONFIG = ConditionalMidiConfig(
    vocab_size=32,
    num_composers=4,
    num_genres=4,
    d_model=16,
    n_heads=4,
    num_layers=1,
    ffn_dim=32,
    max_seq_len=32,
    cond_dim=8,
    tokenization_mode="note",
    num_composer_ids=6,
    num_genre_ids=6,
)

REAL_MODES = [
    "note",
    "note_pedal",
    "note_velocity",
    "note_velocity_beat",
    "note_velocity_pedal",
    "full",
]


def _write_tiny_checkpoint(root: Path) -> Path:
    torch.manual_seed(0)
    model = ConditionalMidiTransformer(TINY_CONFIG)
    model.eval()
    out = root / "note"
    model.save_pretrained(out)
    return out


def test_tiny_checkpoint_converts_with_provenance(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import convert_checkpoint, load_converted

    src = _write_tiny_checkpoint(tmp_path / "src")
    dest = convert_checkpoint(src, tmp_path / "mlx", mode="note", dtype="float16")
    arrays, meta = load_converted(dest)

    assert meta["converter_version"] == "1"
    assert meta["model_revision"] == MODEL_REVISION
    assert meta["upstream_revision"] == UPSTREAM_REVISION
    assert meta["mode"] == "note"
    assert meta["dtype"] == "float16"
    qkv = arrays["layers.0.self_attn.qkv.weight"]
    assert tuple(qkv.shape) == (48, 16)
    assert str(qkv.dtype).replace("torch.", "") in {"float16", "float16"}
    assert set(arrays) == set(meta["tensors"])
    for name, tensor in arrays.items():
        assert list(tensor.shape) == meta["tensors"][name]["shape"]


def test_convert_rejects_missing_tensor(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint

    src = _write_tiny_checkpoint(tmp_path / "src")
    state_path = src / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    del state["layers.0.self_attn.qkv.weight"]
    torch.save(state, state_path)
    with pytest.raises(ConversionError, match="missing"):
        convert_checkpoint(src, tmp_path / "mlx", mode="note")


def test_convert_rejects_unexpected_tensor(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint

    src = _write_tiny_checkpoint(tmp_path / "src")
    state_path = src / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["unexpected.weight"] = torch.zeros(2, 2)
    torch.save(state, state_path)
    with pytest.raises(ConversionError, match="unexpected"):
        convert_checkpoint(src, tmp_path / "mlx", mode="note")


def test_convert_rejects_shape_mismatch(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint

    src = _write_tiny_checkpoint(tmp_path / "src")
    state_path = src / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["layers.0.self_attn.qkv.weight"] = torch.zeros(3, 3)
    torch.save(state, state_path)
    with pytest.raises(ConversionError, match="shape"):
        convert_checkpoint(src, tmp_path / "mlx", mode="note")


def test_load_converted_rejects_stale_provenance(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint, load_converted

    src = _write_tiny_checkpoint(tmp_path / "src")
    dest = convert_checkpoint(src, tmp_path / "mlx", mode="note")
    meta_path = dest / "mlx_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["model_revision"] = "stale"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="stale|revision"):
        load_converted(dest)


def test_linear_weights_keep_out_in_layout(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import convert_checkpoint, load_converted

    src = _write_tiny_checkpoint(tmp_path / "src")
    state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    dest = convert_checkpoint(src, tmp_path / "mlx", mode="note", dtype="float32")
    arrays, _meta = load_converted(dest)
    original = state["condition_projection.0.weight"]
    converted = arrays["condition_projection.0.weight"]
    assert tuple(converted.shape) == tuple(original.shape)
    assert converted.shape[0] == TINY_CONFIG.d_model
    assert converted.shape[1] == TINY_CONFIG.cond_dim


def test_convert_rejects_mode_and_dtype_errors(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint, expected_tensors

    src = _write_tiny_checkpoint(tmp_path / "src")
    with pytest.raises(ConversionError, match="does not match"):
        convert_checkpoint(src, tmp_path / "mlx", mode="full")
    with pytest.raises(ConversionError, match="unsupported conversion dtype"):
        convert_checkpoint(src, tmp_path / "mlx", mode="note", dtype="int8")
    with pytest.raises(ConversionError, match="divisible"):
        expected_tensors({"vocab_size": 8, "d_model": 10, "n_heads": 3, "num_layers": 1, "ffn_dim": 8, "max_seq_len": 8, "cond_dim": 4, "num_composers": 2, "num_genres": 2})
    with pytest.raises(ConversionError, match="missing PyTorch weights"):
        convert_checkpoint(tmp_path / "empty", tmp_path / "mlx", mode="note")


def test_convert_reuses_valid_destination(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import convert_checkpoint

    src = _write_tiny_checkpoint(tmp_path / "src")
    dest = tmp_path / "mlx"
    first = convert_checkpoint(src, dest, mode="note")
    marker = dest / "mlx_model.safetensors"
    mtime = marker.stat().st_mtime_ns
    second = convert_checkpoint(src, dest, mode="note")
    assert first == second == dest
    assert marker.stat().st_mtime_ns == mtime


def test_load_converted_rejects_corrupt_cache(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import ConversionError, convert_checkpoint, load_converted

    src = _write_tiny_checkpoint(tmp_path / "src")
    dest = convert_checkpoint(src, tmp_path / "mlx", mode="note")
    meta_path = dest / "mlx_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["converter_version"] = "0"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="stale converter"):
        load_converted(dest)
    meta["converter_version"] = "1"
    meta["upstream_revision"] = "stale"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="stale upstream"):
        load_converted(dest)
    meta["upstream_revision"] = UPSTREAM_REVISION
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="stale mode"):
        load_converted(dest, mode="full")
    (dest / "mlx_model.safetensors").unlink()
    with pytest.raises(ConversionError, match="missing"):
        load_converted(dest)
    with pytest.raises(ConversionError, match="missing config"):
        from dafx26_demo.mlx_backend.weights import load_config

        load_config(tmp_path / "nope")


def test_invalid_existing_metadata_is_reconverted(tmp_path: Path) -> None:
    from dafx26_demo.mlx_backend.weights import convert_checkpoint, load_converted

    src = _write_tiny_checkpoint(tmp_path / "src")
    dest = convert_checkpoint(src, tmp_path / "mlx", mode="note")
    (dest / "mlx_metadata.json").write_text("{not-json", encoding="utf-8")
    convert_checkpoint(src, dest, mode="note")
    _arrays, meta = load_converted(dest)
    assert meta["converter_version"] == "1"


@pytest.mark.parametrize("mode", REAL_MODES)
def test_real_mode_checkpoint_converts(mode: str) -> None:
    from dafx26_demo.mlx_backend.weights import convert_checkpoint, load_converted

    src = MODELS_ROOT / mode
    if not (src / "pytorch_model.bin").is_file():
        pytest.skip(f"pinned checkpoint missing for {mode}")
    dest = convert_checkpoint(src, src / "mlx", mode=mode, dtype="float16")
    arrays, meta = load_converted(dest, mode=mode)
    assert meta["mode"] == mode
    assert meta["dtype"] == "float16"
    assert len(arrays) == 538
    assert arrays["token_embedding.weight"].shape[1] == 768
    assert list(arrays["lm_head.weight"].shape) == list(arrays["token_embedding.weight"].shape)
