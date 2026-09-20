from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.numpy import save_file

from dafx26_demo.paths import MODEL_REVISION, UPSTREAM_REVISION, ensure_upstream_on_path

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402


def write_converted_fixture(root: Path, *, dtype: str = "float16") -> None:
    config = ConditionalMidiConfig(
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
    torch.manual_seed(0)
    model = ConditionalMidiTransformer(config)
    state = model.state_dict()
    arrays = {
        name: tensor.detach().cpu().numpy().astype(dtype)
        for name, tensor in state.items()
    }
    destination = root / "note" / "mlx"
    destination.mkdir(parents=True)
    save_file(arrays, str(destination / "mlx_model.safetensors"))
    (destination / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    metadata = {
        "converter_version": "1",
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "mode": "note",
        "dtype": dtype,
        "tensors": {
            name: {"shape": list(array.shape), "dtype": dtype}
            for name, array in arrays.items()
        },
    }
    (destination / "mlx_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_select_hub_constructs_mlx_without_torch_fallback(monkeypatch) -> None:
    from dafx26_demo.hubs import hub_backend, select_hub

    monkeypatch.setattr("dafx26_demo.hubs.require_mlx", lambda: None)
    monkeypatch.setattr("dafx26_demo.hubs.require_converted_weights", lambda _root: None)

    hub = select_hub("mlx")

    assert hub.device == "mlx"
    assert hub_backend(hub) == "mlx"


def test_converted_weights_require_current_fp16_metadata(tmp_path: Path) -> None:
    from dafx26_demo.hubs import require_converted_weights

    write_converted_fixture(tmp_path, dtype="float32")

    with pytest.raises(RuntimeError, match="convert_models_mlx.py"):
        require_converted_weights(tmp_path)


def test_converted_weights_ignore_hidden_cache_directories(tmp_path: Path) -> None:
    from dafx26_demo.hubs import require_converted_weights

    write_converted_fixture(tmp_path)
    (tmp_path / ".cache").mkdir()

    require_converted_weights(tmp_path)
