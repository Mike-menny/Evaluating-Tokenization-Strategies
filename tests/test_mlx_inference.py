from __future__ import annotations

from pathlib import Path

import pytest

from dafx26_demo.mlx_backend import mlx_available
from dafx26_demo.schema import GenerationRequest

requires_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX is not installed")


@requires_mlx
def test_mlx_generate_returns_result_contract(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import asdict

    import torch

    from dafx26_demo.mlx_backend import inference as mlx_inf
    from dafx26_demo.mlx_backend.model import model_from_numpy
    from dafx26_demo.paths import ensure_upstream_on_path

    ensure_upstream_on_path()
    from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer
    from src.tokenization.conditional_vocab import get_tokenization_spec

    spec = get_tokenization_spec("note_velocity")
    cfg = ConditionalMidiConfig(
        vocab_size=spec.vocab_size,
        num_composers=16,
        num_genres=32,
        d_model=32,
        n_heads=4,
        num_layers=2,
        ffn_dim=64,
        max_seq_len=64,
        cond_dim=16,
        tokenization_mode=spec.name,
        num_composer_ids=18,
        num_genre_ids=34,
    )
    torch.manual_seed(0)
    pt = ConditionalMidiTransformer(cfg)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt.state_dict().items()}
    model = model_from_numpy(asdict(cfg), arrays, dtype="float32")
    hub = mlx_inf.MlxModelHub(tmp_path)
    hub.models["note_velocity"] = model
    monkeypatch.setattr(mlx_inf, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(mlx_inf, "OUTPUT_ROOT", tmp_path / "out")
    result = mlx_inf.generate(
        hub,
        GenerationRequest(
            mode="note_velocity",
            composer="Chopin",
            genre="etude",
            seed=0,
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            device="mlx",
            min_tokens=4,
        ),
        use_cache=False,
    )
    assert result.ok, result.error
    assert result.n_tokens >= 1
    assert result.midi_path and Path(result.midi_path).is_file()
    assert Path(result.midi_path).read_bytes().startswith(b"MThd")
    assert result.model_revision
    assert result.code_revision


@requires_mlx
def test_mlx_generate_cache_and_error_paths(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import asdict

    import torch

    from dafx26_demo.mlx_backend import inference as mlx_inf
    from dafx26_demo.mlx_backend.model import model_from_numpy
    from dafx26_demo.paths import MODELS_ROOT, ensure_upstream_on_path

    ensure_upstream_on_path()
    from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer
    from src.tokenization.conditional_vocab import get_tokenization_spec

    spec = get_tokenization_spec("note_velocity")
    cfg = ConditionalMidiConfig(
        vocab_size=spec.vocab_size,
        num_composers=16,
        num_genres=32,
        d_model=32,
        n_heads=4,
        num_layers=2,
        ffn_dim=64,
        max_seq_len=64,
        cond_dim=16,
        tokenization_mode=spec.name,
        num_composer_ids=18,
        num_genre_ids=34,
    )
    torch.manual_seed(0)
    pt = ConditionalMidiTransformer(cfg)
    arrays = {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in pt.state_dict().items()}
    model = model_from_numpy(asdict(cfg), arrays, dtype="float32")
    hub = mlx_inf.MlxModelHub(tmp_path)
    hub.models["note_velocity"] = model
    monkeypatch.setattr(mlx_inf, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(mlx_inf, "OUTPUT_ROOT", tmp_path / "out")
    request = GenerationRequest(
        mode="note_velocity",
        composer="Chopin",
        genre="etude",
        seed=1,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        device="mlx",
        min_tokens=4,
    )
    first = mlx_inf.generate(hub, request, use_cache=True)
    second = mlx_inf.generate(hub, request, use_cache=True)
    assert first.ok and second.ok
    assert second.cached is True
    loaded, elapsed = hub.load("note_velocity")
    assert loaded is model
    assert elapsed >= 0

    class BoomHub:
        def load(self, mode: str):
            raise RuntimeError("boom")

    failed = mlx_inf.generate(BoomHub(), request, use_cache=False)
    assert failed.ok is False
    assert "boom" in (failed.error or "")
    hub_real = mlx_inf.default_hub()
    assert hub_real.models_root == MODELS_ROOT
