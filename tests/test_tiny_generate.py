from __future__ import annotations

from pathlib import Path

import torch

from dafx26_demo.inference import ModelHub, generate
from dafx26_demo.paths import ensure_upstream_on_path
from dafx26_demo.schema import GenerationRequest

ensure_upstream_on_path()

from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_vocab import get_tokenization_spec  # noqa: E402


def test_tiny_cpu_generate_writes_midi(tmp_path: Path, monkeypatch) -> None:
    spec = get_tokenization_spec("note_velocity")
    cfg = ConditionalMidiConfig(
        vocab_size=spec.vocab_size,
        num_composers=16,
        num_genres=32,
        d_model=64,
        n_heads=4,
        num_layers=2,
        ffn_dim=128,
        max_seq_len=128,
        cond_dim=32,
        tokenization_mode=spec.name,
        num_composer_ids=18,
        num_genre_ids=34,
    )
    model = ConditionalMidiTransformer(cfg)
    model.eval()
    hub = ModelHub(tmp_path, torch.device("cpu"))
    hub.models["note_velocity"] = model

    from dafx26_demo import inference as inf

    monkeypatch.setattr(inf, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(inf, "OUTPUT_ROOT", tmp_path / "out")

    result = generate(
        hub,
        GenerationRequest(
            mode="note_velocity",
            composer="Chopin",
            genre="etude",
            seed=0,
            max_tokens=16,
            temperature=1.0,
            top_p=0.98,
            device="cpu",
            min_tokens=4,
        ),
    )
    assert result.ok, result.error
    assert result.n_tokens >= 1
    assert result.midi_path and Path(result.midi_path).is_file()
    assert Path(result.midi_path).read_bytes().startswith(b"MThd")
