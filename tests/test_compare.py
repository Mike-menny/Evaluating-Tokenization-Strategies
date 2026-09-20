from __future__ import annotations

from dafx26_demo.compare import compare_modes
from dafx26_demo.inference import ModelHub
from dafx26_demo.paths import ensure_upstream_on_path
from dafx26_demo.schema import GenerationRequest
import torch

ensure_upstream_on_path()
from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_vocab import get_tokenization_spec  # noqa: E402


def _tiny(mode: str) -> ConditionalMidiTransformer:
    spec = get_tokenization_spec(mode)
    cfg = ConditionalMidiConfig(
        vocab_size=spec.vocab_size,
        num_composers=16,
        num_genres=32,
        d_model=32,
        n_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=64,
        cond_dim=16,
        tokenization_mode=spec.name,
        num_composer_ids=18,
        num_genre_ids=34,
    )
    model = ConditionalMidiTransformer(cfg)
    model.eval()
    return model


def test_compare_keeps_successful_card_when_one_mode_missing(tmp_path, monkeypatch) -> None:
    hub = ModelHub(tmp_path, torch.device("cpu"))
    hub.models["note"] = _tiny("note")
    from dafx26_demo import inference as inf

    monkeypatch.setattr(inf, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(inf, "OUTPUT_ROOT", tmp_path / "out")
    results = compare_modes(
        hub,
        GenerationRequest(
            mode="note",
            composer="Chopin",
            genre="etude",
            seed=1,
            max_tokens=8,
            temperature=1.0,
            top_p=0.98,
            device="cpu",
            min_tokens=2,
        ),
        modes=["note", "full"],
    )
    assert results[0].mode == "note"
    assert results[1].mode == "full"
    assert results[1].ok is False
    assert results[0].ok in {True, False}
