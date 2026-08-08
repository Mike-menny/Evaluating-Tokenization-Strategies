#!/usr/bin/env python3
"""Smoke tests for the MIDI_tokenization release package (no multi-GPU train required)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports() -> None:
    from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer
    from src.tokenization.conditional_vocab import TOKENIZATION_MODE_CHOICES, get_tokenization_spec
    from src.tokenization import conditional_convert  # noqa: F401
    from src.dataset import conditional_asap  # noqa: F401
    from src.train import train_conditional  # noqa: F401
    from src.inference import inference_conditional  # noqa: F401
    assert "full" in TOKENIZATION_MODE_CHOICES
    assert ConditionalMidiConfig is not None
    assert ConditionalMidiTransformer is not None
    print("[ok] imports")


def test_vocab_modes() -> None:
    from src.tokenization.conditional_vocab import get_tokenization_spec

    expected = {
        "note": 11131,
        "note_pedal": 11264,
        "note_velocity": 11259,
        "note_velocity_beat": 11262,
        "note_velocity_pedal": 11264,
        "full": 11264,
    }
    for mode, vocab in expected.items():
        spec = get_tokenization_spec(mode)
        assert spec.vocab_size == vocab, (mode, spec.vocab_size, vocab)
    print("[ok] vocab sizes")


def test_cli_help() -> None:
    import runpy
    import io
    from contextlib import redirect_stdout, redirect_stderr

    modules = [
        "src.train.train_conditional",
        "src.inference.inference_conditional",
        "src.train.convert_conditional_cache",
        "src.train.merge_pretrain_cache",
    ]
    for mod in modules:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                sys.argv = [mod, "--help"]
                try:
                    runpy.run_module(mod, run_name="__main__")
                except SystemExit as e:
                    assert e.code in (0, None), (mod, e.code, buf_out.getvalue(), buf_err.getvalue())
        finally:
            sys.argv = [sys.argv[0]]
        print(f"[ok] --help {mod}")


def test_tiny_model_roundtrip() -> None:
    import torch
    from src.model.conditional_transformer import ConditionalMidiConfig, ConditionalMidiTransformer
    from src.tokenization.conditional_vocab import get_tokenization_spec

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
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt"
        model.save_pretrained(path, extra_config={"tokenization_mode": spec.name, "small_vocab": True})
        loaded, payload = ConditionalMidiTransformer.from_pretrained(path, map_location="cpu")
        assert payload.get("tokenization_mode") == "note_velocity"
        assert loaded.config.vocab_size == spec.vocab_size

        bos = torch.tensor([[1]], dtype=torch.long)
        composer = torch.tensor([0], dtype=torch.long)
        genre = torch.tensor([0], dtype=torch.long)
        # one forward step via generate with tiny budget
        out = loaded.generate(
            composer_ids=composer,
            genre_ids=genre,
            max_new_tokens=4,
            temperature=1.0,
            top_p=0.98,
        )
        assert out.ndim == 2 and out.shape[0] == 1 and out.shape[1] >= 1
    print("[ok] tiny model save/load/generate")


def test_token_midi_roundtrip() -> None:
    """Build a minimal note_velocity token sequence and convert to MIDI."""
    import mido
    from src.tokenization.conditional_convert import tokens_to_midi
    from src.tokenization.conditional_vocab import (
        BOS_ID,
        DUR_OFFSET,
        EOS_ID,
        NOTE_OFFSET,
        TIME_OFFSET,
        VELOCITY_OFFSET,
    )

    # time=0.10s, dur=0.20s, pitch=60, vel=64
    tokens = [
        BOS_ID,
        TIME_OFFSET + 10,
        DUR_OFFSET + 20,
        NOTE_OFFSET + 60,
        VELOCITY_OFFSET + 64,
        EOS_ID,
    ]
    mid = tokens_to_midi(tokens, mode="note_velocity")
    assert isinstance(mid, mido.MidiFile)
    assert len(mid.tracks) >= 1
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.mid"
        mid.save(str(path))
        assert path.stat().st_size > 0
    print("[ok] tokens_to_midi")


def main() -> None:
    print(f"ROOT={ROOT}")
    test_imports()
    test_vocab_modes()
    test_token_midi_roundtrip()
    test_tiny_model_roundtrip()
    test_cli_help()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
