from __future__ import annotations

from pathlib import Path
import time
import uuid

import numpy as np

from dafx26_demo.beats import extract_beat_markers
from dafx26_demo.cache import load_cached, store_cached
from dafx26_demo.midi_convert import clean_tokens, tokens_to_midi_bytes
from dafx26_demo.mlx_backend.model import generate_tokens, load_converted_model
from dafx26_demo.mlx_backend.sampling import MidiConstraint
from dafx26_demo.pairs import validate_pair
from dafx26_demo.paths import (
    CACHE_ROOT,
    MODELS_ROOT,
    MODEL_REVISION,
    OUTPUT_ROOT,
    UPSTREAM_REVISION,
    ensure_upstream_on_path,
)
from dafx26_demo.retention import prune_outputs
from dafx26_demo.schema import GenerationRequest, GenerationResult, cache_key

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import composer_id, genre_id  # noqa: E402


def _peak_memory_bytes() -> int | None:
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - optional extra
        return None
    return int(mx.get_peak_memory())


class MlxModelHub:
    backend = "mlx"
    device = "mlx"

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self.models = {}
        self.max_seq_len_by_mode = self._load_max_seq_len_by_mode()

    def _load_max_seq_len_by_mode(self) -> dict[str, int]:
        max_seq_len_by_mode = {}
        if not self.models_root.is_dir():
            return max_seq_len_by_mode
        for mode_dir in self.models_root.iterdir():
            config_path = mode_dir / "mlx" / "config.json"
            if not config_path.is_file():
                continue
            import json

            config = json.loads(config_path.read_text(encoding="utf-8"))
            max_seq_len_by_mode[mode_dir.name] = int(config["max_seq_len"])
        return max_seq_len_by_mode

    def available_modes(self) -> list[str]:
        return sorted(set(self.max_seq_len_by_mode) | set(self.models))

    def max_seq_len(self, mode: str) -> int:
        return self.max_seq_len_by_mode[mode]

    def load(self, mode: str):
        t0 = time.time()
        if mode in self.models:
            return self.models[mode], time.time() - t0
        dest = self.models_root / mode / "mlx"
        model = load_converted_model(dest, dtype="float16")
        self.models[mode] = model
        return model, time.time() - t0


def generate(hub: MlxModelHub, request: GenerationRequest, *, use_cache: bool = True) -> GenerationResult:
    composer, genre = validate_pair(request.composer, request.genre)
    request = GenerationRequest(
        **{
            **request.to_dict(),
            "composer": composer,
            "genre": genre,
            "device": "mlx",
        }
    )
    key = cache_key(request, code_revision=UPSTREAM_REVISION, model_revision=MODEL_REVISION)
    cached = load_cached(CACHE_ROOT, key) if use_cache else None
    if cached is not None:
        return cached

    load_sec = 0.0
    t0 = time.time()
    try:
        import mlx.core as mx

        np.random.seed(request.seed)
        mx.random.seed(request.seed)
        mx.reset_peak_memory()
        model, load_sec = hub.load(request.mode)
        composer_ids = np.array([composer_id(composer)], dtype=np.int32)
        genre_ids = np.array([genre_id(genre)], dtype=np.int32)
        constraint = MidiConstraint(
            batch_size=1,
            vocab_size=int(model.config["vocab_size"]),
            mode=request.mode,
            min_tokens=request.min_tokens,
        )
        t0 = time.time()
        generated = generate_tokens(
            model,
            composer_ids,
            genre_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            constraint=constraint,
            seed=request.seed,
        )
        tokens = clean_tokens(generated[0].tolist())
        elapsed = time.time() - t0
        if not tokens:
            raise RuntimeError("empty generation")
        midi_bytes = tokens_to_midi_bytes(tokens, mode=request.mode)
        job_id = uuid.uuid4().hex[:12]
        out_dir = OUTPUT_ROOT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = out_dir / "generation.mid"
        midi_path.write_bytes(midi_bytes)
        prune_outputs(OUTPUT_ROOT, keep=40)
        import io
        import mido

        duration = float(mido.MidiFile(file=io.BytesIO(midi_bytes)).length)
        result = GenerationResult(
            request_id=job_id,
            mode=request.mode,
            composer=composer,
            genre=genre,
            ok=True,
            error=None,
            midi_path=str(midi_path),
            elapsed_sec=round(elapsed, 3),
            n_tokens=len(tokens),
            load_sec=round(load_sec, 3),
            peak_memory_bytes=_peak_memory_bytes(),
            musical_duration_sec=round(duration, 3),
            beat_markers=extract_beat_markers(tokens, mode=request.mode),
            code_revision=UPSTREAM_REVISION,
            model_revision=MODEL_REVISION,
        )
        if use_cache:
            store_cached(CACHE_ROOT, key, result)
        return result
    except Exception as exc:
        return GenerationResult(
            request_id=uuid.uuid4().hex[:12],
            mode=request.mode,
            composer=composer,
            genre=genre,
            ok=False,
            error=str(exc),
            midi_path=None,
            elapsed_sec=round(time.time() - t0, 3),
            n_tokens=0,
            load_sec=round(load_sec, 3),
            code_revision=UPSTREAM_REVISION,
            model_revision=MODEL_REVISION,
        )


def default_hub() -> MlxModelHub:
    return MlxModelHub(MODELS_ROOT)
