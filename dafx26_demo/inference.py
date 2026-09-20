from __future__ import annotations

from pathlib import Path
import time
import uuid

import torch

from dafx26_demo.beats import extract_beat_markers
from dafx26_demo.devices import dtype_for_device, pick_device, seed_everything
from dafx26_demo.midi_convert import clean_tokens, tokens_to_midi_bytes
from dafx26_demo.pairs import validate_pair
from dafx26_demo.paths import (
    CACHE_ROOT,
    MODELS_ROOT,
    MODEL_REVISION,
    OUTPUT_ROOT,
    UPSTREAM_REVISION,
    ensure_upstream_on_path,
)
from dafx26_demo.schema import GenerationRequest, GenerationResult, cache_key
from dafx26_demo.cache import load_cached, store_cached
from dafx26_demo.retention import prune_outputs

ensure_upstream_on_path()

from src.inference.inference_conditional import FastMidiConstraint  # noqa: E402
from src.model.conditional_transformer import ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_vocab import composer_id, genre_id  # noqa: E402


class ModelHub:
    backend = "pytorch"

    def __init__(
        self,
        models_root: Path,
        device: torch.device,
        *,
        keep_on_device: bool = False,
        quantize: str = "auto",
    ) -> None:
        self.models_root = models_root
        self.device = device
        self.keep_on_device = keep_on_device
        self.quantize = quantize
        self.dtype = dtype_for_device(device, quantize=quantize)
        self.models: dict[str, ConditionalMidiTransformer] = {}
        self.active_mode: str | None = None

    def available_modes(self) -> list[str]:
        if not self.models_root.is_dir():
            return sorted(self.models)
        modes = []
        for child in self.models_root.iterdir():
            if (child / "pytorch_model.bin").is_file():
                modes.append(child.name)
        return sorted(set(modes) | set(self.models))

    def load(self, mode: str) -> tuple[ConditionalMidiTransformer, float]:
        t0 = time.time()
        if mode in self.models:
            model = self._ensure_on_device(mode)
            return model, time.time() - t0
        path = self.models_root / mode
        model, _payload = ConditionalMidiTransformer.from_pretrained(path, map_location="cpu")
        model.eval()
        if self.dtype in {torch.float16, torch.bfloat16}:
            model = model.to(dtype=self.dtype)
        self.models[mode] = model
        model = self._ensure_on_device(mode)
        return model, time.time() - t0

    def _ensure_on_device(self, mode: str) -> ConditionalMidiTransformer:
        model = self.models[mode]
        if self.device.type == "cpu":
            return model
        if self.keep_on_device:
            return model.to(self.device)
        if self.active_mode == mode:
            return model
        if self.active_mode is not None and self.active_mode in self.models:
            self.models[self.active_mode].to("cpu")
        model.to(self.device)
        self.active_mode = mode
        return model


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def generate(hub: ModelHub, request: GenerationRequest, *, use_cache: bool = True) -> GenerationResult:
    composer, genre = validate_pair(request.composer, request.genre)
    request = GenerationRequest(
        **{
            **request.to_dict(),
            "composer": composer,
            "genre": genre,
            "device": str(hub.device),
        }
    )
    key = cache_key(
        request,
        code_revision=UPSTREAM_REVISION,
        model_revision=MODEL_REVISION,
    )
    cached = load_cached(CACHE_ROOT, key) if use_cache else None
    if cached is not None:
        return cached

    seed_everything(request.seed, hub.device)
    t0 = time.time()
    load_sec = 0.0
    try:
        model, load_sec = hub.load(request.mode)
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        composer_ids = torch.tensor([composer_id(composer)], dtype=torch.long, device=device)
        genre_ids = torch.tensor([genre_id(genre)], dtype=torch.long, device=device)
        constraint = FastMidiConstraint(
            batch_size=1,
            vocab_size=model.config.vocab_size,
            mode=request.mode,
            min_tokens=request.min_tokens,
            device=device,
        )
        t0 = time.time()
        use_autocast = device.type in {"cuda", "mps"}
        amp_dtype = hub.dtype if hub.dtype in {torch.float16, torch.bfloat16} else torch.float16
        with torch.inference_mode():
            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    generated = model.generate(
                        composer_ids=composer_ids,
                        genre_ids=genre_ids,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        logits_processor=constraint,
                    )
            else:
                generated = model.generate(
                    composer_ids=composer_ids,
                    genre_ids=genre_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    logits_processor=constraint,
                )
        tokens = clean_tokens(generated[0].detach().cpu().tolist())
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
            peak_memory_bytes=_peak_memory_bytes(device),
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


def default_hub(device_name: str = "auto", *, keep_on_device: bool = False) -> ModelHub:
    device = pick_device(device_name)
    return ModelHub(MODELS_ROOT, device, keep_on_device=keep_on_device)
