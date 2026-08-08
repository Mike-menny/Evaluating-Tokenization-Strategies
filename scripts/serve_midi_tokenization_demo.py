#!/usr/bin/env python3
"""Serve all MIDI tokenization models behind a web demo.

Default memory plan (~2GB-class VRAM):
  - Each checkpoint is ~242M params (~0.49GB bf16 / ~0.24GB int8).
  - All modes stay on CPU; only the active mode is moved to GPU for generate.
  - Peak GPU weight memory ≈ one model (~0.5GB) + KV/activations.

Browser playback/visualization uses Magenta SoundFont + Waterfall (notes + keyboard).
No server-side FluidSynth render.

Example:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 scripts/serve_midi_tokenization_demo.py \\
    --host 0.0.0.0 --port 9310 --quantize bf16
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_file, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference_conditional import FastMidiConstraint  # noqa: E402
from src.model.conditional_transformer import ConditionalMidiTransformer  # noqa: E402
from src.tokenization.conditional_convert import tokens_to_midi  # noqa: E402
from src.tokenization.conditional_vocab import (  # noqa: E402
    BOS_ID,
    COMPOSERS,
    EOS_ID,
    GENRES,
    TOKENIZATION_MODE_CHOICES,
    composer_id,
    genre_id,
)

# Prefer repo-local models/; fall back to exports/ if present.
_MODELS_CANDIDATES = (
    PROJECT_ROOT / "models" / "MIDI_tokenization_models",
    PROJECT_ROOT / "exports" / "MIDI_tokenization_models",
)
DEFAULT_MODELS = next((p for p in _MODELS_CANDIDATES if p.is_dir()), _MODELS_CANDIDATES[0])
DEFAULT_STATIC = PROJECT_ROOT / "demo_static"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo_generations"


class Int8Linear(nn.Module):
    """Weight-only int8 Linear: store int8 + scale, matmul in bf16/fp16."""

    def __init__(self, linear: nn.Linear, compute_dtype: torch.dtype):
        super().__init__()
        weight = linear.weight.detach().float().cpu()
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(weight / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", q)
        self.register_buffer("scale", scale)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().to(compute_dtype), requires_grad=False)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(dtype=self.compute_dtype) * self.scale.to(
            device=x.device, dtype=self.compute_dtype
        ).unsqueeze(1)
        return torch.nn.functional.linear(x.to(self.compute_dtype), w, self.bias)


def apply_int8_weight_only(model: nn.Module, compute_dtype: torch.dtype) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(model, name, Int8Linear(child, compute_dtype))
        else:
            apply_int8_weight_only(child, compute_dtype)
    return model


class ModelHub:
    """All modes resident on CPU; at most one active mode on GPU."""

    def __init__(
        self,
        models_root: Path,
        device: torch.device,
        quantize: str = "bf16",
        modes: Optional[list[str]] = None,
        keep_on_gpu: bool = False,
    ):
        self.device = device
        self.quantize = quantize
        self.keep_on_gpu = keep_on_gpu
        self.models: dict[str, ConditionalMidiTransformer] = {}
        self.configs: dict[str, dict] = {}
        self.active_gpu_mode: Optional[str] = None
        self.lock = threading.Lock()
        modes = modes or list(TOKENIZATION_MODE_CHOICES)
        self.dtype = torch.bfloat16 if quantize in ("bf16", "int8") else torch.float16

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(
            f"Loading models from {models_root} "
            f"(weights on CPU, compute on {device}, quantize={quantize}, keep_on_gpu={keep_on_gpu})...",
            flush=True,
        )
        t0 = time.time()
        for mode in modes:
            ckpt = models_root / mode
            if not (ckpt / "pytorch_model.bin").exists():
                print(f"  skip missing {mode}", flush=True)
                continue
            model, config = ConditionalMidiTransformer.from_pretrained(ckpt, map_location="cpu")
            model.eval()
            if quantize == "int8":
                model = apply_int8_weight_only(model, self.dtype)
            model.to(device="cpu", dtype=self.dtype)
            if keep_on_gpu and device.type == "cuda":
                model.to(device=device)
                self.active_gpu_mode = mode
            self.models[mode] = model
            self.configs[mode] = config
            nparams = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  loaded {mode} params={nparams:.1f}M vocab={model.config.vocab_size}", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated(device) / 1e9
            print(
                f"Ready: {len(self.models)} models in {time.time()-t0:.1f}s · "
                f"GPU alloc={alloc:.2f}GB (weights CPU-resident unless --keep-on-gpu)",
                flush=True,
            )
        else:
            print(f"Ready: {len(self.models)} models in {time.time()-t0:.1f}s (CPU)", flush=True)

    def _ensure_on_device(self, mode: str) -> ConditionalMidiTransformer:
        model = self.models[mode]
        if self.device.type != "cuda":
            return model
        if self.keep_on_gpu:
            return model.to(self.device)

        if self.active_gpu_mode == mode:
            return model

        if self.active_gpu_mode is not None and self.active_gpu_mode in self.models:
            prev = self.models[self.active_gpu_mode]
            prev.to("cpu")
            torch.cuda.empty_cache()
            print(f"  offloaded {self.active_gpu_mode} -> CPU", flush=True)

        model.to(self.device)
        self.active_gpu_mode = mode
        alloc = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"  loaded {mode} -> GPU ({alloc:.2f}GB alloc)", flush=True)
        return model

    def memory_stats(self) -> dict:
        out = {
            "active_gpu_mode": self.active_gpu_mode,
            "keep_on_gpu": self.keep_on_gpu,
            "params_m_each": 242.5,
            "note": "each model ~242M params; bf16≈0.49GB, int8≈0.24GB weights",
        }
        if self.device.type == "cuda":
            out.update(
                {
                    "gpu": self.device.index if self.device.index is not None else 0,
                    "allocated_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(self.device) / 1e9, 2),
                }
            )
        return out

    @torch.inference_mode()
    def generate(
        self,
        mode: str,
        composer: str,
        genre: str,
        *,
        max_tokens: int = 2048,
        min_tokens: int = 64,
        temperature: float = 0.95,
        top_p: float = 0.98,
    ) -> tuple[list[int], float]:
        if mode not in self.models:
            raise KeyError(f"Unknown mode {mode!r}. Available: {sorted(self.models)}")
        with self.lock:
            t0 = time.time()
            model = self._ensure_on_device(mode)
            device = next(model.parameters()).device
            composer_ids = torch.tensor([composer_id(composer)], dtype=torch.long, device=device)
            genre_ids = torch.tensor([genre_id(genre)], dtype=torch.long, device=device)
            constraint = FastMidiConstraint(
                batch_size=1,
                vocab_size=model.config.vocab_size,
                mode=mode,
                min_tokens=min_tokens,
                device=device,
            )
            use_autocast = device.type == "cuda"
            amp_dtype = torch.float16 if self.quantize == "fp16" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                generated = model.generate(
                    composer_ids=composer_ids,
                    genre_ids=genre_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logits_processor=constraint,
                )
            tokens = generated[0].detach().cpu().tolist()
            elapsed = time.time() - t0
        return tokens, elapsed


def tokens_clean(tokens: list[int]) -> list[int]:
    out: list[int] = []
    for t in tokens:
        if t == BOS_ID:
            continue
        if t == EOS_ID:
            break
        out.append(t)
    return out


def create_app(hub: ModelHub, static_dir: Path) -> Flask:
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(
            {
                "modes": sorted(hub.models.keys()),
                "composers": list(COMPOSERS),
                "genres": list(GENRES),
                "quantize": hub.quantize,
                "device": str(hub.device),
                "memory": hub.memory_stats(),
                "defaults": {
                    "mode": "note_velocity" if "note_velocity" in hub.models else next(iter(hub.models), "full"),
                    "composer": "Chopin",
                    "genre": "nocturne",
                    "max_tokens": 2048,
                    "temperature": 0.95,
                    "top_p": 0.98,
                },
            }
        )

    @app.post("/api/generate")
    def generate():
        body = request.get_json(force=True, silent=True) or {}
        mode = str(body.get("mode") or "note_velocity").strip()
        composer = str(body.get("composer") or "Chopin").strip()
        genre = str(body.get("genre") or "nocturne").strip()
        max_tokens = max(64, min(int(body.get("max_tokens") or 2048), 8192))
        min_tokens = int(body.get("min_tokens") or 64)
        temperature = float(body.get("temperature") or 0.95)
        top_p = float(body.get("top_p") or 0.98)

        if mode not in hub.models:
            return jsonify({"error": f"mode not loaded: {mode}"}), 400

        try:
            tokens, elapsed = hub.generate(
                mode,
                composer,
                genre,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        clean = tokens_clean(tokens)
        if not clean:
            return jsonify({"error": "empty generation"}), 500

        job_id = uuid.uuid4().hex[:12]
        out_dir = OUTPUT_ROOT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = out_dir / "generation.mid"
        try:
            tokens_to_midi(clean, mode=mode).save(str(midi_path))
        except Exception as e:
            return jsonify({"error": f"MIDI convert failed: {e}"}), 500

        return jsonify(
            {
                "job_id": job_id,
                "mode": mode,
                "composer": composer,
                "genre": genre,
                "n_tokens": len(clean),
                "elapsed_sec": round(elapsed, 2),
                "midi_url": f"/api/file/{job_id}/generation.mid",
                "memory": hub.memory_stats(),
            }
        )

    @app.get("/api/file/<job_id>/<name>")
    def get_file(job_id: str, name: str):
        if ".." in job_id or ".." in name or "/" in job_id:
            return jsonify({"error": "invalid path"}), 400
        path = (OUTPUT_ROOT / job_id / name).resolve()
        if not str(path).startswith(str(OUTPUT_ROOT.resolve())) or not path.is_file():
            return jsonify({"error": "not found"}), 404
        return send_file(path, mimetype="audio/midi", as_attachment=False, download_name=name)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "modes": sorted(hub.models.keys()), "memory": hub.memory_stats()})

    return app


def pick_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        if requested == "cuda":
            best, best_free = 0, -1
            for i in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(i)
                if free > best_free:
                    best, best_free = i, free
            return torch.device(f"cuda:{best}")
        return torch.device(requested)
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI tokenization multi-model web demo")
    parser.add_argument("--models", type=str, default=str(DEFAULT_MODELS))
    parser.add_argument("--static", type=str, default=str(DEFAULT_STATIC))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9310)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--quantize",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "int8"),
        help="Weight dtype on CPU / compute dtype on GPU",
    )
    parser.add_argument(
        "--keep-on-gpu",
        action="store_true",
        help="Keep ALL models on GPU (~3GB bf16). Default: only active mode on GPU (~0.5GB).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated subset of modes (default: all present on disk)",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()] or None
    hub = ModelHub(
        models_root=Path(args.models),
        device=device,
        quantize=args.quantize,
        modes=modes,
        keep_on_gpu=args.keep_on_gpu,
    )
    if not hub.models:
        raise SystemExit(f"No models found under {args.models}")

    static_dir = Path(args.static)
    if not (static_dir / "index.html").exists():
        raise SystemExit(f"Missing UI at {static_dir / 'index.html'}")

    app = create_app(hub, static_dir)
    print(f"\nDemo UI:  http://127.0.0.1:{args.port}/", flush=True)
    print(f"LAN:      http://10.21.48.25:{args.port}/", flush=True)
    print(
        "Access: same LAN/VPN can open the LAN URL. True public internet needs "
        "campus firewall/NAT to allow this port on 147.8.138.66.",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
