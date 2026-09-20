from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory

from dafx26_demo.artifacts import GeneratedArtifact, index_generated_artifacts
from dafx26_demo.hubs import hub_backend, select_hub
from dafx26_demo.inference import ModelHub, generate as generate_pytorch
from dafx26_demo.live_jobs import LiveJobRegistry
from dafx26_demo.live_routes import register_live_routes
from dafx26_demo.mlx_backend.inference import generate as generate_mlx
from dafx26_demo.pairs import DEFAULT_PAIR, ObservedPairError, get_observed_pairs, validate_pair
from dafx26_demo.paths import (
    MODELS_ROOT,
    MODEL_REVISION,
    REPO_ROOT,
    STATIC_ROOT,
    UPSTREAM_ROOT,
    UPSTREAM_REVISION,
    ensure_upstream_on_path,
)
from dafx26_demo.queueing import GenerationQueue, QueueBusyError
from dafx26_demo.references import load_reference_manifest
from dafx26_demo.schema import request_from_mapping

ensure_upstream_on_path()
from src.tokenization.conditional_vocab import TOKENIZATION_MODE_CHOICES, get_tokenization_spec  # noqa: E402

def mode_features(mode: str) -> dict[str, bool]:
    spec = get_tokenization_spec(mode)
    return {
        "velocity": spec.use_velocity,
        "beat": spec.use_beat,
        "pedal": spec.use_pedal,
    }


def _artifact_root() -> Path:
    return UPSTREAM_ROOT / "artifacts"


def featured_fallback(composer: str, genre: str) -> list[dict[str, Any]]:
    items = index_generated_artifacts(_artifact_root())
    matched = [item for item in items if item.composer == composer and item.genre == genre]
    by_mode: dict[str, GeneratedArtifact] = {}
    for item in matched:
        by_mode.setdefault(item.mode, item)
    return [
        {
            "mode": item.mode,
            "composer": item.composer,
            "genre": item.genre,
            "kind": "generated",
            "midi_url": f"/api/artifact/{item.mode}/{item.composer}/{item.genre}/{item.midi_path.name}",
            "wav_url": (
                f"/api/artifact/{item.mode}/{item.composer}/{item.genre}/{item.wav_path.name}"
                if item.wav_path
                else None
            ),
            "features": mode_features(item.mode),
            "fallback": True,
        }
        for item in by_mode.values()
    ]


def create_app(
    hub: ModelHub | object | None = None,
    *,
    queue: GenerationQueue | None = None,
    jobs: LiveJobRegistry | None = None,
    live_buffer_sec: float = 2.0,
) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="")
    app.extensions["demo_hub"] = hub
    app.extensions["demo_queue"] = queue if queue is not None else GenerationQueue(timeout_sec=180.0)
    app.extensions["demo_jobs"] = jobs if jobs is not None else LiveJobRegistry()
    app.extensions["live_buffer_sec"] = float(live_buffer_sec)
    register_live_routes(app)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.get("/health")
    def health():
        current_hub = app.extensions["demo_hub"]
        device = str(current_hub.device) if current_hub is not None else "uninitialized"
        modes = current_hub.available_modes() if current_hub is not None else []
        missing = []
        if not MODELS_ROOT.is_dir():
            missing.append(f"models not downloaded at {MODELS_ROOT}")
        if not (STATIC_ROOT / "vendor").is_dir():
            missing.append("offline vendor assets missing; run scripts/vendor_static.py")
        return jsonify(
            {
                "ok": True,
                "backend": hub_backend(current_hub) if current_hub is not None else "pytorch",
                "device": device,
                "modes": modes,
                "model_revision": MODEL_REVISION,
                "code_revision": UPSTREAM_REVISION,
                "models_present": MODELS_ROOT.is_dir(),
                "missing": missing,
                "modal": False,
            }
        )

    @app.get("/api/meta")
    def meta():
        current_hub = app.extensions["demo_hub"]
        modes = current_hub.available_modes() if current_hub is not None else list(TOKENIZATION_MODE_CHOICES)
        max_seq_len = getattr(current_hub, "max_seq_len", None)
        max_seq_len_by_mode = (
            {mode: max_seq_len(mode) for mode in modes} if callable(max_seq_len) else {}
        )
        pairs = [{"composer": c, "genre": g} for c, g in get_observed_pairs()]
        composers = sorted({c for c, _g in get_observed_pairs()})
        return jsonify(
            {
                "backend": hub_backend(current_hub) if current_hub is not None else "pytorch",
                "device": str(current_hub.device) if current_hub is not None else "uninitialized",
                "live_buffer_sec": app.extensions["live_buffer_sec"],
                "max_seq_len_by_mode": max_seq_len_by_mode,
                "pairs": pairs,
                "composers": composers,
                "modes": modes,
                "features": {mode: mode_features(mode) for mode in TOKENIZATION_MODE_CHOICES},
                "defaults": {
                    "composer": DEFAULT_PAIR[0],
                    "genre": DEFAULT_PAIR[1],
                    "modes": ["note", "note_velocity_pedal", "full"],
                    "seed": 0,
                    "max_tokens": 256,
                    "temperature": 0.95,
                    "top_p": 0.98,
                },
            }
        )

    @app.get("/api/references")
    def references():
        rows = []
        for item in load_reference_manifest():
            local = REPO_ROOT / item["local_relpath"]
            item = dict(item)
            item["available"] = local.is_file()
            item["midi_url"] = f"/api/reference/{item['id']}" if local.is_file() else None
            rows.append(item)
        return jsonify({"items": rows})

    @app.get("/api/reference/<ref_id>")
    def reference_file(ref_id: str):
        for item in load_reference_manifest():
            if item["id"] != ref_id:
                continue
            path = (REPO_ROOT / item["local_relpath"]).resolve()
            if not path.is_file():
                return jsonify({"error": "reference MIDI is not on this machine"}), 404
            return send_file(path, mimetype="audio/midi")
        return jsonify({"error": "unknown reference"}), 404

    @app.get("/api/fallback")
    def fallback():
        composer = request.args.get("composer", DEFAULT_PAIR[0])
        genre = request.args.get("genre", DEFAULT_PAIR[1])
        try:
            composer, genre = validate_pair(composer, genre)
        except ObservedPairError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"composer": composer, "genre": genre, "items": featured_fallback(composer, genre)})

    @app.get("/api/artifact/<mode>/<composer>/<genre>/<name>")
    def artifact_file(mode: str, composer: str, genre: str, name: str):
        if ".." in name or "/" in name:
            return jsonify({"error": "invalid path"}), 400
        path = (_artifact_root() / f"asap_{mode}" / composer / genre / name).resolve()
        root = _artifact_root().resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            return jsonify({"error": "not found"}), 404
        mime = "audio/wav" if path.suffix == ".wav" else "audio/midi"
        return send_file(path, mimetype=mime)

    @app.post("/api/generate")
    def api_generate():
        current_hub = app.extensions["demo_hub"]
        if current_hub is None:
            return jsonify({"error": "inference hub is not initialized"}), 503
        body = request.get_json(force=True, silent=True) or {}
        try:
            composer, genre = validate_pair(str(body.get("composer") or ""), str(body.get("genre") or ""))
        except ObservedPairError as exc:
            return jsonify({"error": str(exc)}), 400
        gen_request = request_from_mapping(
            {
                **body,
                "composer": composer,
                "genre": genre,
                "device": str(current_hub.device),
            }
        )
        try:
            if hub_backend(current_hub) == "mlx":
                result = app.extensions["demo_queue"].run(
                    lambda: generate_mlx(current_hub, gen_request)
                )
            else:
                result = app.extensions["demo_queue"].run(
                    lambda: generate_pytorch(current_hub, gen_request)
                )
        except QueueBusyError as exc:
            return jsonify({"error": str(exc), "busy": True}), 429
        status = 200 if result.ok else 500
        payload = result.to_dict()
        if result.midi_path:
            payload["midi_url"] = f"/api/file/{Path(result.midi_path).parent.name}/generation.mid"
        payload["features"] = mode_features(result.mode)
        return jsonify(payload), status

    @app.get("/api/file/<job_id>/<name>")
    def generated_file(job_id: str, name: str):
        if ".." in job_id or ".." in name:
            return jsonify({"error": "invalid path"}), 400
        from dafx26_demo.paths import OUTPUT_ROOT

        path = (OUTPUT_ROOT / job_id / name).resolve()
        if not str(path).startswith(str(OUTPUT_ROOT.resolve())) or not path.is_file():
            return jsonify({"error": "not found"}), 404
        return send_file(path, mimetype="audio/midi")

    return app


def _positive_finite_float(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("live buffer must be a finite positive number")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9310)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-on-device", action="store_true")
    parser.add_argument("--live-buffer-sec", type=_positive_finite_float, default=2.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_upstream_on_path()
    hub = select_hub(args.device, keep_on_device=args.keep_on_device)
    app = create_app(hub, live_buffer_sec=args.live_buffer_sec)
    from waitress import serve

    print(f"DAFx demo: http://{args.host}:{args.port}/", flush=True)
    serve(app, host=args.host, port=args.port, threads=4, channel_timeout=300)


if __name__ == "__main__":
    main()
