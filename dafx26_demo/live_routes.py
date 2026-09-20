from __future__ import annotations

import math
from threading import Lock
from typing import Any

from flask import Flask, Response, jsonify, request

from dafx26_demo.hubs import hub_backend
from dafx26_demo.live_jobs import LiveJobRegistry
from dafx26_demo.mlx_backend.live import format_sse, iter_live_sse
from dafx26_demo.pairs import ObservedPairError, validate_pair
from dafx26_demo.paths import ensure_upstream_on_path
from dafx26_demo.queueing import QueueBusyError

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import TOKENIZATION_MODE_CHOICES  # noqa: E402


class LiveRequestError(ValueError):
    pass


def parse_live_body(body: dict[str, Any]) -> dict[str, Any]:
    if "max_tokens" in body:
        raise LiveRequestError("live requests must not include max_tokens")
    mode = str(body.get("mode") or "")
    if mode not in TOKENIZATION_MODE_CHOICES:
        raise LiveRequestError(f"unknown mode {mode!r}")
    composer = str(body.get("composer") or "")
    genre = str(body.get("genre") or "")
    if not composer or not genre:
        raise LiveRequestError("composer and genre are required")
    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise LiveRequestError("seed must be an integer")
    temperature = body.get("temperature", 0.95)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise LiveRequestError("temperature must be a finite number") from exc
    if not math.isfinite(temperature):
        raise LiveRequestError("temperature must be a finite number")
    top_p = body.get("top_p", 0.98)
    try:
        top_p = float(top_p)
    except (TypeError, ValueError) as exc:
        raise LiveRequestError("top_p must be in (0, 1]") from exc
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise LiveRequestError("top_p must be in (0, 1]")
    return {
        "mode": mode,
        "composer": composer,
        "genre": genre,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
    }


def register_live_routes(app: Flask) -> None:
    @app.post("/api/live")
    def api_live():
        hub = app.extensions["demo_hub"]
        queue = app.extensions["demo_queue"]
        jobs: LiveJobRegistry = app.extensions["demo_jobs"]
        if hub is None:
            return jsonify({"error": "inference hub is not initialized"}), 503
        if hub_backend(hub) != "mlx":
            return jsonify({"error": "live playback requires --device mlx"}), 400
        body = request.get_json(force=True, silent=True) or {}
        try:
            parsed = parse_live_body(body)
            composer, genre = validate_pair(parsed["composer"], parsed["genre"])
        except (LiveRequestError, ObservedPairError) as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            lease = queue.acquire()
        except QueueBusyError as exc:
            return jsonify({"error": str(exc), "busy": True}), 429

        job = jobs.create()
        cleanup_lock = Lock()
        cleaned = False

        def cleanup_once() -> None:
            nonlocal cleaned
            with cleanup_lock:
                if cleaned:
                    return
                cleaned = True
            job.cancel_event.set()
            jobs.unregister(job.job_id)
            lease.release()

        stream_fn = app.extensions.get("live_stream_fn") or iter_live_sse

        def body_iter():
            try:
                yield from stream_fn(
                    hub,
                    mode=parsed["mode"],
                    composer=composer,
                    genre=genre,
                    seed=parsed["seed"],
                    live_buffer_sec=float(app.extensions["live_buffer_sec"]),
                    job=job,
                    temperature=parsed["temperature"],
                    top_p=parsed["top_p"],
                )
            except GeneratorExit:
                raise
            except Exception as exc:
                yield format_sse("end", {"reason": "error", "message": str(exc)})
            finally:
                cleanup_once()

        try:
            response = Response(
                body_iter(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
            response.call_on_close(cleanup_once)
            return response
        except Exception:
            cleanup_once()
            raise

    @app.post("/api/live/<job_id>/stop")
    def api_live_stop(job_id: str):
        jobs: LiveJobRegistry = app.extensions["demo_jobs"]
        if jobs.cancel(job_id):
            return jsonify({"ok": True, "job_id": job_id})
        return jsonify({"error": "unknown live job"}), 404
