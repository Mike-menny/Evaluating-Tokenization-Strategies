from __future__ import annotations

from pathlib import Path
import threading

import pytest
import torch

from dafx26_demo.app import build_arg_parser, create_app
from dafx26_demo.inference import ModelHub
from dafx26_demo.live_jobs import LiveJobRegistry
from dafx26_demo.mlx_backend.live import format_sse
from dafx26_demo.queueing import GenerationQueue, QueueBusyError
from tests.test_app import StubMlxHub
from tests.test_mlx_live import decode_sse


def valid_live_body() -> dict[str, object]:
    return {
        "mode": "note",
        "composer": "Chopin",
        "genre": "etude",
        "seed": 0,
        "temperature": 0.95,
        "top_p": 0.98,
    }


def live_test_app(*, stream_fn, hub=None):
    queue = GenerationQueue()
    jobs = LiveJobRegistry()
    app = create_app(
        hub if hub is not None else StubMlxHub(Path(".")),
        queue=queue,
        jobs=jobs,
    )
    app.extensions["live_stream_fn"] = stream_fn
    return app, queue, jobs


def _open_live(app, body: dict[str, object] | None = None):
    with app.test_request_context("/api/live", method="POST", json=body or valid_live_body()):
        return app.make_response(app.dispatch_request())


def _stop(app, job_id: str):
    with app.test_request_context(f"/api/live/{job_id}/stop", method="POST"):
        return app.make_response(app.dispatch_request())


def test_stream_holds_lease_until_response_closes() -> None:
    proceed = threading.Event()

    def blocking_stream(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})
        proceed.wait(timeout=5)
        yield format_sse("end", {"reason": "stop"})

    app, queue, _jobs = live_test_app(stream_fn=blocking_stream)
    response = _open_live(app)
    assert response.status_code == 200
    iterator = iter(response.response)
    first = next(iterator)
    assert "event: meta" in first
    with pytest.raises(QueueBusyError):
        queue.acquire()
    proceed.set()
    list(iterator)
    response.close()
    queue.acquire().release()


def test_stop_sets_cancel_and_unknown_job_is_404() -> None:
    def waiting_stream(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})
        kwargs["job"].cancel_event.wait(timeout=5)
        yield format_sse("end", {"reason": "stop"})

    app, _queue, _jobs = live_test_app(stream_fn=waiting_stream)
    response = _open_live(app)
    iterator = iter(response.response)
    first = next(iterator)
    if isinstance(first, bytes):
        first = first.decode()
    job_id = decode_sse([first])[0].data["job_id"]
    assert _stop(app, job_id).status_code == 200
    assert _stop(app, job_id).status_code == 200
    list(iterator)
    response.close()
    assert _stop(app, job_id).status_code == 404


def test_close_mid_stream_releases_queue() -> None:
    def hanging_stream(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})
        yield format_sse("end", {"reason": "stop"})

    app, queue, _jobs = live_test_app(stream_fn=hanging_stream)
    response = _open_live(app)
    iterator = iter(response.response)
    next(iterator)
    response.close()
    queue.acquire().release()


def test_close_before_first_chunk_cleans_up() -> None:
    def hanging_stream(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})

    app, queue, _jobs = live_test_app(stream_fn=hanging_stream)
    response = _open_live(app)
    assert response.status_code == 200
    response.close()
    queue.acquire().release()


def test_pytorch_hub_rejects_live() -> None:
    app, _queue, _jobs = live_test_app(
        stream_fn=lambda _hub, **kwargs: iter(()),
        hub=ModelHub(Path("."), torch.device("cpu")),
    )
    response = app.test_client().post("/api/live", json=valid_live_body())
    assert response.status_code == 400
    assert "mlx" in response.get_json()["error"].lower()


def test_busy_live_returns_429() -> None:
    queue = GenerationQueue()
    lease = queue.acquire()
    try:
        app = create_app(StubMlxHub(Path(".")), queue=queue, jobs=LiveJobRegistry())
        app.extensions["live_stream_fn"] = lambda _hub, **kwargs: iter(())
        response = app.test_client().post("/api/live", json=valid_live_body())
        assert response.status_code == 429
        assert response.get_json()["busy"] is True
    finally:
        lease.release()


def test_uninitialized_hub_returns_503() -> None:
    app = create_app(None)
    response = app.test_client().post("/api/live", json=valid_live_body())
    assert response.status_code == 503


def test_invalid_live_requests_are_400() -> None:
    app, _queue, _jobs = live_test_app(stream_fn=lambda _hub, **kwargs: iter(()))
    client = app.test_client()
    assert client.post("/api/live", json={**valid_live_body(), "genre": "nocturne"}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "mode": "nope"}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "max_tokens": 256}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "seed": 1.5}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "temperature": float("nan")}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "top_p": 0}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "top_p": 1.5}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "composer": ""}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "temperature": "hot"}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "top_p": "nope"}).status_code == 400
    assert client.post("/api/live", json={**valid_live_body(), "seed": True}).status_code == 400


def test_error_sse_is_still_200() -> None:
    def error_stream(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})
        yield format_sse("end", {"reason": "error", "message": "boom"})

    app, queue, _jobs = live_test_app(stream_fn=error_stream)
    response = app.test_client().post("/api/live", json=valid_live_body())
    assert response.status_code == 200
    events = decode_sse([response.get_data(as_text=True)])
    assert events[-1].data["reason"] == "error"
    queue.acquire().release()


def test_stream_exception_emits_end_error_and_releases() -> None:
    def boom(_hub, **kwargs):
        yield format_sse("meta", {"job_id": kwargs["job"].job_id})
        raise RuntimeError("decode failed")

    app, queue, _jobs = live_test_app(stream_fn=boom)
    response = app.test_client().post("/api/live", json=valid_live_body())
    assert response.status_code == 200
    events = decode_sse([response.get_data(as_text=True)])
    assert events[-1].data["reason"] == "error"
    assert "decode failed" in events[-1].data["message"]
    queue.acquire().release()


def test_live_buffer_cli_must_be_finite_and_positive() -> None:
    parser = build_arg_parser()
    assert parser.parse_args(["--live-buffer-sec", "2"]).live_buffer_sec == 2.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--live-buffer-sec", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--live-buffer-sec", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--live-buffer-sec", "nan"])
