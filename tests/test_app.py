from __future__ import annotations

from pathlib import Path

from dafx26_demo.app import create_app
from dafx26_demo.inference import ModelHub
from dafx26_demo.schema import GenerationResult
import torch


class StubMlxHub:
    backend = "mlx"
    device = "mlx"

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root

    def available_modes(self) -> list[str]:
        return ["note"]

    def max_seq_len(self, mode: str) -> int:
        assert mode == "note"
        return 64


def valid_body() -> dict[str, object]:
    return {
        "mode": "note",
        "composer": "Chopin",
        "genre": "etude",
        "seed": 0,
        "max_tokens": 16,
    }


def ok_result(request) -> GenerationResult:
    return GenerationResult(
        request_id="test",
        mode=request.mode,
        composer=request.composer,
        genre=request.genre,
        ok=True,
        error=None,
        midi_path=None,
        elapsed_sec=0.0,
        n_tokens=1,
    )


def test_meta_exposes_only_observed_pairs() -> None:
    app = create_app(hub=ModelHub(Path("models/MIDI_tokenization_models"), torch.device("cpu")))
    client = app.test_client()
    data = client.get("/api/meta").get_json()
    pairs = {(row["composer"], row["genre"]) for row in data["pairs"]}
    assert ("Chopin", "etude") in pairs
    assert ("Chopin", "nocturne") not in pairs
    assert data["defaults"]["composer"] == "Chopin"
    assert data["defaults"]["genre"] == "etude"


def test_generate_rejects_unseen_pair() -> None:
    app = create_app(hub=ModelHub(Path("models/MIDI_tokenization_models"), torch.device("cpu")))
    client = app.test_client()
    response = client.post(
        "/api/generate",
        json={"mode": "note", "composer": "Chopin", "genre": "nocturne", "seed": 0, "max_tokens": 16},
    )
    assert response.status_code == 400
    assert "nocturne" in response.get_json()["error"]


def test_fallback_returns_generated_kind_for_observed_pair() -> None:
    app = create_app(hub=ModelHub(Path("models/MIDI_tokenization_models"), torch.device("cpu")))
    client = app.test_client()
    data = client.get("/api/fallback?composer=Chopin&genre=etude").get_json()
    assert data["composer"] == "Chopin"
    for item in data["items"]:
        assert item["kind"] == "generated"
        assert item["fallback"] is True


def test_health_endpoint() -> None:
    app = create_app(hub=ModelHub(Path("models/MIDI_tokenization_models"), torch.device("cpu")))
    data = app.test_client().get("/health").get_json()
    assert data["ok"] is True
    assert data["code_revision"]


def test_comparison_dispatches_to_mlx(monkeypatch, tmp_path: Path) -> None:
    hub = StubMlxHub(tmp_path)
    called = []
    monkeypatch.setattr("dafx26_demo.app.generate_mlx", lambda _hub, req: called.append(req) or ok_result(req))

    response = create_app(hub).test_client().post("/api/generate", json=valid_body())

    assert response.status_code == 200
    assert called[0].device == "mlx"


def test_create_app_uses_injected_queue_and_jobs(tmp_path: Path) -> None:
    from dafx26_demo.live_jobs import LiveJobRegistry
    from dafx26_demo.queueing import GenerationQueue

    queue = GenerationQueue()
    jobs = LiveJobRegistry()
    app = create_app(StubMlxHub(tmp_path), queue=queue, jobs=jobs)
    assert app.extensions["demo_queue"] is queue
    assert app.extensions["demo_jobs"] is jobs


def test_apps_retain_their_own_hub_backend(tmp_path: Path) -> None:
    cpu_app = create_app(ModelHub(tmp_path, torch.device("cpu")))
    mlx_app = create_app(StubMlxHub(tmp_path))

    assert cpu_app.test_client().get("/health").get_json()["device"] == "cpu"
    assert cpu_app.test_client().get("/health").get_json()["backend"] == "pytorch"
    assert mlx_app.test_client().get("/health").get_json()["device"] == "mlx"
    assert mlx_app.test_client().get("/health").get_json()["backend"] == "mlx"


def test_meta_exposes_mlx_backend_fields(tmp_path: Path) -> None:
    data = create_app(StubMlxHub(tmp_path)).test_client().get("/api/meta").get_json()

    assert data["backend"] == "mlx"
    assert data["device"] == "mlx"
    assert data["live_buffer_sec"] == 2.0
    assert data["max_seq_len_by_mode"] == {"note": 64}
