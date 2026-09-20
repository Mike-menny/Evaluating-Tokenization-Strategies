from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    mode: str
    composer: str
    genre: str
    seed: int
    max_tokens: int
    temperature: float
    top_p: float
    device: str
    min_tokens: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    request_id: str
    mode: str
    composer: str
    genre: str
    ok: bool
    error: str | None
    midi_path: str | None
    elapsed_sec: float
    n_tokens: int
    load_sec: float = 0.0
    peak_memory_bytes: int | None = None
    musical_duration_sec: float | None = None
    beat_markers: list[dict[str, Any]] | None = None
    code_revision: str | None = None
    model_revision: str | None = None
    cached: bool = False
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def request_from_mapping(data: dict[str, Any]) -> GenerationRequest:
    return GenerationRequest(
        mode=str(data["mode"]),
        composer=str(data["composer"]),
        genre=str(data["genre"]),
        seed=int(data.get("seed", 0)),
        max_tokens=int(data.get("max_tokens", 256)),
        temperature=float(data.get("temperature", 0.95)),
        top_p=float(data.get("top_p", 0.98)),
        device=str(data.get("device", "cpu")),
        min_tokens=int(data.get("min_tokens", 64)),
    )


def cache_key(request: GenerationRequest, *, code_revision: str, model_revision: str) -> str:
    payload = {
        "request": request.to_dict(),
        "code_revision": code_revision,
        "model_revision": model_revision,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
