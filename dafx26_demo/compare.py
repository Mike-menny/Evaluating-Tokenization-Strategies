from __future__ import annotations

from dafx26_demo.inference import generate
from dafx26_demo.schema import GenerationRequest, GenerationResult


def compare_modes(hub, template: GenerationRequest, modes: list[str]) -> list[GenerationResult]:
    """Generate sequentially. A failure does not discard earlier results."""
    results: list[GenerationResult] = []
    for mode in modes:
        request = GenerationRequest(**{**template.to_dict(), "mode": mode})
        try:
            results.append(generate(hub, request))
        except Exception as exc:
            results.append(
                GenerationResult(
                    request_id="",
                    mode=mode,
                    composer=template.composer,
                    genre=template.genre,
                    ok=False,
                    error=str(exc),
                    midi_path=None,
                    elapsed_sec=0.0,
                    n_tokens=0,
                )
            )
    return results
