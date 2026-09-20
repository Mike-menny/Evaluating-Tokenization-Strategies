from __future__ import annotations

from dafx26_demo.schema import GenerationRequest, GenerationResult, cache_key, request_from_mapping


def test_request_roundtrip_and_bounds() -> None:
    req = GenerationRequest(
        mode="note_velocity_pedal",
        composer="Chopin",
        genre="etude",
        seed=7,
        max_tokens=256,
        temperature=0.95,
        top_p=0.98,
        device="cpu",
    )
    data = req.to_dict()
    again = request_from_mapping(data)
    assert again.mode == "note_velocity_pedal"
    assert again.max_tokens == 256


def test_cache_key_changes_with_seed_and_revision() -> None:
    req = GenerationRequest(
        mode="note",
        composer="Bach",
        genre="fugue",
        seed=1,
        max_tokens=256,
        temperature=0.95,
        top_p=0.98,
        device="cpu",
    )
    a = cache_key(req, code_revision="aaa", model_revision="bbb")
    b = cache_key(req, code_revision="aaa", model_revision="ccc")
    c = cache_key(
        GenerationRequest(**{**req.to_dict(), "seed": 2}),
        code_revision="aaa",
        model_revision="bbb",
    )
    assert a != b
    assert a != c
    assert len(a) == 64


def test_result_records_error_without_dropping_fields() -> None:
    result = GenerationResult(
        request_id="abc",
        mode="full",
        composer="Chopin",
        genre="etude",
        ok=False,
        error="timeout",
        midi_path=None,
        elapsed_sec=12.3,
        n_tokens=0,
    )
    payload = result.to_dict()
    assert payload["ok"] is False
    assert payload["error"] == "timeout"
    assert payload["midi_path"] is None
