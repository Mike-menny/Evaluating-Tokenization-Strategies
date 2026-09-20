from __future__ import annotations

from dafx26_demo.cache import load_cached, store_cached
from dafx26_demo.schema import GenerationResult


def test_cache_roundtrip(tmp_path) -> None:
    result = GenerationResult(
        request_id="x",
        mode="note",
        composer="Bach",
        genre="fugue",
        ok=True,
        error=None,
        midi_path="/tmp/x.mid",
        elapsed_sec=1.2,
        n_tokens=10,
    )
    store_cached(tmp_path, "ab" * 32, result)
    loaded = load_cached(tmp_path, "ab" * 32)
    assert loaded is not None
    assert loaded.cached is True
    assert loaded.n_tokens == 10
    assert load_cached(tmp_path, "missing") is None
