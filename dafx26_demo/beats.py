from __future__ import annotations

from typing import Any

from dafx26_demo.paths import ensure_upstream_on_path

ensure_upstream_on_path()

from src.tokenization.conditional_vocab import (  # noqa: E402
    BEAT_B_ID,
    BEAT_BR_ID,
    BEAT_DB_ID,
    TIME_OFFSET,
    TIME_RESOLUTION,
    get_tokenization_spec,
    iter_event_slices,
)


_KIND = {
    BEAT_B_ID: "beat",
    BEAT_DB_ID: "downbeat",
    BEAT_BR_ID: "rubato",
}


def extract_beat_markers(tokens: list[int], mode: str) -> list[dict[str, Any]]:
    spec = get_tokenization_spec(mode)
    if not spec.use_beat:
        return []
    markers: list[dict[str, Any]] = []
    for start, end in iter_event_slices(tokens, mode=mode):
        seg = tokens[start:end]
        if len(seg) == 2 and seg[1] in _KIND:
            ticks = max(0, seg[0] - TIME_OFFSET)
            markers.append(
                {
                    "time_sec": ticks / TIME_RESOLUTION,
                    "kind": _KIND[seg[1]],
                }
            )
    return markers
