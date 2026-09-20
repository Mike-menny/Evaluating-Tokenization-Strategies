"""Shared timing metrics.

RTF uses the ASR convention: processing wall time divided by audio (musical)
duration.

- RTF = 1.0: one second of processing per second of audio (realtime)
- RTF < 1.0: faster than realtime
- RTF > 1.0: slower than realtime
"""

from __future__ import annotations


def rtf(*, wall_sec: float, music_sec: float) -> float | None:
    if not music_sec:
        return None
    return wall_sec / music_sec


def realtime_ok(value: float | None) -> bool:
    return value is not None and value <= 1.0
