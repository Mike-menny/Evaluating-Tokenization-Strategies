from __future__ import annotations

from dafx26_demo.metrics import realtime_ok, rtf


def test_rtf_is_wall_over_music_asr_convention() -> None:
    assert rtf(wall_sec=60.0, music_sec=60.0) == 1.0
    assert rtf(wall_sec=30.0, music_sec=60.0) == 0.5
    assert rtf(wall_sec=2.0, music_sec=1.0) == 2.0
    assert rtf(wall_sec=2.0, music_sec=0.0) is None


def test_realtime_means_rtf_at_most_one() -> None:
    assert realtime_ok(0.5) is True
    assert realtime_ok(1.0) is True
    assert realtime_ok(2.0) is False
    assert realtime_ok(None) is False
