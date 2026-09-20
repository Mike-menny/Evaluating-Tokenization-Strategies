from __future__ import annotations

from pathlib import Path


def test_vendored_piano_samples_exist() -> None:
    piano = Path("dafx26_demo/static/vendor/sgm_plus/acoustic_grand_piano")
    assert (piano / "instrument.json").is_file()
    assert (piano / "p60.mp3").is_file()
    assert (piano / "p21.mp3").is_file()
    assert (piano / "p108.mp3").is_file()
