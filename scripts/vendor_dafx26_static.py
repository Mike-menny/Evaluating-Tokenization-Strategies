#!/usr/bin/env python3
"""Download JS and a local piano SoundFont so the booth UI works offline."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from dafx26_demo.paths import STATIC_ROOT

VENDOR = STATIC_ROOT / "vendor"
SOUNDFONT = VENDOR / "sgm_plus"
PIANO = SOUNDFONT / "acoustic_grand_piano"
SOUNDFONT_BASE = "https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"

FILES = {
    "tone.js": "https://cdn.jsdelivr.net/npm/tone@14.7.77/build/Tone.js",
    "magenta-core.js": "https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/core.js",
    "html-midi-player.js": "https://cdn.jsdelivr.net/npm/html-midi-player@1.5.0/dist/midi-player.min.js",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    print(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "dafx26-demo"})
    with urllib.request.urlopen(req) as response:
        dest.write_bytes(response.read())


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        fetch(url, VENDOR / name)
    fetch(f"{SOUNDFONT_BASE}/soundfont.json", SOUNDFONT / "soundfont.json")
    # Magenta looks up program 0 at {base}/acoustic_grand_piano/{pXX}.mp3
    # when instrument.json has no velocity layers. We flatten v79 samples.
    PIANO.mkdir(parents=True, exist_ok=True)
    (PIANO / "instrument.json").write_text(
        json.dumps(
            {
                "name": "acoustic_grand_piano",
                "minPitch": 21,
                "maxPitch": 108,
                "durationSeconds": 3.0,
                "releaseSeconds": 1.0,
                "percussive": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for pitch in range(21, 109):
        dest = PIANO / f"p{pitch}.mp3"
        fetch(f"{SOUNDFONT_BASE}/acoustic_grand_piano/p{pitch}_v79.mp3", dest)
    print(f"vendor assets in {VENDOR}")


if __name__ == "__main__":
    main()
