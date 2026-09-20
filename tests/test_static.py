from __future__ import annotations

from pathlib import Path
import re


def test_ui_does_not_depend_on_cdn() -> None:
    html = Path("dafx26_demo/static/index.html").read_text(encoding="utf-8")
    assert "jsdelivr" not in html
    assert "googleapis" not in html
    assert "storage.googleapis.com" not in html
    assert "/vendor/" in html


def test_live_controls_and_module_are_offline() -> None:
    html = Path("dafx26_demo/static/index.html").read_text(encoding="utf-8")
    js = Path("dafx26_demo/static/js/live_player.mjs").read_text(encoding="utf-8")
    assert 'id="live-mode"' in html
    assert 'id="live-play"' in html
    assert 'id="live-stop"' in html
    assert 'id="live-audio-device"' in html
    assert 'id="live-indicator"' in html
    assert 'id="live-piano-roll"' in html
    assert 'id="live-speed"' in html
    assert 'step="0.05"' in html
    assert 'value="0.75"' in html
    assert "/js/live_player.mjs" in html or "/js/compare_app.mjs" in html
    assert "max_tokens" not in js
    for path in [Path("dafx26_demo/static/index.html"), *Path("dafx26_demo/static/js").glob("*.mjs")]:
        text = path.read_text(encoding="utf-8")
        assert "https://" not in text
        assert "http://" not in text


def test_local_scripts_and_imports_exist() -> None:
    root = Path("dafx26_demo/static")
    html = (root / "index.html").read_text(encoding="utf-8")
    for src in re.findall(r'<script[^>]+src="([^"]+)"', html):
        assert not src.startswith("//")
        assert (root / src.lstrip("/")).is_file(), src
    for path in (root / "js").glob("*.mjs"):
        text = path.read_text(encoding="utf-8")
        for spec in re.findall(r'''from\s+['"](\./[^'"]+)['"]''', text):
            assert (path.parent / spec).resolve().is_file(), f"{path} -> {spec}"
