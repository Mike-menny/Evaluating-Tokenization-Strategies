from __future__ import annotations

from pathlib import Path
import time

from dafx26_demo.retention import prune_outputs


def test_prune_outputs_keeps_newest(tmp_path: Path) -> None:
    for i in range(5):
        d = tmp_path / f"job{i}"
        d.mkdir()
        (d / "generation.mid").write_bytes(b"MThd")
        time.sleep(0.02)
    removed = prune_outputs(tmp_path, keep=3)
    assert removed == 2
    names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert names == ["job2", "job3", "job4"]
