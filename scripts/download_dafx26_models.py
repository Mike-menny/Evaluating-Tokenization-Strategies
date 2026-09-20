#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from dafx26_demo.paths import MODELS_ROOT, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned MIDI tokenization checkpoints")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--local-dir", type=Path, default=MODELS_ROOT)
    args = parser.parse_args()
    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="cnmat/MIDI_tokenization_models",
        revision=args.revision,
        local_dir=str(args.local_dir),
    )
    modes = sorted(p.name for p in args.local_dir.iterdir() if (p / "pytorch_model.bin").is_file())
    print(f"downloaded revision {args.revision} -> {args.local_dir}")
    print("modes:", ", ".join(modes) or "(none)")
    expected = {
        "note",
        "note_pedal",
        "note_velocity",
        "note_velocity_beat",
        "note_velocity_pedal",
        "full",
    }
    missing = expected - set(modes)
    if missing:
        raise SystemExit(f"missing modes: {sorted(missing)}")


if __name__ == "__main__":
    main()
