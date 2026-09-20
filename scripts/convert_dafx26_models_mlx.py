#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dafx26_demo.mlx_backend.weights import convert_checkpoint
from dafx26_demo.paths import MODELS_ROOT

DEFAULT_MODES = [
    "note",
    "note_pedal",
    "note_velocity",
    "note_velocity_beat",
    "note_velocity_pedal",
    "full",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pinned PyTorch MIDI checkpoints to MLX safetensors")
    parser.add_argument("--modes", default="all", help="Comma-separated modes or 'all'")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    modes = DEFAULT_MODES if args.modes.strip() == "all" else [item.strip() for item in args.modes.split(",") if item.strip()]
    for mode in modes:
        src = MODELS_ROOT / mode
        dest = src / "mlx"
        print(f"converting {mode} -> {dest}")
        convert_checkpoint(src, dest, mode=mode, dtype=args.dtype, force=args.force)
    print("done")


if __name__ == "__main__":
    main()
