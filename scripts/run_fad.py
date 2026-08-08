#!/usr/bin/env python3
"""Compute per-composer/genre FAD for one or more inference WAV roots.

Example:
  python3 scripts/run_fad.py \\
    --reference outputs/asap_test_set_wav \\
    --run name=asap_note_velocity_ft,wav=outputs/inference/asap_note_velocity_ft/SESSION_wav,csv=outputs/inference/asap_note_velocity_ft/SESSION/fad.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse_run(spec: str) -> tuple[str, str, str]:
    parts = dict(kv.split("=", 1) for kv in spec.split(","))
    try:
        return parts["name"], parts["wav"], parts["csv"]
    except KeyError as e:
        raise SystemExit(f"run spec needs name=,wav=,csv=: {spec}") from e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=str, required=True, help="Reference WAV root (composer/genre/*.wav)")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable: name=...,wav=...,csv=...",
    )
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    from src.evaluation.fad import compute_fad_runs_parallel

    runs = [_parse_run(s) for s in args.run]
    compute_fad_runs_parallel(runs=runs, reference_path=args.reference, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
