#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dafx26_demo.paths import MODELS_ROOT, REPO_ROOT, STATIC_ROOT, ensure_upstream_on_path


def check() -> int:
    problems: list[str] = []
    if sys.version_info < (3, 11):
        problems.append(f"Python {sys.version.split()[0]} is too old; need 3.11+")
    ensure_upstream_on_path()
    if not MODELS_ROOT.is_dir():
        problems.append(f"models missing at {MODELS_ROOT}; run uv run python scripts/download_dafx26_models.py")
    vendor = STATIC_ROOT / "vendor"
    if not vendor.is_dir():
        problems.append("offline vendor assets missing; run uv run python scripts/vendor_dafx26_static.py")
    print(f"python: {sys.version.split()[0]}")
    print(f"repo: {REPO_ROOT}")
    print(f"models: {MODELS_ROOT} ({'ok' if MODELS_ROOT.is_dir() else 'missing'})")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda: {torch.cuda.is_available()}")
        print(f"mps: {torch.backends.mps.is_available()}")
    except Exception as exc:
        problems.append(f"torch import failed: {exc}")
    try:
        from dafx26_demo.mlx_backend import mlx_available

        print(f"mlx: {'ok' if mlx_available() else 'not installed (optional extra)'}")
    except Exception as exc:
        print(f"mlx: unavailable ({exc})")
    for disk in [REPO_ROOT]:
        usage = shutil.disk_usage(disk)
        print(f"free disk: {usage.free / 1e9:.1f} GB")
    if problems:
        print("problems:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("environment looks ready")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the DAFx demo environment")
    parser.parse_args()
    raise SystemExit(check())


if __name__ == "__main__":
    main()
