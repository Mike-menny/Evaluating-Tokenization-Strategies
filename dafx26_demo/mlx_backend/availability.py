from __future__ import annotations

import platform
import sys


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def require_mlx() -> None:
    if not mlx_available():
        raise RuntimeError("MLX is not installed. Run: uv sync --extra mlx")


def apple_silicon_available() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def require_apple_silicon_for_benchmark() -> None:
    if not apple_silicon_available():
        raise RuntimeError("MLX benchmark requires Apple Silicon")
    require_mlx()
