from __future__ import annotations

import platform
import sys

import pytest

from dafx26_demo.mlx_backend import (
    apple_silicon_available,
    mlx_available,
    require_apple_silicon_for_benchmark,
    require_mlx,
)


def test_mlx_available_returns_bool() -> None:
    assert isinstance(mlx_available(), bool)


def test_require_mlx_explains_optional_extra(monkeypatch) -> None:
    monkeypatch.setattr("dafx26_demo.mlx_backend.availability.mlx_available", lambda: False)
    with pytest.raises(RuntimeError, match="uv sync --extra mlx"):
        require_mlx()


def test_require_mlx_is_silent_when_available(monkeypatch) -> None:
    monkeypatch.setattr("dafx26_demo.mlx_backend.availability.mlx_available", lambda: True)
    require_mlx()


def test_mlx_available_is_true_when_core_imports(monkeypatch) -> None:
    import types

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    assert mlx_available() is True


def test_apple_silicon_helper_matches_host() -> None:
    expected = sys.platform == "darwin" and platform.machine() == "arm64"
    assert apple_silicon_available() is expected


def test_mlx_benchmark_requires_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr("dafx26_demo.mlx_backend.availability.apple_silicon_available", lambda: False)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        require_apple_silicon_for_benchmark()


def test_mlx_benchmark_proceeds_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr("dafx26_demo.mlx_backend.availability.apple_silicon_available", lambda: True)
    monkeypatch.setattr("dafx26_demo.mlx_backend.availability.mlx_available", lambda: True)
    require_apple_silicon_for_benchmark()


@pytest.mark.skipif(not mlx_available(), reason="MLX is not installed")
def test_mlx_array_smoke_on_available_runtime() -> None:
    require_mlx()
    import mlx.core as mx

    values = mx.array([1.0, 2.0, 3.0])
    mx.eval(values)
    assert values.shape == (3,)
    assert float(values.sum()) == pytest.approx(6.0)
