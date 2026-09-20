from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from safetensors.numpy import load_file

from dafx26_demo.inference import default_hub as torch_default_hub
from dafx26_demo.mlx_backend.availability import require_mlx
from dafx26_demo.mlx_backend.inference import MlxModelHub
from dafx26_demo.mlx_backend.weights import (
    METADATA_NAME,
    WEIGHTS_NAME,
    expected_tensors,
    validate_metadata,
)
from dafx26_demo.paths import MODELS_ROOT


_CONVERSION_COMMAND = "uv run python scripts/convert_models_mlx.py"


def _conversion_error(detail: str) -> RuntimeError:
    return RuntimeError(f"{detail}. Run: {_CONVERSION_COMMAND}")


def require_converted_weights(models_root: Path) -> None:
    if not models_root.is_dir():
        raise _conversion_error(f"model directory is missing at {models_root}")

    modes = [
        path
        for path in models_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not modes:
        raise _conversion_error(f"no model directories found at {models_root}")

    for mode_dir in modes:
        mode = mode_dir.name
        converted_dir = mode_dir / "mlx"
        weights_path = converted_dir / WEIGHTS_NAME
        metadata_path = converted_dir / METADATA_NAME
        config_path = converted_dir / "config.json"
        if not weights_path.is_file() or not metadata_path.is_file() or not config_path.is_file():
            raise _conversion_error(f"converted MLX weights are missing for mode {mode!r}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_metadata(metadata, mode=mode)
            if metadata.get("dtype") != "float16":
                raise ValueError(f"converted dtype is {metadata.get('dtype')!r}, expected 'float16'")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected = set(expected_tensors(config))
            recorded = set(metadata.get("tensors", {}))
            if recorded != expected:
                raise ValueError("converted metadata tensor keys do not match the model configuration")
            arrays = load_file(str(weights_path))
            if set(arrays) != expected:
                raise ValueError("converted tensor keys do not match the model configuration")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise _conversion_error(f"invalid converted MLX weights for mode {mode!r}: {exc}") from exc


def select_hub(device_name: str, *, keep_on_device: bool = False) -> object:
    name = (device_name or "auto").strip().lower()
    if name == "mlx":
        require_mlx()
        require_converted_weights(MODELS_ROOT)
        return MlxModelHub(MODELS_ROOT)
    return torch_default_hub(name, keep_on_device=keep_on_device)


def hub_backend(hub: object) -> Literal["pytorch", "mlx"]:
    backend = str(getattr(hub, "backend", "pytorch"))
    if backend not in {"pytorch", "mlx"}:
        raise ValueError(f"unknown hub backend {backend!r}")
    return backend  # type: ignore[return-value]
