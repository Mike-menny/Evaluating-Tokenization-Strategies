from __future__ import annotations

import pytest
import torch

from dafx26_demo.devices import dtype_for_device, pick_device


def test_cpu_device_and_int8_rejected() -> None:
    assert pick_device("cpu").type == "cpu"
    try:
        dtype_for_device(torch.device("cpu"), quantize="int8")
        raise AssertionError("int8 should be rejected")
    except RuntimeError as exc:
        assert "experimental" in str(exc)


def test_auto_prefers_available_accelerator() -> None:
    device = pick_device("auto")
    assert device.type in {"cpu", "mps", "cuda"}


def test_unknown_torch_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        pick_device("mlx")
