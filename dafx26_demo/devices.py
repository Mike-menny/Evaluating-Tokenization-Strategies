from __future__ import annotations

import torch


def pick_device(requested: str) -> torch.device:
    name = (requested or "auto").strip().lower()
    if name in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available")
    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device {requested!r}")


def dtype_for_device(device: torch.device, *, quantize: str = "auto") -> torch.dtype:
    if quantize == "int8":
        raise RuntimeError("Upstream custom INT8 is experimental and is not enabled")
    if quantize in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if quantize in {"fp16", "float16"}:
        return torch.float16
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int, device: torch.device) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
