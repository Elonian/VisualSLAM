from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GpuInfo:
    enabled: bool
    backend: str
    message: str


def get_array_module(use_gpu: bool):
    """Return cupy when available and requested, otherwise numpy."""
    if not use_gpu:
        return np
    try:
        import cupy as cp

        _ = cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return np


def describe_gpu(use_gpu: bool) -> GpuInfo:
    if not use_gpu:
        return GpuInfo(enabled=False, backend='numpy', message='GPU disabled via CLI flag.')
    try:
        import cupy as cp

        n = int(cp.cuda.runtime.getDeviceCount())
        if n < 1:
            return GpuInfo(enabled=False, backend='numpy', message='No CUDA device found, using CPU.')
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        return GpuInfo(enabled=True, backend='cupy', message=f'Using CUDA device: {name}')
    except Exception as exc:
        return GpuInfo(enabled=False, backend='numpy', message=f'CuPy unavailable, fallback to CPU: {exc}')
