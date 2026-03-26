from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def trajectory_path_length(world_T_imu: np.ndarray) -> float:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def final_translation(world_T_imu: np.ndarray) -> float:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    if p.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]))


def count_initialized_landmarks(initialized: np.ndarray) -> int:
    return int(np.asarray(initialized, dtype=bool).sum())


def nanmean_safe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float('nan')
    finite = np.isfinite(values)
    if not np.any(finite):
        return float('nan')
    return float(np.mean(values[finite]))


def save_metrics(path: Path, metrics: Dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def load_metrics(path: Path) -> Dict[str, float | int | str]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
