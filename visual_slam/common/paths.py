from __future__ import annotations

from pathlib import Path


def dataset_output_dir(output_root: Path, dataset_id: str) -> Path:
    return output_root / f'dataset_{dataset_id}'


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
