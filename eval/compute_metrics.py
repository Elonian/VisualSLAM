#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _dataset_dir(results_root: Path, ds: str) -> Path:
    return results_root / f'dataset_{int(ds):02d}'


def _load_npz(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        return np.array([])
    data = np.load(path)
    if key not in data:
        return np.array([])
    return np.asarray(data[key])


def compute_row(results_root: Path, dataset: str) -> Dict[str, float | int | str]:
    ds = f"{int(dataset):02d}"
    d = _dataset_dir(results_root, ds)

    m1 = _load_json(d / 'metrics_part1.json')
    m2 = _load_json(d / 'metrics_part2.json')
    m3 = _load_json(d / 'metrics_part3.json')
    m4 = _load_json(d / 'metrics_part4.json')

    row: Dict[str, float | int | str] = {
        'dataset': ds,
        'part1_path_length_m': float(m1.get('path_length_m', np.nan)),
        'part1_endpoint_m': float(m1.get('endpoint_distance_m', np.nan)),
        'part2_feature_count': int(m2.get('feature_count', 0)) if m2 else 0,
        'part3_landmarks_initialized': int(m3.get('landmarks_initialized', 0)) if m3 else 0,
        'part3_mean_reproj_px': float(m3.get('mean_reprojection_error_px', np.nan)),
        'part4_path_pred_m': float(m4.get('path_length_pred_m', np.nan)),
        'part4_path_corr_m': float(m4.get('path_length_corr_m', np.nan)),
        'part4_endpoint_pred_m': float(m4.get('endpoint_pred_m', np.nan)),
        'part4_endpoint_corr_m': float(m4.get('endpoint_corr_m', np.nan)),
        'part4_landmarks_initialized': int(m4.get('landmarks_initialized', 0)) if m4 else 0,
        'part4_mean_pose_residual_px': float(m4.get('mean_pose_residual_px', np.nan)),
        'part4_mean_landmark_residual_px': float(m4.get('mean_landmark_residual_px', np.nan)),
    }

    if np.isfinite(row['part4_endpoint_pred_m']) and np.isfinite(row['part4_endpoint_corr_m']):
        row['endpoint_delta_m'] = float(row['part4_endpoint_corr_m'] - row['part4_endpoint_pred_m'])
    else:
        row['endpoint_delta_m'] = np.nan

    # Optional consistency metric from part4 npz.
    slam_npz = d / 'part4_vi_slam.npz'
    traj_pred = _load_npz(slam_npz, 'world_T_imu_pred')
    traj_corr = _load_npz(slam_npz, 'world_T_imu_corr')
    if traj_pred.size and traj_corr.size and traj_pred.shape == traj_corr.shape:
        p0 = traj_pred[:, :3, 3]
        p1 = traj_corr[:, :3, 3]
        row['mean_pose_shift_m'] = float(np.mean(np.linalg.norm(p1 - p0, axis=1)))
    else:
        row['mean_pose_shift_m'] = np.nan

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate VisualSLAM metrics into tables.')
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    parser.add_argument('--datasets', nargs='+', default=['00', '01'])
    parser.add_argument('--output-dir', type=Path, default=Path('eval'))
    args = parser.parse_args()

    rows: List[Dict[str, float | int | str]] = [compute_row(args.results_root, ds) for ds in args.datasets]
    df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'metrics_summary.csv'
    md_path = args.output_dir / 'metrics_summary.md'

    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown_safe(df), encoding='utf-8')

    print(f'[Eval] Wrote {csv_path}')
    print(f'[Eval] Wrote {md_path}')


if __name__ == '__main__':
    main()
