#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    y = df[y_col].to_numpy(dtype=float)
    ax.bar(x, y, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str).tolist())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build report charts from eval/metrics_summary.csv')
    parser.add_argument('--metrics-csv', type=Path, default=Path('eval/metrics_summary.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('eval'))
    parser.add_argument('--output-md', type=Path, default=Path('eval/report_assets.md'))
    args = parser.parse_args()

    if not args.metrics_csv.exists():
        raise FileNotFoundError(
            f'Metrics CSV not found: {args.metrics_csv}. Run eval/compute_metrics.py first.'
        )

    df = pd.read_csv(args.metrics_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = args.output_dir / 'plot_part1_path_length.png'
    fig2 = args.output_dir / 'plot_part3_landmarks_initialized.png'
    fig3 = args.output_dir / 'plot_part4_endpoint_improvement.png'
    fig4 = args.output_dir / 'plot_part4_pose_residual.png'

    _bar_plot(df, 'dataset', 'part1_path_length_m', 'Part 1 IMU Path Length', 'meters', fig1)
    _bar_plot(df, 'dataset', 'part3_landmarks_initialized', 'Part 3 Initialized Landmarks', 'count', fig2)
    _bar_plot(df, 'dataset', 'endpoint_improvement_m', 'Part 4 Endpoint Improvement', 'meters', fig3)
    _bar_plot(df, 'dataset', 'part4_mean_pose_residual_px', 'Part 4 Mean Pose Residual', 'pixels', fig4)

    lines = [
        '# VisualSLAM Evaluation Assets',
        '',
        '## Metrics Table',
        '',
        _to_markdown_safe(df),
        '',
        '## Charts',
        '',
        f'![Part1 Path Length]({fig1.as_posix()})',
        f'![Part3 Initialized Landmarks]({fig2.as_posix()})',
        f'![Part4 Endpoint Improvement]({fig3.as_posix()})',
        f'![Part4 Mean Pose Residual]({fig4.as_posix()})',
    ]
    args.output_md.write_text('\n'.join(lines), encoding='utf-8')

    print(f'[Eval] Wrote {args.output_md}')


if __name__ == '__main__':
    main()
