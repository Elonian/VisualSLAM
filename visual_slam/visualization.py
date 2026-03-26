from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_trajectory_2d(
    world_T_imu: np.ndarray,
    title: str,
    output_path: Path,
    show_orientation: bool = False,
) -> None:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p[:, 0], p[:, 1], "r-", linewidth=1.5, label="trajectory")
    ax.scatter(p[0, 0], p[0, 1], marker="s", s=40, label="start")
    ax.scatter(p[-1, 0], p[-1, 1], marker="o", s=40, label="end")

    if show_orientation and world_T_imu.shape[0] > 1:
        idx = np.linspace(0, world_T_imu.shape[0] - 1, num=min(60, world_T_imu.shape[0]), dtype=int)
        yaw = np.arctan2(world_T_imu[idx, 1, 0], world_T_imu[idx, 0, 0])
        ax.quiver(
            p[idx, 0],
            p[idx, 1],
            np.cos(yaw),
            np.sin(yaw),
            color="b",
            scale=30,
            width=0.002,
        )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_landmarks_xy(
    world_T_imu: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    lm = np.asarray(landmarks_w, dtype=np.float64)
    init = np.asarray(initialized, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p[:, 0], p[:, 1], "r-", linewidth=1.2, label="trajectory")

    if np.any(init):
        ax.scatter(lm[init, 0], lm[init, 1], s=4, c="tab:blue", alpha=0.65, label="landmarks")

    ax.scatter(p[0, 0], p[0, 1], marker="s", s=40, label="start")
    ax.scatter(p[-1, 0], p[-1, 1], marker="o", s=40, label="end")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectory_comparison(
    world_T_pred: np.ndarray,
    world_T_corr: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    p0 = np.asarray(world_T_pred[:, :3, 3], dtype=np.float64)
    p1 = np.asarray(world_T_corr[:, :3, 3], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p0[:, 0], p0[:, 1], "--", linewidth=1.2, label="IMU-only")
    ax.plot(p1[:, 0], p1[:, 1], "-", linewidth=1.5, label="VI-SLAM")

    ax.scatter(p1[0, 0], p1[0, 1], marker="s", s=40, label="start")
    ax.scatter(p1[-1, 0], p1[-1, 1], marker="o", s=40, label="end")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
