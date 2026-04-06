from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

from visual_slam.geometry import StereoCalibration, StereoProjector
from visual_slam.landmark_mapping import LandmarkMapResult


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_imu_dashboard_data(
    world_T_imu: np.ndarray,
    timestamps: np.ndarray,
    v_t: np.ndarray,
    w_t: np.ndarray,
) -> dict:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    v = np.asarray(v_t, dtype=np.float64)
    w = np.asarray(w_t, dtype=np.float64)

    n = int(min(p.shape[0], ts.shape[0], v.shape[0], w.shape[0]))
    p = p[:n]
    ts = ts[:n]
    v = v[:n]
    w = w[:n]

    t_rel = ts - ts[0] if n > 0 else np.zeros(0, dtype=np.float64)
    t_norm = np.linspace(0.0, 1.0, num=max(n, 1))
    linear_speed = np.linalg.norm(v, axis=1) if n > 0 else np.zeros(0, dtype=np.float64)
    angular_speed = np.linalg.norm(w, axis=1) if n > 0 else np.zeros(0, dtype=np.float64)

    if n > 1:
        xy_seg = np.linalg.norm(p[1:, :2] - p[:-1, :2], axis=1)
        xy_path_cum = np.concatenate([[0.0], np.cumsum(xy_seg)])
    else:
        xy_path_cum = np.zeros(n, dtype=np.float64)
    xy_offset = np.linalg.norm(p[:, :2] - p[0, :2], axis=1) if n > 0 else np.zeros(0, dtype=np.float64)
    yaw = np.arctan2(world_T_imu[:n, 1, 0], world_T_imu[:n, 0, 0]) if n > 0 else np.zeros(0, dtype=np.float64)
    yaw_deg = np.rad2deg(np.unwrap(yaw)) if n > 0 else np.zeros(0, dtype=np.float64)

    return {
        'n': n,
        'p': p,
        'ts': ts,
        'v': v,
        'w': w,
        't_rel': t_rel,
        't_norm': t_norm,
        'linear_speed': linear_speed,
        'angular_speed': angular_speed,
        'xy_path_cum': xy_path_cum,
        'xy_offset': xy_offset,
        'yaw': yaw,
        'yaw_deg': yaw_deg,
    }


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


def plot_imu_trajectory_dashboard(
    world_T_imu: np.ndarray,
    timestamps: np.ndarray,
    v_t: np.ndarray,
    w_t: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    data = _prepare_imu_dashboard_data(world_T_imu, timestamps, v_t, w_t)
    n = data['n']
    p = data['p']
    t_rel = data['t_rel']
    t_norm = data['t_norm']
    v = data['v']
    w = data['w']
    yaw_deg = data['yaw_deg']

    fig = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_imu = fig.add_subplot(gs[0, 1])
    ax_pose = fig.add_subplot(gs[1, 1])

    if n >= 2:
        xy = p[:, :2]
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', linewidths=2.4, alpha=0.95)
        lc.set_array(t_norm[:-1])
        ax_xy.add_collection(lc)
        ax_xy.plot(xy[:, 0], xy[:, 1], color='0.82', linewidth=0.9, alpha=0.7, zorder=0)
        cbar = fig.colorbar(lc, ax=ax_xy, fraction=0.045, pad=0.02)
        cbar.set_label('Normalized Time')

        sample_count = min(7, max(2, n // 400))
        idx = np.linspace(0, n - 1, num=sample_count, dtype=int)
        ax_xy.scatter(xy[idx, 0], xy[idx, 1], c=t_norm[idx], cmap='viridis', s=28, edgecolors='white', linewidths=0.5, zorder=3)
        for k in idx[1:-1]:
            ax_xy.annotate(f'{100.0 * t_norm[k]:.0f}%', (xy[k, 0], xy[k, 1]), xytext=(4, 4), textcoords='offset points', fontsize=8, color='0.25')

        yaw = np.arctan2(world_T_imu[:n, 1, 0], world_T_imu[:n, 0, 0])
        q_idx = np.linspace(0, n - 1, num=min(24, n), dtype=int)
        ax_xy.quiver(
            xy[q_idx, 0],
            xy[q_idx, 1],
            np.cos(yaw[q_idx]),
            np.sin(yaw[q_idx]),
            color='0.2',
            alpha=0.35,
            scale=28,
            width=0.0024,
            zorder=2,
        )

    ax_xy.scatter(p[0, 0], p[0, 1], marker='s', s=55, color='tab:green', edgecolors='black', linewidths=0.5, zorder=4)
    ax_xy.scatter(p[-1, 0], p[-1, 1], marker='*', s=110, color='tab:red', edgecolors='black', linewidths=0.5, zorder=4)
    ax_xy.set_title(f'{title} | Top-Down XY Trajectory')
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.axis('equal')
    ax_xy.grid(True, alpha=0.28)

    xy_path = float(data['xy_path_cum'][-1]) if n > 0 else 0.0
    xy_end = float(data['xy_offset'][-1]) if n > 0 else 0.0
    stats = (
        f'Steps: {n}\n'
        f'Duration: {t_rel[-1]:.1f} s\n'
        f'End XY: ({p[-1, 0]:.1f}, {p[-1, 1]:.1f}) m\n'
        f'XY Path: {xy_path:.1f} m\n'
        f'XY End Offset: {xy_end:.1f} m\n'
        f'Final Heading: {yaw_deg[-1]:.1f} deg'
    )
    ax_xy.text(
        0.02,
        0.98,
        stats,
        transform=ax_xy.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.75', alpha=0.95),
    )

    legend_handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor='tab:green', markeredgecolor='black', markersize=8, label='Start'),
        Line2D([0], [0], marker='*', color='none', markerfacecolor='tab:red', markeredgecolor='black', markersize=12, label='End'),
        Line2D([0], [0], color='0.2', lw=1.5, alpha=0.5, label='Heading Samples'),
    ]
    ax_xy.legend(handles=legend_handles, loc='lower right', frameon=True)

    linear_speed = data['linear_speed']
    angular_speed = data['angular_speed']
    ax_imu.plot(t_rel, linear_speed, color='tab:blue', linewidth=1.5, label=r'$\|v_t\|$')
    ax_imu.plot(t_rel, angular_speed, color='tab:orange', linewidth=1.2, label=r'$\|\omega_t\|$')
    ax_imu.set_title('IMU Inputs Used For Propagation')
    ax_imu.set_xlabel('time [s]')
    ax_imu.set_ylabel('m/s and rad/s')
    ax_imu.grid(True, alpha=0.28)
    ax_imu.legend(loc='upper right', fontsize=8)

    ax_pose.plot(t_rel, p[:, 0], linewidth=1.5, label='x')
    ax_pose.plot(t_rel, p[:, 1], linewidth=1.5, label='y')
    ax_pose.plot(t_rel, data['xy_offset'], linewidth=1.2, linestyle='--', color='0.35', label='xy offset')
    ax_pose.set_title('Planar Position Summary')
    ax_pose.set_xlabel('time [s]')
    ax_pose.set_ylabel('meters')
    ax_pose.grid(True, alpha=0.28)
    ax_pose.legend(loc='upper left', fontsize=8)

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_imu_trajectory_gif(
    world_T_imu: np.ndarray,
    timestamps: np.ndarray,
    v_t: np.ndarray,
    w_t: np.ndarray,
    title: str,
    output_path: Path,
    max_frames: int = 72,
    frame_duration_ms: int = 90,
) -> None:
    data = _prepare_imu_dashboard_data(world_T_imu, timestamps, v_t, w_t)
    n = data['n']
    if n == 0:
        return

    p = data['p']
    xy = p[:, :2]
    t_rel = data['t_rel']
    linear_speed = data['linear_speed']
    angular_speed = data['angular_speed']
    yaw = data['yaw']
    yaw_deg = data['yaw_deg']

    x_pad = max(5.0, 0.05 * max(1.0, float(np.ptp(xy[:, 0]))))
    y_pad = max(5.0, 0.05 * max(1.0, float(np.ptp(xy[:, 1]))))
    x_lim = (float(xy[:, 0].min() - x_pad), float(xy[:, 0].max() + x_pad))
    y_lim = (float(xy[:, 1].min() - y_pad), float(xy[:, 1].max() + y_pad))

    imu_max = float(max(np.max(linear_speed), np.max(angular_speed), 1.0))
    planar_series = np.column_stack([p[:, 0], p[:, 1], data['xy_offset']])
    pose_min = float(np.min(planar_series))
    pose_max = float(np.max(planar_series))
    pose_pad = max(5.0, 0.08 * max(1.0, pose_max - pose_min))

    frame_ids = np.unique(np.linspace(0, n - 1, num=min(max_frames, n), dtype=int))
    frames: list[Image.Image] = []

    for idx in frame_ids:
        fig = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)
        ax_xy = fig.add_subplot(gs[:, 0])
        ax_imu = fig.add_subplot(gs[0, 1])
        ax_pose = fig.add_subplot(gs[1, 1])

        ax_xy.plot(xy[:, 0], xy[:, 1], color='0.88', linewidth=1.1, zorder=0, label='Full Trajectory')
        if idx >= 1:
            segments = np.stack([xy[:idx], xy[1 : idx + 1]], axis=1)
            lc = LineCollection(segments, cmap='viridis', linewidths=2.8, alpha=0.98)
            lc.set_array(data['t_norm'][:idx])
            ax_xy.add_collection(lc)

        q_idx = np.linspace(0, idx, num=min(16, idx + 1), dtype=int)
        if q_idx.size > 0:
            ax_xy.quiver(
                xy[q_idx, 0],
                xy[q_idx, 1],
                np.cos(yaw[q_idx]),
                np.sin(yaw[q_idx]),
                color='0.15',
                alpha=0.35,
                scale=28,
                width=0.0022,
                zorder=2,
            )

        ax_xy.scatter(p[0, 0], p[0, 1], marker='s', s=55, color='tab:green', edgecolors='black', linewidths=0.5, zorder=4)
        ax_xy.scatter(p[idx, 0], p[idx, 1], marker='o', s=72, color='gold', edgecolors='black', linewidths=0.7, zorder=5)
        ax_xy.scatter(p[-1, 0], p[-1, 1], marker='*', s=110, color='tab:red', edgecolors='black', linewidths=0.5, alpha=0.5, zorder=3)
        ax_xy.set_title(f'{title} | Top-Down XY Over Time')
        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_xlim(*x_lim)
        ax_xy.set_ylim(*y_lim)
        ax_xy.axis('equal')
        ax_xy.grid(True, alpha=0.28)

        stats = (
            f'Step: {idx + 1}/{n}\n'
            f'Time: {t_rel[idx]:.1f} s\n'
            f'XY Position: ({p[idx, 0]:.1f}, {p[idx, 1]:.1f}) m\n'
            f'XY Path So Far: {data["xy_path_cum"][idx]:.1f} m\n'
            f'XY Offset So Far: {data["xy_offset"][idx]:.1f} m\n'
            f'Heading: {yaw_deg[idx]:.1f} deg\n'
            f'|v_t|: {linear_speed[idx]:.2f} m/s\n'
            f'|ω_t|: {angular_speed[idx]:.2f} rad/s'
        )
        ax_xy.text(
            0.02,
            0.98,
            stats,
            transform=ax_xy.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.75', alpha=0.96),
        )

        legend_handles = [
            Line2D([0], [0], marker='s', color='none', markerfacecolor='tab:green', markeredgecolor='black', markersize=8, label='Start'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='gold', markeredgecolor='black', markersize=8, label='Current'),
            Line2D([0], [0], marker='*', color='none', markerfacecolor='tab:red', markeredgecolor='black', markersize=12, alpha=0.6, label='Final Endpoint'),
        ]
        ax_xy.legend(handles=legend_handles, loc='lower right', frameon=True, title='Trajectory Markers')

        ax_imu.plot(t_rel, linear_speed, color='tab:blue', linewidth=1.4, label=r'$\|v_t\|$')
        ax_imu.plot(t_rel, angular_speed, color='tab:orange', linewidth=1.2, label=r'$\|\omega_t\|$')
        ax_imu.axvline(t_rel[idx], color='0.2', linestyle='--', linewidth=1.0)
        ax_imu.scatter(t_rel[idx], linear_speed[idx], color='tab:blue', s=24, zorder=3)
        ax_imu.scatter(t_rel[idx], angular_speed[idx], color='tab:orange', s=24, zorder=3)
        ax_imu.set_title('IMU Inputs Driving The Curve')
        ax_imu.set_xlabel('time [s]')
        ax_imu.set_ylabel('m/s and rad/s')
        ax_imu.set_xlim(float(t_rel[0]), float(t_rel[-1]) if n > 1 else 1.0)
        ax_imu.set_ylim(0.0, imu_max * 1.08)
        ax_imu.grid(True, alpha=0.28)
        ax_imu.legend(loc='upper right', fontsize=8)

        ax_pose.plot(t_rel, p[:, 0], linewidth=1.4, label='x')
        ax_pose.plot(t_rel, p[:, 1], linewidth=1.4, label='y')
        ax_pose.plot(t_rel, data['xy_offset'], linewidth=1.2, linestyle='--', color='0.35', label='xy offset')
        ax_pose.axvline(t_rel[idx], color='0.2', linestyle='--', linewidth=1.0)
        ax_pose.scatter(t_rel[idx], p[idx, 0], s=22, zorder=3)
        ax_pose.scatter(t_rel[idx], p[idx, 1], s=22, zorder=3)
        ax_pose.scatter(t_rel[idx], data['xy_offset'][idx], s=22, color='0.25', zorder=3)
        ax_pose.set_title('Planar Position Summary')
        ax_pose.set_xlabel('time [s]')
        ax_pose.set_ylabel('meters')
        ax_pose.set_xlim(float(t_rel[0]), float(t_rel[-1]) if n > 1 else 1.0)
        ax_pose.set_ylim(pose_min - pose_pad, pose_max + pose_pad)
        ax_pose.grid(True, alpha=0.28)
        ax_pose.legend(loc='upper left', fontsize=8)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(frame[:, :, :3]))
        plt.close(fig)

    if not frames:
        return

    _ensure_parent(output_path)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


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
    valid_ids = _landmark_display_ids(world_T_imu, landmarks_w, initialized)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p[:, 0], p[:, 1], "r-", linewidth=2.5, label="trajectory", zorder=3)

    if valid_ids.size > 0:
        ax.scatter(lm[valid_ids, 0], lm[valid_ids, 1], s=6, c="tab:blue", alpha=0.72, label="landmarks", zorder=2)

    ax.scatter(p[0, 0], p[0, 1], marker="s", s=56, label="start", zorder=4)
    ax.scatter(p[-1, 0], p[-1, 1], marker="o", s=56, label="end", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    if valid_ids.size > 0:
        min_xy, max_xy = _compute_xy_bounds(world_T_imu, landmarks_w, initialized, display_ids=valid_ids)
        ax.set_xlim(float(min_xy[0]), float(max_xy[0]))
        ax.set_ylim(float(min_xy[1]), float(max_xy[1]))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _vis_to_uint8_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
        if v_max <= v_min:
            return np.zeros_like(arr, dtype=np.uint8)
        return (255.0 * (arr - v_min) / (v_max - v_min)).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        import cv2

        if arr.dtype != np.uint8:
            arr = _vis_to_uint8_gray(arr[..., 0])
            return arr
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return _vis_to_uint8_gray(arr[..., 0])
    raise ValueError(f'Unsupported image shape {arr.shape}')


def _iter_gray_frames_from_arrays(left_images: np.ndarray) -> Iterator[np.ndarray]:
    frames = np.asarray(left_images)
    for idx in range(int(frames.shape[0])):
        yield _vis_to_uint8_gray(frames[idx])


def _iter_gray_frames_from_video(left_video_path: Path) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(left_video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {left_video_path}')
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield _vis_to_uint8_gray(frame)
    finally:
        cap.release()


def _sample_indices(count: int, target: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, num=min(count, target), dtype=int))


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    if w == width:
        return image
    scale = float(width) / float(max(w, 1))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def _make_planar_trajectory_panel(
    primary_trajectory: np.ndarray,
    frame_idx: int,
    title: str,
    subtitle: str,
    panel_size: Tuple[int, int] = (240, 236),
    secondary_trajectory: np.ndarray | None = None,
    primary_color: Tuple[int, int, int] = (88, 196, 255),
    secondary_color: Tuple[int, int, int] = (72, 72, 72),
    current_primary_color: Tuple[int, int, int] = (255, 220, 90),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)

    primary = np.asarray(primary_trajectory, dtype=np.float64)
    if primary.ndim != 3 or primary.shape[0] == 0:
        return panel
    primary_xy = primary[:, :2, 3]
    idx = int(np.clip(frame_idx, 0, primary_xy.shape[0] - 1))

    secondary_xy = None
    if secondary_trajectory is not None:
        secondary = np.asarray(secondary_trajectory, dtype=np.float64)
        if secondary.ndim == 3 and secondary.shape[0] > 0:
            secondary_xy = secondary[: min(secondary.shape[0], primary_xy.shape[0]), :2, 3]

    draw_x0 = 10
    draw_y0 = 54
    draw_w = panel_w - 20
    draw_h = panel_h - draw_y0 - 34

    pts = [primary_xy]
    if secondary_xy is not None:
        pts.append(secondary_xy)
    stacked = np.vstack(pts)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    span_xy = np.maximum(max_xy - min_xy, 1.0)
    scale = 0.90 * min(draw_w / span_xy[0], draw_h / span_xy[1])

    def map_xy(pt: np.ndarray) -> Tuple[int, int]:
        u = int(round(draw_x0 + 0.5 * draw_w + scale * float(pt[0] - center_xy[0])))
        v = int(round(draw_y0 + 0.5 * draw_h - scale * float(pt[1] - center_xy[1])))
        return u, v

    for frac in np.linspace(0.0, 1.0, 5):
        gx = int(round(draw_x0 + frac * draw_w))
        gy = int(round(draw_y0 + frac * draw_h))
        cv2.line(panel, (gx, draw_y0), (gx, draw_y0 + draw_h), (36, 36, 36), 1, cv2.LINE_AA)
        cv2.line(panel, (draw_x0, gy), (draw_x0 + draw_w, gy), (36, 36, 36), 1, cv2.LINE_AA)

    if secondary_xy is not None and secondary_xy.shape[0] >= 2:
        sec_pts = np.array([map_xy(pt) for pt in secondary_xy], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [sec_pts], False, secondary_color, 1, cv2.LINE_AA)
        sec_cur_pt = map_xy(secondary_xy[min(idx, secondary_xy.shape[0] - 1)])
        cv2.circle(panel, sec_cur_pt, 4, secondary_color, -1, cv2.LINE_AA)

    all_pts = np.array([map_xy(pt) for pt in primary_xy], dtype=np.int32).reshape(-1, 1, 2)
    if all_pts.shape[0] >= 2:
        cv2.polylines(panel, [all_pts], False, (44, 72, 108), 1, cv2.LINE_AA)
    past_pts = np.array([map_xy(pt) for pt in primary_xy[: idx + 1]], dtype=np.int32).reshape(-1, 1, 2)
    if past_pts.shape[0] >= 2:
        cv2.polylines(panel, [past_pts], False, primary_color, 2, cv2.LINE_AA)

    start_pt = map_xy(primary_xy[0])
    cur_pt = map_xy(primary_xy[idx])
    end_pt = map_xy(primary_xy[-1])
    cv2.drawMarker(panel, start_pt, (90, 210, 110), cv2.MARKER_SQUARE, 10, 2, cv2.LINE_AA)
    cv2.circle(panel, cur_pt, 6, current_primary_color, -1, cv2.LINE_AA)
    cv2.drawMarker(panel, end_pt, (255, 110, 110), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    yaw = float(np.arctan2(primary[idx, 1, 0], primary[idx, 0, 0]))
    arrow_len = 15
    arrow_tip = (
        int(round(cur_pt[0] + arrow_len * np.cos(yaw))),
        int(round(cur_pt[1] - arrow_len * np.sin(yaw))),
    )
    cv2.arrowedLine(panel, cur_pt, arrow_tip, current_primary_color, 2, cv2.LINE_AA, tipLength=0.32)

    cur_xy = primary_xy[idx]
    cv2.putText(panel, f'Step {idx + 1}/{primary_xy.shape[0]}', (12, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, f'XY ({cur_xy[0]:.1f}, {cur_xy[1]:.1f}) m', (12, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (188, 188, 188), 1, cv2.LINE_AA)
    return panel


def _compute_xy_bounds(
    trajectory: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    display_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    traj_xy = np.asarray(trajectory[:, :2, 3], dtype=np.float64)
    pts = [traj_xy]
    lm = np.asarray(landmarks_w, dtype=np.float64)
    init = np.asarray(initialized, dtype=bool)
    if display_ids is not None and np.asarray(display_ids).size > 0:
        pts.append(lm[np.asarray(display_ids, dtype=np.int64), :2])
    else:
        finite_lm = init & np.isfinite(lm).all(axis=1)
        if np.any(finite_lm):
            pts.append(lm[finite_lm, :2])
    stacked = np.vstack(pts)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    pad = np.maximum(5.0, 0.07 * span)
    return min_xy - pad, max_xy + pad


def _landmark_display_ids(
    trajectory: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    lo_pct: float = 0.2,
    hi_pct: float = 99.8,
) -> np.ndarray:
    lm = np.asarray(landmarks_w, dtype=np.float64)
    init = np.asarray(initialized, dtype=bool)
    valid_ids = np.where(init & np.isfinite(lm).all(axis=1))[0]
    if valid_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if valid_ids.size <= 600:
        return valid_ids

    traj_xy = np.asarray(trajectory[:, :2, 3], dtype=np.float64)
    traj_min = np.min(traj_xy, axis=0)
    traj_max = np.max(traj_xy, axis=0)
    traj_span = np.maximum(traj_max - traj_min, 1.0)

    pts = lm[valid_ids, :2]
    q_lo = np.percentile(pts, lo_pct, axis=0)
    q_hi = np.percentile(pts, hi_pct, axis=0)
    margin = np.maximum(25.0, 0.45 * traj_span)
    low = np.minimum(q_lo, traj_min - margin)
    high = np.maximum(q_hi, traj_max + margin)

    keep = np.all((pts >= low) & (pts <= high), axis=1)
    kept_ids = valid_ids[keep]
    if kept_ids.size >= 600:
        return kept_ids
    return valid_ids


def _make_landmark_observation_panel(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    z_frame: np.ndarray,
    pose: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    projector: StereoProjector,
    frame_idx: int,
    elapsed_s: float,
    accepted_updates: int,
    frame_mean_err_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    left_rgb = cv2.cvtColor(_vis_to_uint8_gray(left_frame), cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(_vis_to_uint8_gray(right_frame), cv2.COLOR_GRAY2RGB)
    h, w = left_rgb.shape[:2]
    gap = 12
    panel = np.full((h, 2 * w + gap, 3), 18, dtype=np.uint8)
    panel[:, :w] = left_rgb
    panel[:, w + gap :] = right_rgb
    valid_visible = np.where(
        np.all(z_frame >= 0.0, axis=0)
        & np.asarray(initialized, dtype=bool)
        & np.isfinite(np.asarray(landmarks_w, dtype=np.float64)).all(axis=1)
    )[0]
    draw_ids = valid_visible[_sample_indices(valid_visible.size, 96)]
    current_visible_ids: list[int] = []

    for lid in draw_ids:
        z_hat = projector.predict_stereo(pose, landmarks_w[int(lid)])
        if not np.isfinite(z_hat).all():
            continue
        p_l_meas = (int(round(float(z_frame[0, lid]))), int(round(float(z_frame[1, lid]))))
        p_l_pred = (int(round(float(z_hat[0]))), int(round(float(z_hat[1]))))
        p_r_meas = (int(round(float(z_frame[2, lid] + w + gap))), int(round(float(z_frame[3, lid]))))
        p_r_pred = (int(round(float(z_hat[2] + w + gap))), int(round(float(z_hat[3]))))
        cv2.line(panel, p_l_meas, p_l_pred, (92, 220, 120), 1, cv2.LINE_AA)
        cv2.line(panel, p_r_meas, p_r_pred, (92, 220, 120), 1, cv2.LINE_AA)
        cv2.circle(panel, p_l_meas, 3, (92, 214, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, p_r_meas, 3, (255, 166, 84), -1, cv2.LINE_AA)
        cv2.drawMarker(panel, p_l_pred, (255, 230, 120), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        cv2.drawMarker(panel, p_r_pred, (255, 230, 120), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        current_visible_ids.append(int(lid))

    overlay = panel.copy()
    cv2.rectangle(overlay, (10, 10), (334, 112), (8, 8, 8), -1)
    panel = cv2.addWeighted(overlay, 0.64, panel, 0.36, 0.0)
    text_color = (245, 245, 245)
    subtle = (188, 188, 188)
    cv2.putText(panel, f'Frame {frame_idx:04d}', (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)
    cv2.putText(panel, f'Elapsed {elapsed_s:6.1f} s', (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, text_color, 2, cv2.LINE_AA)
    cv2.putText(panel, f'Initialized + visible {len(current_visible_ids):3d}', (22, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.56, text_color, 2, cv2.LINE_AA)
    if np.isfinite(frame_mean_err_px):
        err_text = f'Frame mean reproj err {frame_mean_err_px:5.1f} px'
    else:
        err_text = 'Frame mean reproj err   n/a'
    cv2.putText(panel, err_text, (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.54, subtle, 2, cv2.LINE_AA)
    cv2.putText(panel, f'Accepted EKF updates {accepted_updates:3d}', (panel.shape[1] - 262, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (110, 230, 150), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Left camera: measured vs projected landmarks', (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (92, 214, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Right camera: measured vs projected landmarks', (w + gap + 16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 166, 84), 2, cv2.LINE_AA)
    return panel, np.asarray(current_visible_ids, dtype=np.int64)


def _make_landmark_map_panel(
    trajectory: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    display_ids: np.ndarray,
    current_visible_ids: np.ndarray,
    frame_idx: int,
    bounds_min_xy: np.ndarray,
    bounds_max_xy: np.ndarray,
    panel_size: Tuple[int, int] = (320, 236),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'Landmark Map Evolution', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Fixed IMU path with evolving EKF landmarks', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)

    draw_x0 = 10
    draw_y0 = 54
    draw_w = panel_w - 20
    draw_h = panel_h - draw_y0 - 32

    def map_xy(pt_xy: np.ndarray) -> Tuple[int, int]:
        span = np.maximum(bounds_max_xy - bounds_min_xy, 1e-6)
        u = draw_x0 + int(round((float(pt_xy[0]) - float(bounds_min_xy[0])) / float(span[0]) * draw_w))
        v = draw_y0 + draw_h - int(round((float(pt_xy[1]) - float(bounds_min_xy[1])) / float(span[1]) * draw_h))
        return u, v

    for frac in np.linspace(0.0, 1.0, 5):
        gx = draw_x0 + int(round(frac * draw_w))
        gy = draw_y0 + int(round(frac * draw_h))
        cv2.line(panel, (gx, draw_y0), (gx, draw_y0 + draw_h), (36, 36, 36), 1, cv2.LINE_AA)
        cv2.line(panel, (draw_x0, gy), (draw_x0 + draw_w, gy), (36, 36, 36), 1, cv2.LINE_AA)

    traj_xy = np.asarray(trajectory[:, :2, 3], dtype=np.float64)
    all_traj_pts = np.array([map_xy(pt) for pt in traj_xy], dtype=np.int32).reshape(-1, 1, 2)
    if all_traj_pts.shape[0] >= 2:
        cv2.polylines(panel, [all_traj_pts], False, (72, 72, 72), 1, cv2.LINE_AA)

    past_traj_pts = np.array([map_xy(pt) for pt in traj_xy[: frame_idx + 1]], dtype=np.int32).reshape(-1, 1, 2)
    if past_traj_pts.shape[0] >= 2:
        cv2.polylines(panel, [past_traj_pts], False, (88, 196, 255), 2, cv2.LINE_AA)

    lm = np.asarray(landmarks_w, dtype=np.float64)
    init = np.asarray(initialized, dtype=bool) & np.isfinite(lm).all(axis=1)
    display_active = 0
    for lid in display_ids:
        lid_i = int(lid)
        if 0 <= lid_i < lm.shape[0] and init[lid_i]:
            pt = lm[lid_i, :2]
            cv2.circle(panel, map_xy(pt), 2, (110, 130, 255), -1, cv2.LINE_AA)
            display_active += 1
    for lid in np.asarray(current_visible_ids, dtype=np.int64):
        if 0 <= int(lid) < lm.shape[0] and init[int(lid)]:
            cv2.circle(panel, map_xy(lm[int(lid), :2]), 3, (255, 220, 90), -1, cv2.LINE_AA)

    start_pt = map_xy(traj_xy[0])
    cur_pose = trajectory[min(frame_idx, trajectory.shape[0] - 1)]
    cur_pt = map_xy(traj_xy[min(frame_idx, traj_xy.shape[0] - 1)])
    end_pt = map_xy(traj_xy[-1])
    cv2.drawMarker(panel, start_pt, (90, 210, 110), cv2.MARKER_SQUARE, 10, 2, cv2.LINE_AA)
    cv2.circle(panel, cur_pt, 5, (255, 220, 90), -1, cv2.LINE_AA)
    cv2.drawMarker(panel, end_pt, (255, 110, 110), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
    yaw = float(np.arctan2(cur_pose[1, 0], cur_pose[0, 0]))
    arrow_len = 14
    arrow_tip = (
        int(round(cur_pt[0] + arrow_len * np.cos(yaw))),
        int(round(cur_pt[1] - arrow_len * np.sin(yaw))),
    )
    cv2.arrowedLine(panel, cur_pt, arrow_tip, (255, 220, 90), 2, cv2.LINE_AA, tipLength=0.35)

    total_mapped = int(np.sum(init))
    cv2.putText(panel, f'Showing {display_active} / {total_mapped} landmarks', (12, panel_h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Blue: IMU path', (12, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (88, 196, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Purple: mapped landmarks', (108, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 130, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Yellow: robot + current obs', (12, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 90), 1, cv2.LINE_AA)
    return panel


def _draw_series_chart(
    panel: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    title: str,
    footer: str,
) -> None:
    import cv2

    x0, y0 = top_left
    w, h = size
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (62, 62, 62), 1)
    cv2.putText(panel, title, (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    chart_x0 = x0 + 10
    chart_x1 = x0 + w - 10
    chart_y0 = y0 + 28
    chart_y1 = y0 + h - 12
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    span = max(v_max - v_min, 1.0)
    xs = np.linspace(chart_x0, chart_x1, num=values.size).astype(np.int32)
    ys = chart_y1 - ((values - v_min) / span * (chart_y1 - chart_y0)).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], False, (68, 68, 68), 1, cv2.LINE_AA)
    if frame_idx > 0:
        cv2.polylines(panel, [pts[: frame_idx + 1]], False, color, 2, cv2.LINE_AA)
    cursor_x = int(xs[min(frame_idx, values.size - 1)])
    cv2.line(panel, (cursor_x, chart_y0), (cursor_x, chart_y1), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, footer, (x0 + 10, y0 + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _draw_stat_card(
    panel: np.ndarray,
    rect: Tuple[int, int, int, int],
    title: str,
    value_text: str,
    sub_text: str,
    accent: Tuple[int, int, int],
) -> None:
    import cv2

    x0, y0, w, h = rect
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (44, 44, 44), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (92, 92, 92), 1)
    cv2.rectangle(panel, (x0, y0), (x0 + 5, y0 + h), accent, -1)
    cv2.putText(panel, title, (x0 + 12, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, value_text, (x0 + 12, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, accent, 2, cv2.LINE_AA)
    cv2.putText(panel, sub_text, (x0 + 12, y0 + h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA)


def _make_landmark_stats_panel(
    map_result: LandmarkMapResult,
    visible_now: int,
    frame_idx: int,
    panel_size: Tuple[int, int] = (440, 236),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'Landmark EKF Over Time', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)

    initialized_count = np.asarray(map_result.initialized_count_per_frame, dtype=np.float64)
    accepted_updates = np.asarray(map_result.accepted_updates_per_frame, dtype=np.float64)
    current_err = float(map_result.mean_reprojection_error_per_frame[min(frame_idx, map_result.mean_reprojection_error_per_frame.shape[0] - 1)])

    chart_y0 = 28
    chart_h = 62
    _draw_series_chart(
        panel,
        initialized_count,
        frame_idx,
        (10, chart_y0),
        (panel_w - 20, chart_h),
        (110, 130, 255),
        'Initialized landmarks',
        f'now {int(initialized_count[min(frame_idx, initialized_count.size - 1)])} / final {int(initialized_count[-1])}',
    )
    _draw_series_chart(
        panel,
        accepted_updates,
        frame_idx,
        (10, chart_y0 + chart_h + 8),
        (panel_w - 20, chart_h),
        (110, 230, 150),
        'Accepted EKF updates per frame',
        f'now {int(accepted_updates[min(frame_idx, accepted_updates.size - 1)])} / peak {int(np.max(accepted_updates))}',
    )

    card_y0 = chart_y0 + 2 * chart_h + 22
    card_gap = 8
    card_w = (panel_w - 20 - 3 * card_gap) // 4
    card_h = panel_h - card_y0 - 10
    _draw_stat_card(panel, (10, card_y0, card_w, card_h), 'Visible Now', f'{int(visible_now)}', 'stereo obs', (92, 214, 255))
    _draw_stat_card(panel, (10 + (card_w + card_gap), card_y0, card_w, card_h), 'Initialized', f'{int(initialized_count[min(frame_idx, initialized_count.size - 1)])}', f'final {int(initialized_count[-1])}', (110, 130, 255))
    _draw_stat_card(panel, (10 + 2 * (card_w + card_gap), card_y0, card_w, card_h), 'Updates', f'{int(accepted_updates[min(frame_idx, accepted_updates.size - 1)])}', f'peak {int(np.max(accepted_updates))}', (110, 230, 150))
    err_text = 'n/a' if not np.isfinite(current_err) else f'{current_err:.1f}'
    err_sub = 'frame reproj px'
    _draw_stat_card(panel, (10 + 3 * (card_w + card_gap), card_y0, card_w, card_h), 'Mean Err', err_text, err_sub, (255, 182, 90))
    return panel


def _compose_landmark_dashboard(
    observation_panel: np.ndarray,
    map_panel: np.ndarray,
    trajectory_panel: np.ndarray,
    stats_panel: np.ndarray,
) -> np.ndarray:
    import cv2

    dashboard_width = 900
    observation_resized = _resize_to_width(observation_panel, dashboard_width)
    bottom_gap = 10
    bottom_h = max(map_panel.shape[0], trajectory_panel.shape[0], stats_panel.shape[0])
    dashboard = np.full((observation_resized.shape[0] + bottom_gap + bottom_h, dashboard_width, 3), 10, dtype=np.uint8)
    dashboard[: observation_resized.shape[0], :dashboard_width] = observation_resized
    y0 = observation_resized.shape[0] + bottom_gap
    x_map = 0
    x_traj = x_map + map_panel.shape[1] + 10
    x_stats = dashboard_width - stats_panel.shape[1]
    dashboard[y0 : y0 + map_panel.shape[0], x_map : x_map + map_panel.shape[1]] = map_panel
    dashboard[y0 : y0 + trajectory_panel.shape[0], x_traj : x_traj + trajectory_panel.shape[1]] = trajectory_panel
    dashboard[y0 : y0 + stats_panel.shape[0], x_stats : x_stats + stats_panel.shape[1]] = stats_panel
    cv2.line(dashboard, (x_traj - 5, y0 + 6), (x_traj - 5, y0 + bottom_h - 6), (44, 44, 44), 1, cv2.LINE_AA)
    cv2.line(dashboard, (x_stats - 5, y0 + 6), (x_stats - 5, y0 + bottom_h - 6), (44, 44, 44), 1, cv2.LINE_AA)
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1] - 1, dashboard.shape[0] - 1), (112, 112, 112), 1)
    return dashboard


def plot_landmark_mapping_stats(
    map_result: LandmarkMapResult,
    output_path: Path,
) -> None:
    init_count = np.asarray(map_result.initialized_count_per_frame, dtype=np.float64)
    updates = np.asarray(map_result.accepted_updates_per_frame, dtype=np.float64)
    errs = np.asarray(map_result.mean_reprojection_error_per_frame, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0].plot(init_count, color='tab:purple', linewidth=1.3)
    axes[0].set_title('Initialized Landmarks Over Time')
    axes[0].set_xlabel('frame index')
    axes[0].set_ylabel('count')
    axes[0].grid(True, alpha=0.28)

    axes[1].plot(updates, color='tab:green', linewidth=1.2)
    axes[1].set_title('Accepted Landmark EKF Updates Per Frame')
    axes[1].set_xlabel('frame index')
    axes[1].set_ylabel('count')
    axes[1].grid(True, alpha=0.28)

    finite = np.isfinite(errs)
    axes[2].plot(np.where(finite)[0], errs[finite], color='tab:orange', linewidth=1.1)
    axes[2].set_title('Frame Mean Reprojection Error')
    axes[2].set_xlabel('frame index')
    axes[2].set_ylabel('pixels')
    axes[2].grid(True, alpha=0.28)

    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_landmark_mapping_visuals_core(
    output_dir: Path,
    frame_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    map_result: LandmarkMapResult,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None = None,
    prefix: str = 'part3',
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f'{prefix}_landmark_mapping.gif'
    montage_path = output_dir / f'{prefix}_landmark_mapping_montage.png'
    stats_path = output_dir / f'{prefix}_landmark_stats.png'

    plot_landmark_mapping_stats(map_result, stats_path)

    snapshot_ids = np.asarray(map_result.snapshot_frame_ids, dtype=np.int64)
    if snapshot_ids.size == 0:
        return
    snapshot_lookup = {int(fid): idx for idx, fid in enumerate(snapshot_ids)}
    if snapshot_ids.size >= 6:
        montage_snapshot_indices = [int(i) for i in np.linspace(0, snapshot_ids.size - 1, num=6, dtype=int)]
    else:
        montage_snapshot_indices = list(range(int(snapshot_ids.size)))
        while len(montage_snapshot_indices) < 6 and montage_snapshot_indices:
            montage_snapshot_indices.append(montage_snapshot_indices[-1])
    montage_index_set = set(montage_snapshot_indices)
    montage_frames_by_index: dict[int, np.ndarray] = {}

    projector = StereoProjector(calib)
    display_ids = _landmark_display_ids(
        map_result.trajectory,
        map_result.landmarks_w,
        map_result.initialized,
    )
    bounds_min_xy, bounds_max_xy = _compute_xy_bounds(
        map_result.trajectory,
        map_result.landmarks_w,
        map_result.initialized,
        display_ids=display_ids,
    )

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == map_result.trajectory.shape[0]:
            ts = ts - float(ts[0])
        else:
            ts = None
    else:
        ts = None

    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('imageio is required to write Part 3 GIF outputs.') from exc

    with imageio.get_writer(gif_path, mode='I', duration=0.05, loop=0, palettesize=32, quantizer=0) as writer:
        for idx, (left_frame, right_frame) in enumerate(frame_iter):
            snap_idx = snapshot_lookup.get(int(idx))
            if snap_idx is None:
                continue
            elapsed_s = float(ts[idx]) if ts is not None else float(idx)
            z_frame = np.asarray(features[:, :, idx], dtype=np.float64)
            landmarks_snapshot = np.asarray(map_result.landmark_snapshots_w[snap_idx], dtype=np.float64)
            initialized_snapshot = np.asarray(map_result.initialized_snapshots[snap_idx], dtype=bool)

            observation_panel, current_visible_ids = _make_landmark_observation_panel(
                left_frame=left_frame,
                right_frame=right_frame,
                z_frame=z_frame,
                pose=map_result.trajectory[idx],
                landmarks_w=landmarks_snapshot,
                initialized=initialized_snapshot,
                projector=projector,
                frame_idx=int(idx),
                elapsed_s=elapsed_s,
                accepted_updates=int(map_result.accepted_updates_per_frame[idx]),
                frame_mean_err_px=float(map_result.mean_reprojection_error_per_frame[idx]),
            )
            map_panel = _make_landmark_map_panel(
                trajectory=map_result.trajectory,
                landmarks_w=landmarks_snapshot,
                initialized=initialized_snapshot,
                display_ids=display_ids,
                current_visible_ids=current_visible_ids,
                frame_idx=int(idx),
                bounds_min_xy=bounds_min_xy,
                bounds_max_xy=bounds_max_xy,
            )
            trajectory_panel = _make_planar_trajectory_panel(
                primary_trajectory=map_result.trajectory,
                frame_idx=int(idx),
                title='Robot Trajectory',
                subtitle='Fixed IMU path with current robot heading',
                panel_size=(250, 236),
            )
            stats_panel = _make_landmark_stats_panel(
                map_result=map_result,
                visible_now=int(current_visible_ids.size),
                frame_idx=int(idx),
                panel_size=(320, 236),
            )
            dashboard = _compose_landmark_dashboard(observation_panel, map_panel, trajectory_panel, stats_panel)
            writer.append_data(np.asarray(dashboard, dtype=np.uint8))
            if snap_idx in montage_index_set:
                montage_frames_by_index[int(snap_idx)] = dashboard.copy()

    montage_frames: list[np.ndarray] = []
    if montage_snapshot_indices:
        available_indices = sorted(montage_frames_by_index)
        for snap_idx in montage_snapshot_indices:
            frame = montage_frames_by_index.get(int(snap_idx))
            if frame is None and available_indices:
                nearest_idx = min(available_indices, key=lambda idx: abs(idx - int(snap_idx)))
                frame = montage_frames_by_index.get(int(nearest_idx))
            if frame is not None:
                montage_frames.append(frame)

    if montage_frames:
        cols = 3
        rows = 2
        tile_h, tile_w = montage_frames[0].shape[:2]
        board = np.full((rows * tile_h, cols * tile_w, 3), 12, dtype=np.uint8)
        for k, frame in enumerate(montage_frames):
            r = k // cols
            c = k % cols
            board[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = frame
        Image.fromarray(board).save(montage_path)


def save_landmark_mapping_visuals(
    output_dir: Path,
    left_images: np.ndarray,
    right_images: np.ndarray,
    map_result: LandmarkMapResult,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None = None,
    prefix: str = 'part3',
) -> None:
    _save_landmark_mapping_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_arrays(left_images, right_images),
        map_result=map_result,
        features=features,
        calib=calib,
        timestamps=timestamps,
        prefix=prefix,
    )


def save_landmark_mapping_visuals_from_videos(
    output_dir: Path,
    left_video_path: Path,
    right_video_path: Path,
    map_result: LandmarkMapResult,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None = None,
    prefix: str = 'part3',
) -> None:
    left_count = _video_frame_count(left_video_path)
    right_count = _video_frame_count(right_video_path)
    frame_count = min(
        c for c in (
            left_count,
            right_count,
            int(map_result.trajectory.shape[0]),
            int(features.shape[2]),
        ) if c is not None
    )
    snap_ids = np.asarray(map_result.snapshot_frame_ids, dtype=np.int64)
    snap_mask = snap_ids < frame_count
    trimmed_result = LandmarkMapResult(
        landmarks_w=np.asarray(map_result.landmarks_w, dtype=np.float64),
        covariances=np.asarray(map_result.covariances, dtype=np.float64),
        initialized=np.asarray(map_result.initialized, dtype=bool),
        mean_reprojection_error_px=float(map_result.mean_reprojection_error_px),
        trajectory=np.asarray(map_result.trajectory[:frame_count], dtype=np.float64),
        initialized_count_per_frame=np.asarray(map_result.initialized_count_per_frame[:frame_count]),
        accepted_updates_per_frame=np.asarray(map_result.accepted_updates_per_frame[:frame_count]),
        mean_reprojection_error_per_frame=np.asarray(map_result.mean_reprojection_error_per_frame[:frame_count]),
        snapshot_frame_ids=snap_ids[snap_mask],
        landmark_snapshots_w=np.asarray(map_result.landmark_snapshots_w[snap_mask], dtype=np.float32),
        initialized_snapshots=np.asarray(map_result.initialized_snapshots[snap_mask], dtype=bool),
    )
    _save_landmark_mapping_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_videos(left_video_path, right_video_path),
        map_result=trimmed_result,
        features=features[:, :, :frame_count],
        calib=calib,
        timestamps=None if timestamps is None else timestamps[:frame_count],
        prefix=prefix,
    )


def _iter_gray_stereo_pairs_from_arrays(left_images: np.ndarray, right_images: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    left = np.asarray(left_images)
    right = np.asarray(right_images)
    n = int(min(left.shape[0], right.shape[0]))
    for idx in range(n):
        yield _vis_to_uint8_gray(left[idx]), _vis_to_uint8_gray(right[idx])


def _iter_gray_stereo_pairs_from_videos(left_video_path: Path, right_video_path: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    import cv2

    cap_left = cv2.VideoCapture(str(left_video_path))
    cap_right = cv2.VideoCapture(str(right_video_path))
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError(f'Unable to open stereo videos: {left_video_path}, {right_video_path}')
    try:
        while True:
            ok_left, frame_left = cap_left.read()
            ok_right, frame_right = cap_right.read()
            if not ok_left or not ok_right or frame_left is None or frame_right is None:
                break
            yield _vis_to_uint8_gray(frame_left), _vis_to_uint8_gray(frame_right)
    finally:
        cap_left.release()
        cap_right.release()


def _video_frame_count(video_path: Path) -> int | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None
    finally:
        cap.release()


def _make_vi_slam_stereo_panel(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    z_frame: np.ndarray,
    pose: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    projector: StereoProjector,
    frame_idx: int,
    elapsed_s: float,
    accepted_joint_updates: int,
    frame_pose_err_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    left_rgb = cv2.cvtColor(_vis_to_uint8_gray(left_frame), cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(_vis_to_uint8_gray(right_frame), cv2.COLOR_GRAY2RGB)
    h, w = left_rgb.shape[:2]
    gap = 12
    canvas = np.full((h, 2 * w + gap, 3), 18, dtype=np.uint8)
    canvas[:, :w] = left_rgb
    canvas[:, w + gap :] = right_rgb

    valid_visible = np.where(
        np.all(z_frame >= 0.0, axis=0)
        & np.asarray(initialized, dtype=bool)
        & np.isfinite(np.asarray(landmarks_w, dtype=np.float64)).all(axis=1)
    )[0]
    draw_ids = valid_visible[_sample_indices(valid_visible.size, 96)]
    current_visible_ids: list[int] = []

    left_meas_color = (92, 214, 255)
    right_meas_color = (255, 166, 84)
    pred_color = (255, 230, 120)
    residual_color = (110, 225, 145)

    for lid in draw_ids:
        z_hat = projector.predict_stereo(pose, landmarks_w[int(lid)])
        if not np.isfinite(z_hat).all():
            continue
        lx, ly, rx, ry = z_frame[:, int(lid)]
        p_l_meas = (int(round(float(lx))), int(round(float(ly))))
        p_l_pred = (int(round(float(z_hat[0]))), int(round(float(z_hat[1]))))
        p_r_meas = (int(round(float(rx + w + gap))), int(round(float(ry))))
        p_r_pred = (int(round(float(z_hat[2] + w + gap))), int(round(float(z_hat[3]))))

        cv2.line(canvas, p_l_meas, p_l_pred, residual_color, 1, cv2.LINE_AA)
        cv2.line(canvas, p_r_meas, p_r_pred, residual_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p_l_meas, 3, left_meas_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_r_meas, 3, right_meas_color, -1, cv2.LINE_AA)
        cv2.drawMarker(canvas, p_l_pred, pred_color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, p_r_pred, pred_color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        current_visible_ids.append(int(lid))

    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (334, 112), (8, 8, 8), -1)
    canvas = cv2.addWeighted(overlay, 0.64, canvas, 0.36, 0.0)
    text_color = (245, 245, 245)
    subtle = (188, 188, 188)
    cv2.putText(canvas, f'Frame {frame_idx:04d}', (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Elapsed {elapsed_s:6.1f} s', (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Visible selected landmarks {len(current_visible_ids):3d}', (22, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.54, text_color, 2, cv2.LINE_AA)
    pose_text = 'Frame pose residual   n/a' if not np.isfinite(frame_pose_err_px) else f'Frame pose residual {frame_pose_err_px:5.1f} px'
    cv2.putText(canvas, pose_text, (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, subtle, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Accepted joint updates {accepted_joint_updates:3d}', (canvas.shape[1] - 276, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (110, 230, 150), 2, cv2.LINE_AA)

    cv2.putText(canvas, 'Left camera: measured vs projected landmarks', (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, left_meas_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Right camera: measured vs projected landmarks', (w + gap + 16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, right_meas_color, 2, cv2.LINE_AA)
    return canvas, np.asarray(current_visible_ids, dtype=np.int64)


def _compute_vi_slam_bounds(
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    display_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_xy, max_xy = _compute_xy_bounds(world_T_imu_corr, landmarks_w, initialized, display_ids=display_ids)
    pred_xy = np.asarray(world_T_imu_pred[:, :2, 3], dtype=np.float64)
    min_xy = np.minimum(min_xy, np.min(pred_xy, axis=0) - 5.0)
    max_xy = np.maximum(max_xy, np.max(pred_xy, axis=0) + 5.0)
    span = np.maximum(max_xy - min_xy, 1.0)
    pad = np.maximum(5.0, 0.05 * span)
    return min_xy - pad, max_xy + pad


def _make_vi_slam_map_panel(
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    display_ids: np.ndarray,
    current_visible_ids: np.ndarray,
    frame_idx: int,
    bounds_min_xy: np.ndarray,
    bounds_max_xy: np.ndarray,
    panel_size: Tuple[int, int] = (320, 236),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'VI-SLAM Map Evolution', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Gray: IMU-only, Blue: corrected trajectory', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)

    draw_x0 = 10
    draw_y0 = 54
    draw_w = panel_w - 20
    draw_h = panel_h - draw_y0 - 32

    def map_xy(pt_xy: np.ndarray) -> Tuple[int, int]:
        span = np.maximum(bounds_max_xy - bounds_min_xy, 1e-6)
        u = draw_x0 + int(round((float(pt_xy[0]) - float(bounds_min_xy[0])) / float(span[0]) * draw_w))
        v = draw_y0 + draw_h - int(round((float(pt_xy[1]) - float(bounds_min_xy[1])) / float(span[1]) * draw_h))
        return u, v

    for frac in np.linspace(0.0, 1.0, 5):
        gx = draw_x0 + int(round(frac * draw_w))
        gy = draw_y0 + int(round(frac * draw_h))
        cv2.line(panel, (gx, draw_y0), (gx, draw_y0 + draw_h), (36, 36, 36), 1, cv2.LINE_AA)
        cv2.line(panel, (draw_x0, gy), (draw_x0 + draw_w, gy), (36, 36, 36), 1, cv2.LINE_AA)

    pred_xy = np.asarray(world_T_imu_pred[:, :2, 3], dtype=np.float64)
    corr_xy = np.asarray(world_T_imu_corr[:, :2, 3], dtype=np.float64)
    pred_pts = np.array([map_xy(pt) for pt in pred_xy], dtype=np.int32).reshape(-1, 1, 2)
    corr_all_pts = np.array([map_xy(pt) for pt in corr_xy], dtype=np.int32).reshape(-1, 1, 2)
    corr_past_pts = np.array([map_xy(pt) for pt in corr_xy[: frame_idx + 1]], dtype=np.int32).reshape(-1, 1, 2)
    if pred_pts.shape[0] >= 2:
        cv2.polylines(panel, [pred_pts], False, (72, 72, 72), 1, cv2.LINE_AA)
    if corr_all_pts.shape[0] >= 2:
        cv2.polylines(panel, [corr_all_pts], False, (40, 90, 150), 1, cv2.LINE_AA)
    if corr_past_pts.shape[0] >= 2:
        cv2.polylines(panel, [corr_past_pts], False, (88, 196, 255), 2, cv2.LINE_AA)

    lm = np.asarray(landmarks_w, dtype=np.float64)
    init = np.asarray(initialized, dtype=bool) & np.isfinite(lm).all(axis=1)
    display_active = 0
    for lid in np.asarray(display_ids, dtype=np.int64):
        lid_i = int(lid)
        if 0 <= lid_i < lm.shape[0] and init[lid_i]:
            cv2.circle(panel, map_xy(lm[lid_i, :2]), 2, (122, 120, 255), -1, cv2.LINE_AA)
            display_active += 1
    for lid in np.asarray(current_visible_ids, dtype=np.int64):
        lid_i = int(lid)
        if 0 <= lid_i < lm.shape[0] and init[lid_i]:
            cv2.circle(panel, map_xy(lm[lid_i, :2]), 3, (255, 220, 90), -1, cv2.LINE_AA)

    start_pt = map_xy(corr_xy[0])
    cur_pose = world_T_imu_corr[min(frame_idx, world_T_imu_corr.shape[0] - 1)]
    cur_pt = map_xy(corr_xy[min(frame_idx, corr_xy.shape[0] - 1)])
    end_pt = map_xy(corr_xy[-1])
    cv2.drawMarker(panel, start_pt, (90, 210, 110), cv2.MARKER_SQUARE, 10, 2, cv2.LINE_AA)
    cv2.circle(panel, cur_pt, 5, (255, 220, 90), -1, cv2.LINE_AA)
    cv2.drawMarker(panel, end_pt, (255, 110, 110), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
    yaw = float(np.arctan2(cur_pose[1, 0], cur_pose[0, 0]))
    arrow_len = 14
    arrow_tip = (
        int(round(cur_pt[0] + arrow_len * np.cos(yaw))),
        int(round(cur_pt[1] - arrow_len * np.sin(yaw))),
    )
    cv2.arrowedLine(panel, cur_pt, arrow_tip, (255, 220, 90), 2, cv2.LINE_AA, tipLength=0.35)

    total_mapped = int(np.count_nonzero(init))
    cv2.putText(panel, f'Showing {display_active} / {total_mapped} landmarks', (12, panel_h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Gray IMU-only  Blue corrected', (12, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Purple map  Yellow current obs', (12, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    return panel


def _make_vi_slam_stats_panel(
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    accepted_joint_updates_per_frame: np.ndarray,
    mean_pose_residual_per_frame: np.ndarray,
    initialized_count_per_frame: np.ndarray,
    mean_landmark_residual_per_frame: np.ndarray,
    visible_now: int,
    used_landmarks: int,
    frame_idx: int,
    panel_size: Tuple[int, int] = (440, 236),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'VI-SLAM Corrections Over Time', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)

    pred_xy = np.asarray(world_T_imu_pred[:, :2, 3], dtype=np.float64)
    corr_xy = np.asarray(world_T_imu_corr[:, :2, 3], dtype=np.float64)
    pose_shift = np.linalg.norm(corr_xy - pred_xy, axis=1)

    chart_y0 = 28
    chart_h = 62
    _draw_series_chart(
        panel,
        pose_shift,
        frame_idx,
        (10, chart_y0),
        (panel_w - 20, chart_h),
        (88, 196, 255),
        'XY correction from IMU-only',
        f'now {pose_shift[min(frame_idx, pose_shift.size - 1)]:.2f} m / max {np.max(pose_shift):.2f} m',
    )
    _draw_series_chart(
        panel,
        np.asarray(accepted_joint_updates_per_frame, dtype=np.float64),
        frame_idx,
        (10, chart_y0 + chart_h + 8),
        (panel_w - 20, chart_h),
        (110, 230, 150),
        'Accepted joint EKF updates per frame',
        f'now {int(accepted_joint_updates_per_frame[min(frame_idx, accepted_joint_updates_per_frame.shape[0] - 1)])} / peak {int(np.max(accepted_joint_updates_per_frame))}',
    )

    card_y0 = chart_y0 + 2 * chart_h + 22
    card_gap = 8
    card_w = (panel_w - 20 - 3 * card_gap) // 4
    card_h = panel_h - card_y0 - 10
    pose_err = float(mean_pose_residual_per_frame[min(frame_idx, mean_pose_residual_per_frame.shape[0] - 1)])
    map_err = float(mean_landmark_residual_per_frame[min(frame_idx, mean_landmark_residual_per_frame.shape[0] - 1)])
    init_count = int(initialized_count_per_frame[min(frame_idx, initialized_count_per_frame.shape[0] - 1)])

    _draw_stat_card(panel, (10, card_y0, card_w, card_h), 'Visible Now', f'{int(visible_now)}', 'selected obs', (92, 214, 255))
    _draw_stat_card(panel, (10 + (card_w + card_gap), card_y0, card_w, card_h), 'Used Map', f'{int(used_landmarks)}', f'init {init_count}', (122, 120, 255))
    pose_text = 'n/a' if not np.isfinite(pose_err) else f'{pose_err:.1f}'
    _draw_stat_card(panel, (10 + 2 * (card_w + card_gap), card_y0, card_w, card_h), 'Pose Err', pose_text, 'frame px', (255, 182, 90))
    map_text = 'n/a' if not np.isfinite(map_err) else f'{map_err:.1f}'
    _draw_stat_card(panel, (10 + 3 * (card_w + card_gap), card_y0, card_w, card_h), 'Map Err', map_text, 'frame px', (110, 230, 150))
    return panel


def _compose_vi_slam_dashboard(
    stereo_panel: np.ndarray,
    map_panel: np.ndarray,
    trajectory_panel: np.ndarray,
    stats_panel: np.ndarray,
) -> np.ndarray:
    import cv2

    dashboard_width = 900
    stereo_resized = _resize_to_width(stereo_panel, dashboard_width)
    bottom_gap = 10
    bottom_h = max(map_panel.shape[0], trajectory_panel.shape[0], stats_panel.shape[0])
    dashboard = np.full((stereo_resized.shape[0] + bottom_gap + bottom_h, dashboard_width, 3), 10, dtype=np.uint8)
    dashboard[: stereo_resized.shape[0], :dashboard_width] = stereo_resized
    y0 = stereo_resized.shape[0] + bottom_gap
    x_map = 0
    x_traj = x_map + map_panel.shape[1] + 10
    x_stats = dashboard_width - stats_panel.shape[1]
    dashboard[y0 : y0 + map_panel.shape[0], x_map : x_map + map_panel.shape[1]] = map_panel
    dashboard[y0 : y0 + trajectory_panel.shape[0], x_traj : x_traj + trajectory_panel.shape[1]] = trajectory_panel
    dashboard[y0 : y0 + stats_panel.shape[0], x_stats : x_stats + stats_panel.shape[1]] = stats_panel
    cv2.line(dashboard, (x_traj - 5, y0 + 6), (x_traj - 5, y0 + bottom_h - 6), (44, 44, 44), 1, cv2.LINE_AA)
    cv2.line(dashboard, (x_stats - 5, y0 + 6), (x_stats - 5, y0 + bottom_h - 6), (44, 44, 44), 1, cv2.LINE_AA)
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1] - 1, dashboard.shape[0] - 1), (112, 112, 112), 1)
    return dashboard


def _save_vi_slam_visuals_core(
    output_dir: Path,
    frame_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None,
    accepted_joint_updates_per_frame: np.ndarray,
    mean_pose_residual_per_frame: np.ndarray,
    initialized_count_per_frame: np.ndarray,
    accepted_landmark_updates_per_frame: np.ndarray,
    mean_landmark_residual_per_frame: np.ndarray,
    snapshot_frame_ids: np.ndarray,
    landmark_snapshots_w: np.ndarray,
    initialized_snapshots: np.ndarray,
    prefix: str = 'part4',
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f'{prefix}_vi_slam.gif'
    montage_path = output_dir / f'{prefix}_vi_slam_montage.png'

    snapshot_ids = np.asarray(snapshot_frame_ids, dtype=np.int64)
    if snapshot_ids.size == 0:
        return
    snapshot_lookup = {int(fid): idx for idx, fid in enumerate(snapshot_ids)}
    if snapshot_ids.size >= 6:
        montage_snapshot_indices = [int(i) for i in np.linspace(0, snapshot_ids.size - 1, num=6, dtype=int)]
    else:
        montage_snapshot_indices = list(range(int(snapshot_ids.size)))
        while len(montage_snapshot_indices) < 6 and montage_snapshot_indices:
            montage_snapshot_indices.append(montage_snapshot_indices[-1])
    montage_index_set = set(montage_snapshot_indices)
    montage_frames_by_index: dict[int, np.ndarray] = {}

    projector = StereoProjector(calib)
    display_ids = _landmark_display_ids(world_T_imu_corr, landmarks_w, initialized)
    bounds_min_xy, bounds_max_xy = _compute_vi_slam_bounds(
        world_T_imu_pred,
        world_T_imu_corr,
        landmarks_w,
        initialized,
        display_ids,
    )

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == world_T_imu_corr.shape[0]:
            ts = ts - float(ts[0])
        else:
            ts = None
    else:
        ts = None

    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('imageio is required to write Part 4 GIF outputs.') from exc

    with imageio.get_writer(gif_path, mode='I', duration=0.05, loop=0, palettesize=32, quantizer=0) as writer:
        for idx, (left_frame, right_frame) in enumerate(frame_iter):
            snap_idx = snapshot_lookup.get(int(idx))
            if snap_idx is None:
                continue
            elapsed_s = float(ts[idx]) if ts is not None else float(idx)
            z_frame = np.asarray(features[:, :, idx], dtype=np.float64)
            landmarks_snapshot = np.asarray(landmark_snapshots_w[snap_idx], dtype=np.float64)
            initialized_snapshot = np.asarray(initialized_snapshots[snap_idx], dtype=bool)

            stereo_panel, current_visible_ids = _make_vi_slam_stereo_panel(
                left_frame=left_frame,
                right_frame=right_frame,
                z_frame=z_frame,
                pose=world_T_imu_corr[idx],
                landmarks_w=landmarks_snapshot,
                initialized=initialized_snapshot,
                projector=projector,
                frame_idx=int(idx),
                elapsed_s=elapsed_s,
                accepted_joint_updates=int(accepted_joint_updates_per_frame[idx]),
                frame_pose_err_px=float(mean_pose_residual_per_frame[idx]),
            )
            map_panel = _make_vi_slam_map_panel(
                world_T_imu_pred=world_T_imu_pred,
                world_T_imu_corr=world_T_imu_corr,
                landmarks_w=landmarks_snapshot,
                initialized=initialized_snapshot,
                display_ids=display_ids,
                current_visible_ids=current_visible_ids,
                frame_idx=int(idx),
                bounds_min_xy=bounds_min_xy,
                bounds_max_xy=bounds_max_xy,
            )
            trajectory_panel = _make_planar_trajectory_panel(
                primary_trajectory=world_T_imu_corr,
                secondary_trajectory=world_T_imu_pred,
                frame_idx=int(idx),
                title='Trajectory Comparison',
                subtitle='Gray IMU-only, blue corrected, yellow robot',
                panel_size=(250, 236),
            )
            stats_panel = _make_vi_slam_stats_panel(
                world_T_imu_pred=world_T_imu_pred,
                world_T_imu_corr=world_T_imu_corr,
                accepted_joint_updates_per_frame=accepted_joint_updates_per_frame,
                mean_pose_residual_per_frame=mean_pose_residual_per_frame,
                initialized_count_per_frame=initialized_count_per_frame,
                mean_landmark_residual_per_frame=mean_landmark_residual_per_frame,
                visible_now=int(current_visible_ids.size),
                used_landmarks=int(np.count_nonzero(initialized)),
                frame_idx=int(idx),
                panel_size=(320, 236),
            )
            dashboard = _compose_vi_slam_dashboard(stereo_panel, map_panel, trajectory_panel, stats_panel)
            writer.append_data(np.asarray(dashboard, dtype=np.uint8))
            if snap_idx in montage_index_set:
                montage_frames_by_index[int(snap_idx)] = dashboard.copy()

    montage_frames: list[np.ndarray] = []
    if montage_snapshot_indices:
        available_indices = sorted(montage_frames_by_index)
        for snap_idx in montage_snapshot_indices:
            frame = montage_frames_by_index.get(int(snap_idx))
            if frame is None and available_indices:
                nearest_idx = min(available_indices, key=lambda idx: abs(idx - int(snap_idx)))
                frame = montage_frames_by_index.get(int(nearest_idx))
            if frame is not None:
                montage_frames.append(frame)

    if montage_frames:
        cols = 3
        rows = 2
        tile_h, tile_w = montage_frames[0].shape[:2]
        board = np.full((rows * tile_h, cols * tile_w, 3), 12, dtype=np.uint8)
        for k, frame in enumerate(montage_frames):
            r = k // cols
            c = k % cols
            board[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = frame
        Image.fromarray(board).save(montage_path)


def save_vi_slam_visuals(
    output_dir: Path,
    left_images: np.ndarray,
    right_images: np.ndarray,
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None,
    accepted_joint_updates_per_frame: np.ndarray,
    mean_pose_residual_per_frame: np.ndarray,
    initialized_count_per_frame: np.ndarray,
    accepted_landmark_updates_per_frame: np.ndarray,
    mean_landmark_residual_per_frame: np.ndarray,
    snapshot_frame_ids: np.ndarray,
    landmark_snapshots_w: np.ndarray,
    initialized_snapshots: np.ndarray,
    prefix: str = 'part4',
) -> None:
    _save_vi_slam_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_arrays(left_images, right_images),
        world_T_imu_pred=world_T_imu_pred,
        world_T_imu_corr=world_T_imu_corr,
        landmarks_w=landmarks_w,
        initialized=initialized,
        features=features,
        calib=calib,
        timestamps=timestamps,
        accepted_joint_updates_per_frame=accepted_joint_updates_per_frame,
        mean_pose_residual_per_frame=mean_pose_residual_per_frame,
        initialized_count_per_frame=initialized_count_per_frame,
        accepted_landmark_updates_per_frame=accepted_landmark_updates_per_frame,
        mean_landmark_residual_per_frame=mean_landmark_residual_per_frame,
        snapshot_frame_ids=snapshot_frame_ids,
        landmark_snapshots_w=landmark_snapshots_w,
        initialized_snapshots=initialized_snapshots,
        prefix=prefix,
    )


def save_vi_slam_visuals_from_videos(
    output_dir: Path,
    left_video_path: Path,
    right_video_path: Path,
    world_T_imu_pred: np.ndarray,
    world_T_imu_corr: np.ndarray,
    landmarks_w: np.ndarray,
    initialized: np.ndarray,
    features: np.ndarray,
    calib: StereoCalibration,
    timestamps: np.ndarray | None,
    accepted_joint_updates_per_frame: np.ndarray,
    mean_pose_residual_per_frame: np.ndarray,
    initialized_count_per_frame: np.ndarray,
    accepted_landmark_updates_per_frame: np.ndarray,
    mean_landmark_residual_per_frame: np.ndarray,
    snapshot_frame_ids: np.ndarray,
    landmark_snapshots_w: np.ndarray,
    initialized_snapshots: np.ndarray,
    prefix: str = 'part4',
) -> None:
    left_count = _video_frame_count(left_video_path)
    right_count = _video_frame_count(right_video_path)
    frame_count = min(
        c for c in (
            left_count,
            right_count,
            int(world_T_imu_corr.shape[0]),
            int(features.shape[2]),
        ) if c is not None
    )
    snap_ids = np.asarray(snapshot_frame_ids, dtype=np.int64)
    snap_mask = snap_ids < frame_count
    _save_vi_slam_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_videos(left_video_path, right_video_path),
        world_T_imu_pred=world_T_imu_pred[:frame_count],
        world_T_imu_corr=world_T_imu_corr[:frame_count],
        landmarks_w=landmarks_w,
        initialized=initialized,
        features=features[:, :, :frame_count],
        calib=calib,
        timestamps=None if timestamps is None else timestamps[:frame_count],
        accepted_joint_updates_per_frame=accepted_joint_updates_per_frame[:frame_count],
        mean_pose_residual_per_frame=mean_pose_residual_per_frame[:frame_count],
        initialized_count_per_frame=initialized_count_per_frame[:frame_count],
        accepted_landmark_updates_per_frame=accepted_landmark_updates_per_frame[:frame_count],
        mean_landmark_residual_per_frame=mean_landmark_residual_per_frame[:frame_count],
        snapshot_frame_ids=snap_ids[snap_mask],
        landmark_snapshots_w=landmark_snapshots_w[snap_mask],
        initialized_snapshots=initialized_snapshots[snap_mask],
        prefix=prefix,
    )


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
