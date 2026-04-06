from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visual_slam.geometry import OPTICAL_T_CAM, StereoCalibration


class FeatureTrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureTrackingConfig:
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 7
    lk_win_size: int = 21
    lk_max_level: int = 3
    min_tracked_per_frame: int = 250
    fb_max_error_px: float = 1.0
    stereo_epipolar_tol_px: float = 2.0
    min_disparity_px: float = 0.5
    min_track_length: int = 10


@dataclass(frozen=True)
class FeatureTracks:
    z: np.ndarray  # (4, M, T)
    frame_count: int
    feature_count: int


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    v_min, v_max = float(np.min(arr)), float(np.max(arr))
    if v_max <= v_min:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (255.0 * (arr - v_min) / (v_max - v_min)).astype(np.uint8)
    return arr


def _ensure_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return _to_uint8(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return _to_uint8(arr[..., 0])
    if arr.ndim == 3 and arr.shape[2] == 3:
        import cv2

        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    raise FeatureTrackingError(f"Expected grayscale or BGR frame, got shape {arr.shape}")


def _detect_new_features(
    cv2,
    image: np.ndarray,
    existing_points: np.ndarray,
    cfg: FeatureTrackingConfig,
    max_new_points: int | None = None,
) -> np.ndarray:
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    if existing_points.size > 0:
        for x, y in existing_points.reshape(-1, 2):
            cv2.circle(mask, (int(round(x)), int(round(y))), int(cfg.min_distance), 0, -1)

    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=int(cfg.max_corners if max_new_points is None else max_new_points),
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        blockSize=cfg.block_size,
        mask=mask,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def _track_bidirectional(cv2, img_a: np.ndarray, img_b: np.ndarray, pts_a: np.ndarray, cfg: FeatureTrackingConfig):
    if pts_a.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    lk_params = dict(
        winSize=(cfg.lk_win_size, cfg.lk_win_size),
        maxLevel=cfg.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-2),
    )

    pts_b, st_fwd, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, pts_a.reshape(-1, 1, 2), None, **lk_params)
    if pts_b is None:
        return np.zeros_like(pts_a), np.zeros((pts_a.shape[0],), dtype=bool)
    pts_b = pts_b.reshape(-1, 2)
    st_fwd = st_fwd.reshape(-1).astype(bool)

    pts_a_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img_b, img_a, pts_b.reshape(-1, 1, 2), None, **lk_params)
    if pts_a_back is None:
        return pts_b, np.zeros((pts_a.shape[0],), dtype=bool)
    pts_a_back = pts_a_back.reshape(-1, 2)
    st_back = st_back.reshape(-1).astype(bool)

    fb_err = np.linalg.norm(pts_a_back - pts_a, axis=1)
    ok = st_fwd & st_back & np.isfinite(pts_b).all(axis=1) & np.isfinite(pts_a_back).all(axis=1)
    ok &= fb_err <= float(cfg.fb_max_error_px)
    return pts_b.astype(np.float32), ok


def _in_bounds(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (pts[:, 0] >= 0.0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] < height)
    )


def _stereo_match(cv2, img_left: np.ndarray, img_right: np.ndarray, pts_left: np.ndarray, cfg: FeatureTrackingConfig):
    pts_right, ok = _track_bidirectional(cv2, img_left, img_right, pts_left, cfg)
    if pts_left.size == 0:
        return pts_right, ok
    ok &= _in_bounds(pts_left, img_left.shape[1], img_left.shape[0])
    ok &= _in_bounds(pts_right, img_right.shape[1], img_right.shape[0])
    ok &= np.abs(pts_left[:, 1] - pts_right[:, 1]) <= float(cfg.stereo_epipolar_tol_px)
    ok &= (pts_left[:, 0] - pts_right[:, 0]) > float(cfg.min_disparity_px)
    return pts_right, ok


def _finalize_feature_tracks(
    frame_obs: List[Dict[int, np.ndarray]],
    obs_counts: List[int],
    next_id: int,
    min_track_length: int,
    dtype: np.dtype = np.float32,
) -> FeatureTracks:
    frame_count = len(frame_obs)
    if next_id == 0:
        return FeatureTracks(z=-np.ones((4, 0, frame_count), dtype=dtype), frame_count=frame_count, feature_count=0)

    obs_counts_arr = np.asarray(obs_counts, dtype=np.int64)
    keep_ids = np.where(obs_counts_arr >= int(min_track_length))[0]
    kept_count = int(keep_ids.size)

    z = -np.ones((4, kept_count, frame_count), dtype=dtype)
    if kept_count > 0:
        remap = -np.ones((next_id,), dtype=np.int64)
        remap[keep_ids] = np.arange(kept_count, dtype=np.int64)
        for t, obs in enumerate(frame_obs):
            if not obs:
                continue
            ids = np.fromiter(obs.keys(), dtype=np.int64)
            new_ids = remap[ids]
            keep_mask = new_ids >= 0
            if not np.any(keep_mask):
                continue
            ids = ids[keep_mask]
            new_ids = new_ids[keep_mask]
            vals = np.stack([obs[int(i)] for i in ids], axis=1).astype(dtype, copy=False)
            z[:, new_ids, t] = vals

    return FeatureTracks(z=z, frame_count=frame_count, feature_count=kept_count)


def _sample_indices(n: int, max_count: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= max_count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=max_count, dtype=int)


def _feature_color(fid: int) -> Tuple[int, int, int]:
    palette = np.array(
        [
            [255, 99, 71],
            [255, 215, 0],
            [135, 206, 235],
            [50, 205, 50],
            [255, 105, 180],
            [0, 191, 255],
            [255, 140, 0],
            [186, 85, 211],
            [0, 250, 154],
            [255, 182, 193],
            [173, 255, 47],
            [240, 230, 140],
        ],
        dtype=np.uint8,
    )
    color = palette[int(fid) % len(palette)]
    return int(color[0]), int(color[1]), int(color[2])


def _map_to_panel(
    pt: np.ndarray | Tuple[float, float],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    pad: int = 16,
) -> Tuple[int, int]:
    x = float(np.clip(pt[0], 0.0, max(src_w - 1, 0)))
    y = float(np.clip(pt[1], 0.0, max(src_h - 1, 0)))
    draw_w = max(dst_w - 2 * pad - 1, 1)
    draw_h = max(dst_h - 2 * pad - 1, 1)
    u = int(round(pad + draw_w * x / max(src_w - 1, 1)))
    v = int(round(pad + draw_h * y / max(src_h - 1, 1)))
    return u, v


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    target_height = max(1, int(round(h * target_width / max(w, 1))))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _make_stereo_panel(
    left_img: np.ndarray,
    right_img: np.ndarray,
    z_frame: np.ndarray,
    z_prev: np.ndarray | None,
    frame_idx: int,
    elapsed_s: float,
    retained_so_far: int,
) -> np.ndarray:
    import cv2

    left_rgb = cv2.cvtColor(_to_uint8(left_img), cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(_to_uint8(right_img), cv2.COLOR_GRAY2RGB)

    h, w = left_rgb.shape[:2]
    gap = 12
    canvas = np.full((h, 2 * w + gap, 3), 18, dtype=np.uint8)
    canvas[:, :w] = left_rgb
    canvas[:, w + gap :] = right_rgb

    valid = np.where(np.all(z_frame >= 0.0, axis=0))[0]
    draw_ids = valid[_sample_indices(valid.size, 96)]
    left_color = (92, 214, 255)
    right_color = (255, 166, 84)
    stereo_color = (114, 220, 128)

    prev_lookup: Dict[int, Tuple[float, float]] = {}
    if z_prev is not None:
        valid_prev = np.where(np.all(z_prev >= 0.0, axis=0))[0]
        prev_lookup = {int(i): (float(z_prev[0, i]), float(z_prev[1, i])) for i in valid_prev}

    for lid in draw_ids:
        lid_i = int(lid)
        lx, ly, rx, ry = z_frame[:, lid_i]
        p_l = (int(round(lx)), int(round(ly)))
        p_r = (int(round(rx + w + gap)), int(round(ry)))

        cv2.circle(canvas, p_l, 3, left_color, -1)
        cv2.circle(canvas, p_r, 3, right_color, -1)
        cv2.line(canvas, p_l, p_r, stereo_color, 1)

        if lid_i in prev_lookup:
            px, py = prev_lookup[lid_i]
            p_prev = (int(round(px)), int(round(py)))
            cv2.arrowedLine(canvas, p_prev, p_l, left_color, 1, tipLength=0.22)

    legend_w = 318
    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (10 + legend_w, 102), (8, 8, 8), -1)
    canvas = cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0)

    text_color = (245, 245, 245)
    subtle = (190, 190, 190)
    cv2.putText(canvas, f'Frame {frame_idx:04d}', (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Elapsed {elapsed_s:6.1f} s', (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Active stereo tracks {valid.size:3d}', (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Retained stable tracks so far {retained_so_far:5d}', (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.58, subtle, 2, cv2.LINE_AA)

    cv2.putText(canvas, 'Left: temporal motion arrows', (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Right: stereo correspondences', (w + gap + 16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 120), 2, cv2.LINE_AA)
    return canvas


def _make_track_trajectory_panel(
    accum_canvas: np.ndarray,
    current_points: Dict[int, Tuple[float, float]],
    selected_track_count: int,
) -> np.ndarray:
    import cv2

    panel = accum_canvas.copy()
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'Feature Trajectories', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Long-lived tracks in left-image coordinates', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (188, 188, 188), 1, cv2.LINE_AA)

    for _, pt in current_points.items():
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(panel, p, 4, (255, 216, 92), -1)

    cv2.putText(panel, f'Displayed tracks: {selected_track_count}', (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (188, 188, 188), 1, cv2.LINE_AA)
    return panel


def _make_robot_trajectory_panel(
    world_T_imu: np.ndarray,
    frame_idx: int,
    panel_size: Tuple[int, int] = (236, 176),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'Robot Trajectory', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Top-down XY path with current pose', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)

    poses = np.asarray(world_T_imu, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[0] == 0:
        return panel

    xy = poses[:, :2, 3]
    idx = int(np.clip(frame_idx, 0, xy.shape[0] - 1))

    draw_x0 = 8
    draw_y0 = 52
    draw_w = panel_w - 16
    draw_h = panel_h - draw_y0 - 34

    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    span_xy = np.maximum(max_xy - min_xy, 1e-6)
    scale = 0.92 * min(draw_w / span_xy[0], draw_h / span_xy[1])

    def map_xy(pt: np.ndarray) -> Tuple[int, int]:
        u = int(round(draw_x0 + 0.5 * draw_w + scale * float(pt[0] - center_xy[0])))
        v = int(round(draw_y0 + 0.5 * draw_h - scale * float(pt[1] - center_xy[1])))
        return u, v

    for frac in np.linspace(0.0, 1.0, 5):
        gx = int(round(draw_x0 + frac * draw_w))
        gy = int(round(draw_y0 + frac * draw_h))
        cv2.line(panel, (gx, draw_y0), (gx, draw_y0 + draw_h), (36, 36, 36), 1, cv2.LINE_AA)
        cv2.line(panel, (draw_x0, gy), (draw_x0 + draw_w, gy), (36, 36, 36), 1, cv2.LINE_AA)

    all_pts = np.array([map_xy(pt) for pt in xy], dtype=np.int32).reshape(-1, 1, 2)
    if all_pts.shape[0] >= 2:
        cv2.polylines(panel, [all_pts], False, (78, 78, 78), 1, cv2.LINE_AA)

    past_xy = xy[: idx + 1]
    past_pts = np.array([map_xy(pt) for pt in past_xy], dtype=np.int32).reshape(-1, 1, 2)
    if past_pts.shape[0] >= 2:
        cv2.polylines(panel, [past_pts], False, (84, 214, 255), 2, cv2.LINE_AA)

    start_pt = map_xy(xy[0])
    cur_pt = map_xy(xy[idx])
    end_pt = map_xy(xy[-1])
    cv2.drawMarker(panel, start_pt, (90, 210, 110), cv2.MARKER_SQUARE, 10, 2, cv2.LINE_AA)
    cv2.circle(panel, cur_pt, 5, (255, 220, 90), -1, cv2.LINE_AA)
    cv2.drawMarker(panel, end_pt, (255, 110, 110), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    if idx < poses.shape[0]:
        yaw = float(np.arctan2(poses[idx, 1, 0], poses[idx, 0, 0]))
        arrow_len = 14
        arrow_tip = (
            int(round(cur_pt[0] + arrow_len * np.cos(yaw))),
            int(round(cur_pt[1] - arrow_len * np.sin(yaw))),
        )
        cv2.arrowedLine(panel, cur_pt, arrow_tip, (255, 220, 90), 2, cv2.LINE_AA, tipLength=0.3)

    cur_xy = xy[idx]
    cv2.putText(panel, f'Step {idx + 1}/{xy.shape[0]}', (12, panel_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, f'XY ({cur_xy[0]:.1f}, {cur_xy[1]:.1f}) m', (12, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (188, 188, 188), 1, cv2.LINE_AA)
    return panel


def _project_virtual_camera(
    points_3d: np.ndarray,
    dst_w: int,
    dst_h: int,
    camera_pos: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray,
    focal_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    forward = np.asarray(look_at, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= max(np.linalg.norm(right), 1e-9)
    cam_up = np.cross(right, forward)

    rel = pts - camera_pos.reshape(1, 3)
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    pix = np.zeros((pts.shape[0], 2), dtype=np.float64)
    pix[:, 0] = 0.5 * dst_w + focal_px * x_cam / np.maximum(z_cam, 1e-6)
    pix[:, 1] = 0.62 * dst_h - focal_px * y_cam / np.maximum(z_cam, 1e-6)
    return pix, valid


def _stereo_points_in_imu(
    z_frame: np.ndarray,
    selected_ids: np.ndarray,
    calib: StereoCalibration,
) -> Dict[int, np.ndarray]:
    if selected_ids.size == 0:
        return {}

    K = np.asarray(calib.K_left, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    baseline = float(np.linalg.norm(np.asarray(calib.camL_T_imu[:3, 3]) - np.asarray(calib.camR_T_imu[:3, 3])))
    if baseline <= 1e-9:
        return {}

    cam_R_opt = np.asarray(OPTICAL_T_CAM[:3, :3], dtype=np.float64).T
    camL_T_imu = np.asarray(calib.camL_T_imu, dtype=np.float64)

    points_imu: Dict[int, np.ndarray] = {}
    for lid in selected_ids:
        lid_i = int(lid)
        if not np.all(z_frame[:, lid_i] >= 0.0):
            continue
        lx, ly, rx, _ = np.asarray(z_frame[:, lid_i], dtype=np.float64)
        disparity = lx - rx
        if disparity <= 0.5:
            continue
        z_opt = fx * baseline / disparity
        if not np.isfinite(z_opt) or z_opt <= 0.05 or z_opt > 80.0:
            continue
        x_opt = (lx - cx) * z_opt / fx
        y_opt = (ly - cy) * z_opt / fy
        p_opt = np.array([x_opt, y_opt, z_opt], dtype=np.float64)
        p_cam = cam_R_opt @ p_opt
        p_imu = camL_T_imu[:3, :3] @ p_cam + camL_T_imu[:3, 3]
        if np.isfinite(p_imu).all():
            points_imu[lid_i] = p_imu
    return points_imu


def _make_feature_3d_panel(
    history_3d: Dict[int, List[np.ndarray]],
    current_3d: Dict[int, np.ndarray],
    selected_track_count: int,
    panel_size: Tuple[int, int] = (236, 176),
) -> np.ndarray:
    import cv2

    panel_w, panel_h = panel_size
    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (92, 92, 92), 1)
    cv2.putText(panel, '3D Feature Space', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Ego-centered stereo depth in IMU frame', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)

    draw_y0 = 54
    draw_h = panel_h - draw_y0 - 14
    draw_w = panel_w - 12
    draw_x0 = 6

    camera_pos = np.array([-7.0, -14.0, 7.0], dtype=np.float64)
    look_at = np.array([12.0, 0.0, 1.2], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    focal_px = 170.0

    def draw_poly(points: np.ndarray, color: Tuple[int, int, int], thickness: int = 1) -> None:
        if points.shape[0] < 2:
            return
        pix, valid = _project_virtual_camera(points, draw_w, draw_h, camera_pos, look_at, up, focal_px)
        keep = valid & np.isfinite(pix).all(axis=1)
        if np.count_nonzero(keep) < 2:
            return
        pts = pix[keep]
        pts[:, 0] += draw_x0
        pts[:, 1] += draw_y0
        cv2.polylines(panel, [pts.astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

    for x in np.linspace(0.0, 25.0, 6):
        line = np.array([[x, -10.0, 0.0], [x, 10.0, 0.0]], dtype=np.float64)
        draw_poly(line, (40, 40, 40), 1)
    for y in np.linspace(-10.0, 10.0, 5):
        line = np.array([[0.0, y, 0.0], [25.0, y, 0.0]], dtype=np.float64)
        draw_poly(line, (40, 40, 40), 1)

    robot = np.array(
        [
            [0.8, 0.0, 0.0],
            [-0.8, 0.55, 0.0],
            [-0.8, -0.55, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    draw_poly(robot, (255, 210, 90), 2)

    for lid, hist in history_3d.items():
        if len(hist) < 2:
            continue
        draw_poly(np.asarray(hist, dtype=np.float64), (78, 212, 214), 1)

    if current_3d:
        pts = np.stack(list(current_3d.values()), axis=0)
        pix, valid = _project_virtual_camera(pts, draw_w, draw_h, camera_pos, look_at, up, focal_px)
        depths = np.linalg.norm(pts, axis=1)
        order = np.argsort(depths)[::-1]
        for idx in order:
            if not valid[idx] or not np.isfinite(pix[idx]).all():
                continue
            u = int(round(draw_x0 + pix[idx, 0]))
            v = int(round(draw_y0 + pix[idx, 1]))
            if u < 0 or u >= panel_w or v < 0 or v >= panel_h:
                continue
            depth_norm = float(np.clip(depths[idx] / 30.0, 0.0, 1.0))
            color = (
                int(round(255 * (1.0 - depth_norm))),
                int(round(180 + 60 * (1.0 - depth_norm))),
                int(round(80 + 120 * depth_norm)),
            )
            cv2.circle(panel, (u, v), 3, color, -1, cv2.LINE_AA)

    cv2.putText(panel, 'x forward, y left, z up', (12, panel_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (188, 188, 188), 1, cv2.LINE_AA)
    cv2.putText(panel, f'Displayed tracks: {selected_track_count}', (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (188, 188, 188), 1, cv2.LINE_AA)
    return panel


def _plot_series_on_panel(
    panel: np.ndarray,
    values: np.ndarray,
    frame_idx: int,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    title: str,
    current_label: str,
) -> None:
    import cv2

    x0, y0 = top_left
    w, h = size
    x1 = x0 + w
    y1 = y0 + h
    cv2.rectangle(panel, (x0, y0), (x1, y1), (62, 62, 62), 1)
    cv2.putText(panel, title, (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

    if values.size == 0:
        return

    chart_x0 = x0 + 10
    chart_x1 = x1 - 10
    chart_y0 = y0 + 28
    chart_y1 = y1 - 10
    v_max = float(np.max(values))
    v_min = float(np.min(values))
    span = max(v_max - v_min, 1.0)

    xs = np.linspace(chart_x0, chart_x1, num=values.size).astype(np.int32)
    ys = chart_y1 - ((values - v_min) / span * (chart_y1 - chart_y0)).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], False, (68, 68, 68), 1, cv2.LINE_AA)

    if frame_idx > 0:
        current_pts = pts[: frame_idx + 1]
        cv2.polylines(panel, [current_pts], False, color, 2, cv2.LINE_AA)

    cursor_x = int(xs[min(frame_idx, values.size - 1)])
    cv2.line(panel, (cursor_x, chart_y0), (cursor_x, chart_y1), (235, 235, 235), 1, cv2.LINE_AA)

    current_value = float(values[min(frame_idx, values.size - 1)])
    cv2.putText(
        panel,
        current_label.format(current=current_value, max=v_max),
        (x0 + 10, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _make_feature_metrics_panel(
    visible_per_frame: np.ndarray,
    discovered_per_frame: np.ndarray,
    frame_idx: int,
    panel_width: int = 472,
    panel_height: int = 208,
) -> np.ndarray:
    import cv2

    panel = np.full((panel_height, panel_width, 3), 16, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (92, 92, 92), 1)
    cv2.putText(panel, 'Tracking Counts Over Time', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)

    top_y = 28
    chart_gap = 8
    summary_gap = 10
    summary_h = 48
    chart_h = max(42, (panel_height - top_y - 12 - summary_h - summary_gap - chart_gap) // 2)
    chart_w = panel_width - 20

    _plot_series_on_panel(
        panel,
        visible_per_frame.astype(np.float64),
        frame_idx,
        (10, top_y),
        (chart_w, chart_h),
        (88, 196, 255),
        'Visible stereo tracks',
        'now {current:.0f}',
    )
    _plot_series_on_panel(
        panel,
        discovered_per_frame.astype(np.float64),
        frame_idx,
        (10, top_y + chart_h + chart_gap),
        (chart_w, chart_h),
        (110, 230, 150),
        'Cumulative stable tracks discovered',
        'now {current:.0f}',
    )

    def draw_stat_card(
        x0: int,
        y0: int,
        w: int,
        h: int,
        title: str,
        value_text: str,
        sub_text: str,
        accent: Tuple[int, int, int],
    ) -> None:
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (46, 46, 46), -1)
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (92, 92, 92), 1)
        cv2.rectangle(panel, (x0, y0), (x0 + 5, y0 + h), accent, -1)
        cv2.putText(panel, title, (x0 + 14, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (188, 188, 188), 1, cv2.LINE_AA)
        cv2.putText(panel, value_text, (x0 + 14, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, accent, 2, cv2.LINE_AA)
        cv2.putText(panel, sub_text, (x0 + 14, y0 + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (208, 208, 208), 1, cv2.LINE_AA)

    cur_idx = int(np.clip(frame_idx, 0, max(visible_per_frame.size - 1, 0)))
    visible_now = int(visible_per_frame[cur_idx]) if visible_per_frame.size else 0
    visible_max = int(np.max(visible_per_frame)) if visible_per_frame.size else 0
    stable_now = int(discovered_per_frame[cur_idx]) if discovered_per_frame.size else 0
    stable_total = int(np.max(discovered_per_frame)) if discovered_per_frame.size else 0

    card_y0 = top_y + 2 * chart_h + chart_gap + summary_gap
    card_w = (panel_width - 30) // 2
    draw_stat_card(10, card_y0, card_w, summary_h, 'Visible Now', f'{visible_now}', f'peak {visible_max}', (88, 196, 255))
    draw_stat_card(20 + card_w, card_y0, card_w, summary_h, 'Stable Tracks', f'{stable_now}', f'total {stable_total}', (110, 230, 150))
    return panel


def _compose_tracking_dashboard(
    stereo_panel: np.ndarray,
    trajectory_panel: np.ndarray,
    metrics_panel: np.ndarray,
) -> np.ndarray:
    import cv2

    dashboard_width = 720
    stereo_resized = _resize_to_width(stereo_panel, dashboard_width)
    bottom_gap = 10
    bottom_h = max(int(trajectory_panel.shape[0]), int(metrics_panel.shape[0]))
    dashboard = np.full((stereo_resized.shape[0] + bottom_gap + bottom_h, dashboard_width, 3), 10, dtype=np.uint8)
    dashboard[: stereo_resized.shape[0], :dashboard_width] = stereo_resized
    y0 = stereo_resized.shape[0] + bottom_gap
    dashboard[y0 : y0 + trajectory_panel.shape[0], : trajectory_panel.shape[1]] = trajectory_panel
    metrics_x0 = dashboard_width - metrics_panel.shape[1]
    dashboard[y0 : y0 + metrics_panel.shape[0], metrics_x0:] = metrics_panel
    cv2.line(dashboard, (trajectory_panel.shape[1] + 6, y0 + 6), (trajectory_panel.shape[1] + 6, y0 + bottom_h - 6), (44, 44, 44), 1, cv2.LINE_AA)
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1] - 1, dashboard.shape[0] - 1), (112, 112, 112), 1)
    return dashboard


def _compose_tracking_story_tile(
    stereo_panel: np.ndarray,
    trajectory_panel: np.ndarray,
    frame_idx: int,
    elapsed_s: float,
    visible_tracks: int,
    discovered_tracks: int,
) -> np.ndarray:
    import cv2

    stereo_resized = _resize_to_width(stereo_panel, 640)
    side_w = 240
    tile_h = max(int(stereo_resized.shape[0]), 238)
    tile = np.full((tile_h, stereo_resized.shape[1] + 12 + side_w, 3), 10, dtype=np.uint8)
    tile[: stereo_resized.shape[0], : stereo_resized.shape[1]] = stereo_resized

    side_x0 = stereo_resized.shape[1] + 12
    cv2.rectangle(tile, (side_x0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (92, 92, 92), 1)
    cv2.putText(tile, f'Frame {frame_idx:04d}', (side_x0 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(tile, f'Time {elapsed_s:6.1f} s', (side_x0 + 12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(tile, f'Visible {visible_tracks:3d}', (side_x0 + 12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, f'Stable total {discovered_tracks:5d}', (side_x0 + 12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (110, 230, 150), 1, cv2.LINE_AA)

    traj = cv2.resize(trajectory_panel, (side_w - 20, int(round(trajectory_panel.shape[0] * (side_w - 20) / max(trajectory_panel.shape[1], 1)))), interpolation=cv2.INTER_AREA)
    traj_y0 = min(max(114, tile_h - traj.shape[0] - 12), tile_h - traj.shape[0] - 12)
    tile[traj_y0 : traj_y0 + traj.shape[0], side_x0 + 10 : side_x0 + 10 + traj.shape[1]] = traj
    return tile


def _overlay_tracking_frame(
    left_img: np.ndarray,
    right_img: np.ndarray,
    z_frame: np.ndarray,
    z_prev: np.ndarray | None,
    frame_idx: int,
) -> np.ndarray:
    import cv2

    left_rgb = cv2.cvtColor(_to_uint8(left_img), cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(_to_uint8(right_img), cv2.COLOR_GRAY2RGB)

    h, w = left_rgb.shape[:2]
    gap = 12
    canvas = np.full((h, 2 * w + gap, 3), 18, dtype=np.uint8)
    canvas[:, :w] = left_rgb
    canvas[:, w + gap :] = right_rgb

    valid = np.where(np.all(z_frame >= 0.0, axis=0))[0]
    draw_ids = valid[_sample_indices(valid.size, 90)]

    prev_lookup: Dict[int, Tuple[float, float]] = {}
    if z_prev is not None:
        valid_prev = np.where(np.all(z_prev >= 0.0, axis=0))[0]
        prev_lookup = {int(i): (float(z_prev[0, i]), float(z_prev[1, i])) for i in valid_prev}

    for lid in draw_ids:
        lx, ly, rx, ry = z_frame[:, lid]
        p_l = (int(round(lx)), int(round(ly)))
        p_r = (int(round(rx + w + gap)), int(round(ry)))

        cv2.circle(canvas, p_l, 3, (0, 255, 255), -1)
        cv2.circle(canvas, p_r, 3, (255, 120, 40), -1)
        cv2.line(canvas, p_l, p_r, (70, 220, 120), 1)

        if int(lid) in prev_lookup:
            px, py = prev_lookup[int(lid)]
            p_prev = (int(round(px)), int(round(py)))
            cv2.arrowedLine(canvas, p_prev, p_l, (220, 80, 220), 1, tipLength=0.22)

    text_color = (245, 245, 245)
    cv2.putText(canvas, f'Frame {frame_idx}', (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Active Stereo Tracks: {valid.size}', (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Left Image + temporal arrows', (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Right Image + stereo matches', (w + gap + 14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 120, 40), 2, cv2.LINE_AA)
    return canvas


def save_feature_tracking_visuals(
    output_dir: Path,
    left_images: np.ndarray,
    right_images: np.ndarray,
    tracks: FeatureTracks,
    timestamps: np.ndarray | None = None,
    calib: StereoCalibration | None = None,
    world_T_imu: np.ndarray | None = None,
    prefix: str = 'part2',
) -> None:
    _save_feature_tracking_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_arrays(left_images, right_images),
        tracks=tracks,
        timestamps=timestamps,
        calib=calib,
        world_T_imu=world_T_imu,
        prefix=prefix,
    )


def _iter_gray_stereo_pairs_from_arrays(left_images: np.ndarray, right_images: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    left = np.asarray(left_images)
    right = np.asarray(right_images)
    n = int(min(left.shape[0], right.shape[0]))
    for idx in range(n):
        yield _ensure_gray_u8(left[idx]), _ensure_gray_u8(right[idx])


def _iter_gray_stereo_pairs_from_videos(left_video_path: Path, right_video_path: Path) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    import cv2

    cap_left = cv2.VideoCapture(str(left_video_path))
    cap_right = cv2.VideoCapture(str(right_video_path))
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise FeatureTrackingError(f"Unable to open stereo videos: {left_video_path}, {right_video_path}")

    try:
        while True:
            ok_left, frame_left = cap_left.read()
            ok_right, frame_right = cap_right.read()
            if not ok_left or not ok_right or frame_left is None or frame_right is None:
                break
            yield _ensure_gray_u8(frame_left), _ensure_gray_u8(frame_right)
    finally:
        cap_left.release()
        cap_right.release()


def _save_feature_tracking_visuals_core(
    output_dir: Path,
    frame_iter: Iterable[Tuple[np.ndarray, np.ndarray]],
    tracks: FeatureTracks,
    timestamps: np.ndarray | None = None,
    calib: StereoCalibration | None = None,
    world_T_imu: np.ndarray | None = None,
    prefix: str = 'part2',
) -> None:
    z = np.asarray(tracks.z)
    valid = np.all(z >= 0.0, axis=0)
    visible_per_frame = valid.sum(axis=0)
    obs_per_track = valid.sum(axis=1)

    stats_fig = output_dir / f'{prefix}_feature_stats.png'
    gif_path = output_dir / f'{prefix}_feature_tracking.gif'
    montage_path = output_dir / f'{prefix}_feature_tracking_montage.png'

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), constrained_layout=True)
    axes[0].plot(visible_per_frame, color='tab:blue', linewidth=1.1)
    axes[0].set_title('Visible Tracked Features Per Frame')
    axes[0].set_xlabel('frame index')
    axes[0].set_ylabel('count')
    axes[0].grid(True, alpha=0.28)

    finite_obs = obs_per_track[obs_per_track > 0]
    bins = min(40, max(10, int(np.sqrt(max(1, finite_obs.size)))))
    axes[1].hist(finite_obs, bins=bins, color='tab:green', alpha=0.85)
    axes[1].axvline(10.0, color='tab:red', linestyle='--', linewidth=1.2, label='min track length')
    axes[1].set_title('Track-Length Distribution')
    axes[1].set_xlabel('observed frames per track')
    axes[1].set_ylabel('count')
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc='upper right')
    fig.savefig(stats_fig, dpi=170)
    plt.close(fig)

    if tracks.frame_count <= 0:
        return

    first_seen = np.full((tracks.feature_count,), tracks.frame_count, dtype=np.int64)
    has_obs = obs_per_track > 0
    if np.any(has_obs):
        first_seen[has_obs] = np.argmax(valid[has_obs], axis=1)

    births = np.zeros((tracks.frame_count,), dtype=np.int64)
    for idx in first_seen[has_obs]:
        births[int(idx)] += 1
    discovered_per_frame = np.cumsum(births)

    selected_pool = np.argsort(obs_per_track)[::-1]
    selected_pool = selected_pool[obs_per_track[selected_pool] > 0]
    selected_ids = selected_pool[_sample_indices(selected_pool.size, 16)] if selected_pool.size else np.zeros((0,), dtype=np.int64)
    history_3d: Dict[int, List[np.ndarray]] = {int(lid): [] for lid in selected_ids}

    traj_h, traj_w = 176, 236
    import cv2
    traj_base = np.full((traj_h, traj_w, 3), 16, dtype=np.uint8)
    for x in np.linspace(16, traj_w - 16, 4, dtype=int):
        cv2.line(traj_base, (int(x), 16), (int(x), traj_h - 16), (36, 36, 36), 1, cv2.LINE_AA)
    for y in np.linspace(16, traj_h - 16, 4, dtype=int):
        cv2.line(traj_base, (16, int(y)), (traj_w - 16, int(y)), (36, 36, 36), 1, cv2.LINE_AA)

    gif_frame_ids = np.unique(np.linspace(0, tracks.frame_count - 1, num=min(tracks.frame_count, 480), dtype=int))
    gif_frame_set = set(int(i) for i in gif_frame_ids)
    montage_ids = set(int(i) for i in np.unique(np.linspace(0, tracks.frame_count - 1, num=min(tracks.frame_count, 6), dtype=int)))
    montage_frames: List[np.ndarray] = []

    if timestamps is not None:
        time_s = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if time_s.size != tracks.frame_count:
            time_s = None
        else:
            time_s = time_s - float(time_s[0])
    else:
        time_s = None

    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise FeatureTrackingError(
            "imageio is required to write the continuous Part 2 GIF. Install imageio>=2.34."
        ) from exc

    with imageio.get_writer(gif_path, mode='I', duration=0.04, loop=0, palettesize=32, quantizer=0) as writer:
        for idx, (left_frame, right_frame) in enumerate(frame_iter):
            if idx >= tracks.frame_count:
                break
            if idx > 0:
                for lid in selected_ids:
                    lid_i = int(lid)
                    if valid[lid_i, idx - 1] and valid[lid_i, idx]:
                        p0 = _map_to_panel(
                            (z[0, lid_i, idx - 1], z[1, lid_i, idx - 1]),
                            src_w=int(left_frame.shape[1]),
                            src_h=int(left_frame.shape[0]),
                            dst_w=traj_w,
                            dst_h=traj_h,
                        )
                        p1 = _map_to_panel(
                            (z[0, lid_i, idx], z[1, lid_i, idx]),
                            src_w=int(left_frame.shape[1]),
                            src_h=int(left_frame.shape[0]),
                            dst_w=traj_w,
                            dst_h=traj_h,
                        )
                        cv2.line(traj_base, p0, p1, (78, 212, 214), 1, cv2.LINE_AA)

            if idx not in gif_frame_set and idx not in montage_ids:
                continue

            current_points = {}
            for lid in selected_ids:
                lid_i = int(lid)
                if valid[lid_i, idx]:
                    current_points[lid_i] = _map_to_panel(
                        (z[0, lid_i, idx], z[1, lid_i, idx]),
                        src_w=int(left_frame.shape[1]),
                        src_h=int(left_frame.shape[0]),
                        dst_w=traj_w,
                        dst_h=traj_h,
                    )

            prev = z[:, :, idx - 1] if idx > 0 else None
            stereo_panel = _make_stereo_panel(
                left_frame,
                right_frame,
                z[:, :, idx],
                prev,
                frame_idx=int(idx),
                elapsed_s=float(time_s[idx]) if time_s is not None else float(idx),
                retained_so_far=int(discovered_per_frame[idx]),
            )
            if world_T_imu is not None:
                trajectory_panel = _make_robot_trajectory_panel(
                    world_T_imu=np.asarray(world_T_imu, dtype=np.float64),
                    frame_idx=idx,
                    panel_size=(traj_w, traj_h),
                )
            else:
                current_3d = _stereo_points_in_imu(z[:, :, idx], selected_ids, calib) if calib is not None else {}
                for lid in selected_ids:
                    lid_i = int(lid)
                    if lid_i in current_3d:
                        hist = history_3d.setdefault(lid_i, [])
                        hist.append(np.asarray(current_3d[lid_i], dtype=np.float64))
                        if len(hist) > 60:
                            del hist[:-60]
                trajectory_panel = (
                    _make_feature_3d_panel(
                        history_3d=history_3d,
                        current_3d=current_3d,
                        selected_track_count=int(selected_ids.size),
                        panel_size=(traj_w, traj_h),
                    )
                    if calib is not None
                    else _make_track_trajectory_panel(
                        traj_base,
                        current_points=current_points,
                        selected_track_count=int(selected_ids.size),
                    )
                )
            if idx in gif_frame_set:
                metrics_panel = _make_feature_metrics_panel(
                    visible_per_frame=visible_per_frame,
                    discovered_per_frame=discovered_per_frame,
                    frame_idx=idx,
                )
                dashboard = _compose_tracking_dashboard(stereo_panel, trajectory_panel, metrics_panel)
                writer.append_data(np.asarray(dashboard, dtype=np.uint8))

            if idx in montage_ids:
                montage_frames.append(
                    _compose_tracking_story_tile(
                        stereo_panel=stereo_panel,
                        trajectory_panel=trajectory_panel,
                        frame_idx=int(idx),
                        elapsed_s=float(time_s[idx]) if time_s is not None else float(idx),
                        visible_tracks=int(visible_per_frame[idx]),
                        discovered_tracks=int(discovered_per_frame[idx]),
                    )
                )

    if montage_frames:
        cols = min(3, len(montage_frames))
        rows = (len(montage_frames) + cols - 1) // cols
        tile_h, tile_w = montage_frames[0].shape[:2]
        board = np.full((rows * tile_h, cols * tile_w, 3), 12, dtype=np.uint8)
        for k, frame in enumerate(montage_frames):
            r = k // cols
            c = k % cols
            board[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = frame
        Image.fromarray(board).save(montage_path)


def save_feature_tracking_visuals_from_videos(
    output_dir: Path,
    left_video_path: Path,
    right_video_path: Path,
    tracks: FeatureTracks,
    timestamps: np.ndarray | None = None,
    calib: StereoCalibration | None = None,
    world_T_imu: np.ndarray | None = None,
    prefix: str = 'part2',
) -> None:
    _save_feature_tracking_visuals_core(
        output_dir=output_dir,
        frame_iter=_iter_gray_stereo_pairs_from_videos(left_video_path, right_video_path),
        tracks=tracks,
        timestamps=timestamps,
        calib=calib,
        world_T_imu=world_T_imu,
        prefix=prefix,
    )


def save_feature_tracking_reference_comparison(
    output_path: Path,
    tracks: FeatureTracks,
    reference_features: np.ndarray,
) -> None:
    z = np.asarray(tracks.z)
    ref = np.asarray(reference_features, dtype=np.float64)
    if ref.ndim != 3 or ref.shape[0] != 4:
        return

    valid = np.all(z >= 0.0, axis=0) if z.size else np.zeros((tracks.feature_count, tracks.frame_count), dtype=bool)
    ref_valid = np.all(ref >= 0.0, axis=0)

    visible = valid.sum(axis=0) if valid.size else np.zeros((tracks.frame_count,), dtype=np.int64)
    ref_visible = ref_valid.sum(axis=0)
    obs = valid.sum(axis=1) if valid.size else np.zeros((tracks.feature_count,), dtype=np.int64)
    ref_obs = ref_valid.sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), constrained_layout=True)
    axes[0].plot(visible, color='tab:blue', linewidth=1.1, label='image-based tracker')
    axes[0].plot(ref_visible, color='tab:red', linewidth=1.0, alpha=0.75, label='provided features')
    axes[0].set_title('Visible Features Per Frame: Tracker vs Provided Features')
    axes[0].set_xlabel('frame index')
    axes[0].set_ylabel('count')
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc='upper right')

    bins = np.arange(0, max(int(obs.max()) if obs.size else 0, int(ref_obs.max()) if ref_obs.size else 0, 10) + 20, 10)
    axes[1].hist(ref_obs[ref_obs > 0], bins=bins, color='tab:red', alpha=0.45, label='provided')
    axes[1].hist(obs[obs > 0], bins=bins, color='tab:blue', alpha=0.45, label='tracked')
    axes[1].axvline(10.0, color='0.2', linestyle='--', linewidth=1.1, label='min track length')
    axes[1].set_title('Track-Length Distribution Comparison')
    axes[1].set_xlabel('observed frames per track')
    axes[1].set_ylabel('count')
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc='upper right')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def track_features_stereo_temporal(
    left_images: np.ndarray,
    right_images: np.ndarray,
    cfg: FeatureTrackingConfig = FeatureTrackingConfig(),
) -> FeatureTracks:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise FeatureTrackingError(
            "OpenCV is required for Part 2 feature tracking. Install opencv-python-headless."
        ) from exc

    left = np.asarray(left_images)
    right = np.asarray(right_images)
    if left.shape != right.shape or left.ndim != 3:
        raise FeatureTrackingError(
            f"Expected left/right image tensors with matching shape (T,H,W), got {left.shape}, {right.shape}"
        )

    return _track_feature_sequence(cv2, _iter_gray_stereo_pairs_from_arrays(left, right), cfg)


def track_features_stereo_temporal_from_videos(
    left_video_path: Path,
    right_video_path: Path,
    cfg: FeatureTrackingConfig = FeatureTrackingConfig(),
) -> FeatureTracks:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise FeatureTrackingError(
            "OpenCV is required for Part 2 feature tracking. Install opencv-python-headless."
        ) from exc

    return _track_feature_sequence(cv2, _iter_gray_stereo_pairs_from_videos(left_video_path, right_video_path), cfg)


def _track_feature_sequence(
    cv2,
    frame_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    cfg: FeatureTrackingConfig,
) -> FeatureTracks:
    first_pair = next(frame_iter, None)
    if first_pair is None:
        return FeatureTracks(z=-np.ones((4, 0, 0), dtype=np.float32), frame_count=0, feature_count=0)

    img_left, img_right = first_pair
    height, width = map(int, img_left.shape[:2])

    frame_obs: List[Dict[int, np.ndarray]] = []
    obs_counts: List[int] = []
    next_id = 0

    active_ids = np.zeros((0,), dtype=np.int64)
    active_pts = np.zeros((0, 2), dtype=np.float32)

    seed = _detect_new_features(cv2, img_left, active_pts, cfg, max_new_points=cfg.max_corners)
    if seed.size > 0:
        ids = np.arange(next_id, next_id + seed.shape[0], dtype=np.int64)
        next_id += seed.shape[0]
        obs_counts.extend([0] * seed.shape[0])
        active_ids = ids
        active_pts = seed

    while True:
        current_obs: Dict[int, np.ndarray] = {}
        stereo_ids = np.zeros((0,), dtype=np.int64)
        stereo_left = np.zeros((0, 2), dtype=np.float32)

        if active_pts.size > 0:
            pts_right, stereo_ok = _stereo_match(cv2, img_left, img_right, active_pts, cfg)
            stereo_ids = active_ids[stereo_ok]
            stereo_left = active_pts[stereo_ok]
            stereo_right = pts_right[stereo_ok]

            for lid, left_pt, right_pt in zip(stereo_ids, stereo_left, stereo_right):
                current_obs[int(lid)] = np.array([left_pt[0], left_pt[1], right_pt[0], right_pt[1]], dtype=np.float32)
                obs_counts[int(lid)] += 1

        frame_obs.append(current_obs)

        next_pair = next(frame_iter, None)
        if next_pair is None:
            break

        next_left_img, next_right_img = next_pair
        next_ids = np.zeros((0,), dtype=np.int64)
        next_pts = np.zeros((0, 2), dtype=np.float32)
        if stereo_left.size > 0:
            pts_next, temporal_ok = _track_bidirectional(cv2, img_left, next_left_img, stereo_left, cfg)
            temporal_ok &= _in_bounds(pts_next, width, height)
            next_ids = stereo_ids[temporal_ok]
            next_pts = pts_next[temporal_ok]

        active_ids = next_ids
        active_pts = next_pts

        if active_pts.shape[0] < cfg.min_tracked_per_frame:
            needed = int(cfg.max_corners - active_pts.shape[0])
            if needed > 0:
                refill = _detect_new_features(cv2, next_left_img, active_pts, cfg, max_new_points=needed)
                if refill.size > 0:
                    refill_ids = np.arange(next_id, next_id + refill.shape[0], dtype=np.int64)
                    next_id += refill.shape[0]
                    obs_counts.extend([0] * refill.shape[0])
                    active_ids = np.concatenate([active_ids, refill_ids])
                    active_pts = np.vstack([active_pts, refill]).astype(np.float32) if active_pts.size else refill

        img_left, img_right = next_left_img, next_right_img

    return _finalize_feature_tracks(frame_obs, obs_counts, next_id, cfg.min_track_length, dtype=np.float32)


def save_feature_tracks(path: Path, tracks: FeatureTracks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=tracks.z,
        frame_count=np.array([tracks.frame_count], dtype=np.int32),
        feature_count=np.array([tracks.feature_count], dtype=np.int32),
    )


def load_feature_tracks(path: Path) -> FeatureTracks:
    if not path.exists():
        raise FeatureTrackingError(f"Track file not found: {path}")
    data = np.load(path)
    z = np.asarray(data["features"])
    frame_count = int(np.asarray(data["frame_count"]).reshape(-1)[0])
    feature_count = int(np.asarray(data["feature_count"]).reshape(-1)[0])
    return FeatureTracks(z=z, frame_count=frame_count, feature_count=feature_count)
