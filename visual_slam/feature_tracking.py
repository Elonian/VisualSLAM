from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class FeatureTrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureTrackingConfig:
    max_corners: int = 1500
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7
    lk_win_size: int = 21
    lk_max_level: int = 3
    min_tracked_per_frame: int = 300


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


def _detect_new_features(
    cv2,
    image: np.ndarray,
    existing_points: np.ndarray,
    cfg: FeatureTrackingConfig,
) -> np.ndarray:
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    if existing_points.size > 0:
        for x, y in existing_points.reshape(-1, 2):
            cv2.circle(mask, (int(round(x)), int(round(y))), int(cfg.min_distance), 0, -1)

    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        blockSize=cfg.block_size,
        mask=mask,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_features_stereo_temporal(
    left_images: np.ndarray,
    right_images: np.ndarray,
    cfg: FeatureTrackingConfig = FeatureTrackingConfig(),
) -> FeatureTracks:
    """Part 2: build z_t in shape (4, M, T) with persistent feature IDs."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise FeatureTrackingError(
            "OpenCV is required for Part 2 feature tracking. Install opencv-python."
        ) from exc

    left = np.asarray(left_images)
    right = np.asarray(right_images)
    if left.shape != right.shape or left.ndim != 3:
        raise FeatureTrackingError(
            f"Expected left/right image tensors with matching shape (T,H,W), got {left.shape}, {right.shape}"
        )

    T = int(left.shape[0])
    left_u8 = np.stack([_to_uint8(im) for im in left], axis=0)
    right_u8 = np.stack([_to_uint8(im) for im in right], axis=0)

    frame_obs: List[Dict[int, np.ndarray]] = [dict() for _ in range(T)]
    next_id = 0

    active_ids: np.ndarray = np.zeros((0,), dtype=np.int64)
    active_pts: np.ndarray = np.zeros((0, 2), dtype=np.float32)

    # Seed features at t=0
    new_pts = _detect_new_features(cv2, left_u8[0], active_pts, cfg)
    if new_pts.size > 0:
        ids = np.arange(next_id, next_id + new_pts.shape[0], dtype=np.int64)
        next_id += new_pts.shape[0]
        active_ids = np.concatenate([active_ids, ids])
        active_pts = np.vstack([active_pts, new_pts]) if active_pts.size else new_pts

    lk_params = dict(
        winSize=(cfg.lk_win_size, cfg.lk_win_size),
        maxLevel=cfg.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
    )

    for t in range(T):
        imgL = left_u8[t]
        imgR = right_u8[t]

        if active_pts.size > 0:
            ptsR, stR, _ = cv2.calcOpticalFlowPyrLK(imgL, imgR, active_pts.reshape(-1, 1, 2), None, **lk_params)
            if ptsR is None:
                stR = np.zeros((active_pts.shape[0], 1), dtype=np.uint8)
                ptsR = np.zeros((active_pts.shape[0], 1, 2), dtype=np.float32)
            stR = stR.reshape(-1).astype(bool)
            ptsR = ptsR.reshape(-1, 2)

            for i, ok in enumerate(stR):
                if not ok:
                    continue
                lx, ly = active_pts[i]
                rx, ry = ptsR[i]
                frame_obs[t][int(active_ids[i])] = np.array([lx, ly, rx, ry], dtype=np.float64)

        if t == T - 1:
            break

        # Temporal tracking L_t -> L_{t+1}
        if active_pts.size > 0:
            pts_next, stT, _ = cv2.calcOpticalFlowPyrLK(
                imgL,
                left_u8[t + 1],
                active_pts.reshape(-1, 1, 2),
                None,
                **lk_params,
            )
            if pts_next is None:
                stT = np.zeros((active_pts.shape[0], 1), dtype=np.uint8)
                pts_next = np.zeros((active_pts.shape[0], 1, 2), dtype=np.float32)
            stT = stT.reshape(-1).astype(bool)
            pts_next = pts_next.reshape(-1, 2)

            active_ids = active_ids[stT]
            active_pts = pts_next[stT]

        # Refill tracks if too sparse.
        if active_pts.shape[0] < cfg.min_tracked_per_frame:
            needed = cfg.max_corners - active_pts.shape[0]
            if needed > 0:
                refill_cfg = FeatureTrackingConfig(
                    max_corners=needed,
                    quality_level=cfg.quality_level,
                    min_distance=cfg.min_distance,
                    block_size=cfg.block_size,
                    lk_win_size=cfg.lk_win_size,
                    lk_max_level=cfg.lk_max_level,
                    min_tracked_per_frame=cfg.min_tracked_per_frame,
                )
                add_pts = _detect_new_features(cv2, left_u8[t + 1], active_pts, refill_cfg)
                if add_pts.size > 0:
                    add_ids = np.arange(next_id, next_id + add_pts.shape[0], dtype=np.int64)
                    next_id += add_pts.shape[0]
                    active_ids = np.concatenate([active_ids, add_ids])
                    active_pts = np.vstack([active_pts, add_pts]) if active_pts.size else add_pts

    M = int(next_id)
    z = -np.ones((4, M, T), dtype=np.float64)
    for t in range(T):
        if not frame_obs[t]:
            continue
        ids = np.fromiter(frame_obs[t].keys(), dtype=np.int64)
        vals = np.stack([frame_obs[t][int(i)] for i in ids], axis=1)
        z[:, ids, t] = vals

    return FeatureTracks(z=z, frame_count=T, feature_count=M)


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
    z = np.asarray(data["features"], dtype=np.float64)
    frame_count = int(np.asarray(data["frame_count"]).reshape(-1)[0])
    feature_count = int(np.asarray(data["feature_count"]).reshape(-1)[0])
    return FeatureTracks(z=z, frame_count=frame_count, feature_count=feature_count)
