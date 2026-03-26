from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from visual_slam.geometry import StereoCalibration


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    v_t: np.ndarray
    w_t: np.ndarray
    timestamps: np.ndarray
    features: Optional[np.ndarray]
    calib: StereoCalibration


@dataclass(frozen=True)
class StereoImages:
    left: np.ndarray
    right: np.ndarray


class DataLoadError(RuntimeError):
    pass


def normalize_dataset_id(dataset: str | int) -> str:
    return f"{int(dataset):02d}"


def dataset_file_path(data_dir: Path, dataset: str | int) -> Path:
    ds = normalize_dataset_id(dataset)
    return data_dir / f"dataset{ds}" / f"dataset{ds}.npy"


def dataset_images_path(data_dir: Path, dataset: str | int) -> Path:
    ds = normalize_dataset_id(dataset)
    return data_dir / f"dataset{ds}" / f"dataset{ds}_imgs.npy"


def _ensure_time_major(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise DataLoadError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    raise DataLoadError(f"Cannot infer velocity array orientation from shape {arr.shape}")


def _ensure_features_shape(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise DataLoadError(f"Expected features shape (4, M, T), got {features.shape}")
    if features.shape[0] == 4:
        return features
    if features.shape[1] == 4:
        return np.transpose(features, (1, 0, 2))
    if features.shape[2] == 4:
        return np.transpose(features, (2, 1, 0))
    raise DataLoadError(f"Cannot infer feature array orientation from shape {features.shape}")


def _extract_images_from_dict(blob: Dict[str, np.ndarray]) -> StereoImages:
    key_pairs = [
        ("left", "right"),
        ("img_left", "img_right"),
        ("left_imgs", "right_imgs"),
        ("cam0", "cam1"),
    ]
    for k_l, k_r in key_pairs:
        if k_l in blob and k_r in blob:
            return StereoImages(left=np.asarray(blob[k_l]), right=np.asarray(blob[k_r]))

    if "images" in blob:
        arr = np.asarray(blob["images"])
        return _extract_images_from_array(arr)

    raise DataLoadError(
        "Unsupported image npy format. Expected dict keys like (left,right), (cam0,cam1), or 'images'."
    )


def _extract_images_from_array(arr: np.ndarray) -> StereoImages:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 2:
        # (T, 2, H, W)
        return StereoImages(left=arr[:, 0], right=arr[:, 1])
    if arr.ndim == 4 and arr.shape[0] == 2:
        # (2, T, H, W)
        return StereoImages(left=arr[0], right=arr[1])
    if arr.ndim == 3 and arr.shape[0] == 2:
        # (2, H, W) single pair
        return StereoImages(left=arr[0][None, ...], right=arr[1][None, ...])
    raise DataLoadError(f"Unsupported image array shape: {arr.shape}")


def load_dataset(data_dir: Path, dataset: str | int) -> DatasetBundle:
    ds = normalize_dataset_id(dataset)
    path = dataset_file_path(data_dir, ds)
    if not path.exists():
        raise DataLoadError(
            f"Dataset file not found: {path}. "
            "Place the UCSD dataset under data/datasetXX/datasetXX.npy."
        )

    data = np.load(path, allow_pickle=True).item()
    v_t = _ensure_time_major(np.asarray(data["v_t"]))
    w_t = _ensure_time_major(np.asarray(data["w_t"]))
    timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)

    features = None
    if "features" in data and data["features"] is not None:
        features = _ensure_features_shape(np.asarray(data["features"], dtype=np.float64))

    calib = StereoCalibration(
        K_left=np.asarray(data["K_l"], dtype=np.float64),
        K_right=np.asarray(data["K_r"], dtype=np.float64),
        camL_T_imu=np.asarray(data["extL_T_imu"], dtype=np.float64),
        camR_T_imu=np.asarray(data["extR_T_imu"], dtype=np.float64),
    )

    n = min(v_t.shape[0], w_t.shape[0], timestamps.shape[0])
    v_t = v_t[:n]
    w_t = w_t[:n]
    timestamps = timestamps[:n]
    if features is not None and features.shape[2] != n:
        t = min(n, features.shape[2])
        v_t = v_t[:t]
        w_t = w_t[:t]
        timestamps = timestamps[:t]
        features = features[:, :, :t]

    return DatasetBundle(
        dataset_id=ds,
        v_t=v_t,
        w_t=w_t,
        timestamps=timestamps,
        features=features,
        calib=calib,
    )


def load_stereo_images(data_dir: Path, dataset: str | int) -> StereoImages:
    ds = normalize_dataset_id(dataset)
    path = dataset_images_path(data_dir, ds)
    if not path.exists():
        raise DataLoadError(
            f"Image file not found: {path}. "
            "Expected data/datasetXX/datasetXX_imgs.npy for Part 2 feature tracking."
        )

    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        blob = raw.item()
        if isinstance(blob, dict):
            imgs = _extract_images_from_dict(blob)
        else:
            imgs = _extract_images_from_array(np.asarray(blob))
    elif isinstance(raw, np.ndarray):
        imgs = _extract_images_from_array(raw)
    else:
        raise DataLoadError(f"Unsupported npy content type: {type(raw)}")

    left = np.asarray(imgs.left)
    right = np.asarray(imgs.right)

    if left.shape[0] != right.shape[0]:
        raise DataLoadError(f"Stereo left/right frame mismatch: {left.shape} vs {right.shape}")

    if left.ndim != 3 or right.ndim != 3:
        raise DataLoadError(f"Expected image tensors of shape (T,H,W), got {left.shape}, {right.shape}")

    return StereoImages(left=left, right=right)
