from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from visual_slam.common.paths import dataset_output_dir, ensure_dir
from visual_slam.config import PipelineConfig
from visual_slam.data_loader import DatasetBundle, load_dataset, load_stereo_images, normalize_dataset_id
from visual_slam.feature_tracking import (
    FeatureTrackingConfig,
    load_feature_tracks,
    save_feature_tracks,
    track_features_stereo_temporal,
)
from visual_slam.imu_localization import ImuEkfPredictor, trajectory_path_length
from visual_slam.landmark_mapping import LandmarkMapper
from visual_slam.metrics import count_initialized_landmarks, final_translation, save_metrics
from visual_slam.vi_slam import ViSlamRunner
from visual_slam.visualization import plot_landmarks_xy, plot_trajectory_2d, plot_trajectory_comparison


@dataclass(frozen=True)
class RunArtifacts:
    dataset_id: str
    output_dir: Path


def _resolve_features(
    bundle: DatasetBundle,
    track_file: Optional[Path],
) -> np.ndarray:
    if track_file is not None and track_file.exists():
        return load_feature_tracks(track_file).z
    if bundle.features is None:
        raise RuntimeError(
            "No features found in dataset and no --track-file provided. "
            "Run Part 2 first or provide dataset with precomputed features."
        )
    return bundle.features


def run_part1(
    data_dir: Path,
    output_root: Path,
    dataset: str | int,
    cfg: PipelineConfig,
) -> RunArtifacts:
    bundle = load_dataset(data_dir, dataset)
    ds = normalize_dataset_id(bundle.dataset_id)
    out_dir = ensure_dir(dataset_output_dir(output_root, ds))

    pred = ImuEkfPredictor(cfg.imu).run(bundle.v_t, bundle.w_t, bundle.timestamps)

    np.savez_compressed(
        out_dir / "part1_imu_prediction.npz",
        world_T_imu=pred.world_T_imu,
        pose_cov=pred.pose_cov,
        dt=pred.dt,
    )

    plot_trajectory_2d(
        pred.world_T_imu,
        title=f"Part 1 IMU Localization (Dataset {ds})",
        output_path=out_dir / "part1_imu_trajectory.png",
        show_orientation=True,
    )

    metrics = {
        "dataset": ds,
        "part": 1,
        "num_steps": int(pred.world_T_imu.shape[0]),
        "path_length_m": trajectory_path_length(pred.world_T_imu),
        "endpoint_distance_m": final_translation(pred.world_T_imu),
        "mean_dt_s": float(np.mean(pred.dt[1:])) if pred.dt.shape[0] > 1 else 0.0,
    }
    save_metrics(out_dir / "metrics_part1.json", metrics)
    return RunArtifacts(dataset_id=ds, output_dir=out_dir)


def run_part2(
    data_dir: Path,
    output_root: Path,
    dataset: str | int,
    cfg: FeatureTrackingConfig = FeatureTrackingConfig(),
) -> RunArtifacts:
    ds = normalize_dataset_id(dataset)
    out_dir = ensure_dir(dataset_output_dir(output_root, ds))
    imgs = load_stereo_images(data_dir, ds)

    tracks = track_features_stereo_temporal(imgs.left, imgs.right, cfg=cfg)
    save_feature_tracks(out_dir / "part2_feature_tracks.npz", tracks)

    metrics = {
        "dataset": ds,
        "part": 2,
        "frame_count": int(tracks.frame_count),
        "feature_count": int(tracks.feature_count),
    }
    save_metrics(out_dir / "metrics_part2.json", metrics)
    return RunArtifacts(dataset_id=ds, output_dir=out_dir)


def run_part3(
    data_dir: Path,
    output_root: Path,
    dataset: str | int,
    cfg: PipelineConfig,
    track_file: Optional[Path] = None,
) -> RunArtifacts:
    bundle = load_dataset(data_dir, dataset)
    ds = normalize_dataset_id(bundle.dataset_id)
    out_dir = ensure_dir(dataset_output_dir(output_root, ds))

    features = _resolve_features(bundle, track_file if track_file else out_dir / "part2_feature_tracks.npz")

    pred = ImuEkfPredictor(cfg.imu).run(bundle.v_t, bundle.w_t, bundle.timestamps)
    mapper = LandmarkMapper(bundle.calib, cfg.landmark)
    map_result = mapper.run(pred.world_T_imu, features)

    np.savez_compressed(
        out_dir / "part3_landmark_mapping.npz",
        world_T_imu=map_result.trajectory,
        landmarks_w=map_result.landmarks_w,
        landmark_cov=map_result.covariances,
        initialized=map_result.initialized,
    )

    plot_landmarks_xy(
        map_result.trajectory,
        map_result.landmarks_w,
        map_result.initialized,
        title=f"Part 3 Landmark Mapping (Dataset {ds})",
        output_path=out_dir / "part3_landmarks_xy.png",
    )

    metrics = {
        "dataset": ds,
        "part": 3,
        "num_steps": int(map_result.trajectory.shape[0]),
        "landmarks_total": int(map_result.landmarks_w.shape[0]),
        "landmarks_initialized": count_initialized_landmarks(map_result.initialized),
        "mean_reprojection_error_px": float(map_result.mean_reprojection_error_px),
        "path_length_m": trajectory_path_length(map_result.trajectory),
    }
    save_metrics(out_dir / "metrics_part3.json", metrics)
    return RunArtifacts(dataset_id=ds, output_dir=out_dir)


def run_part4(
    data_dir: Path,
    output_root: Path,
    dataset: str | int,
    cfg: PipelineConfig,
    track_file: Optional[Path] = None,
) -> RunArtifacts:
    bundle = load_dataset(data_dir, dataset)
    ds = normalize_dataset_id(bundle.dataset_id)
    out_dir = ensure_dir(dataset_output_dir(output_root, ds))

    features = _resolve_features(bundle, track_file if track_file else out_dir / "part2_feature_tracks.npz")

    runner = ViSlamRunner(
        bundle.calib,
        cfg.imu,
        cfg.landmark,
        cfg.slam,
        use_gpu=cfg.runtime.use_gpu,
    )
    slam = runner.run(bundle.v_t, bundle.w_t, bundle.timestamps, features)

    np.savez_compressed(
        out_dir / "part4_vi_slam.npz",
        world_T_imu_pred=slam.world_T_imu_pred,
        world_T_imu_corr=slam.world_T_imu_corr,
        pose_cov_corr=slam.pose_cov_corr,
        landmarks_w=slam.landmarks_w,
        landmark_cov=slam.landmark_covariances,
        initialized=slam.landmarks_initialized,
    )

    plot_trajectory_comparison(
        slam.world_T_imu_pred,
        slam.world_T_imu_corr,
        title=f"Part 4 Trajectory Comparison (Dataset {ds})",
        output_path=out_dir / "part4_trajectory_comparison.png",
    )
    plot_landmarks_xy(
        slam.world_T_imu_corr,
        slam.landmarks_w,
        slam.landmarks_initialized,
        title=f"Part 4 VI-SLAM Map (Dataset {ds})",
        output_path=out_dir / "part4_landmarks_xy.png",
    )

    metrics = {
        "dataset": ds,
        "part": 4,
        "num_steps": int(slam.world_T_imu_corr.shape[0]),
        "path_length_pred_m": trajectory_path_length(slam.world_T_imu_pred),
        "path_length_corr_m": trajectory_path_length(slam.world_T_imu_corr),
        "endpoint_pred_m": final_translation(slam.world_T_imu_pred),
        "endpoint_corr_m": final_translation(slam.world_T_imu_corr),
        "landmarks_total": int(slam.landmarks_w.shape[0]),
        "landmarks_initialized": count_initialized_landmarks(slam.landmarks_initialized),
        "mean_pose_residual_px": float(slam.mean_pose_residual_px),
        "mean_landmark_residual_px": float(slam.mean_landmark_residual_px),
    }
    save_metrics(out_dir / "metrics_part4.json", metrics)
    return RunArtifacts(dataset_id=ds, output_dir=out_dir)


def run_pipeline(
    data_dir: Path,
    output_root: Path,
    dataset: str | int,
    cfg: PipelineConfig,
    max_part: int = 4,
    run_part2_if_missing: bool = False,
) -> RunArtifacts:
    ds = normalize_dataset_id(dataset)
    out_dir = ensure_dir(dataset_output_dir(output_root, ds))

    run_part1(data_dir, output_root, ds, cfg)

    track_file = out_dir / "part2_feature_tracks.npz"
    if max_part >= 2 and run_part2_if_missing and not track_file.exists():
        try:
            run_part2(data_dir, output_root, ds)
        except Exception as exc:
            if cfg.runtime.verbose:
                print(f"[Part2] Dataset {ds}: skipped ({exc})")

    if max_part == 2:
        return RunArtifacts(dataset_id=ds, output_dir=out_dir)

    if max_part >= 3:
        run_part3(data_dir, output_root, ds, cfg, track_file=track_file)

    if max_part >= 4:
        run_part4(data_dir, output_root, ds, cfg, track_file=track_file)

    return RunArtifacts(dataset_id=ds, output_dir=out_dir)
