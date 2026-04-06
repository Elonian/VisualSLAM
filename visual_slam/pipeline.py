from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from visual_slam.common.paths import dataset_output_dir, ensure_dir
from visual_slam.config import ImuEkfConfig, PipelineConfig
from visual_slam.data_loader import DatasetBundle, find_stereo_video_paths, load_dataset, load_stereo_images, normalize_dataset_id
from visual_slam.feature_tracking import (
    FeatureTrackingConfig,
    FeatureTracks,
    load_feature_tracks,
    save_feature_tracking_reference_comparison,
    save_feature_tracking_visuals,
    save_feature_tracking_visuals_from_videos,
    save_feature_tracks,
    track_features_stereo_temporal,
    track_features_stereo_temporal_from_videos,
)
from visual_slam.imu_localization import ImuEkfPredictor, trajectory_path_length
from visual_slam.landmark_mapping import LandmarkMapper
from visual_slam.metrics import (
    count_initialized_landmarks,
    final_translation,
    final_translation_xy,
    save_metrics,
    trajectory_path_length_xy,
)
from visual_slam.vi_slam import ViSlamRunner
from visual_slam.visualization import (
    save_landmark_mapping_visuals,
    save_landmark_mapping_visuals_from_videos,
    save_vi_slam_visuals,
    save_vi_slam_visuals_from_videos,
    plot_imu_trajectory_dashboard,
    plot_imu_trajectory_gif,
    plot_landmarks_xy,
    plot_trajectory_2d,
    plot_trajectory_comparison,
)


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


def _default_track_file(bundle: DatasetBundle, out_dir: Path) -> Optional[Path]:
    candidate = out_dir / "part2_feature_tracks.npz"
    ds = normalize_dataset_id(bundle.dataset_id)
    if candidate.exists() and (bundle.features is None or ds in {"01", "02"}):
        return candidate
    return None


def _dataset_part4_cfg(cfg: PipelineConfig, dataset: str | int) -> PipelineConfig:
    ds = normalize_dataset_id(dataset)
    slam_cfg = cfg.slam
    if ds == "01":
        slam_cfg = replace(
            slam_cfg,
            landmark_keep_fraction=0.25,
            large_landmark_keep_fraction=0.25,
        )
    elif ds == "02":
        slam_cfg = replace(
            slam_cfg,
            landmark_keep_fraction=0.12,
            large_landmark_keep_fraction=0.12,
        )
    if slam_cfg is cfg.slam:
        return cfg
    return replace(cfg, slam=slam_cfg)


def _load_or_compute_part1_trajectory(
    out_dir: Path,
    bundle: DatasetBundle,
    frame_count: int,
    imu_cfg: ImuEkfConfig,
) -> np.ndarray:
    pred_path = out_dir / "part1_imu_prediction.npz"
    if pred_path.exists():
        pred_data = np.load(pred_path)
        world_T_imu = np.asarray(pred_data["world_T_imu"], dtype=np.float64)
        if world_T_imu.ndim == 3 and world_T_imu.shape[1:] == (4, 4):
            return world_T_imu[:frame_count]
    pred = ImuEkfPredictor(imu_cfg).run(bundle.v_t, bundle.w_t, bundle.timestamps)
    return pred.world_T_imu[:frame_count]


def _load_or_compute_raw_trajectory(out_dir: Path, bundle: DatasetBundle, frame_count: int) -> np.ndarray:
    pred_path = out_dir / "part1_imu_prediction.npz"
    if pred_path.exists():
        pred_data = np.load(pred_path)
        world_T_imu = np.asarray(pred_data["world_T_imu"], dtype=np.float64)
        ground_plane = bool(int(np.asarray(pred_data["ground_plane"]).reshape(-1)[0])) if "ground_plane" in pred_data.files else False
        if not ground_plane and world_T_imu.ndim == 3 and world_T_imu.shape[1:] == (4, 4):
            return world_T_imu[:frame_count]

    pred = ImuEkfPredictor(ImuEkfConfig(ground_plane=False)).run(bundle.v_t, bundle.w_t, bundle.timestamps)
    return pred.world_T_imu[:frame_count]


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
    title_suffix = " Ground-Plane" if cfg.imu.ground_plane else ""

    np.savez_compressed(
        out_dir / "part1_imu_prediction.npz",
        world_T_imu=pred.world_T_imu,
        pose_cov=pred.pose_cov,
        dt=pred.dt,
        ground_plane=np.array([int(cfg.imu.ground_plane)], dtype=np.int32),
    )

    plot_imu_trajectory_dashboard(
        pred.world_T_imu,
        bundle.timestamps,
        bundle.v_t,
        bundle.w_t,
        title=f"Part 1 IMU Localization{title_suffix} (Dataset {ds})",
        output_path=out_dir / "part1_imu_trajectory.png",
    )
    plot_imu_trajectory_gif(
        pred.world_T_imu,
        bundle.timestamps,
        bundle.v_t,
        bundle.w_t,
        title=f"Part 1 IMU Localization{title_suffix} (Dataset {ds})",
        output_path=out_dir / "part1_imu_trajectory.gif",
    )

    metrics = {
        "dataset": ds,
        "part": 1,
        "num_steps": int(pred.world_T_imu.shape[0]),
        "path_length_m": trajectory_path_length(pred.world_T_imu),
        "path_length_xy_m": trajectory_path_length_xy(pred.world_T_imu),
        "endpoint_distance_m": final_translation(pred.world_T_imu),
        "endpoint_distance_xy_m": final_translation_xy(pred.world_T_imu),
        "mean_dt_s": float(np.mean(pred.dt[1:])) if pred.dt.shape[0] > 1 else 0.0,
        "ground_plane": bool(cfg.imu.ground_plane),
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
    track_source = "images"
    bundle = None
    video_paths = find_stereo_video_paths(data_dir, ds)
    if video_paths is not None:
        tracks = track_features_stereo_temporal_from_videos(video_paths[0], video_paths[1], cfg=cfg)
        track_source = "videos"
        bundle = load_dataset(data_dir, ds)
        world_T_imu = _load_or_compute_raw_trajectory(out_dir, bundle, tracks.frame_count)
        save_feature_tracking_visuals_from_videos(
            out_dir,
            video_paths[0],
            video_paths[1],
            tracks,
            timestamps=bundle.timestamps[: tracks.frame_count],
            calib=bundle.calib,
            world_T_imu=world_T_imu,
            prefix="part2",
        )
        if bundle.features is not None:
            save_feature_tracking_reference_comparison(
                out_dir / "part2_feature_comparison.png",
                tracks,
                bundle.features[:, :, : tracks.frame_count],
            )
    else:
        try:
            imgs = load_stereo_images(data_dir, ds)
        except Exception:
            bundle = load_dataset(data_dir, ds)
            if bundle.features is None:
                raise
            tracks = FeatureTracks(
                z=bundle.features,
                frame_count=int(bundle.features.shape[2]),
                feature_count=int(bundle.features.shape[1]),
            )
            track_source = "dataset_features"
        else:
            tracks = track_features_stereo_temporal(imgs.left, imgs.right, cfg=cfg)
            bundle = load_dataset(data_dir, ds)
            world_T_imu = _load_or_compute_raw_trajectory(out_dir, bundle, tracks.frame_count)
            save_feature_tracking_visuals(
                out_dir,
                imgs.left[: tracks.frame_count],
                imgs.right[: tracks.frame_count],
                tracks,
                timestamps=bundle.timestamps[: tracks.frame_count],
                calib=bundle.calib,
                world_T_imu=world_T_imu,
                prefix="part2",
            )
            if bundle.features is not None:
                save_feature_tracking_reference_comparison(
                    out_dir / "part2_feature_comparison.png",
                    tracks,
                    bundle.features[:, :, : tracks.frame_count],
                )
    save_feature_tracks(out_dir / "part2_feature_tracks.npz", tracks)

    valid = np.all(tracks.z >= 0.0, axis=0) if tracks.z.size else np.zeros((tracks.feature_count, tracks.frame_count), dtype=bool)
    obs_per_track = valid.sum(axis=1) if valid.size else np.zeros((tracks.feature_count,), dtype=np.int64)
    visible_per_frame = valid.sum(axis=0) if valid.size else np.zeros((tracks.frame_count,), dtype=np.int64)

    metrics = {
        "dataset": ds,
        "part": 2,
        "frame_count": int(tracks.frame_count),
        "feature_count": int(tracks.feature_count),
        "track_source": track_source,
        "median_track_length_frames": float(np.median(obs_per_track)) if obs_per_track.size else 0.0,
        "max_track_length_frames": int(obs_per_track.max()) if obs_per_track.size else 0,
        "median_visible_features_per_frame": float(np.median(visible_per_frame)) if visible_per_frame.size else 0.0,
        "max_visible_features_per_frame": int(visible_per_frame.max()) if visible_per_frame.size else 0,
    }
    if bundle is not None and bundle.features is not None:
        ref_valid = np.all(bundle.features >= 0.0, axis=0)
        ref_obs = ref_valid.sum(axis=1)
        ref_visible = ref_valid.sum(axis=0)
        metrics["reference_feature_count"] = int(bundle.features.shape[1])
        metrics["reference_median_track_length_frames"] = float(np.median(ref_obs)) if ref_obs.size else 0.0
        metrics["reference_median_visible_features_per_frame"] = float(np.median(ref_visible)) if ref_visible.size else 0.0
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

    resolved_track_file = track_file if track_file is not None else _default_track_file(bundle, out_dir)
    use_track_features = resolved_track_file is not None and resolved_track_file.exists()
    features = _resolve_features(bundle, resolved_track_file)

    pred_world_T_imu = _load_or_compute_part1_trajectory(out_dir, bundle, features.shape[2], cfg.imu)
    mapper = LandmarkMapper(bundle.calib, cfg.landmark)
    gif_frame_ids = np.unique(np.linspace(0, min(pred_world_T_imu.shape[0], features.shape[2]) - 1, num=min(features.shape[2], 480), dtype=int))
    map_result = mapper.run(pred_world_T_imu, features, snapshot_frame_ids=gif_frame_ids)

    np.savez_compressed(
        out_dir / "part3_landmark_mapping.npz",
        world_T_imu=map_result.trajectory,
        landmarks_w=map_result.landmarks_w,
        landmark_cov=map_result.covariances,
        initialized=map_result.initialized,
        initialized_count_per_frame=map_result.initialized_count_per_frame,
        accepted_updates_per_frame=map_result.accepted_updates_per_frame,
        mean_reprojection_error_per_frame=map_result.mean_reprojection_error_per_frame,
    )

    plot_landmarks_xy(
        map_result.trajectory,
        map_result.landmarks_w,
        map_result.initialized,
        title=f"Part 3 Landmark Mapping (Dataset {ds})",
        output_path=out_dir / "part3_landmarks_xy.png",
    )

    video_paths = find_stereo_video_paths(data_dir, ds)
    if video_paths is not None:
        save_landmark_mapping_visuals_from_videos(
            output_dir=out_dir,
            left_video_path=video_paths[0],
            right_video_path=video_paths[1],
            map_result=map_result,
            features=features[:, :, : map_result.trajectory.shape[0]],
            calib=bundle.calib,
            timestamps=bundle.timestamps[: map_result.trajectory.shape[0]],
            prefix="part3",
        )
    else:
        imgs = load_stereo_images(data_dir, ds)
        save_landmark_mapping_visuals(
            output_dir=out_dir,
            left_images=imgs.left[: map_result.trajectory.shape[0]],
            right_images=imgs.right[: map_result.trajectory.shape[0]],
            map_result=map_result,
            features=features[:, :, : map_result.trajectory.shape[0]],
            calib=bundle.calib,
            timestamps=bundle.timestamps[: map_result.trajectory.shape[0]],
            prefix="part3",
        )

    metrics = {
        "dataset": ds,
        "part": 3,
        "num_steps": int(map_result.trajectory.shape[0]),
        "landmarks_total": int(map_result.landmarks_w.shape[0]),
        "landmarks_initialized": count_initialized_landmarks(map_result.initialized),
        "mean_reprojection_error_px": float(map_result.mean_reprojection_error_px),
        "path_length_m": trajectory_path_length(map_result.trajectory),
        "feature_stride": int(cfg.landmark.feature_stride),
        "feature_source": "track_file" if use_track_features and track_file is not None else ("dataset_features" if bundle.features is not None and not use_track_features else "part2_tracks"),
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

    resolved_track_file = track_file if track_file is not None else _default_track_file(bundle, out_dir)
    use_track_features = resolved_track_file is not None and resolved_track_file.exists()
    features = _resolve_features(bundle, resolved_track_file)

    dataset_cfg = _dataset_part4_cfg(cfg, ds)
    runner = ViSlamRunner(
        bundle.calib,
        dataset_cfg.imu,
        dataset_cfg.landmark,
        dataset_cfg.slam,
        use_gpu=dataset_cfg.runtime.use_gpu,
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
        selected_feature_ids=slam.selected_feature_ids,
        observation_counts=slam.observation_counts,
        accepted_joint_updates_per_frame=slam.accepted_joint_updates_per_frame,
        mean_pose_residual_per_frame=slam.mean_pose_residual_per_frame,
        initialized_count_per_frame=slam.initialized_count_per_frame,
        accepted_landmark_updates_per_frame=slam.accepted_landmark_updates_per_frame,
        mean_landmark_residual_per_frame=slam.mean_landmark_residual_per_frame,
        snapshot_frame_ids=slam.snapshot_frame_ids,
        landmark_snapshots_w=slam.landmark_snapshots_w,
        initialized_snapshots=slam.initialized_snapshots,
    )

    used_features = features[:, slam.selected_feature_ids, :]

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

    video_paths = find_stereo_video_paths(data_dir, ds)
    if video_paths is not None:
        save_vi_slam_visuals_from_videos(
            output_dir=out_dir,
            left_video_path=video_paths[0],
            right_video_path=video_paths[1],
            world_T_imu_pred=slam.world_T_imu_pred,
            world_T_imu_corr=slam.world_T_imu_corr,
            landmarks_w=slam.landmarks_w,
            initialized=slam.landmarks_initialized,
            features=used_features,
            calib=bundle.calib,
            timestamps=bundle.timestamps,
            accepted_joint_updates_per_frame=slam.accepted_joint_updates_per_frame,
            mean_pose_residual_per_frame=slam.mean_pose_residual_per_frame,
            initialized_count_per_frame=slam.initialized_count_per_frame,
            accepted_landmark_updates_per_frame=slam.accepted_landmark_updates_per_frame,
            mean_landmark_residual_per_frame=slam.mean_landmark_residual_per_frame,
            snapshot_frame_ids=slam.snapshot_frame_ids,
            landmark_snapshots_w=slam.landmark_snapshots_w,
            initialized_snapshots=slam.initialized_snapshots,
            prefix="part4",
        )
    else:
        imgs = load_stereo_images(data_dir, ds)
        save_vi_slam_visuals(
            output_dir=out_dir,
            left_images=imgs.left,
            right_images=imgs.right,
            world_T_imu_pred=slam.world_T_imu_pred,
            world_T_imu_corr=slam.world_T_imu_corr,
            landmarks_w=slam.landmarks_w,
            initialized=slam.landmarks_initialized,
            features=used_features,
            calib=bundle.calib,
            timestamps=bundle.timestamps,
            accepted_joint_updates_per_frame=slam.accepted_joint_updates_per_frame,
            mean_pose_residual_per_frame=slam.mean_pose_residual_per_frame,
            initialized_count_per_frame=slam.initialized_count_per_frame,
            accepted_landmark_updates_per_frame=slam.accepted_landmark_updates_per_frame,
            mean_landmark_residual_per_frame=slam.mean_landmark_residual_per_frame,
            snapshot_frame_ids=slam.snapshot_frame_ids,
            landmark_snapshots_w=slam.landmark_snapshots_w,
            initialized_snapshots=slam.initialized_snapshots,
            prefix="part4",
        )

    metrics = {
        "dataset": ds,
        "part": 4,
        "feature_source": "track_file" if use_track_features and track_file is not None else ("dataset_features" if bundle.features is not None and not use_track_features else "part2_tracks"),
        "num_steps": int(slam.world_T_imu_corr.shape[0]),
        "features_total_input": int(features.shape[1]),
        "features_used_slam": int(slam.selected_feature_ids.size),
        "path_length_pred_m": trajectory_path_length(slam.world_T_imu_pred),
        "path_length_corr_m": trajectory_path_length(slam.world_T_imu_corr),
        "endpoint_pred_m": final_translation(slam.world_T_imu_pred),
        "endpoint_corr_m": final_translation(slam.world_T_imu_corr),
        "landmarks_total": int(slam.landmarks_w.shape[0]),
        "landmarks_initialized": count_initialized_landmarks(slam.landmarks_initialized),
        "mean_pose_residual_px": float(slam.mean_pose_residual_px),
        "mean_landmark_residual_px": float(slam.mean_landmark_residual_px),
        "landmark_keep_fraction": float(dataset_cfg.slam.landmark_keep_fraction),
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
        run_part3(data_dir, output_root, ds, cfg)

    if max_part >= 4:
        run_part4(data_dir, output_root, ds, cfg)

    return RunArtifacts(dataset_id=ds, output_dir=out_dir)
