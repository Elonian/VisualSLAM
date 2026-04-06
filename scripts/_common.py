from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from visual_slam.config import ImuEkfConfig, LandmarkEkfConfig, PipelineConfig, RuntimeConfig, SlamConfig
from visual_slam.data_loader import discover_dataset_ids, normalize_dataset_id
from visual_slam.feature_tracking import FeatureTrackingConfig


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--datasets', nargs='+', default=None, help='Dataset IDs, e.g. 00 01 02. Defaults to datasets found under --data-dir.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Root data directory')
    parser.add_argument('--output-root', type=Path, default=Path('results'), help='Output root directory')
    parser.add_argument('--use-gpu', action='store_true', help='Enable optional CuPy acceleration when available')

    parser.add_argument('--feature-stride', type=int, default=1, help='Use every Nth valid feature for mapping/SLAM')
    parser.add_argument('--init-landmark-sigma', type=float, default=100.0, help='Initial landmark covariance scale')
    parser.add_argument('--landmark-gate-px', type=float, default=50.0, help='Landmark EKF reprojection gate (px)')
    parser.add_argument('--landmark-mahalanobis-gate', type=float, default=13.277, help='Landmark EKF Mahalanobis gate (chi-square threshold)')

    parser.add_argument('--max-pose-features', type=int, default=40, help='Maximum features per timestep for pose update')
    parser.add_argument('--min-pose-features', type=int, default=6, help='Minimum features per timestep for pose update')
    parser.add_argument('--pose-gate-px', type=float, default=60.0, help='Pose update gate (px)')
    parser.add_argument('--slam-passes', type=int, default=2, help='Number of alternating map/pose refinement passes for Part 4')

    parser.add_argument('--process-sigma-linear', type=float, default=5e-2, help='IMU process sigma for linear velocity')
    parser.add_argument('--process-sigma-angular', type=float, default=5e-2, help='IMU process sigma for angular velocity')
    parser.add_argument('--ground-plane', action='store_true', help='Constrain Part 1 pose propagation to planar x-y motion with yaw only')


def add_part2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--max-corners', type=int, default=500)
    parser.add_argument('--quality-level', type=float, default=0.01)
    parser.add_argument('--min-distance', type=float, default=10.0)
    parser.add_argument('--block-size', type=int, default=7)
    parser.add_argument('--lk-win-size', type=int, default=21)
    parser.add_argument('--lk-max-level', type=int, default=3)
    parser.add_argument('--min-tracked-per-frame', type=int, default=250)
    parser.add_argument('--fb-max-error-px', type=float, default=1.0)
    parser.add_argument('--stereo-epipolar-tol-px', type=float, default=2.0)
    parser.add_argument('--min-disparity-px', type=float, default=0.5)
    parser.add_argument('--min-track-length', type=int, default=10)


def tracking_cfg_from_args(args: argparse.Namespace) -> FeatureTrackingConfig:
    return FeatureTrackingConfig(
        max_corners=max(100, int(args.max_corners)),
        quality_level=float(args.quality_level),
        min_distance=float(args.min_distance),
        block_size=max(3, int(args.block_size)),
        lk_win_size=max(5, int(args.lk_win_size)),
        lk_max_level=max(0, int(args.lk_max_level)),
        min_tracked_per_frame=max(20, int(args.min_tracked_per_frame)),
        fb_max_error_px=float(args.fb_max_error_px),
        stereo_epipolar_tol_px=float(args.stereo_epipolar_tol_px),
        min_disparity_px=float(args.min_disparity_px),
        min_track_length=max(1, int(args.min_track_length)),
    )


def resolve_datasets(args: argparse.Namespace) -> List[str]:
    if args.datasets:
        return [normalize_dataset_id(ds) for ds in args.datasets]
    found = discover_dataset_ids(Path(args.data_dir))
    if not found:
        raise RuntimeError(f'No datasets found under {args.data_dir}')
    return found


def pipeline_cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    imu_cfg = ImuEkfConfig(
        process_sigma_linear=float(args.process_sigma_linear),
        process_sigma_angular=float(args.process_sigma_angular),
        ground_plane=bool(args.ground_plane),
    )
    landmark_cfg = LandmarkEkfConfig(
        feature_stride=max(1, int(args.feature_stride)),
        init_landmark_sigma=float(args.init_landmark_sigma),
        reprojection_gate_px=float(args.landmark_gate_px),
        mahalanobis_gate_chi2=float(args.landmark_mahalanobis_gate),
    )
    slam_cfg = SlamConfig(
        max_pose_features_per_step=max(4, int(args.max_pose_features)),
        min_pose_features_per_step=max(2, int(args.min_pose_features)),
        pose_gate_px=float(args.pose_gate_px),
        refinement_passes=max(1, int(args.slam_passes)),
    )
    runtime_cfg = RuntimeConfig(use_gpu=bool(args.use_gpu), verbose=True)
    return PipelineConfig(imu=imu_cfg, landmark=landmark_cfg, slam=slam_cfg, runtime=runtime_cfg)
