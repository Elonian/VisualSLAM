from __future__ import annotations

import argparse
from pathlib import Path

from visual_slam.config import ImuEkfConfig, LandmarkEkfConfig, PipelineConfig, RuntimeConfig, SlamConfig
from visual_slam.feature_tracking import FeatureTrackingConfig


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--datasets', nargs='+', default=['00', '01'], help='Dataset IDs, e.g. 00 01 02')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Root data directory')
    parser.add_argument('--output-root', type=Path, default=Path('results'), help='Output root directory')
    parser.add_argument('--use-gpu', action='store_true', help='Enable optional CuPy acceleration when available')

    parser.add_argument('--feature-stride', type=int, default=4, help='Use every Nth valid feature for mapping/SLAM')
    parser.add_argument('--init-landmark-sigma', type=float, default=100.0, help='Initial landmark covariance scale')
    parser.add_argument('--landmark-gate-px', type=float, default=50.0, help='Landmark EKF reprojection gate (px)')

    parser.add_argument('--max-pose-features', type=int, default=40, help='Maximum features per timestep for pose update')
    parser.add_argument('--min-pose-features', type=int, default=6, help='Minimum features per timestep for pose update')
    parser.add_argument('--pose-gate-px', type=float, default=60.0, help='Pose update gate (px)')

    parser.add_argument('--process-sigma-linear', type=float, default=5e-2, help='IMU process sigma for linear velocity')
    parser.add_argument('--process-sigma-angular', type=float, default=5e-2, help='IMU process sigma for angular velocity')


def add_part2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--max-corners', type=int, default=1500)
    parser.add_argument('--quality-level', type=float, default=0.01)
    parser.add_argument('--min-distance', type=float, default=7.0)
    parser.add_argument('--block-size', type=int, default=7)
    parser.add_argument('--lk-win-size', type=int, default=21)
    parser.add_argument('--lk-max-level', type=int, default=3)
    parser.add_argument('--min-tracked-per-frame', type=int, default=300)


def tracking_cfg_from_args(args: argparse.Namespace) -> FeatureTrackingConfig:
    return FeatureTrackingConfig(
        max_corners=max(100, int(args.max_corners)),
        quality_level=float(args.quality_level),
        min_distance=float(args.min_distance),
        block_size=max(3, int(args.block_size)),
        lk_win_size=max(5, int(args.lk_win_size)),
        lk_max_level=max(0, int(args.lk_max_level)),
        min_tracked_per_frame=max(20, int(args.min_tracked_per_frame)),
    )


def pipeline_cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    imu_cfg = ImuEkfConfig(
        process_sigma_linear=float(args.process_sigma_linear),
        process_sigma_angular=float(args.process_sigma_angular),
    )
    landmark_cfg = LandmarkEkfConfig(
        feature_stride=max(1, int(args.feature_stride)),
        init_landmark_sigma=float(args.init_landmark_sigma),
        reprojection_gate_px=float(args.landmark_gate_px),
    )
    slam_cfg = SlamConfig(
        max_pose_features_per_step=max(4, int(args.max_pose_features)),
        min_pose_features_per_step=max(2, int(args.min_pose_features)),
        pose_gate_px=float(args.pose_gate_px),
    )
    runtime_cfg = RuntimeConfig(use_gpu=bool(args.use_gpu), verbose=True)
    return PipelineConfig(imu=imu_cfg, landmark=landmark_cfg, slam=slam_cfg, runtime=runtime_cfg)
