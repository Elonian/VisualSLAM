from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ImuEkfConfig:
    """Configuration for Part 1 IMU EKF prediction."""

    init_pose_sigma: float = 1e-2
    process_sigma_linear: float = 5e-2
    process_sigma_angular: float = 5e-2

    @property
    def process_noise(self) -> np.ndarray:
        q = np.array(
            [
                self.process_sigma_linear,
                self.process_sigma_linear,
                self.process_sigma_linear,
                self.process_sigma_angular,
                self.process_sigma_angular,
                self.process_sigma_angular,
            ],
            dtype=np.float64,
        )
        return np.diag(q**2)


@dataclass(frozen=True)
class LandmarkEkfConfig:
    """Configuration for Part 3 landmark EKF updates."""

    feature_stride: int = 4
    init_landmark_sigma: float = 100.0
    meas_sigma_left_px: float = 2.0
    meas_sigma_right_px: float = 2.0
    reprojection_gate_px: float = 50.0

    @property
    def meas_noise(self) -> np.ndarray:
        return np.diag(
            np.array(
                [
                    self.meas_sigma_left_px,
                    self.meas_sigma_left_px,
                    self.meas_sigma_right_px,
                    self.meas_sigma_right_px,
                ],
                dtype=np.float64,
            )
            ** 2
        )


@dataclass(frozen=True)
class SlamConfig:
    """Configuration for Part 4 visual-inertial SLAM pose correction."""

    max_pose_features_per_step: int = 40
    min_pose_features_per_step: int = 6
    pose_meas_sigma_px: float = 3.0
    pose_jac_eps: float = 1e-4
    pose_gate_px: float = 60.0

    @property
    def pose_meas_noise(self) -> np.ndarray:
        return np.diag(np.array([self.pose_meas_sigma_px] * 4, dtype=np.float64) ** 2)


@dataclass(frozen=True)
class RuntimeConfig:
    """Global runtime controls."""

    use_gpu: bool = False
    save_debug_images: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline config."""

    imu: ImuEkfConfig = field(default_factory=ImuEkfConfig)
    landmark: LandmarkEkfConfig = field(default_factory=LandmarkEkfConfig)
    slam: SlamConfig = field(default_factory=SlamConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
