from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from visual_slam.config import ImuEkfConfig
from visual_slam.geometry import ad_se3, se3_exp


@dataclass(frozen=True)
class ImuPredictionResult:
    world_T_imu: np.ndarray
    pose_cov: np.ndarray
    dt: np.ndarray


class ImuEkfPredictor:
    def __init__(self, cfg: ImuEkfConfig) -> None:
        self.cfg = cfg

    def step(
        self,
        world_T_imu: np.ndarray,
        pose_cov: np.ndarray,
        v_t: np.ndarray,
        w_t: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = float(max(dt, 1e-6))
        xi = np.concatenate([np.asarray(v_t, dtype=np.float64), np.asarray(w_t, dtype=np.float64)])
        delta_T = se3_exp(dt * xi)
        pred_T = world_T_imu @ delta_T

        F = expm(-dt * ad_se3(xi))
        Q = dt * self.cfg.process_noise
        pred_cov = F @ pose_cov @ F.T + Q
        pred_cov = 0.5 * (pred_cov + pred_cov.T)

        return pred_T, pred_cov

    def run(self, v_t: np.ndarray, w_t: np.ndarray, timestamps: np.ndarray) -> ImuPredictionResult:
        n = int(min(v_t.shape[0], w_t.shape[0], timestamps.shape[0]))
        world_T_imu = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
        pose_cov = np.repeat(np.eye(6, dtype=np.float64)[None, :, :], n, axis=0)
        pose_cov[0] *= self.cfg.init_pose_sigma

        dt_all = np.zeros(n, dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        S = self.cfg.init_pose_sigma * np.eye(6, dtype=np.float64)

        for i in range(n - 1):
            dt = float(timestamps[i + 1] - timestamps[i])
            dt_all[i + 1] = dt
            T, S = self.step(T, S, v_t[i], w_t[i], dt)
            world_T_imu[i + 1] = T
            pose_cov[i + 1] = S

        return ImuPredictionResult(world_T_imu=world_T_imu, pose_cov=pose_cov, dt=dt_all)


def trajectory_path_length(world_T_imu: np.ndarray) -> float:
    p = np.asarray(world_T_imu[:, :3, 3], dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    return float(np.sum(seg))
