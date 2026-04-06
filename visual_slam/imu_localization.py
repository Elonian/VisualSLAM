from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_slam.config import ImuEkfConfig
from visual_slam.geometry import adjoint_pose, se3_exp


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
        if self.cfg.ground_plane:
            return self._step_ground_plane(world_T_imu, pose_cov, v_t, w_t, dt)
        xi = np.concatenate([np.asarray(v_t, dtype=np.float64), np.asarray(w_t, dtype=np.float64)])
        delta_T = se3_exp(dt * xi)
        pred_T = world_T_imu @ delta_T

        # Use the simplified Phi = I propagation described in the report.
        Ad_T = adjoint_pose(world_T_imu)
        Q = (dt * dt) * (Ad_T @ self.cfg.process_noise @ Ad_T.T)
        pred_cov = pose_cov + Q
        pred_cov = 0.5 * (pred_cov + pred_cov.T)

        return pred_T, pred_cov

    def _step_ground_plane(
        self,
        world_T_imu: np.ndarray,
        pose_cov: np.ndarray,
        v_t: np.ndarray,
        w_t: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_t = np.asarray(v_t, dtype=np.float64).reshape(3)
        w_t = np.asarray(w_t, dtype=np.float64).reshape(3)

        yaw = float(np.arctan2(world_T_imu[1, 0], world_T_imu[0, 0]))
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)
        R_yaw = np.array(
            [
                [c_yaw, -s_yaw],
                [s_yaw, c_yaw],
            ],
            dtype=np.float64,
        )
        v_world_xy = R_yaw @ v_t[:2]
        yaw_next = yaw + dt * float(w_t[2])

        pred_T = np.eye(4, dtype=np.float64)
        c_next = np.cos(yaw_next)
        s_next = np.sin(yaw_next)
        pred_T[:3, :3] = np.array(
            [
                [c_next, -s_next, 0.0],
                [s_next, c_next, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pred_T[:3, 3] = world_T_imu[:3, 3]
        pred_T[0, 3] += dt * v_world_xy[0]
        pred_T[1, 3] += dt * v_world_xy[1]
        pred_T[2, 3] = self.cfg.ground_height

        pred_cov = np.array(pose_cov, dtype=np.float64, copy=True)
        pred_cov[2, :] = 0.0
        pred_cov[:, 2] = 0.0
        pred_cov[3, :] = 0.0
        pred_cov[:, 3] = 0.0
        pred_cov[4, :] = 0.0
        pred_cov[:, 4] = 0.0

        q_xy = (dt * self.cfg.process_sigma_linear) ** 2 * np.eye(2, dtype=np.float64)
        q_xy_world = R_yaw @ q_xy @ R_yaw.T
        pred_cov[:2, :2] += q_xy_world
        pred_cov[5, 5] += (dt * self.cfg.process_sigma_angular) ** 2
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
