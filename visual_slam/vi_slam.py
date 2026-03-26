from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from visual_slam.config import ImuEkfConfig, LandmarkEkfConfig, SlamConfig
from visual_slam.geometry import StereoCalibration, StereoProjector, se3_exp, triangulate_stereo
from visual_slam.imu_localization import ImuEkfPredictor
from visual_slam.utils import get_array_module


@dataclass(frozen=True)
class ViSlamResult:
    world_T_imu_pred: np.ndarray
    world_T_imu_corr: np.ndarray
    pose_cov_corr: np.ndarray
    landmarks_w: np.ndarray
    landmark_covariances: np.ndarray
    landmarks_initialized: np.ndarray
    mean_pose_residual_px: float
    mean_landmark_residual_px: float


class ViSlamRunner:
    def __init__(
        self,
        calib: StereoCalibration,
        imu_cfg: ImuEkfConfig,
        landmark_cfg: LandmarkEkfConfig,
        slam_cfg: SlamConfig,
        use_gpu: bool = False,
    ) -> None:
        self.calib = calib
        self.imu_cfg = imu_cfg
        self.landmark_cfg = landmark_cfg
        self.slam_cfg = slam_cfg

        self.imu_predictor = ImuEkfPredictor(imu_cfg)
        self.projector = StereoProjector(calib)
        self.xp = get_array_module(use_gpu)

    def run(self, v_t: np.ndarray, w_t: np.ndarray, timestamps: np.ndarray, features: np.ndarray) -> ViSlamResult:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 3 or feats.shape[0] != 4:
            raise ValueError(f"Expected features shape (4,M,T), got {feats.shape}")

        n = int(min(v_t.shape[0], w_t.shape[0], timestamps.shape[0], feats.shape[2]))
        v_t = v_t[:n]
        w_t = w_t[:n]
        ts = timestamps[:n]
        feats = feats[:, :, :n]

        M = feats.shape[1]
        landmarks = np.full((M, 3), np.nan, dtype=np.float64)
        landmark_cov = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], M, axis=0)
        landmark_cov *= self.landmark_cfg.init_landmark_sigma
        landmark_seen = np.zeros(M, dtype=bool)

        world_T_imu_pred = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
        world_T_imu_corr = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
        pose_cov_corr = np.repeat(np.eye(6, dtype=np.float64)[None, :, :], n, axis=0)
        pose_cov_corr[0] *= self.imu_cfg.init_pose_sigma

        T = np.eye(4, dtype=np.float64)
        S = self.imu_cfg.init_pose_sigma * np.eye(6, dtype=np.float64)

        pose_residuals: List[float] = []
        landmark_residuals: List[float] = []

        stride = max(1, int(self.landmark_cfg.feature_stride))
        I6 = np.eye(6, dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)

        for t in range(n - 1):
            dt = float(ts[t + 1] - ts[t])
            T_pred, S_pred = self.imu_predictor.step(T, S, v_t[t], w_t[t], dt)
            world_T_imu_pred[t + 1] = T_pred

            z_t = feats[:, :, t + 1]
            valid_ids = np.where(np.all(z_t > 0.0, axis=0))[0]
            if valid_ids.size > 0:
                valid_ids = valid_ids[::stride]

            # Lazy landmark initialization.
            for lid in valid_ids:
                if landmark_seen[lid]:
                    continue
                xyz, ok = triangulate_stereo(z_t[:2, lid], z_t[2:, lid], T_pred, self.calib)
                if not ok:
                    continue
                landmarks[lid] = xyz
                landmark_seen[lid] = True

            corr_ids = [int(i) for i in valid_ids if landmark_seen[int(i)]]

            # Pose correction (Part 4 requirement).
            T_corr = T_pred
            S_corr = S_pred
            if len(corr_ids) >= self.slam_cfg.min_pose_features_per_step:
                corr_ids = corr_ids[: self.slam_cfg.max_pose_features_per_step]
                H_rows: List[np.ndarray] = []
                r_rows: List[np.ndarray] = []
                for lid in corr_ids:
                    z = z_t[:, lid]
                    z_hat = self.projector.predict_stereo(T_pred, landmarks[lid])
                    if not np.isfinite(z_hat).all():
                        continue
                    residual = z - z_hat
                    if float(np.linalg.norm(residual)) > self.slam_cfg.pose_gate_px:
                        continue
                    H_pose = self.projector.pose_jacobian_numeric(
                        T_pred,
                        landmarks[lid],
                        eps=self.slam_cfg.pose_jac_eps,
                    )
                    if not np.isfinite(H_pose).all():
                        continue
                    H_rows.append(H_pose)
                    r_rows.append(residual)

                if len(H_rows) >= self.slam_cfg.min_pose_features_per_step:
                    H = np.vstack(H_rows)
                    r = np.concatenate(r_rows)
                    R_block = np.kron(np.eye(len(H_rows), dtype=np.float64), self.slam_cfg.pose_meas_noise)
                    try:
                        K = self._kalman_gain(S_pred, H, R_block)
                        dxi = K @ r
                        T_corr = T_pred @ se3_exp(dxi)
                        S_corr = (I6 - K @ H) @ S_pred @ (I6 - K @ H).T + K @ R_block @ K.T
                        S_corr = 0.5 * (S_corr + S_corr.T)
                        pose_residuals.extend(float(np.linalg.norm(rr)) for rr in r_rows)
                    except Exception:
                        T_corr = T_pred
                        S_corr = S_pred

            # Landmark correction using corrected pose.
            for lid in corr_ids:
                z = z_t[:, lid]
                z_hat = self.projector.predict_stereo(T_corr, landmarks[lid])
                if not np.isfinite(z_hat).all():
                    continue
                residual = z - z_hat
                err = float(np.linalg.norm(residual))
                if err > self.landmark_cfg.reprojection_gate_px:
                    continue
                H_lm = self.projector.landmark_jacobian(T_corr, landmarks[lid])
                if not np.isfinite(H_lm).all():
                    continue

                P = landmark_cov[lid]
                S_lm = H_lm @ P @ H_lm.T + self.landmark_cfg.meas_noise
                try:
                    K_lm = self._kalman_gain(P, H_lm, self.landmark_cfg.meas_noise)
                except Exception:
                    continue

                landmarks[lid] = landmarks[lid] + K_lm @ residual
                P_new = (I3 - K_lm @ H_lm) @ P @ (I3 - K_lm @ H_lm).T + K_lm @ self.landmark_cfg.meas_noise @ K_lm.T
                landmark_cov[lid] = 0.5 * (P_new + P_new.T)
                landmark_residuals.append(err)

            T, S = T_corr, S_corr
            world_T_imu_corr[t + 1] = T
            pose_cov_corr[t + 1] = S

        mean_pose = float(np.mean(pose_residuals)) if pose_residuals else float('nan')
        mean_lm = float(np.mean(landmark_residuals)) if landmark_residuals else float('nan')

        return ViSlamResult(
            world_T_imu_pred=world_T_imu_pred,
            world_T_imu_corr=world_T_imu_corr,
            pose_cov_corr=pose_cov_corr,
            landmarks_w=landmarks,
            landmark_covariances=landmark_cov,
            landmarks_initialized=landmark_seen,
            mean_pose_residual_px=mean_pose,
            mean_landmark_residual_px=mean_lm,
        )

    def _kalman_gain(self, P: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
        xp = self.xp
        if xp.__name__ != "numpy":
            P_gpu = xp.asarray(P)
            H_gpu = xp.asarray(H)
            R_gpu = xp.asarray(R)
            S_gpu = H_gpu @ P_gpu @ H_gpu.T + R_gpu
            K_gpu = P_gpu @ H_gpu.T @ xp.linalg.inv(S_gpu)
            return xp.asnumpy(K_gpu)
        S = H @ P @ H.T + R
        return P @ H.T @ np.linalg.inv(S)
