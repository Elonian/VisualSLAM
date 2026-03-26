from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from visual_slam.config import LandmarkEkfConfig
from visual_slam.geometry import StereoCalibration, StereoProjector, triangulate_stereo


@dataclass(frozen=True)
class LandmarkMapResult:
    landmarks_w: np.ndarray  # (M,3), NaN for unseen
    covariances: np.ndarray  # (M,3,3)
    initialized: np.ndarray  # (M,)
    mean_reprojection_error_px: float
    trajectory: np.ndarray   # (T,4,4)


class LandmarkMapper:
    def __init__(self, calib: StereoCalibration, cfg: LandmarkEkfConfig) -> None:
        self.calib = calib
        self.cfg = cfg
        self.projector = StereoProjector(calib)

    def run(self, world_T_imu: np.ndarray, features: np.ndarray) -> LandmarkMapResult:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 3 or feats.shape[0] != 4:
            raise ValueError(f"Expected features shape (4,M,T), got {feats.shape}")

        T = min(world_T_imu.shape[0], feats.shape[2])
        M = feats.shape[1]

        landmarks = np.full((M, 3), np.nan, dtype=np.float64)
        covariances = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], M, axis=0)
        covariances *= self.cfg.init_landmark_sigma
        initialized = np.zeros(M, dtype=bool)

        errors: List[float] = []
        R_meas = self.cfg.meas_noise
        I3 = np.eye(3, dtype=np.float64)

        stride = max(1, int(self.cfg.feature_stride))

        for t in range(T):
            pose = world_T_imu[t]
            valid = np.where(np.all(feats[:, :, t] > 0.0, axis=0))[0]
            if valid.size == 0:
                continue
            valid = valid[::stride]

            for lid in valid:
                z = feats[:, lid, t]
                if not initialized[lid]:
                    xyz, ok = triangulate_stereo(z[:2], z[2:], pose, self.calib)
                    if not ok:
                        continue
                    landmarks[lid] = xyz
                    initialized[lid] = True
                    continue

                z_hat = self.projector.predict_stereo(pose, landmarks[lid])
                if not np.isfinite(z_hat).all():
                    continue
                residual = z - z_hat
                err = float(np.linalg.norm(residual))
                if err > self.cfg.reprojection_gate_px:
                    continue

                H = self.projector.landmark_jacobian(pose, landmarks[lid])
                if not np.isfinite(H).all():
                    continue

                P = covariances[lid]
                S = H @ P @ H.T + R_meas
                try:
                    K = P @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    continue

                landmarks[lid] = landmarks[lid] + K @ residual
                P_new = (I3 - K @ H) @ P @ (I3 - K @ H).T + K @ R_meas @ K.T
                covariances[lid] = 0.5 * (P_new + P_new.T)
                errors.append(err)

        mean_err = float(np.mean(errors)) if errors else float('nan')
        return LandmarkMapResult(
            landmarks_w=landmarks,
            covariances=covariances,
            initialized=initialized,
            mean_reprojection_error_px=mean_err,
            trajectory=world_T_imu[:T],
        )
