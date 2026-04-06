from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from visual_slam.config import LandmarkEkfConfig
from visual_slam.geometry import OPTICAL_T_CAM, StereoCalibration, StereoProjector, pose_inverse, triangulate_stereo


@dataclass(frozen=True)
class LandmarkMapResult:
    landmarks_w: np.ndarray  # (M,3), NaN for unseen
    covariances: np.ndarray  # (M,3,3)
    initialized: np.ndarray  # (M,)
    mean_reprojection_error_px: float
    trajectory: np.ndarray   # (T,4,4)
    initialized_count_per_frame: np.ndarray  # (T,)
    accepted_updates_per_frame: np.ndarray  # (T,)
    mean_reprojection_error_per_frame: np.ndarray  # (T,)
    snapshot_frame_ids: np.ndarray  # (S,)
    landmark_snapshots_w: np.ndarray  # (S,M,3)
    initialized_snapshots: np.ndarray  # (S,M)


class LandmarkMapper:
    def __init__(self, calib: StereoCalibration, cfg: LandmarkEkfConfig) -> None:
        self.calib = calib
        self.cfg = cfg
        self.projector = StereoProjector(calib)
        self.fx = float(np.asarray(calib.K_left, dtype=np.float64)[0, 0])
        self.fy = float(np.asarray(calib.K_left, dtype=np.float64)[1, 1])
        self.cx = float(np.asarray(calib.K_left, dtype=np.float64)[0, 2])
        self.cy = float(np.asarray(calib.K_left, dtype=np.float64)[1, 2])
        self.baseline = float(np.linalg.norm(np.asarray(calib.camL_T_imu[:3, 3]) - np.asarray(calib.camR_T_imu[:3, 3])))
        self.imu_T_camL = pose_inverse(np.asarray(calib.camL_T_imu, dtype=np.float64))
        self.cam_T_opt = pose_inverse(OPTICAL_T_CAM)

    def _initialize_landmark(self, z: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, bool]:
        z = np.asarray(z, dtype=np.float64).reshape(4)
        disparity = float(z[0] - z[2])
        if disparity > 0.5 and self.baseline > 1e-9:
            depth = self.fx * self.baseline / disparity
            if np.isfinite(depth) and 0.1 < depth <= 150.0:
                x_opt = (float(z[0]) - self.cx) * depth / self.fx
                y_opt = (float(z[1]) - self.cy) * depth / self.fy
                p_opt = np.array([x_opt, y_opt, depth, 1.0], dtype=np.float64)
                p_w = np.asarray(pose, dtype=np.float64) @ self.imu_T_camL @ self.cam_T_opt @ p_opt
                if np.isfinite(p_w[:3]).all():
                    return np.asarray(p_w[:3], dtype=np.float64), True
        return triangulate_stereo(z[:2], z[2:], pose, self.calib)

    def run(
        self,
        world_T_imu: np.ndarray,
        features: np.ndarray,
        snapshot_frame_ids: np.ndarray | None = None,
        initial_landmarks: np.ndarray | None = None,
        initial_covariances: np.ndarray | None = None,
        initial_initialized: np.ndarray | None = None,
    ) -> LandmarkMapResult:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 3 or feats.shape[0] != 4:
            raise ValueError(f"Expected features shape (4,M,T), got {feats.shape}")

        T = min(world_T_imu.shape[0], feats.shape[2])
        M = feats.shape[1]
        snapshot_ids = np.unique(np.asarray(snapshot_frame_ids, dtype=np.int64).reshape(-1)) if snapshot_frame_ids is not None else np.zeros((0,), dtype=np.int64)
        snapshot_ids = snapshot_ids[(snapshot_ids >= 0) & (snapshot_ids < T)]
        snapshot_lookup = {int(fid): idx for idx, fid in enumerate(snapshot_ids)}

        if initial_landmarks is not None:
            landmarks = np.asarray(initial_landmarks, dtype=np.float64).copy()
            if landmarks.shape != (M, 3):
                raise ValueError(f"Expected initial_landmarks shape {(M, 3)}, got {landmarks.shape}")
        else:
            landmarks = np.full((M, 3), np.nan, dtype=np.float64)

        if initial_covariances is not None:
            covariances = np.asarray(initial_covariances, dtype=np.float64).copy()
            if covariances.shape != (M, 3, 3):
                raise ValueError(f"Expected initial_covariances shape {(M, 3, 3)}, got {covariances.shape}")
        else:
            covariances = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], M, axis=0)
            covariances *= self.cfg.init_landmark_sigma

        if initial_initialized is not None:
            initialized = np.asarray(initial_initialized, dtype=bool).copy()
            if initialized.shape != (M,):
                raise ValueError(f"Expected initial_initialized shape {(M,)}, got {initialized.shape}")
        else:
            initialized = np.zeros(M, dtype=bool)
        initialized_count_per_frame = np.zeros((T,), dtype=np.int32)
        accepted_updates_per_frame = np.zeros((T,), dtype=np.int32)
        mean_reprojection_error_per_frame = np.full((T,), np.nan, dtype=np.float64)
        landmark_snapshots_w = np.full((snapshot_ids.size, M, 3), np.nan, dtype=np.float32)
        initialized_snapshots = np.zeros((snapshot_ids.size, M), dtype=bool)

        errors: List[float] = []
        R_meas = self.cfg.meas_noise
        I3 = np.eye(3, dtype=np.float64)

        stride = max(1, int(self.cfg.feature_stride))

        for t in range(T):
            pose = world_T_imu[t]
            valid = np.where(np.all(feats[:, :, t] >= 0.0, axis=0))[0]
            frame_errors: List[float] = []
            accepted_updates = 0
            if valid.size > 0:
                valid = valid[::stride]

                for lid in valid:
                    z = feats[:, lid, t]
                    if not initialized[lid]:
                        xyz, ok = self._initialize_landmark(z, pose)
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
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        continue
                    nis = float(residual.T @ S_inv @ residual)
                    if nis > float(self.cfg.mahalanobis_gate_chi2):
                        continue
                    K = P @ H.T @ S_inv

                    landmarks[lid] = landmarks[lid] + K @ residual
                    P_new = (I3 - K @ H) @ P @ (I3 - K @ H).T + K @ R_meas @ K.T
                    covariances[lid] = 0.5 * (P_new + P_new.T)
                    errors.append(err)
                    frame_errors.append(err)
                    accepted_updates += 1

            initialized_count_per_frame[t] = int(np.count_nonzero(initialized))
            accepted_updates_per_frame[t] = int(accepted_updates)
            if frame_errors:
                mean_reprojection_error_per_frame[t] = float(np.mean(frame_errors))
            snap_idx = snapshot_lookup.get(int(t))
            if snap_idx is not None:
                landmark_snapshots_w[snap_idx] = landmarks.astype(np.float32)
                initialized_snapshots[snap_idx] = initialized.copy()

        mean_err = float(np.mean(errors)) if errors else float('nan')
        return LandmarkMapResult(
            landmarks_w=landmarks,
            covariances=covariances,
            initialized=initialized,
            mean_reprojection_error_px=mean_err,
            trajectory=world_T_imu[:T],
            initialized_count_per_frame=initialized_count_per_frame,
            accepted_updates_per_frame=accepted_updates_per_frame,
            mean_reprojection_error_per_frame=mean_reprojection_error_per_frame,
            snapshot_frame_ids=snapshot_ids,
            landmark_snapshots_w=landmark_snapshots_w,
            initialized_snapshots=initialized_snapshots,
        )
