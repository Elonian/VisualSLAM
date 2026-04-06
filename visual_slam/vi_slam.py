from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_slam.config import ImuEkfConfig, LandmarkEkfConfig, SlamConfig
from visual_slam.geometry import StereoCalibration, StereoProjector, se3_exp
from visual_slam.imu_localization import ImuEkfPredictor
from visual_slam.landmark_mapping import LandmarkMapper


@dataclass(frozen=True)
class ViSlamResult:
    world_T_imu_pred: np.ndarray
    world_T_imu_corr: np.ndarray
    pose_cov_corr: np.ndarray
    landmarks_w: np.ndarray
    landmark_covariances: np.ndarray
    landmarks_initialized: np.ndarray
    selected_feature_ids: np.ndarray
    observation_counts: np.ndarray
    mean_pose_residual_px: float
    mean_landmark_residual_px: float
    accepted_joint_updates_per_frame: np.ndarray
    mean_pose_residual_per_frame: np.ndarray
    initialized_count_per_frame: np.ndarray
    accepted_landmark_updates_per_frame: np.ndarray
    mean_landmark_residual_per_frame: np.ndarray
    snapshot_frame_ids: np.ndarray
    landmark_snapshots_w: np.ndarray
    initialized_snapshots: np.ndarray


@dataclass(frozen=True)
class JointCorrectionPass:
    world_T_imu_corr: np.ndarray
    pose_cov_corr: np.ndarray
    landmarks_w: np.ndarray
    landmark_covariances: np.ndarray
    mean_pose_residual_px: float
    accepted_updates_per_frame: np.ndarray
    mean_pose_residual_per_frame: np.ndarray


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
        self.use_gpu = bool(use_gpu)

        self.imu_predictor = ImuEkfPredictor(imu_cfg)
        self.mapper = LandmarkMapper(calib, landmark_cfg)
        self.projector = StereoProjector(calib)

    def run(self, v_t: np.ndarray, w_t: np.ndarray, timestamps: np.ndarray, features: np.ndarray) -> ViSlamResult:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 3 or feats.shape[0] != 4:
            raise ValueError(f"Expected features shape (4,M,T), got {feats.shape}")

        n = int(min(v_t.shape[0], w_t.shape[0], timestamps.shape[0], feats.shape[2]))
        v_t = np.asarray(v_t[:n], dtype=np.float64)
        w_t = np.asarray(w_t[:n], dtype=np.float64)
        ts = np.asarray(timestamps[:n], dtype=np.float64)
        feats = feats[:, :, :n]

        valid_obs = np.all(feats >= 0.0, axis=0)
        obs_counts_all = np.sum(valid_obs, axis=1)
        selected_ids = self._select_feature_ids(obs_counts_all, valid_obs)
        feats_sel = feats[:, selected_ids, :]
        obs_counts = obs_counts_all[selected_ids]

        imu_only = self.imu_predictor.run(v_t, w_t, ts)
        snapshot_ids = self._snapshot_frame_ids(n)

        map_result = self.mapper.run(imu_only.world_T_imu, feats_sel)
        pose_pass = JointCorrectionPass(
            world_T_imu_corr=imu_only.world_T_imu.copy(),
            pose_cov_corr=imu_only.pose_cov.copy(),
            landmarks_w=map_result.landmarks_w.copy(),
            landmark_covariances=map_result.covariances.copy(),
            mean_pose_residual_px=float("nan"),
            accepted_updates_per_frame=np.zeros((n,), dtype=np.int32),
            mean_pose_residual_per_frame=np.full((n,), np.nan, dtype=np.float64),
        )

        for pass_idx in range(max(1, int(self.slam_cfg.refinement_passes))):
            pose_pass = self._joint_correction_pass(
                v_t=v_t,
                w_t=w_t,
                timestamps=ts,
                features=feats_sel,
                landmarks_w=map_result.landmarks_w,
                landmark_covariances=map_result.covariances,
                initialized=map_result.initialized,
                observation_counts=obs_counts,
            )
            want_snapshots = pass_idx == max(1, int(self.slam_cfg.refinement_passes)) - 1
            map_result = self.mapper.run(
                pose_pass.world_T_imu_corr,
                feats_sel,
                snapshot_frame_ids=snapshot_ids if want_snapshots else None,
                initial_landmarks=pose_pass.landmarks_w,
                initial_covariances=pose_pass.landmark_covariances,
                initial_initialized=map_result.initialized,
            )

        return ViSlamResult(
            world_T_imu_pred=imu_only.world_T_imu,
            world_T_imu_corr=pose_pass.world_T_imu_corr,
            pose_cov_corr=pose_pass.pose_cov_corr,
            landmarks_w=map_result.landmarks_w,
            landmark_covariances=map_result.covariances,
            landmarks_initialized=map_result.initialized,
            selected_feature_ids=selected_ids,
            observation_counts=obs_counts,
            mean_pose_residual_px=pose_pass.mean_pose_residual_px,
            mean_landmark_residual_px=map_result.mean_reprojection_error_px,
            accepted_joint_updates_per_frame=pose_pass.accepted_updates_per_frame,
            mean_pose_residual_per_frame=pose_pass.mean_pose_residual_per_frame,
            initialized_count_per_frame=map_result.initialized_count_per_frame,
            accepted_landmark_updates_per_frame=map_result.accepted_updates_per_frame,
            mean_landmark_residual_per_frame=map_result.mean_reprojection_error_per_frame,
            snapshot_frame_ids=map_result.snapshot_frame_ids,
            landmark_snapshots_w=map_result.landmark_snapshots_w,
            initialized_snapshots=map_result.initialized_snapshots,
        )

    def _snapshot_frame_ids(self, frame_count: int, target: int = 160) -> np.ndarray:
        if frame_count <= 0:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.linspace(0, frame_count - 1, num=min(frame_count, target), dtype=int))

    def _select_feature_ids(self, observation_counts: np.ndarray, visibility_mask: np.ndarray | None = None) -> np.ndarray:
        obs = np.asarray(observation_counts, dtype=np.int64).reshape(-1)
        min_obs = max(1, int(self.slam_cfg.min_landmark_observations))
        candidate_ids = np.where(obs >= min_obs)[0]
        if candidate_ids.size == 0:
            candidate_ids = np.arange(obs.shape[0], dtype=np.int64)
        if candidate_ids.size == 0:
            raise RuntimeError("Part 4 received no landmarks to process.")

        frac = float(self.slam_cfg.landmark_keep_fraction)
        if candidate_ids.size >= int(self.slam_cfg.large_landmark_threshold):
            frac = float(self.slam_cfg.large_landmark_keep_fraction)
        keep = max(int(self.slam_cfg.max_pose_features_per_step), int(np.ceil(frac * candidate_ids.size)))
        keep = min(keep, candidate_ids.size)

        order = np.argsort(obs[candidate_ids], kind="stable")[::-1]
        ranked_ids = candidate_ids[order]

        # Long custom-tracked sequences can cluster high-count landmarks into a few time windows.
        # For long runs, prefer a subset that still covers the whole sequence.
        if visibility_mask is None:
            return ranked_ids[:keep]
        vis = np.asarray(visibility_mask, dtype=bool)
        if vis.ndim != 2 or vis.shape[0] != obs.shape[0] or vis.shape[1] < 5000 or ranked_ids.size <= keep:
            return ranked_ids[:keep]

        bin_count = int(min(64, max(16, vis.shape[1] // 100)))
        bin_edges = np.linspace(0, vis.shape[1], num=bin_count + 1, dtype=int)
        cand_bins = np.zeros((ranked_ids.size, bin_count), dtype=bool)
        for b in range(bin_count):
            s = int(bin_edges[b])
            e = int(bin_edges[b + 1])
            if e <= s:
                continue
            cand_bins[:, b] = np.any(vis[ranked_ids, s:e], axis=1)

        coverage = np.zeros((bin_count,), dtype=np.int32)
        target_per_bin = max(
            int(self.slam_cfg.min_pose_features_per_step),
            int(np.ceil(0.5 * keep / max(bin_count, 1))),
        )
        obs_rank = obs[ranked_ids].astype(np.float64)
        obs_rank = obs_rank / max(float(np.max(obs_rank)), 1.0)
        remaining = np.ones((ranked_ids.size,), dtype=bool)
        picked: list[int] = []

        for _ in range(keep):
            deficit = np.maximum(target_per_bin - coverage, 0)
            gains = cand_bins.astype(np.int16) @ deficit.astype(np.int16)
            scores = gains.astype(np.float64) + 0.02 * obs_rank
            scores[~remaining] = -1.0
            best = int(np.argmax(scores))
            if scores[best] < 0.0:
                break
            picked.append(best)
            coverage += cand_bins[best].astype(np.int32)
            remaining[best] = False

        if len(picked) < keep:
            extra = np.where(remaining)[0]
            picked.extend(extra[: keep - len(picked)].tolist())

        return ranked_ids[np.asarray(picked[:keep], dtype=np.int64)]

    def _joint_correction_pass(
        self,
        v_t: np.ndarray,
        w_t: np.ndarray,
        timestamps: np.ndarray,
        features: np.ndarray,
        landmarks_w: np.ndarray,
        landmark_covariances: np.ndarray,
        initialized: np.ndarray,
        observation_counts: np.ndarray,
    ) -> JointCorrectionPass:
        n = int(min(v_t.shape[0], w_t.shape[0], timestamps.shape[0], features.shape[2]))
        m = int(landmarks_w.shape[0])

        world_T_imu_corr = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
        pose_cov_corr = np.repeat(np.eye(6, dtype=np.float64)[None, :, :], n, axis=0)
        accepted_updates_per_frame = np.zeros((n,), dtype=np.int32)
        mean_pose_residual_per_frame = np.full((n,), np.nan, dtype=np.float64)

        I6 = np.eye(6, dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        P_xx = float(self.slam_cfg.init_pose_sigma) * I6

        landmarks = np.asarray(landmarks_w, dtype=np.float64).copy()
        P_mm = np.asarray(landmark_covariances, dtype=np.float64).copy()
        if P_mm.shape != (m, 3, 3):
            raise ValueError(f"Expected landmark covariances shape {(m, 3, 3)}, got {P_mm.shape}")

        initialized_mask = np.asarray(initialized, dtype=bool).copy()
        stable_mask = (
            initialized_mask
            & np.isfinite(landmarks).all(axis=1)
            & (np.asarray(observation_counts, dtype=np.int64) >= int(self.slam_cfg.min_landmark_observations))
        )
        R_pose = self.slam_cfg.pose_meas_noise
        R_landmark = self.landmark_cfg.meas_noise
        pose_residuals: list[float] = []

        pose_cov_corr[0] = P_xx
        world_T_imu_corr[0] = T

        for t in range(n - 1):
            dt = float(timestamps[t + 1] - timestamps[t])
            T_pred, P_xx_pred = self.imu_predictor.step(T, P_xx, v_t[t], w_t[t], dt)

            T_cur = T_pred
            P_xx_cur = P_xx_pred
            frame_errors: list[float] = []
            accepted = 0

            z_t = np.asarray(features[:, :, t + 1], dtype=np.float64)
            candidate_ids = np.where(np.all(z_t >= 0.0, axis=0) & stable_mask)[0]

            if candidate_ids.size >= int(self.slam_cfg.min_pose_features_per_step):
                quality = np.asarray(observation_counts[candidate_ids], dtype=np.int64)
                uncertainty = np.trace(P_mm[candidate_ids], axis1=1, axis2=2)
                order = np.lexsort((uncertainty, -quality))
                candidate_ids = candidate_ids[order[: int(self.slam_cfg.max_pose_features_per_step)]]

                for lid in candidate_ids:
                    z = z_t[:, lid]
                    z_hat = self.projector.predict_stereo(T_cur, landmarks[lid])
                    if not np.isfinite(z_hat).all():
                        continue

                    residual = z - z_hat
                    err = float(np.linalg.norm(residual))
                    if err > float(self.slam_cfg.pose_gate_px):
                        continue

                    H_x = self.projector.pose_jacobian_left(T_cur, landmarks[lid])
                    if not np.isfinite(H_x).all():
                        continue

                    S = H_x @ P_xx_cur @ H_x.T + R_pose
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        continue

                    nis = float(residual.T @ S_inv @ residual)
                    if nis > float(self.landmark_cfg.mahalanobis_gate_chi2):
                        continue

                    K_x = P_xx_cur @ H_x.T @ S_inv
                    d_xi = K_x @ residual
                    T_cur = se3_exp(d_xi) @ T_cur
                    P_new = (I6 - K_x @ H_x) @ P_xx_cur @ (I6 - K_x @ H_x).T + K_x @ R_pose @ K_x.T
                    P_xx_cur = 0.5 * (P_new + P_new.T)

                    frame_errors.append(err)
                    pose_residuals.append(err)
                    accepted += 1

                    H_m = self.projector.landmark_jacobian(T_cur, landmarks[lid])
                    if not np.isfinite(H_m).all():
                        continue
                    z_hat_lm = self.projector.predict_stereo(T_cur, landmarks[lid])
                    if not np.isfinite(z_hat_lm).all():
                        continue
                    residual_lm = z - z_hat_lm
                    err_lm = float(np.linalg.norm(residual_lm))
                    if err_lm > float(self.landmark_cfg.reprojection_gate_px):
                        continue
                    P_lm = P_mm[lid]
                    S_lm = H_m @ P_lm @ H_m.T + R_landmark
                    try:
                        S_lm_inv = np.linalg.inv(S_lm)
                    except np.linalg.LinAlgError:
                        continue
                    nis_lm = float(residual_lm.T @ S_lm_inv @ residual_lm)
                    if nis_lm > float(self.landmark_cfg.mahalanobis_gate_chi2):
                        continue
                    K_lm = P_lm @ H_m.T @ S_lm_inv
                    landmarks[lid] = landmarks[lid] + K_lm @ residual_lm
                    P_lm_new = (I3 - K_lm @ H_m) @ P_lm @ (I3 - K_lm @ H_m).T + K_lm @ R_landmark @ K_lm.T
                    P_mm[lid] = 0.5 * (P_lm_new + P_lm_new.T)

            T = T_cur
            P_xx = P_xx_cur
            world_T_imu_corr[t + 1] = T
            pose_cov_corr[t + 1] = P_xx
            accepted_updates_per_frame[t + 1] = int(accepted)
            if frame_errors:
                mean_pose_residual_per_frame[t + 1] = float(np.mean(frame_errors))

        mean_pose = float(np.mean(pose_residuals)) if pose_residuals else float("nan")
        return JointCorrectionPass(
            world_T_imu_corr=world_T_imu_corr,
            pose_cov_corr=pose_cov_corr,
            landmarks_w=landmarks,
            landmark_covariances=P_mm,
            mean_pose_residual_px=mean_pose,
            accepted_updates_per_frame=accepted_updates_per_frame,
            mean_pose_residual_per_frame=mean_pose_residual_per_frame,
        )
