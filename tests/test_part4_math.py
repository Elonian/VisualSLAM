from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from visual_slam.config import PipelineConfig
from visual_slam.data_loader import load_dataset
from visual_slam.geometry import OPTICAL_T_CAM, StereoProjector, project_jacobian, se3_exp, skew, pose_inverse
from visual_slam.vi_slam import ViSlamRunner


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class Part4MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_dataset(DATA_DIR, "00")
        cls.projector = StereoProjector(cls.bundle.calib)

    def test_left_pose_jacobian_matches_numeric(self) -> None:
        T = se3_exp(np.array([0.7, -0.4, 0.2, 0.15, -0.08, 0.1], dtype=np.float64))
        landmark_w = np.array([5.0, -1.0, 0.5], dtype=np.float64)
        J_analytic = self.projector.pose_jacobian_left(T, landmark_w)
        J_numeric = self.projector.pose_jacobian_left_numeric(T, landmark_w, eps=1e-6)
        max_err = float(np.max(np.abs(J_analytic - J_numeric)))
        self.assertLess(max_err, 1e-5)

    def test_wrong_body_frame_jacobian_is_not_equivalent(self) -> None:
        T = se3_exp(np.array([0.5, -0.2, 0.1, 0.08, 0.03, -0.05], dtype=np.float64))
        landmark_w = np.array([8.0, 1.5, 0.8], dtype=np.float64)

        imu_T_world = pose_inverse(T)
        oL_T_imu = OPTICAL_T_CAM @ self.bundle.calib.camL_T_imu
        oR_T_imu = OPTICAL_T_CAM @ self.bundle.calib.camR_T_imu
        m_h = np.r_[landmark_w, 1.0]
        p_imu = (imu_T_world @ m_h)[:3]
        qL = (oL_T_imu @ imu_T_world @ m_h)[:3]
        qR = (oR_T_imu @ imu_T_world @ m_h)[:3]
        JL = project_jacobian(self.bundle.calib.K_left, qL)
        JR = project_jacobian(self.bundle.calib.K_right, qR)
        wrong_body = np.hstack([np.eye(3, dtype=np.float64), -skew(p_imu)])
        J_wrong = np.vstack(
            [
                JL @ oL_T_imu[:3, :3] @ wrong_body,
                JR @ oR_T_imu[:3, :3] @ wrong_body,
            ]
        )

        J_numeric = self.projector.pose_jacobian_left_numeric(T, landmark_w, eps=1e-6)
        max_err = float(np.max(np.abs(J_wrong - J_numeric)))
        self.assertGreater(max_err, 1.0)

    def test_left_update_reduces_reprojection_error(self) -> None:
        landmark_w = np.array([7.5, -1.2, 0.6], dtype=np.float64)
        T_true = np.eye(4, dtype=np.float64)
        z_true = self.projector.predict_stereo(T_true, landmark_w)

        T_est = se3_exp(np.array([0.15, -0.05, 0.02, 0.03, -0.01, 0.025], dtype=np.float64)) @ T_true
        z_hat = self.projector.predict_stereo(T_est, landmark_w)
        residual = z_true - z_hat
        H = self.projector.pose_jacobian_left(T_est, landmark_w)
        dxi = np.linalg.lstsq(H, residual, rcond=None)[0] * 0.2

        err_before = float(np.linalg.norm(residual))
        err_left_plus = float(np.linalg.norm(z_true - self.projector.predict_stereo(se3_exp(dxi) @ T_est, landmark_w)))
        err_left_minus = float(np.linalg.norm(z_true - self.projector.predict_stereo(se3_exp(-dxi) @ T_est, landmark_w)))

        self.assertLess(err_left_plus, err_before)
        self.assertGreater(err_left_minus, err_left_plus)

    def test_vi_slam_smoke_dataset00_prefix(self) -> None:
        prefix = 240
        cfg = PipelineConfig()
        runner = ViSlamRunner(self.bundle.calib, cfg.imu, cfg.landmark, cfg.slam, use_gpu=False)
        result = runner.run(
            self.bundle.v_t[:prefix],
            self.bundle.w_t[:prefix],
            self.bundle.timestamps[:prefix],
            self.bundle.features[:, :, :prefix],
        )
        self.assertGreater(int(result.selected_feature_ids.size), 100)
        self.assertTrue(np.isfinite(result.mean_pose_residual_px))
        self.assertLess(float(result.mean_pose_residual_px), 10.0)
        self.assertGreater(int(np.max(result.accepted_joint_updates_per_frame)), 0)


if __name__ == "__main__":
    unittest.main()
