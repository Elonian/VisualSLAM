from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import expm


OPTICAL_T_CAM = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """Map twist vector [v, w] to se(3) matrix."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v = xi[:3]
    w = xi[3:]
    T = np.zeros((4, 4), dtype=np.float64)
    T[:3, :3] = skew(w)
    T[:3, 3] = v
    return T


def ad_se3(xi: np.ndarray) -> np.ndarray:
    """Adjoint representation ad_xi for se(3)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v = xi[:3]
    w = xi[3:]
    A = np.zeros((6, 6), dtype=np.float64)
    A[:3, :3] = skew(w)
    A[:3, 3:] = skew(v)
    A[3:, 3:] = skew(w)
    return A


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) twist vector to SE(3) pose."""
    return expm(hat_se3(xi))


def pose_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    iT = np.eye(4, dtype=np.float64)
    iT[:3, :3] = R.T
    iT[:3, 3] = -R.T @ t
    return iT


def homogenize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)


def project_point(K: np.ndarray, q_optical: np.ndarray) -> np.ndarray:
    q = np.asarray(q_optical, dtype=np.float64).reshape(3)
    z = q[2]
    if z <= 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float64)
    uvw = K @ (q / z)
    return uvw[:2]


def project_jacobian(K: np.ndarray, q_optical: np.ndarray) -> np.ndarray:
    """Return d(pixel)/d(q_optical_xyz), shape (2,3)."""
    x, y, z = np.asarray(q_optical, dtype=np.float64).reshape(3)
    if z <= 1e-8:
        return np.full((2, 3), np.nan, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    J = np.array(
        [
            [fx / z, 0.0, -fx * x / (z * z)],
            [0.0, fy / z, -fy * y / (z * z)],
        ],
        dtype=np.float64,
    )
    return J


@dataclass(frozen=True)
class StereoCalibration:
    K_left: np.ndarray
    K_right: np.ndarray
    camL_T_imu: np.ndarray
    camR_T_imu: np.ndarray


@dataclass(frozen=True)
class StereoProjector:
    calib: StereoCalibration

    def optical_T_world(self, world_T_imu: np.ndarray, cam_T_imu: np.ndarray) -> np.ndarray:
        imu_T_world = pose_inverse(world_T_imu)
        return OPTICAL_T_CAM @ cam_T_imu @ imu_T_world

    def predict_stereo(self, world_T_imu: np.ndarray, landmark_w: np.ndarray) -> np.ndarray:
        m_h = homogenize(landmark_w)
        oL_T_w = self.optical_T_world(world_T_imu, self.calib.camL_T_imu)
        oR_T_w = self.optical_T_world(world_T_imu, self.calib.camR_T_imu)
        qL = (oL_T_w @ m_h)[:3]
        qR = (oR_T_w @ m_h)[:3]

        uvL = project_point(self.calib.K_left, qL)
        uvR = project_point(self.calib.K_right, qR)
        return np.array([uvL[0], uvL[1], uvR[0], uvR[1]], dtype=np.float64)

    def landmark_jacobian(self, world_T_imu: np.ndarray, landmark_w: np.ndarray) -> np.ndarray:
        """Return d(z_stereo)/d(landmark_xyz), shape (4,3)."""
        m_h = homogenize(landmark_w)
        oL_T_w = self.optical_T_world(world_T_imu, self.calib.camL_T_imu)
        oR_T_w = self.optical_T_world(world_T_imu, self.calib.camR_T_imu)
        qL = (oL_T_w @ m_h)[:3]
        qR = (oR_T_w @ m_h)[:3]

        JL = project_jacobian(self.calib.K_left, qL)
        JR = project_jacobian(self.calib.K_right, qR)
        dql_dm = oL_T_w[:3, :3]
        dqr_dm = oR_T_w[:3, :3]
        HL = JL @ dql_dm
        HR = JR @ dqr_dm
        return np.vstack([HL, HR])

    def pose_jacobian_numeric(
        self,
        world_T_imu: np.ndarray,
        landmark_w: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Return d(z_stereo)/d(xi_pose), xi is body-frame se(3) perturbation, shape (4,6)."""
        J = np.zeros((4, 6), dtype=np.float64)
        for j in range(6):
            d = np.zeros(6, dtype=np.float64)
            d[j] = eps
            T_plus = world_T_imu @ se3_exp(d)
            T_minus = world_T_imu @ se3_exp(-d)
            z_plus = self.predict_stereo(T_plus, landmark_w)
            z_minus = self.predict_stereo(T_minus, landmark_w)
            J[:, j] = (z_plus - z_minus) / (2.0 * eps)
        return J


def triangulate_stereo(
    uv_left: np.ndarray,
    uv_right: np.ndarray,
    world_T_imu: np.ndarray,
    calib: StereoCalibration,
) -> Tuple[np.ndarray, bool]:
    """Triangulate one landmark into world coordinates. Returns (xyz, is_valid)."""
    uv_left = np.asarray(uv_left, dtype=np.float64).reshape(2)
    uv_right = np.asarray(uv_right, dtype=np.float64).reshape(2)

    oL_T_w = OPTICAL_T_CAM @ calib.camL_T_imu @ pose_inverse(world_T_imu)
    oR_T_w = OPTICAL_T_CAM @ calib.camR_T_imu @ pose_inverse(world_T_imu)
    P_L = calib.K_left @ oL_T_w[:3, :]
    P_R = calib.K_right @ oR_T_w[:3, :]

    A = np.vstack(
        [
            uv_left[0] * P_L[2, :] - P_L[0, :],
            uv_left[1] * P_L[2, :] - P_L[1, :],
            uv_right[0] * P_R[2, :] - P_R[0, :],
            uv_right[1] * P_R[2, :] - P_R[1, :],
        ]
    )
    try:
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        if abs(X[3]) < 1e-10:
            return np.zeros(3, dtype=np.float64), False
        xyz = X[:3] / X[3]
        if not np.isfinite(xyz).all():
            return np.zeros(3, dtype=np.float64), False
        zL = (oL_T_w @ homogenize(xyz))[2]
        zR = (oR_T_w @ homogenize(xyz))[2]
        valid = bool(zL > 1e-6 and zR > 1e-6)
        return xyz, valid
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64), False
