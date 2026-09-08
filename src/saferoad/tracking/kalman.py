"""Kalman filter cho bounding box, dùng trong ByteTrack.

Vector trạng thái 8 chiều::

    x = [cx, cy, a, h, vx, vy, va, vh]^T

trong đó ``(cx, cy)`` là tâm bbox, ``a`` là tỉ lệ khung (w/h), ``h`` là chiều
cao, và 4 thành phần còn lại là đạo hàm theo thời gian.

Mô hình vận tốc không đổi (constant velocity) — đủ tốt cho khoảng thời gian
giữa 2 frame (~33 ms) kể cả khi xe đang tăng/giảm tốc.
"""

from __future__ import annotations

import numpy as np

#: Bảng phân vị chi-square 0.95 theo bậc tự do — dùng cho gating Mahalanobis.
CHI2_INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592}


class KalmanBoxFilter:
    """Kalman filter cho bbox với mô hình vận tốc không đổi.

    Nhiễu quá trình và nhiễu đo được đặt **tỉ lệ với chiều cao bbox**: đối tượng
    ở gần camera (bbox lớn) có sai số pixel tuyệt đối lớn hơn đối tượng ở xa,
    nên scale theo ``h`` giữ cho filter ổn định trên toàn khung hình.
    """

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        ndim, dt = 4, 1.0
        # Ma trận chuyển trạng thái: vị trí cộng thêm vận tốc * dt.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Ma trận quan sát: chỉ đo được 4 thành phần vị trí.
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity

    # ------------------------------------------------------------------ #
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Khởi tạo từ phép đo đầu tiên ``[cx, cy, a, h]``.

        Vận tốc khởi tạo bằng 0 nhưng với hiệp phương sai lớn, để filter nhanh
        chóng "học" được vận tốc thật từ vài phép đo kế tiếp.
        """
        mean_pos = np.asarray(measurement, dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bước dự đoán — đẩy trạng thái tiến 1 frame."""
        h = mean[3]
        std_pos = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h]
        std_vel = [self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chiếu trạng thái sang không gian đo."""
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        innovation_cov = np.diag(np.square(std))

        mean_p = self._update_mat @ mean
        cov_p = self._update_mat @ covariance @ self._update_mat.T
        return mean_p, cov_p + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bước hiệu chỉnh — hợp nhất phép đo mới vào trạng thái."""
        proj_mean, proj_cov = self.project(mean, covariance)

        # Giải hệ thay vì nghịch đảo ma trận: ổn định số học hơn.
        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._update_mat.T).T)
        ).T
        innovation = np.asarray(measurement, dtype=float) - proj_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Khoảng cách Mahalanobis bình phương giữa trạng thái và các phép đo."""
        proj_mean, proj_cov = self.project(mean, covariance)
        chol = np.linalg.cholesky(proj_cov)
        d = np.atleast_2d(measurements) - proj_mean
        z = np.linalg.solve(chol, d.T)
        return np.sum(z * z, axis=0)


def xyxy_to_xyah(bbox) -> np.ndarray:
    """(x1, y1, x2, y2) → (cx, cy, aspect, height)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    h = max(h, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=float)


def xyah_to_xyxy(mean) -> tuple[float, float, float, float]:
    """(cx, cy, aspect, height) → (x1, y1, x2, y2)."""
    cx, cy, a, h = mean[:4]
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
