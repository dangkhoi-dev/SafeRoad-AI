"""Quy đổi toạ độ ảnh → mặt đất (bird's-eye view) bằng homography.

Vì sao bắt buộc phải có bước này
--------------------------------
TTC và PET là đại lượng **vật lý** (giây), tính từ khoảng cách (mét) và vận tốc
(m/s). Nếu tính thẳng trên pixel, cùng một khoảng cách thực tế sẽ cho ra số pixel
khác nhau tuỳ vị trí trong khung hình (xa camera thì nhỏ lại), khiến TTC ở nửa
trên khung hình sai lệch có hệ thống so với nửa dưới. Homography khử đúng hiệu
ứng phối cảnh đó.

Giả thiết: mọi đối tượng di chuyển trên **một mặt phẳng** (mặt đường). Với camera
đặt cao 10-12 m nhìn xuống giao lộ, giả thiết này đủ chính xác — sai số chủ yếu
đến từ việc bbox không chạm đất chính xác, chứ không phải từ giả thiết mặt phẳng.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config import HomographyConfig

log = logging.getLogger(__name__)


class GroundPlane:
    """Bộ quy đổi pixel ↔ mét.

    Có 2 chế độ:

    * **Calibrated** — người dùng cung cấp ≥4 cặp điểm ảnh/thế giới trong config.
      Đây là chế độ nên dùng cho dữ liệu thật: đo 4 điểm mốc trên mặt đường
      (vạch kẻ, góc đảo giao thông) bằng Google Maps hoặc thước, nhập vào YAML.
    * **Fallback** — không có calibration thì dùng một tỉ lệ phẳng
      ``fallback_scale`` (m/px). Pipeline vẫn chạy nhưng TTC chỉ đúng tương đối;
      báo cáo phải ghi rõ điều này.
    """

    def __init__(self, cfg: HomographyConfig, frame_size: tuple[int, int] | None = None):
        self.cfg = cfg
        self.frame_size = frame_size
        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None
        self.calibrated = False

        img_pts = cfg.image_points
        wld_pts = cfg.world_points
        if len(img_pts) >= 4 and len(img_pts) == len(wld_pts):
            src = np.asarray(img_pts, dtype=np.float32)
            dst = np.asarray(wld_pts, dtype=np.float32)
            if len(img_pts) == 4:
                self.H = cv2.getPerspectiveTransform(src, dst)
            else:
                # Nhiều hơn 4 điểm → dùng RANSAC để chịu được điểm đo lệch.
                self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if self.H is not None:
                self.H_inv = np.linalg.inv(self.H)
                self.calibrated = True
                log.info("Homography đã calibrate từ %d cặp điểm", len(img_pts))
        if not self.calibrated:
            log.warning(
                "Chưa có calibration — dùng fallback_scale=%.4f m/px. "
                "TTC/PET chỉ chính xác tương đối.",
                cfg.fallback_scale,
            )

    # ------------------------------------------------------------------ #
    def to_ground(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Một điểm ảnh (px) → toạ độ mặt đất (m)."""
        if self.calibrated and self.H is not None:
            src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.H)
            return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))
        s = self.cfg.fallback_scale
        return (pt[0] * s, pt[1] * s)

    def to_ground_batch(self, pts: np.ndarray) -> np.ndarray:
        """Quy đổi hàng loạt — ``pts`` dạng (N, 2)."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if self.calibrated and self.H is not None:
            return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return (pts.reshape(-1, 2) * self.cfg.fallback_scale)

    def to_image(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Toạ độ mặt đất (m) → điểm ảnh (px). Dùng để vẽ overlay."""
        if self.calibrated and self.H_inv is not None:
            src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.H_inv)
            return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))
        s = self.cfg.fallback_scale
        return (pt[0] / s, pt[1] / s)

    def reprojection_error(self) -> float | None:
        """Sai số tái chiếu trung bình (m) trên chính các điểm calibration.

        Đây là chỉ số bắt buộc phải báo cáo: nếu sai số > ~0.5 m thì TTC không
        còn đáng tin và phải đo lại điểm mốc.
        """
        if not self.calibrated or self.H is None:
            return None
        src = np.asarray(self.cfg.image_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(self.cfg.world_points, dtype=np.float32)
        proj = cv2.perspectiveTransform(src, self.H).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj - dst, axis=1)))

    def scale_at(self, pt: tuple[float, float]) -> float:
        """Ước lượng số mét trên mỗi pixel tại một vị trí trong ảnh.

        Dùng để chọn độ dày nét vẽ overlay theo khoảng cách, và để cảnh báo khi
        đối tượng ở quá xa (1 px ≈ nhiều mét ⇒ vị trí rất nhiễu).
        """
        p0 = self.to_ground(pt)
        p1 = self.to_ground((pt[0] + 1.0, pt[1]))
        p2 = self.to_ground((pt[0], pt[1] + 1.0))
        return 0.5 * (float(np.hypot(*(np.subtract(p1, p0)))) +
                      float(np.hypot(*(np.subtract(p2, p0)))))


def make_default_homography(
    frame_w: int, frame_h: int, camera_height_m: float = 12.0, fov_width_m: float = 40.0
) -> HomographyConfig:
    """Sinh homography xấp xỉ khi chưa có calibration thực địa.

    Mô hình hoá camera nhìn nghiêng xuống: một hình thang trong ảnh (đáy rộng ở
    gần camera, đỉnh hẹp ở xa) tương ứng với một hình chữ nhật trên mặt đất.
    Đây **không thay thế** calibration thật, nhưng cho ra tỉ lệ hợp lý theo độ
    sâu và tốt hơn nhiều so với dùng một hằng số m/px.

    Tham số:
        frame_w, frame_h: kích thước khung hình (px).
        camera_height_m: độ cao camera (m).
        fov_width_m: bề rộng thực tế (m) mà cạnh đáy khung hình bao phủ.
    """
    # Hình thang trong ảnh: 60% chiều cao dưới của khung hình.
    top_y = frame_h * 0.40
    bot_y = frame_h * 0.98
    top_half = frame_w * 0.18   # ở xa: hẹp
    bot_half = frame_w * 0.50   # ở gần: rộng hết khung
    cx = frame_w / 2.0

    image_points = [
        [cx - bot_half, bot_y],
        [cx + bot_half, bot_y],
        [cx + top_half, top_y],
        [cx - top_half, top_y],
    ]
    # Trên mặt đất là hình chữ nhật. Chiều sâu ước lượng từ độ cao camera:
    # góc nhìn nghiêng ~40° cho vùng phủ sâu gấp ~2.2 lần bề rộng đáy.
    depth = fov_width_m * 2.2 * (camera_height_m / 12.0)
    half_w = fov_width_m / 2.0
    world_points = [
        [-half_w, 0.0],
        [half_w, 0.0],
        [half_w, depth],
        [-half_w, depth],
    ]
    return HomographyConfig(image_points=image_points, world_points=world_points)
