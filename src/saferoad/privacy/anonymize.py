"""Ẩn danh dữ liệu — làm mờ khuôn mặt và biển số.

Tuân thủ Điều 5 Thể lệ cuộc thi và nguyên tắc bảo vệ dữ liệu cá nhân: video thu
tại nơi công cộng vẫn chứa thông tin nhận dạng (mặt người, biển số xe), nên phải
ẩn danh **trước khi** lưu trữ hoặc đưa vào hồ sơ dự thi.

Chiến lược không cần thêm model
--------------------------------
Thay vì chạy thêm một detector khuôn mặt/biển số (tốn FPS và lại cần dữ liệu
huấn luyện riêng), ta tận dụng ngay bbox mà detector chính đã sinh ra:

* **Khuôn mặt** nằm ở ~28% phía trên của bbox người đi bộ.
* **Biển số** nằm ở dải 55-95% chiều cao bbox phương tiện, giữa theo chiều ngang.

Cách này bắt được vùng nhạy cảm với chi phí gần như bằng 0. Vì làm mờ dư ra một
chút xung quanh, nó thiên về **an toàn** (ẩn nhiều hơn cần thiết) — đúng hướng
mong muốn khi xử lý dữ liệu cá nhân.

Với dữ liệu công bố ra ngoài, nên chạy thêm một detector khuôn mặt chuyên dụng;
xem ``docs/protocol_du_lieu.md``.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PrivacyConfig
from ..types import Detection, VehicleClass


def _blur_region(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, kernel: int
) -> None:
    """Làm mờ Gaussian một vùng chữ nhật, ngay trên ``frame`` (in-place)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return

    roi = frame[y1:y2, x1:x2]
    # Kernel phải là số lẻ và không lớn hơn vùng cần làm mờ.
    k = min(kernel, (x2 - x1) // 2 * 2 - 1, (y2 - y1) // 2 * 2 - 1)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


class Anonymizer:
    """Ẩn danh khung hình dựa trên bbox của detector chính."""

    def __init__(self, cfg: PrivacyConfig):
        self.cfg = cfg
        self.faces_blurred = 0
        self.plates_blurred = 0

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Trả về khung hình đã ẩn danh. Không sửa ``frame`` gốc."""
        if not self.cfg.enabled:
            return frame

        out = frame.copy()
        cfg = self.cfg

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            if w < 4 or h < 4:
                continue

            if det.cls is VehicleClass.PEDESTRIAN:
                if not cfg.blur_faces:
                    continue
                # Vùng đầu: phần trên bbox, thu hẹp hai bên cho sát khuôn mặt.
                face_h = h * cfg.face_ratio
                pad_x = w * 0.15
                _blur_region(
                    out,
                    int(x1 + pad_x), int(y1),
                    int(x2 - pad_x), int(y1 + face_h),
                    cfg.blur_kernel,
                )
                self.faces_blurred += 1
            else:
                if not cfg.blur_plates:
                    continue
                # Vùng biển số: dải ngang phía dưới thân xe.
                lo, hi = cfg.plate_band
                pad_x = w * 0.22
                _blur_region(
                    out,
                    int(x1 + pad_x), int(y1 + h * lo),
                    int(x2 - pad_x), int(y1 + h * hi),
                    cfg.blur_kernel,
                )
                self.plates_blurred += 1

                # Người ngồi trên xe máy: làm mờ thêm phần trên bbox.
                if det.cls in (VehicleClass.MOTORCYCLE, VehicleClass.BICYCLE) and cfg.blur_faces:
                    _blur_region(
                        out,
                        int(x1 + w * 0.25), int(y1),
                        int(x2 - w * 0.25), int(y1 + h * 0.30),
                        cfg.blur_kernel,
                    )
                    self.faces_blurred += 1

        return out

    @property
    def stats(self) -> dict[str, int]:
        return {
            "faces_blurred": self.faces_blurred,
            "plates_blurred": self.plates_blurred,
        }


def anonymize_video(
    src_path: str,
    dst_path: str,
    detector,
    cfg: PrivacyConfig,
    progress_every: int = 300,
) -> dict[str, int]:
    """Chạy ẩn danh trên toàn bộ một video và ghi ra file mới.

    Đây là bước tiền xử lý bắt buộc cho mọi video tự quay trước khi đưa vào hồ sơ.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    anon = Anonymizer(cfg)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(frame, idx)
            writer.write(anon.apply(frame, dets))
            idx += 1
            if progress_every and idx % progress_every == 0:
                print(f"  ẩn danh {idx} frames...", flush=True)
    finally:
        cap.release()
        writer.release()

    stats = anon.stats
    stats["frames"] = idx
    return stats
