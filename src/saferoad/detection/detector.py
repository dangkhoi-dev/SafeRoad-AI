"""Khối Detection — phát hiện phương tiện & người đi bộ.

Mặc định dùng YOLO11n với trọng số COCO: 5 lớp cần thiết (person, bicycle,
motorcycle, car, bus/truck) đã có sẵn trong COCO nên MVP chạy được ngay mà
không cần train. Notebook ``notebooks/01_finetune_yolo11n_colab.ipynb`` cho
phép fine-tune trên dữ liệu giao thông Việt Nam rồi thay trọng số vào đây.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import DetectionConfig
from ..types import COCO_TO_SAFEROAD, Detection, VehicleClass

log = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Giao diện chung — cho phép thay detector mà không đụng pipeline."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """Trả về danh sách Detection cho một khung hình BGR."""

    @property
    def name(self) -> str:
        return type(self).__name__


class YoloDetector(BaseDetector):
    """Detector dựa trên Ultralytics YOLO (mặc định YOLO11n).

    Chỉ giữ lại các lớp thuộc ``COCO_TO_SAFEROAD``; mọi lớp COCO khác (ghế, cây,
    đèn giao thông...) bị loại ngay ở tầng này để không lọt vào tracking.
    """

    def __init__(self, cfg: DetectionConfig):
        from ultralytics import YOLO  # import trễ để module này nạp nhanh

        weights = Path(cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Không tìm thấy trọng số '{weights}'.\n"
                f"Tải bằng: python scripts/download_assets.py"
            )
        self.cfg = cfg
        self.model = YOLO(str(weights))
        self._keep = set(COCO_TO_SAFEROAD)
        log.info("Đã nạp YOLO từ %s (device=%s)", weights, cfg.device)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            half=self.cfg.half,
            max_det=self.cfg.max_det,
            classes=sorted(self._keep),
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        clsid = boxes.cls.cpu().numpy().astype(int)

        for box, score, cid in zip(xyxy, conf, clsid):
            vclass = COCO_TO_SAFEROAD.get(int(cid))
            if vclass is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if (x2 - x1) * (y2 - y1) < self.cfg.min_box_area:
                continue
            out.append(
                Detection(bbox=(x1, y1, x2, y2), score=float(score), cls=vclass)
            )
        return out


class ReplayDetector(BaseDetector):
    """Phát lại detection đã có sẵn — dùng cho video synthetic.

    Simulator sinh ra bbox tuyệt đối chính xác. Khi chấm ablation ta cần tách
    được sai số của *detector* khỏi sai số của *conflict logic*; detector này
    cho phép chạy pipeline với detection hoàn hảo để cô lập phần còn lại.

    ``noise_px`` và ``miss_rate`` mô phỏng detector không hoàn hảo, giúp đo
    độ nhạy của TTC/PET với sai số đầu vào.
    """

    def __init__(
        self,
        per_frame: dict[int, list[Detection]],
        noise_px: float = 0.0,
        miss_rate: float = 0.0,
        seed: int = 0,
    ):
        self.per_frame = per_frame
        self.noise_px = noise_px
        self.miss_rate = miss_rate
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        dets = self.per_frame.get(frame_idx, [])
        if self.noise_px <= 0 and self.miss_rate <= 0:
            return list(dets)

        out: list[Detection] = []
        for d in dets:
            if self.miss_rate > 0 and self.rng.random() < self.miss_rate:
                continue
            if self.noise_px > 0:
                jitter = self.rng.normal(0.0, self.noise_px, size=4)
                bbox = tuple(float(v + j) for v, j in zip(d.bbox, jitter))
            else:
                bbox = d.bbox
            out.append(Detection(bbox=bbox, score=d.score, cls=d.cls))
        return out


def build_detector(cfg: DetectionConfig, **kwargs) -> BaseDetector:
    """Factory — chọn detector theo ``cfg.backend``."""
    if cfg.backend == "yolo":
        return YoloDetector(cfg)
    if cfg.backend == "replay":
        return ReplayDetector(**kwargs)
    raise ValueError(f"Backend detection không hợp lệ: {cfg.backend!r}")


__all__ = [
    "BaseDetector",
    "YoloDetector",
    "ReplayDetector",
    "build_detector",
    "VehicleClass",
]
