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



def _precision_kwargs(quantize: int | None) -> dict:
    """Chọn tham số độ chính xác đúng với phiên bản Ultralytics đang cài.

    Ultralytics thay cờ ``half`` bằng ``quantize`` và in một dòng cảnh báo cho
    MỖI lần gọi ``predict`` nếu còn dùng tên cũ — chạy một video 2.441 frame là
    2.441 dòng cảnh báo lấp kín toàn bộ log, che mất những thông báo thật sự
    cần đọc. Đồng thời, truyền ``half=False`` chẳng làm gì ngoài việc kích hoạt
    cảnh báo đó, nên khi chạy FP32 ta không truyền tham số nào cả.
    """
    if not quantize:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": int(quantize)}
    except Exception:  # pragma: no cover - phụ thuộc phiên bản
        pass
    return {"half": int(quantize) == 16}


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
        self._precision_kwargs = _precision_kwargs(cfg.quantize)
        log.info("Đã nạp YOLO từ %s (device=%s)", weights, cfg.device)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            **self._precision_kwargs,
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



def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """IoU của hai hộp (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _merge(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """Triệt tiêu trùng lặp giữa các ô, xét riêng từng lớp.

    Không gộp chéo lớp: một người đi bộ đứng cạnh xe máy có hộp chồng nhau
    nhiều, nhưng đó là hai đối tượng thật chứ không phải một vật bị đếm hai lần.
    """
    out: list[Detection] = []
    for cls in {d.cls for d in dets}:
        same = sorted((d for d in dets if d.cls is cls),
                      key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []
        for d in same:
            if all(_iou(d.bbox, k.bbox) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


class TiledYoloDetector(BaseDetector):
    """Chạy YOLO trên từng ô của khung hình rồi gộp kết quả lại.

    Vì sao cần: camera giao thông đặt cao nhìn chếch xuống, phương tiện ở xa chỉ
    chiếm vài chục pixel. Một lượt suy luận trên cả khung hình 1024×640 thu nhỏ
    về 640 px làm chúng teo thêm một lần nữa, và detector bỏ sót gần hết — đo
    trên tập MVTI, cách chạy một lượt chỉ đạt recall@0,5 ≈ 0,20.

    Cắt khung hình thành lưới ô chồng lấn và chạy detector trên từng ô ở độ phân
    giải gốc của ô giúp mỗi vật thể chiếm nhiều pixel hơn trong ảnh đưa vào model.
    Ý tưởng này lấy từ SAHI (Slicing Aided Hyper Inference).

    Ba chi tiết quyết định kết quả:

    * **Ô phải chồng lấn nhau.** Không chồng lấn thì một chiếc xe nằm đúng trên
      đường cắt bị xẻ làm đôi và không ô nào nhìn thấy nó trọn vẹn.
    * **Vẫn phải chạy một lượt toàn khung.** Ô nhỏ không chứa nổi vật thể lớn ở
      gần camera; lượt toàn khung bù đúng phần đó.
    * **Gộp kết quả theo từng lớp.** Vùng chồng lấn khiến cùng một vật xuất hiện
      ở hai ô, phải triệt tiêu bớt; nhưng gộp chéo lớp sẽ xoá nhầm hai đối tượng
      thật đứng sát nhau.

    Cái giá phải trả là tốc độ: với lưới 2×3 kèm lượt toàn khung, mỗi khung hình
    tốn bảy lần suy luận thay vì một.
    """

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.inner = YoloDetector(cfg)
        self.rows = max(1, int(cfg.tile_rows))
        self.cols = max(1, int(cfg.tile_cols))
        self.overlap = min(max(float(cfg.tile_overlap), 0.0), 0.9)
        log.info(
            "Detector chia ô: lưới %d×%d, chồng lấn %.0f%%, lượt toàn khung: %s",
            self.rows, self.cols, self.overlap * 100,
            "có" if cfg.tile_full_frame else "không",
        )

    def _tiles(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """Toạ độ các ô, đã tính phần chồng lấn và kẹp trong khung hình."""
        th, tw = h / self.rows, w / self.cols
        oh, ow = th * self.overlap, tw * self.overlap
        boxes = []
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = int(max(0, c * tw - ow))
                y1 = int(max(0, r * th - oh))
                x2 = int(min(w, (c + 1) * tw + ow))
                y2 = int(min(h, (r + 1) * th + oh))
                if x2 - x1 > 1 and y2 - y1 > 1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        h, w = frame.shape[:2]
        found: list[Detection] = []

        for (x1, y1, x2, y2) in self._tiles(h, w):
            tile = frame[y1:y2, x1:x2]
            for d in self.inner.detect(tile, frame_idx):
                bx1, by1, bx2, by2 = d.bbox
                found.append(
                    Detection(
                        bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                        score=d.score, cls=d.cls,
                    )
                )

        if self.cfg.tile_full_frame:
            found.extend(self.inner.detect(frame, frame_idx))

        return _merge(found, self.cfg.tile_merge_iou)


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
    if cfg.backend == "tiled":
        return TiledYoloDetector(cfg)
    if cfg.backend == "replay":
        return ReplayDetector(**kwargs)
    raise ValueError(f"Backend detection không hợp lệ: {cfg.backend!r}")


__all__ = [
    "BaseDetector",
    "YoloDetector",
    "TiledYoloDetector",
    "ReplayDetector",
    "build_detector",
    "VehicleClass",
]
