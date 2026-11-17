"""Chấm điểm detection theo chuẩn COCO (mAP@0.5, mAP@0.5:0.95).

Dùng cho dữ liệu **thật** có nhãn bbox (MVTI). Đây là chỉ số mà poster cam kết
(mAP@0.5 ≥ 0.70) và là thứ duy nhất trong bộ chỉ số có thể đo trực tiếp trên
ảnh thật - vì dataset công khai có nhãn bbox nhưng không có nhãn xung đột.

Cài đặt theo đúng định nghĩa COCO: với mỗi lớp, sắp dự đoán theo điểm tin cậy
giảm dần, ghép tham lam với ground truth theo IoU, rồi lấy Average Precision
bằng phép nội suy 101 điểm trên đường cong Precision-Recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import Detection


@dataclass
class DetectionMetrics:
    """Kết quả đánh giá detection."""

    ap_per_class: dict[str, float] = field(default_factory=dict)
    ap50_per_class: dict[str, float] = field(default_factory=dict)
    n_gt_per_class: dict[str, int] = field(default_factory=dict)
    #: Precision/Recall ở IoU 0,5 trên TOÀN BỘ danh sách dự đoán được chấm. Khi
    #: chấm mAP ở ngưỡng tin cậy rất thấp (0,001 theo chuẩn COCO), đây là điểm
    #: cuối của đường precision-recall: recall cao nhất có thể, precision thấp
    #: nhất. Không phải hiệu năng lúc vận hành.
    precision50: float = 0.0
    recall50: float = 0.0
    #: Precision/Recall ở IoU 0,5 khi chỉ giữ dự đoán có điểm ≥ ngưỡng vận hành
    #: của pipeline. Đây mới là con số mô tả hệ thống lúc chạy thật.
    operating_conf: float | None = None
    precision50_op: float | None = None
    recall50_op: float | None = None
    n_pred: int = 0
    n_gt: int = 0
    #: Số hộp chuẩn bị bỏ qua vì nhỏ hơn ``min_area`` (0 nếu chấm toàn bộ).
    n_gt_ignored: int = 0

    @property
    def map50(self) -> float:
        vals = [v for v in self.ap50_per_class.values() if v == v]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def map(self) -> float:
        vals = [v for v in self.ap_per_class.values() if v == v]
        return float(np.mean(vals)) if vals else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mAP50": round(self.map50, 4),
            "mAP50_95": round(self.map, 4),
            "precision50": round(self.precision50, 4),
            "recall50": round(self.recall50, 4),
            "operating_conf": self.operating_conf,
            "precision50_op": (None if self.precision50_op is None
                               else round(self.precision50_op, 4)),
            "recall50_op": (None if self.recall50_op is None
                            else round(self.recall50_op, 4)),
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "n_gt_ignored": self.n_gt_ignored,
            "per_class": {
                k: {
                    "AP50": round(self.ap50_per_class.get(k, float("nan")), 4),
                    "AP50_95": round(self.ap_per_class.get(k, float("nan")), 4),
                    "n_gt": self.n_gt_per_class.get(k, 0),
                }
                for k in sorted(self.n_gt_per_class)
            },
        }


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """AP theo nội suy 101 điểm của COCO."""
    if n_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Bao lồi trên: precision tại mỗi mức recall là max của mọi mức cao hơn.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, grid, side="left")
    out = np.zeros_like(grid)
    valid = idx < len(precision)
    out[valid] = precision[idx[valid]]
    return float(out.mean())


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp hộp ``a`` (P, 4) và ``b`` (G, 4), dạng x1y1x2y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _greedy_tp(ious: np.ndarray, thr: float) -> np.ndarray:
    """Ghép tham lam theo thứ tự điểm giảm dần (hàng đã sắp), một-một."""
    n_p, n_g = ious.shape
    tp = np.zeros(n_p, dtype=bool)
    if n_g == 0:
        return tp
    matched = np.zeros(n_g, dtype=bool)
    for i in range(n_p):
        row = np.where(matched, -1.0, ious[i])
        j = n_g - 1 - int(np.argmax(row[::-1]))      # hoà thì lấy hộp sau cùng
        if row[j] >= thr:
            matched[j] = True
            tp[i] = True
    return tp


def evaluate_detection(
    predictions: dict[int, list[Detection]],
    ground_truth: dict[int, list[Detection]],
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
    operating_conf: float | None = None,
    min_area: float = 0.0,
) -> DetectionMetrics:
    """So khớp dự đoán với nhãn chuẩn theo từng lớp và tính mAP.

    ``min_area`` > 0: hộp chuẩn nhỏ hơn ngưỡng (pixel²) thành vùng "bỏ qua"
    theo quy ước COCO (iscrowd/ignore) - không tính là bỏ sót, và dự đoán khớp
    với chúng không tính là báo nhầm. Dùng khi pipeline cố ý loại vật thể nhỏ
    hơn ``detection.min_box_area``; con số trên toàn bộ nhãn vẫn được báo riêng.

    Ghép theo đúng quy ước COCO: trong mỗi ảnh, dự đoán xét theo điểm giảm dần,
    nhận hộp chuẩn chưa ghép có IoU lớn nhất (≥ ngưỡng). Ma trận IoU tính một lần
    mỗi ảnh bằng numpy - bản trước gọi hàm IoU Python cho từng cặp, chạy hàng
    chục phút trên video 1080p nhiều đối tượng.
    """
    metrics = DetectionMetrics()
    classes = {d.cls for dets in ground_truth.values() for d in dets}
    classes |= {d.cls for dets in predictions.values() for d in dets}

    metrics.n_pred = sum(len(v) for v in predictions.values())
    metrics.n_gt = sum(len(v) for v in ground_truth.values())

    tp50_total = fp50_total = 0
    gt50_total = 0
    tp50_op = fp50_op = 0

    def _big(b) -> bool:
        return (b[2] - b[0]) * (b[3] - b[1]) >= min_area

    if min_area > 0:
        metrics.n_gt = sum(1 for v in ground_truth.values() for d in v if _big(d.bbox))
        metrics.n_gt_ignored = sum(len(v) for v in ground_truth.values()) - metrics.n_gt

    for cls in sorted(classes, key=lambda c: c.value):
        n_gt = sum(1 for dets in ground_truth.values() for d in dets
                   if d.cls is cls and _big(d.bbox))
        metrics.n_gt_per_class[cls.value] = n_gt
        if n_gt == 0:
            continue

        scores_all: list[np.ndarray] = []
        tps_all: dict[float, list[np.ndarray]] = {float(t): [] for t in iou_thresholds}
        ign_all: dict[float, list[np.ndarray]] = {float(t): [] for t in iou_thresholds}
        for f, dets in predictions.items():
            p = [d for d in dets if d.cls is cls]
            if not p:
                continue
            sc = np.array([d.score for d in p], dtype=float)
            order = np.argsort(-sc, kind="stable")
            pb = np.array([p[i].bbox for i in order], dtype=float)
            gts = [d.bbox for d in ground_truth.get(f, []) if d.cls is cls]
            g = [b for b in gts if _big(b)]
            small = [b for b in gts if not _big(b)]
            ious = _iou_matrix(pb, np.array(g, dtype=float).reshape(-1, 4))
            ious_s = (_iou_matrix(pb, np.array(small, dtype=float).reshape(-1, 4))
                      if small else None)
            scores_all.append(sc[order])
            for thr in tps_all:
                tp = _greedy_tp(ious, thr)
                tps_all[thr].append(tp)
                if ious_s is not None:
                    ign_all[thr].append(~tp & (ious_s.max(axis=1) >= thr))
                else:
                    ign_all[thr].append(np.zeros(len(tp), dtype=bool))

        if scores_all:
            scores = np.concatenate(scores_all)
            glob = np.argsort(-scores, kind="stable")
            scores = scores[glob]
        else:
            scores = np.zeros(0)
            glob = np.zeros(0, dtype=int)

        aps = []
        for thr, parts in tps_all.items():
            tp = (np.concatenate(parts)[glob] if parts else np.zeros(0, dtype=bool)).astype(float)
            keep = ~(np.concatenate(ign_all[thr])[glob] if parts else np.zeros(0, dtype=bool))
            tp = tp[keep]
            sc_thr = scores[keep]
            fp = 1.0 - tp
            aps.append(_average_precision(tp, fp, n_gt))
            if abs(thr - 0.5) < 1e-6:
                metrics.ap50_per_class[cls.value] = aps[-1]
                tp50_total += int(tp.sum())
                fp50_total += int(fp.sum())
                gt50_total += n_gt
                if operating_conf is not None:
                    k = int((sc_thr >= operating_conf).sum())
                    tp50_op += int(tp[:k].sum())
                    fp50_op += int(fp[:k].sum())

        metrics.ap_per_class[cls.value] = float(np.nanmean(aps))

    denom = tp50_total + fp50_total
    metrics.precision50 = tp50_total / denom if denom else 0.0
    metrics.recall50 = tp50_total / gt50_total if gt50_total else 0.0
    if operating_conf is not None:
        metrics.operating_conf = float(operating_conf)
        d_op = tp50_op + fp50_op
        metrics.precision50_op = tp50_op / d_op if d_op else 0.0
        metrics.recall50_op = tp50_op / gt50_total if gt50_total else 0.0
    return metrics
