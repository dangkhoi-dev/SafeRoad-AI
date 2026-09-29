"""Chấm điểm detection theo chuẩn COCO (mAP@0.5, mAP@0.5:0.95).

Dùng cho dữ liệu **thật** có nhãn bbox (MVTI). Đây là chỉ số mà poster cam kết
(mAP@0.5 ≥ 0.70) và là thứ duy nhất trong bộ chỉ số có thể đo trực tiếp trên
ảnh thật - vì dataset công khai có nhãn bbox nhưng không có nhãn xung đột.

Cài đặt theo đúng định nghĩa COCO: với mỗi lớp, sắp dự đoán theo điểm tin cậy
giảm dần, ghép tham lam với ground truth theo IoU, rồi lấy Average Precision
bằng phép nội suy 101 điểm trên đường cong Precision-Recall.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import Detection, VehicleClass, iou


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


def evaluate_detection(
    predictions: dict[int, list[Detection]],
    ground_truth: dict[int, list[Detection]],
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
    operating_conf: float | None = None,
) -> DetectionMetrics:
    """So khớp dự đoán với nhãn chuẩn theo từng lớp và tính mAP.

    Tham số:
        predictions: ``{frame_idx: [Detection, ...]}`` do detector sinh.
        ground_truth: ``{frame_idx: [Detection, ...]}`` nhãn chuẩn.
    """
    metrics = DetectionMetrics()
    classes = {d.cls for dets in ground_truth.values() for d in dets}
    classes |= {d.cls for dets in predictions.values() for d in dets}

    metrics.n_pred = sum(len(v) for v in predictions.values())
    metrics.n_gt = sum(len(v) for v in ground_truth.values())

    tp50_total = fp50_total = 0
    gt50_total = 0
    tp50_op = fp50_op = 0

    for cls in sorted(classes, key=lambda c: c.value):
        gt_by_frame = {
            f: [d for d in dets if d.cls is cls] for f, dets in ground_truth.items()
        }
        n_gt = sum(len(v) for v in gt_by_frame.values())
        metrics.n_gt_per_class[cls.value] = n_gt
        if n_gt == 0:
            continue

        # Gom mọi dự đoán của lớp này, sắp theo điểm tin cậy giảm dần.
        preds: list[tuple[float, int, Detection]] = []
        for f, dets in predictions.items():
            for d in dets:
                if d.cls is cls:
                    preds.append((d.score, f, d))
        preds.sort(key=lambda p: -p[0])

        aps = []
        for thr in iou_thresholds:
            matched: dict[int, set[int]] = defaultdict(set)
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, (_score, frame, det) in enumerate(preds):
                gts = gt_by_frame.get(frame, [])
                best_j, best_iou = -1, thr
                for j, gt in enumerate(gts):
                    if j in matched[frame]:
                        continue
                    v = iou(det.bbox, gt.bbox)
                    if v >= best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    matched[frame].add(best_j)
                    tp[i] = 1
                else:
                    fp[i] = 1

            aps.append(_average_precision(tp, fp, n_gt))
            if abs(thr - 0.5) < 1e-6:
                metrics.ap50_per_class[cls.value] = aps[-1]
                tp50_total += int(tp.sum())
                fp50_total += int(fp.sum())
                gt50_total += n_gt
                if operating_conf is not None:
                    # Dự đoán đã xếp theo điểm giảm dần và được ghép tham lam
                    # theo đúng thứ tự đó, nên kết quả ghép của k dự đoán đầu
                    # (điểm ≥ ngưỡng) trùng khít với việc chạy riêng tập con ấy.
                    k = sum(1 for s, _f, _d in preds if s >= operating_conf)
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
