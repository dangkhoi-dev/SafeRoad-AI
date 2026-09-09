"""Đánh giá trên dữ liệu **thật** (MVTI) — detection + tracking + tốc độ.

Bổ sung cho :mod:`saferoad.evaluation.ablation` (chạy trên tập mô phỏng). Ranh
giới giữa hai bên là có chủ đích và phải nêu rõ trong báo cáo:

* dữ liệu thật có nhãn **bbox và ID đối tượng** ⇒ đo được mAP, IDF1, MOTA, FPS;
* dữ liệu thật **không có nhãn xung đột** ⇒ Precision/Recall của near-miss chỉ
  đo được trên tập mô phỏng, nơi ta biết chính xác quỹ đạo giải tích.

Trộn lẫn hai thứ đó rồi báo cáo một con số duy nhất sẽ là gian dối về mặt
phương pháp.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..data.mvti import MvtiSequence
from ..types import Track
from .metrics import TrackingMetrics

log = logging.getLogger(__name__)


def tracking_metrics_from_boxes(
    tracks: dict[int, Track],
    seq: MvtiSequence,
    iou_threshold: float = 0.3,
) -> tuple[dict[int, int], TrackingMetrics]:
    """Chấm tracking trên dữ liệu thật, ghép theo IoU của bbox.

    Khác với tập mô phỏng (ghép theo khoảng cách mặt đất), ở đây ta ghép trong
    **không gian ảnh**: dữ liệu thật không có toạ độ mặt đất chuẩn, và ghép theo
    IoU chính là quy ước của các benchmark MOT.
    """
    from ..types import iou as box_iou

    states_by_frame: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid, track in tracks.items():
        for st in track.history:
            states_by_frame[st.frame_idx].append((tid, st.bbox))

    votes: dict[int, Counter] = defaultdict(Counter)
    gt_hits: dict[int, list[tuple[int, int]]] = defaultdict(list)
    total_gt = total_matched = false_pos = 0

    for frame_idx, gt_dets in seq.detections.items():
        ids = seq.id_map.get(frame_idx, {})
        truth = [
            (ids.get(i, -1), d.bbox) for i, d in enumerate(gt_dets) if ids.get(i, -1) >= 0
        ]
        total_gt += len(truth)
        preds = states_by_frame.get(frame_idx, [])

        pairs: list[tuple[float, int, int]] = []
        for tid, pbox in preds:
            for oid, gbox in truth:
                v = box_iou(pbox, gbox)
                if v >= iou_threshold:
                    pairs.append((-v, tid, oid))
        pairs.sort()

        used_t: set[int] = set()
        used_g: set[int] = set()
        for _neg_iou, tid, oid in pairs:
            if tid in used_t or oid in used_g:
                continue
            used_t.add(tid)
            used_g.add(oid)
            votes[tid][oid] += 1
            gt_hits[oid].append((frame_idx, tid))
            total_matched += 1
        false_pos += len(preds) - len(used_t)

    mapping = {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}

    switches = 0
    frags = []
    for oid, hits in gt_hits.items():
        hits.sort()
        seq_ids = [tid for _f, tid in hits]
        changes = sum(1 for a, b in zip(seq_ids, seq_ids[1:]) if a != b)
        switches += changes
        frags.append(changes + 1)

    gt_frames = Counter()
    for frame_idx, gt_dets in seq.detections.items():
        ids = seq.id_map.get(frame_idx, {})
        for i in range(len(gt_dets)):
            oid = ids.get(i, -1)
            if oid >= 0:
                gt_frames[oid] += 1
    mostly = sum(
        1 for oid, n in gt_frames.items() if len(gt_hits.get(oid, [])) >= 0.8 * n
    )

    recall = total_matched / total_gt if total_gt else 0.0
    precision = (
        total_matched / (total_matched + false_pos) if (total_matched + false_pos) else 0.0
    )
    idf1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mota = max(
        0.0,
        1.0 - ((total_gt - total_matched) + false_pos + switches) / max(total_gt, 1),
    )

    return mapping, TrackingMetrics(
        idf1=round(idf1, 4), mota=round(mota, 4), id_switches=switches,
        n_gt_tracks=len(gt_frames), n_pred_tracks=len(tracks),
        mostly_tracked=mostly,
        fragmentation=float(np.mean(frags)) if frags else 0.0,
    )


def conflict_statistics(events: list) -> dict[str, Any]:
    """Thống kê mô tả về near-miss trên dữ liệu thật.

    Không có nhãn chuẩn nên **không** báo cáo Precision/Recall ở đây — chỉ mô tả
    những gì hệ thống quan sát được, đúng như một báo cáo khảo sát hiện trường.
    """
    if not events:
        return {"total": 0}

    ttcs = [e.ttc for e in events if e.ttc is not None]
    pets = [e.pet for e in events if e.pet is not None]
    risks = [e.risk_score for e in events]

    def stats(v: list[float]) -> dict[str, float] | None:
        if not v:
            return None
        a = np.asarray(v, dtype=float)
        return {
            "min": round(float(a.min()), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "median": round(float(np.median(a)), 3),
            "p75": round(float(np.percentile(a, 75)), 3),
            "max": round(float(a.max()), 3),
        }

    return {
        "total": len(events),
        "severe_ttc_lt_1_5": sum(1 for t in ttcs if t < 1.5),
        "by_type": dict(Counter(e.conflict_type.value for e in events)),
        "by_level": dict(Counter(e.risk_level.value for e in events)),
        "ttc": stats(ttcs),
        "pet": stats(pets),
        "risk_score": stats(risks),
    }
