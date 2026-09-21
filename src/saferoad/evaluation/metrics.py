"""Chấm điểm hệ thống so với nhãn chuẩn.

Ba nhóm chỉ số, tương ứng ba cam kết trong poster:

* **Conflict Precision / Recall / F1** — hệ thống có bắt đúng các near-miss thật
  và có ít báo động giả không.
* **TTC MAE** — sai số ước lượng thời gian tới va chạm, tính trên các sự kiện
  khớp đúng.
* **Tracking IDF1 / MOTA** — chất lượng gán ID, vì mọi thứ phía sau đều phụ
  thuộc vào việc một chiếc xe giữ nguyên một ID.

Bài toán ghép cặp
-----------------
Một sự kiện dự đoán và một nhãn chuẩn được coi là **khớp** khi chúng nói về cùng
một cặp đối tượng và lệch nhau không quá ``time_tol`` giây. Vì hệ thống dùng ID
do tracker sinh ra còn nhãn chuẩn dùng ID của simulator, ta phải **ánh xạ ID**
trước — xem :func:`build_id_mapping`.

Ghép được thực hiện bằng thuật toán tham lam theo thứ tự lệch thời gian tăng
dần, mỗi nhãn và mỗi dự đoán chỉ được dùng một lần. Cách này tránh việc một dự
đoán "ăn" nhiều nhãn và làm Recall bị thổi phồng.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from ..types import ConflictEvent, GroundTruthConflict, Track, pair_key


@dataclass
class ConflictMetrics:
    """Kết quả chấm điểm phát hiện xung đột."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    ttc_errors: list[float] = field(default_factory=list)
    pet_errors: list[float] = field(default_factory=list)
    time_errors: list[float] = field(default_factory=list)
    type_correct: int = 0
    type_total: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def ttc_mae(self) -> float:
        return float(np.mean(self.ttc_errors)) if self.ttc_errors else float("nan")

    @property
    def ttc_rmse(self) -> float:
        if not self.ttc_errors:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self.ttc_errors))))

    @property
    def pet_mae(self) -> float:
        return float(np.mean(self.pet_errors)) if self.pet_errors else float("nan")

    @property
    def time_mae(self) -> float:
        return float(np.mean(self.time_errors)) if self.time_errors else float("nan")

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / self.type_total if self.type_total else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "ttc_mae": round(self.ttc_mae, 4) if self.ttc_errors else None,
            "ttc_rmse": round(self.ttc_rmse, 4) if self.ttc_errors else None,
            "pet_mae": round(self.pet_mae, 4) if self.pet_errors else None,
            "time_mae": round(self.time_mae, 4) if self.time_errors else None,
            "type_accuracy": (
                round(self.type_accuracy, 4) if self.type_total else None
            ),
            "n_matched_ttc": len(self.ttc_errors),
        }


@dataclass
class TrackingMetrics:
    """Chỉ số chất lượng tracking."""

    idf1: float = 0.0
    mota: float = 0.0
    id_switches: int = 0
    n_gt_tracks: int = 0
    n_pred_tracks: int = 0
    mostly_tracked: int = 0
    fragmentation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "idf1": round(self.idf1, 4),
            "mota": round(self.mota, 4),
            "id_switches": self.id_switches,
            "n_gt_tracks": self.n_gt_tracks,
            "n_pred_tracks": self.n_pred_tracks,
            "mostly_tracked": self.mostly_tracked,
            "fragmentation": round(self.fragmentation, 3),
        }


# --------------------------------------------------------------------------- #
def build_gt_points(
    detections: dict[int, list],
    id_map: dict[int, dict[int, int]],
    vehicles: list,
    fps: float,
    min_box_area: float = 150.0,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Tập điểm chuẩn dùng để chấm tracking: ``{frame: {vid: (x, y)}}``.

    Chỉ giữ các đối tượng mà hệ thống **thực sự có cơ hội nhìn thấy**:

    * đúng tần số khung hình của video (30 Hz), không phải tần số mô phỏng
      (60 Hz) — nếu không, IDF1 bị chặn trên ở 0.5 một cách nhân tạo;
    * bbox đủ lớn (``min_box_area``) — đối tượng nhỏ hơn nằm ngoài khả năng
      phát hiện của detector, nên tính chúng vào mẫu số là chấm điểm hệ thống
      trên thứ nó đã chủ động loại bỏ.

    Giới hạn phạm vi đánh giá theo vùng cảm biến thật là thực hành chuẩn trong
    benchmark tracking (MOT bỏ qua vùng ``ignore``); ở đây vùng đó được định
    nghĩa tường minh bằng kích thước bbox.
    """
    veh_by_id = {v.vid: v for v in vehicles}
    out: dict[int, dict[int, tuple[float, float]]] = {}

    for frame_idx, dets in detections.items():
        t = frame_idx / fps
        per_frame: dict[int, tuple[float, float]] = {}
        for i, det in enumerate(dets):
            if det.area < min_box_area:
                continue
            vid = id_map.get(frame_idx, {}).get(i)
            veh = veh_by_id.get(vid) if vid is not None else None
            if veh is None or veh.times.size == 0:
                continue
            if t < veh.times[0] or t > veh.times[-1]:
                continue
            idx = min(int(np.searchsorted(veh.times, t)), len(veh.times) - 1)
            per_frame[vid] = (float(veh.positions[idx, 0]), float(veh.positions[idx, 1]))
        if per_frame:
            out[frame_idx] = per_frame
    return out


def build_id_mapping(
    tracks: dict[int, Track],
    gt_points: dict[int, dict[int, tuple[float, float]]],
    max_distance: float = 3.0,
) -> tuple[dict[int, int], TrackingMetrics]:
    """Ánh xạ ``track_id`` (hệ thống) → ``vid`` (simulator), kèm chỉ số tracking.

    Nguyên tắc: ở mỗi frame, ghép mỗi track với chiếc xe thật **gần nhất** trên
    mặt đất (trong bán kính ``max_distance``), mỗi xe chỉ nhận một track. Một
    track được gán cho chiếc xe mà nó khớp ở **nhiều frame nhất**.

    ID switch được đếm khi cùng một chiếc xe thật lần lượt được gán cho các
    ``track_id`` khác nhau theo thời gian.
    """
    # Gom trạng thái của mọi track theo frame.
    states_by_frame: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for tid, track in tracks.items():
        for state in track.history:
            states_by_frame[state.frame_idx].append((tid, state.ground))

    votes: dict[int, Counter] = defaultdict(Counter)
    gt_hits: dict[int, list[tuple[int, int]]] = defaultdict(list)
    total_gt_points = 0
    total_matched = 0
    false_positives = 0

    for frame_idx, truth in gt_points.items():
        total_gt_points += len(truth)
        preds = states_by_frame.get(frame_idx, [])

        # Ghép tham lam theo khoảng cách tăng dần, một-một.
        pairs: list[tuple[float, int, int]] = []
        for tid, pos in preds:
            for vid, gt_pos in truth.items():
                d = math.dist(pos, gt_pos)
                if d <= max_distance:
                    pairs.append((d, tid, vid))
        pairs.sort()

        used_t: set[int] = set()
        used_v: set[int] = set()
        for _d, tid, vid in pairs:
            if tid in used_t or vid in used_v:
                continue
            used_t.add(tid)
            used_v.add(vid)
            votes[tid][vid] += 1
            gt_hits[vid].append((frame_idx, tid))
            total_matched += 1

        false_positives += len(preds) - len(used_t)

    mapping: dict[int, int] = {}
    for tid, counter in votes.items():
        if counter:
            mapping[tid] = counter.most_common(1)[0][0]

    switches = 0
    frag_counts = []
    for vid, hits in gt_hits.items():
        hits.sort()
        seq = [tid for _f, tid in hits]
        changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        switches += changes
        frag_counts.append(changes + 1)

    # Số frame mà mỗi xe thật xuất hiện trong vùng đánh giá.
    gt_frames_per_vid: Counter = Counter()
    for truth in gt_points.values():
        for vid in truth:
            gt_frames_per_vid[vid] += 1
    mostly = sum(
        1 for vid, n in gt_frames_per_vid.items()
        if len(gt_hits.get(vid, [])) >= 0.8 * n
    )

    recall = total_matched / total_gt_points if total_gt_points else 0.0
    precision = (
        total_matched / (total_matched + false_positives)
        if (total_matched + false_positives)
        else 0.0
    )
    idf1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    mota = max(
        0.0,
        1.0
        - (
            (total_gt_points - total_matched) + false_positives + switches
        ) / max(total_gt_points, 1),
    )

    metrics = TrackingMetrics(
        idf1=round(idf1, 4),
        mota=round(mota, 4),
        id_switches=switches,
        n_gt_tracks=len(gt_frames_per_vid),
        n_pred_tracks=len(tracks),
        mostly_tracked=mostly,
        fragmentation=float(np.mean(frag_counts)) if frag_counts else 0.0,
    )
    return mapping, metrics


# --------------------------------------------------------------------------- #
def evaluate_conflicts(
    predictions: list[ConflictEvent],
    ground_truth: list[GroundTruthConflict],
    id_mapping: dict[int, int],
    time_tol: float = 2.0,
    region: tuple[float, float, float, float] | None = None,
) -> ConflictMetrics:
    """Ghép dự đoán với nhãn chuẩn và tính Precision/Recall/TTC-MAE.

    Tham số:
        predictions: sự kiện do pipeline sinh ra (dùng ``track_id``).
        ground_truth: nhãn chuẩn (dùng ``vid`` của simulator).
        id_mapping: ``track_id → vid``, lấy từ :func:`build_id_mapping`.
        time_tol: dung sai thời gian khi ghép (giây).
        region: vùng camera bao phủ. Dự đoán nằm ngoài vùng này được **bỏ qua**
            (không tính TP cũng không tính FP), vì nhãn chuẩn cũng chỉ được sinh
            trong vùng đó — chấm chúng là FP thì đơn thuần là phạt hệ thống vì
            hai bên đo trên hai phạm vi khác nhau. Đây chính là cơ chế "ignore
            region" quen thuộc trong các benchmark MOT/detection.
    """
    metrics = ConflictMetrics()

    # Quy các dự đoán về không gian ID của simulator.
    mapped: list[tuple[tuple[int, int], ConflictEvent]] = []
    unmapped = 0
    for ev in predictions:
        if region is not None:
            x_min, y_min, x_max, y_max = region
            x, y = ev.location
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue          # ngoài vùng đánh giá — bỏ qua
        va = id_mapping.get(ev.track_a)
        vb = id_mapping.get(ev.track_b)
        if va is None or vb is None or va == vb:
            unmapped += 1
            continue
        mapped.append((pair_key(va, vb), ev))

    # Track không ánh xạ được về xe thật ⇒ không thể là true positive.
    metrics.fp += unmapped

    gt_by_pair: dict[tuple[int, int], list[GroundTruthConflict]] = defaultdict(list)
    for g in ground_truth:
        gt_by_pair[pair_key(g.track_a, g.track_b)].append(g)

    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # Sinh mọi ứng viên ghép rồi sắp theo lệch thời gian tăng dần (tham lam).
    candidates: list[tuple[float, int, int]] = []
    gt_index = {id(g): i for i, g in enumerate(ground_truth)}
    for pi, (key, ev) in enumerate(mapped):
        for g in gt_by_pair.get(key, []):
            dt = abs(g.t - ev.t)
            if dt <= time_tol:
                candidates.append((dt, pi, gt_index[id(g)]))
    candidates.sort()

    for dt, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)

        ev = mapped[pi][1]
        g = ground_truth[gi]
        metrics.tp += 1
        metrics.time_errors.append(dt)

        if ev.ttc is not None and math.isfinite(g.ttc):
            metrics.ttc_errors.append(abs(ev.ttc - g.ttc))
        if ev.pet is not None and g.pet is not None:
            metrics.pet_errors.append(abs(ev.pet - g.pet))

        metrics.type_total += 1
        if ev.conflict_type == g.conflict_type:
            metrics.type_correct += 1

    metrics.fp += len(mapped) - len(used_pred)
    metrics.fn = len(ground_truth) - len(used_gt)
    return metrics


def severity_breakdown(
    predictions: list[ConflictEvent],
    ground_truth: list[GroundTruthConflict],
    id_mapping: dict[int, int],
    bands: tuple[tuple[str, float, float], ...] = (
        # Nhãn viết bằng dấu phẩy thập phân cho khớp phần còn lại của báo cáo
        # tiếng Việt — các nhãn này được in nguyên văn vào bảng.
        ("Rất nghiêm trọng (TTC < 1,0 s)", 0.0, 1.0),
        ("Nghiêm trọng (1,0–1,5 s)", 1.0, 1.5),
        ("Trung bình (1,5–2,5 s)", 1.5, 2.5),
        ("Nhẹ (2,5–3,0 s)", 2.5, 3.0),
    ),
    region: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Chấm điểm riêng theo từng dải mức nghiêm trọng của TTC.

    Đây là bảng quan trọng nhất về mặt an toàn: bỏ sót một near-miss TTC < 1 s
    nguy hiểm hơn nhiều so với bỏ sót một near-miss TTC 2.8 s, nên Recall phải
    được báo cáo tách theo dải chứ không chỉ một con số gộp.
    """
    rows: list[dict[str, Any]] = []
    for label, lo, hi in bands:
        subset = [g for g in ground_truth if math.isfinite(g.ttc) and lo <= g.ttc < hi]
        if not subset:
            rows.append({"band": label, "n_gt": 0, "recall": None})
            continue
        m = evaluate_conflicts(predictions, subset, id_mapping, region=region)
        rows.append({
            "band": label,
            "n_gt": len(subset),
            "detected": m.tp,
            "recall": round(m.recall, 4),
            "ttc_mae": round(m.ttc_mae, 4) if m.ttc_errors else None,
        })
    return rows
