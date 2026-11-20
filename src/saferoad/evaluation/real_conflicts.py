"""Cảnh báo xung đột trên video THẬT: tham chiếu, đối chiếu, và rà soát của người.

Vì sao cần ba lớp
-----------------
Video thật có nhãn bbox + ID (MVTI, đại lộ Kaggle) nhưng **không có nhãn xung
đột**. Báo cáo cũ vì thế chỉ đưa ra "thống kê mô tả" số cảnh báo - không biết bao
nhiêu cảnh báo là đúng. Module này chia bài toán làm ba tầng, mỗi tầng trả lời
một câu hỏi khác nhau và được báo cáo riêng:

1. **Cảnh báo tham chiếu** (:func:`reference_events`) - chạy *đúng phần sau phát
   hiện* của pipeline (homography, làm mượt, TTC/PET, cổng, gom episode) trên
   **quỹ đạo do người gán nhãn** (bbox + ID chuẩn). Kết quả là: "nếu detector và
   tracker hoàn hảo thì hệ thống sẽ cảnh báo gì". So pipeline đầy đủ với tham
   chiếu cho ra sai số **do detection + tracking gây ra** - đúng thứ hội đồng
   muốn thấy trên cùng một dữ liệu: bảng detection → tracking → cảnh báo.
2. **Danh sách rà soát** (:func:`export_review_candidates`) - hợp của cảnh báo
   hệ thống và cảnh báo tham chiếu, kèm thời điểm để người xem lại video và gán
   nhãn: near-miss thật / va chạm / bình thường / chồng thân do sai số / không rõ.
3. **Chấm theo nhãn đã rà soát** (:func:`score_reviewed`) - khi đã có nhãn của
   ít nhất hai người độc lập: tính Cohen's kappa, nhãn đồng thuận, rồi Precision,
   Recall, báo nhầm/giờ và bỏ sót/giờ của hệ thống so với **nhãn người**.

Tầng 1 không thay thế tầng 3: cảnh báo tham chiếu vẫn là đầu ra của thuật toán,
chỉ là trên đầu vào sạch. Nhãn xung đột đáng tin duy nhất là nhãn người rà soát.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


from ..config import Config
from ..conflict.detector import ConflictDetector
from ..data.mvti import MvtiSequence
from ..geometry.homography import GroundPlane, make_default_homography
from ..geometry.trajectory import TrajectoryProcessor
from ..risk.scoring import RiskScorer
from ..types import ConflictEvent, GroundTruthConflict, Track, TrackState
from .scoring import ratio, reconciliation, score_events, summary_metrics

#: Nhãn rà soát hợp lệ.
REVIEW_LABELS = ("near_miss", "collision", "normal", "overlap_artifact", "unclear")
REVIEW_LABELS_VI = {
    "near_miss": "Suýt va chạm thật",
    "collision": "Va chạm",
    "normal": "Giao thông bình thường (báo nhầm)",
    "overlap_artifact": "Chồng thân do sai số đo/che khuất",
    "unclear": "Không rõ",
}
POSITIVE = {"near_miss", "collision"}


def reference_events(
    cfg: Config,
    seq: MvtiSequence,
    fps: float,
    min_box_area: float | None = None,
    homography_override: Any = None,
) -> tuple[list[ConflictEvent], dict[str, Any], dict[int, Track]]:
    """Chạy tầng xung đột của pipeline trên quỹ đạo gán nhãn tay."""
    min_area = cfg.detection.min_box_area if min_box_area is None else min_box_area
    w, h = seq.width, seq.height
    homo = homography_override or cfg.homography
    if not homo.image_points:
        homo = make_default_homography(w, h, camera_height_m=cfg.site.camera_height_m)
    ground = GroundPlane(homo, (w, h))
    traj = TrajectoryProcessor(ground, homo, cfg.tracking)
    det = ConflictDetector(cfg.conflict)
    scorer = RiskScorer(cfg.risk)

    by_frame: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for oid, gt in seq.tracks.items():
        for f, box in zip(gt.frames, gt.boxes):
            if (box[2] - box[0]) * (box[3] - box[1]) >= min_area:
                by_frame[f].append((oid, box))

    tracks: dict[int, Track] = {}
    events: list[ConflictEvent] = []
    n = seq.n_frames
    last_t = 0.0
    for f in range(n):
        t = f / fps
        last_t = t
        active: list[Track] = []
        for oid, box in by_frame.get(f, []):
            tr = tracks.get(oid)
            if tr is None:
                tr = Track(track_id=oid, cls=seq.tracks[oid].cls, confirmed=True)
                tracks[oid] = tr
            anchor = (0.5 * (box[0] + box[2]), box[3])
            tr.history.append(TrackState(frame_idx=f, t=t, bbox=box, anchor=anchor,
                                         ground=(0.0, 0.0), score=1.0))
            traj.process(tr)
            active.append(tr)
        events += det.update(active, f, t, risk_scorer=scorer)
    events += det.flush(risk_scorer=scorer, now=last_t)
    return events, det.funnel.to_dict(), tracks


def as_reference_labels(events: list[ConflictEvent]) -> list[GroundTruthConflict]:
    """Đổi cảnh báo tham chiếu sang dạng nhãn để dùng chung hàm chấm."""
    out = []
    for e in events:
        out.append(GroundTruthConflict(
            t=e.t, track_a=e.track_a, track_b=e.track_b,
            ttc=e.ttc if e.ttc is not None else math.inf, pet=e.pet,
            conflict_type=e.conflict_type, source="reference-gt-tracks",
            kind="collision" if e.contact == "collision_suspected" else "near_miss",
            pet_status=e.pet_status, pet_times=e.pet_times, min_gap=e.gap,
        ))
    return out


def compare_to_reference(
    events: list[ConflictEvent],
    funnel: dict[str, int],
    ref_events: list[ConflictEvent],
    mapping: dict[int, int],
    duration_s: float,
) -> dict[str, Any]:
    """Pipeline đầy đủ so với tham chiếu trên quỹ đạo người gán, cùng một video."""
    ref = as_reference_labels(ref_events)
    detail = score_events(events, ref, mapping, time_tol=2.0)
    return {
        "metrics_vs_reference": summary_metrics(detail),
        "reconciliation": reconciliation(detail, funnel, len(events), duration_s),
        "n_reference": len(ref_events),
        "reference_contact": dict(Counter(e.contact for e in ref_events)),
        "system_contact": dict(Counter(e.contact for e in events)),
        "_detail": detail,
    }


def export_review_candidates(
    path: str | Path,
    events: list[ConflictEvent],
    ref_events: list[ConflictEvent],
    detail,
    mapping: dict[int, int],
    video: str,
    fps: float,
    clip_pad_s: float = 3.0,
) -> int:
    """Ghi CSV các tình huống cần người xem lại (hợp hệ thống ∪ tham chiếu)."""
    rows = []
    matched_ref = {id(g) for _e, g, _d in detail.matched}
    ref_labels = as_reference_labels(ref_events)
    matched_sys = {e.event_id: g for e, g, _d in detail.matched}
    k = 0
    for e in events:
        k += 1
        g = matched_sys.get(e.event_id)
        rows.append({
            "candidate_id": f"C{k:04d}", "source": "both" if g is not None else "system",
            "t_s": e.t, "frame": int(round(e.t * fps)),
            "clip_start_s": round(max(0.0, e.t - clip_pad_s), 2),
            "clip_end_s": round(e.t + clip_pad_s, 2),
            "system_event": e.event_id, "track_a": e.track_a, "track_b": e.track_b,
            "gt_object_a": mapping.get(e.track_a, ""), "gt_object_b": mapping.get(e.track_b, ""),
            "ttc_s": e.ttc if e.ttc is not None else "", "pet_s": e.pet if e.pet is not None else "",
            "pet_status": e.pet_status, "contact": e.contact,
            "type": e.conflict_type.value, "risk": e.risk_score,
            "px_x": round(e.location_px[0], 1), "px_y": round(e.location_px[1], 1),
            "video": video, "label": "", "reviewer": "", "note": "",
        })
    for g, e in zip(ref_labels, ref_events):
        if id(g) in matched_ref:
            continue
        k += 1
        rows.append({
            "candidate_id": f"C{k:04d}", "source": "reference",
            "t_s": e.t, "frame": int(round(e.t * fps)),
            "clip_start_s": round(max(0.0, e.t - clip_pad_s), 2),
            "clip_end_s": round(e.t + clip_pad_s, 2),
            "system_event": "", "track_a": "", "track_b": "",
            "gt_object_a": e.track_a, "gt_object_b": e.track_b,
            "ttc_s": e.ttc if e.ttc is not None else "", "pet_s": e.pet if e.pet is not None else "",
            "pet_status": e.pet_status, "contact": e.contact,
            "type": e.conflict_type.value, "risk": e.risk_score,
            "px_x": round(e.location_px[0], 1), "px_y": round(e.location_px[1], 1),
            "video": video, "label": "", "reviewer": "", "note": "",
        })
    rows.sort(key=lambda r: r["t_s"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["candidate_id"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------- #
def _read_labels(path: str | Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        out = {}
        for r in csv.DictReader(fh):
            lab = (r.get("label") or "").strip()
            if lab:
                if lab not in REVIEW_LABELS:
                    raise ValueError(f"{path}: nhãn không hợp lệ '{lab}' ở {r.get('candidate_id')}; "
                                     f"hợp lệ: {', '.join(REVIEW_LABELS)}")
                out[r["candidate_id"]] = lab
        return out


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa cho hai người gán nhãn trên cùng tập phần tử.

    Trả ``None`` khi kappa không xác định: cả hai người chỉ dùng đúng một nhãn
    giống nhau (đồng thuận ngẫu nhiên kỳ vọng = 1, mẫu số bằng 0). Khi đó đọc
    tỉ lệ đồng thuận thô (``raw_agreement``) thay cho kappa.
    """
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return None if pe >= 1.0 else (po - pe) / (1.0 - pe)


def raw_agreement(a: list[str], b: list[str]) -> float | None:
    """Tỉ lệ phần tử hai người gán trùng nhãn."""
    return None if not a else sum(1 for x, y in zip(a, b) if x == y) / len(a)


def score_reviewed(
    candidates_csv: str | Path,
    label_files: list[str | Path],
    duration_s: float,
    extra_missed: int | None = 0,
) -> dict[str, Any]:
    """Chấm hệ thống theo nhãn người rà soát.

    * Tính kappa giữa hai người đầu (trên các ứng viên cả hai cùng gán, bỏ
      ``unclear``) theo hai cách: đủ 5 nhãn và gộp thành có/không xung đột.
    * Nhãn đồng thuận = các ứng viên mà MỌI người gán cùng một nhãn (không lấy
      trung bình khi bất đồng - đúng protocol mục 8).
    * Precision = cảnh báo hệ thống được đồng thuận là xung đột / cảnh báo hệ
      thống có nhãn đồng thuận. Recall = xung đột đồng thuận mà hệ thống báo /
      tổng xung đột đồng thuận (+ ``extra_missed`` - các tình huống người xem phát
      hiện mà không nằm trong danh sách ứng viên).
    """
    with open(candidates_csv, newline="", encoding="utf-8-sig") as fh:
        cands = {r["candidate_id"]: r for r in csv.DictReader(fh)}
    labels = [_read_labels(p) for p in label_files]

    kappa = kappa_bin = agree = None
    n_pair = 0
    if len(labels) >= 2:
        common = [c for c in cands if c in labels[0] and c in labels[1]
                  and labels[0][c] != "unclear" and labels[1][c] != "unclear"]
        n_pair = len(common)
        a = [labels[0][c] for c in common]
        b = [labels[1][c] for c in common]
        kappa = cohen_kappa(a, b)
        kappa_bin = cohen_kappa([str(x in POSITIVE) for x in a], [str(x in POSITIVE) for x in b])
        agree = raw_agreement(a, b)
    # Protocol mục 8: kappa >= 0,6. Kappa không xác định chỉ xảy ra khi hai người
    # cùng dùng một nhãn duy nhất, tức là trùng nhau hoàn toàn - không cần gán lại.
    if kappa is not None:
        kappa_ok = kappa >= 0.6
    elif agree is not None:
        kappa_ok = agree == 1.0
    else:
        kappa_ok = None

    consensus: dict[str, str] = {}
    for cid in cands:
        labs = [lab[cid] for lab in labels if cid in lab]
        if len(labs) == len(labels) and len(set(labs)) == 1 and labs[0] != "unclear":
            consensus[cid] = labs[0]

    sys_ids = [c for c, r in cands.items() if r.get("source") in ("system", "both")]
    sys_lab = [c for c in sys_ids if c in consensus]
    tp = sum(1 for c in sys_lab if consensus[c] in POSITIVE)
    fp = len(sys_lab) - tp
    pos = [c for c, lab in consensus.items() if lab in POSITIVE]
    hit = sum(1 for c in pos if cands[c].get("source") in ("system", "both"))
    fn = len(pos) - hit + int(extra_missed or 0)
    hours = duration_s / 3600.0 if duration_s else float("nan")
    return {
        "n_candidates": len(cands),
        "n_reviewers": len(labels),
        "n_labeled_by_each": [len(lab) for lab in labels],
        "kappa_5class": None if kappa is None else round(kappa, 3),
        "kappa_conflict_vs_not": None if kappa_bin is None else round(kappa_bin, 3),
        "raw_agreement": None if agree is None else round(agree, 3),
        "n_pairs_for_kappa": n_pair,
        "kappa_ok": kappa_ok,
        "consensus_counts": dict(Counter(consensus.values())),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(hit, hit + fn),
        "false_alarms_per_hour": round(fp / hours, 1) if hours == hours else None,
        # None khi chưa ai xem lại toàn bộ video để đếm bỏ sót: không ghi 0 thay cho "chưa đếm".
        "misses_per_hour": (round(fn / hours, 1) if hours == hours else None)
        if extra_missed is not None else None,
        "system_alerts_unlabeled_or_disputed": len(sys_ids) - len(sys_lab),
        "n_system_alerts": len(sys_ids),
        "n_reference_conflicts_judged_normal": sum(
            1 for c, r in cands.items() if r.get("source") in ("reference", "both")
            and consensus.get(c) not in (None, *POSITIVE)),
        "n_reference_alerts": sum(1 for r in cands.values()
                                  if r.get("source") in ("reference", "both")),
        "extra_missed": None if extra_missed is None else int(extra_missed),
        "duration_s": duration_s,
    }


def export_system_candidates(path: str | Path, events: list[ConflictEvent], video: str,
                             fps: float) -> int:
    """Danh sách rà soát cho video KHÔNG có nhãn: chỉ gồm cảnh báo của hệ thống.

    Dùng cho chạy thử đầu-cuối trên video độc lập. Người rà soát vẫn phải xem cả
    video để ghi thêm các xung đột hệ thống bỏ sót (tham số ``--extra-missed``
    của ``saferoad review-score``).
    """
    from .scoring import ScoreDetail

    return export_review_candidates(path, events, [], ScoreDetail(), {}, video, fps)
