"""Chấm điểm có đối soát: mỗi sự kiện và mỗi nhãn đều được đếm vào đúng một ô.

Vấn đề hồ sơ cũ gặp phải
------------------------
Báo cáo cũ có ba con số không khớp nhau mà không giải thích: dashboard hiện 214
sự kiện, bảng đánh giá ghi 211, còn Precision được tính trên 174 (= 82 TP + 92
FP). Chênh lệch đến từ (i) hai lệnh ``run`` và ``evaluate`` dùng hai seed khác
nhau cho detector mô phỏng, (ii) 37 sự kiện nằm ngoài vùng camera bao phủ bị
**bỏ qua** khi chấm nhưng không được ghi ra ở đâu cả.

Module này trả về đầy đủ luồng đếm::

    episode mở ─┬─ loại ở cổng khoảng cách
                ├─ loại vì quá ngắn
                ├─ loại vì PET đo ≥ ngưỡng
                ├─ gộp vào sự kiện trước của cùng cặp
                └─ phát ra = (ngoài vùng đánh giá) + (track không ánh xạ được = FP)
                             + (ánh xạ được: TP + FP)

    nhãn chuẩn = TP + FN

và tự kiểm tra các đẳng thức đó (``checks``). Mọi tỉ lệ đều đi kèm tử số và
mẫu số để người đọc tự tính lại được.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..types import ConflictEvent, GroundTruthConflict, pair_key

Region = tuple[float, float, float, float]


@dataclass
class ScoreDetail:
    """Kết quả ghép dự đoán ↔ nhãn chuẩn, giữ lại từng phần tử."""

    n_pred_total: int = 0
    ignored_outside: list[ConflictEvent] = field(default_factory=list)
    unmapped: list[ConflictEvent] = field(default_factory=list)
    matched: list[tuple[ConflictEvent, GroundTruthConflict, float]] = field(default_factory=list)
    fp_mapped: list[ConflictEvent] = field(default_factory=list)
    missed: list[GroundTruthConflict] = field(default_factory=list)
    n_gt: int = 0

    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fp(self) -> int:
        return len(self.fp_mapped) + len(self.unmapped)

    @property
    def fn(self) -> int:
        return len(self.missed)

    @property
    def n_scored(self) -> int:
        return self.tp + self.fp


def _inside(ev: ConflictEvent, region: Region | None,
            visible: Callable[[ConflictEvent], bool] | None) -> bool:
    if region is not None:
        x0, y0, x1, y1 = region
        x, y = ev.location
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
    if visible is not None and not visible(ev):
        return False
    return True


def score_events(
    predictions: list[ConflictEvent],
    ground_truth: list[GroundTruthConflict],
    id_mapping: dict[int, int],
    time_tol: float = 2.0,
    region: Region | None = None,
    visible: Callable[[ConflictEvent], bool] | None = None,
) -> ScoreDetail:
    """Ghép tham lam theo lệch thời gian tăng dần, một-một."""
    d = ScoreDetail(n_pred_total=len(predictions), n_gt=len(ground_truth))
    mapped: list[tuple[tuple[int, int], ConflictEvent]] = []
    for ev in predictions:
        if not _inside(ev, region, visible):
            d.ignored_outside.append(ev)
            continue
        va = id_mapping.get(ev.track_a)
        vb = id_mapping.get(ev.track_b)
        if va is None or vb is None or va == vb:
            d.unmapped.append(ev)
            continue
        mapped.append((pair_key(va, vb), ev))

    gt_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gi, g in enumerate(ground_truth):
        gt_by_pair[pair_key(g.track_a, g.track_b)].append(gi)

    cands: list[tuple[float, int, int]] = []
    for pi, (key, ev) in enumerate(mapped):
        for gi in gt_by_pair.get(key, []):
            dt = abs(ground_truth[gi].t - ev.t)
            if dt <= time_tol:
                cands.append((dt, pi, gi))
    cands.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    for dt, pi, gi in cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        d.matched.append((mapped[pi][1], ground_truth[gi], dt))
    d.fp_mapped = [ev for pi, (_k, ev) in enumerate(mapped) if pi not in used_p]
    d.missed = [g for gi, g in enumerate(ground_truth) if gi not in used_g]
    return d


def ratio(num: int, den: int) -> dict[str, Any]:
    """Tỉ lệ kèm tử số/mẫu số - để không bao giờ in một con số không truy được."""
    return {"value": round(num / den, 4) if den else None, "num": num, "den": den}


def _err_stats(errs: list[float], signed: list[float]) -> dict[str, Any] | None:
    if not errs:
        return None
    a = np.asarray(errs, dtype=float)
    s = np.asarray(signed, dtype=float)
    return {
        "n": int(a.size),
        "mae": round(float(a.mean()), 3),
        "median_ae": round(float(np.median(a)), 3),
        "p90_ae": round(float(np.percentile(a, 90)), 3),
        "bias": round(float(s.mean()), 3),
        "within_0_5s": ratio(int((a <= 0.5).sum()), int(a.size)),
    }


def pet_error_table(detail: ScoreDetail, threshold: float = 1.5) -> dict[str, Any]:
    """Sai số PET trên các cặp khớp, tách PET ngoại suy / PET đo / PET dùng để chốt.

    ``threshold_agreement`` đếm số cặp mà dự đoán và nhãn chuẩn nằm **cùng phía**
    ngưỡng 1,5 s - đây mới là thứ quyết định một tình huống có bị gắn cờ hay
    không, và vì sao sai số trung bình lớn hơn ngưỡng là đáng lo.
    """
    out: dict[str, Any] = {}
    for name, attr in (("predicted", "pet_pred"), ("measured", "pet_measured"),
                       ("final", "pet")):
        errs, signed = [], []
        agree = total = 0
        for ev, g, _dt in detail.matched:
            p = getattr(ev, attr)
            if p is None or g.pet is None:
                continue
            errs.append(abs(p - g.pet))
            signed.append(p - g.pet)
            total += 1
            agree += int((p < threshold) == (g.pet < threshold))
        stats = _err_stats(errs, signed)
        if stats is not None:
            stats["threshold_agreement"] = ratio(agree, total)
        out[name] = stats
    status = Counter(ev.pet_status or "none" for ev, _g, _dt in detail.matched)
    gt_status = Counter(g.pet_status or "none" for _ev, g, _dt in detail.matched)
    out["status_pred_matched"] = dict(status)
    out["status_gt_matched"] = dict(gt_status)
    return out


def contact_confusion(detail: ScoreDetail) -> dict[str, Any]:
    """Bảng nhầm lẫn: loại nhãn chuẩn × kết cục hệ thống phân loại.

    Hàng ``không có nhãn`` là các cảnh báo sai (FP): người đọc thấy được hệ thống
    gắn "chồng thân xe" hay "nghi va chạm" cho bao nhiêu tình huống bình thường.
    """
    classes = ("near_miss", "body_overlap", "collision_suspected")
    table: dict[str, dict[str, int]] = {}
    for kind in ("near_miss", "collision"):
        row = Counter(ev.contact for ev, g, _dt in detail.matched if g.kind == kind)
        table[kind] = {c: int(row.get(c, 0)) for c in classes}
    row = Counter(ev.contact for ev in detail.fp_mapped + detail.unmapped)
    table["no_label"] = {c: int(row.get(c, 0)) for c in classes}
    missed = Counter(g.kind for g in detail.missed)
    return {"matrix": table, "missed_by_kind": dict(missed), "classes": list(classes)}


def by_kind(detail: ScoreDetail, ground_truth: list[GroundTruthConflict]) -> dict[str, Any]:
    out = {}
    for kind in ("near_miss", "collision"):
        n = sum(1 for g in ground_truth if g.kind == kind)
        hit = sum(1 for _ev, g, _dt in detail.matched if g.kind == kind)
        out[kind] = {"recall": ratio(hit, n)}
    return out


def ttc_bands(detail: ScoreDetail, ground_truth: list[GroundTruthConflict]) -> list[dict[str, Any]]:
    """Recall theo dải TTC, CHỈ trên nhãn near-miss (va chạm báo riêng)."""
    bands = (
        ("Rất nghiêm trọng (TTC < 1,0 s)", 0.0, 1.0),
        ("Nghiêm trọng (1,0-1,5 s)", 1.0, 1.5),
        ("Trung bình (1,5-2,5 s)", 1.5, 2.5),
        ("Nhẹ (2,5-3,0 s)", 2.5, 3.0),
        ("Chỉ thoả PET (TTC ≥ 3,0 s)", 3.0, math.inf),
    )
    matched = {id(g): ev for ev, g, _dt in detail.matched}
    rows = []
    for label, lo, hi in bands:
        sub = [g for g in ground_truth if g.kind == "near_miss" and lo <= g.ttc < hi]
        hit = [g for g in sub if id(g) in matched]
        errs = [abs(matched[id(g)].ttc - g.ttc) for g in hit
                if matched[id(g)].ttc is not None and math.isfinite(g.ttc)]
        rows.append({
            "band": label,
            "recall": ratio(len(hit), len(sub)),
            "ttc_mae": round(float(np.mean(errs)), 3) if errs else None,
        })
    return rows


def reconciliation(
    detail: ScoreDetail,
    funnel: dict[str, int],
    n_exported: int,
    duration_s: float,
) -> dict[str, Any]:
    """Bảng đối soát số sự kiện + các đẳng thức tự kiểm."""
    emitted = int(funnel.get("emitted", 0))
    opened = int(funnel.get("episodes_opened", 0))
    rejected = {k: int(v) for k, v in funnel.items() if k.startswith("rejected_")}
    merged = int(funnel.get("merged_into_previous", 0))
    hours = duration_s / 3600.0 if duration_s else float("nan")
    rows = {
        "episodes_opened": opened,
        **rejected,
        "merged_into_previous": merged,
        "emitted": emitted,
        "exported_to_results": n_exported,
        "ignored_outside_eval_region": len(detail.ignored_outside),
        "scored": detail.n_scored,
        "scored_unmapped_track_fp": len(detail.unmapped),
        "scored_mapped_fp": len(detail.fp_mapped),
        "scored_tp": detail.tp,
        "gt_total": detail.n_gt,
        "gt_matched_tp": detail.tp,
        "gt_missed_fn": detail.fn,
    }
    checks = {
        "opened = rejected + merged + emitted":
            opened == sum(rejected.values()) + merged + emitted,
        "emitted = exported": emitted == n_exported,
        "exported = ignored + scored":
            n_exported == len(detail.ignored_outside) + detail.n_scored,
        "scored = TP + FP": detail.n_scored == detail.tp + detail.fp,
        "gt = TP + FN": detail.n_gt == detail.tp + detail.fn,
    }
    per_hour = {
        "false_alarms_per_hour": round(detail.fp / hours, 1) if hours == hours else None,
        "misses_per_hour": round(detail.fn / hours, 1) if hours == hours else None,
        "alerts_per_hour": round(n_exported / hours, 1) if hours == hours else None,
        "hours": round(hours, 4),
    }
    return {"rows": rows, "checks": checks, "all_checks_pass": all(checks.values()),
            "per_hour": per_hour}


def summary_metrics(detail: ScoreDetail) -> dict[str, Any]:
    """Precision/Recall/F1 kèm tử số, mẫu số."""
    p = ratio(detail.tp, detail.tp + detail.fp)
    r = ratio(detail.tp, detail.tp + detail.fn)
    pv, rv = p["value"] or 0.0, r["value"] or 0.0
    f1 = 2 * pv * rv / (pv + rv) if (pv + rv) else 0.0
    ttc_err = [abs(ev.ttc - g.ttc) for ev, g, _dt in detail.matched
               if ev.ttc is not None and math.isfinite(g.ttc)]
    type_ok = sum(1 for ev, g, _dt in detail.matched if ev.conflict_type == g.conflict_type)
    return {
        "precision": p,
        "recall": r,
        "f1": round(f1, 4),
        "ttc_mae": round(float(np.mean(ttc_err)), 4) if ttc_err else None,
        "ttc_mae_n": len(ttc_err),
        "time_mae": round(float(np.mean([dt for *_x, dt in detail.matched])), 4)
        if detail.matched else None,
        "type_accuracy": ratio(type_ok, detail.tp),
    }
