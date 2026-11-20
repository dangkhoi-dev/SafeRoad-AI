"""Cảnh báo tham chiếu trên video đại lộ thay đổi thế nào theo từng bước lọc.

    python scripts/street_reference_sensitivity.py

Chạy tầng xung đột trên QUỸ ĐẠO GÁN NHÃN TAY (detector/tracker hoàn hảo) với
các cấu hình tăng dần, để thấy số cảnh báo đến từ đâu:

  A. toàn khung hình, như cấu hình cũ (xét cả cặp hai người đi bộ, không kẹp
     vận tốc xe đứng yên)
  B. A + bỏ cặp hai người đi bộ (đúng quy ước của nhãn chuẩn protocol v2)
  C. B + chỉ xét phần mặt đường cách gốc hiệu chuẩn ≤ 65 m
  D. C + kẹp vận tốc xe đứng yên (xe đỗ / hàng chờ)
  E. D + chỉ xét lòng đường đã hiệu chuẩn
  F. E + cổng cùng làn: cặp đi gần song song phải lệch ngang < 1,75 m
     (cấu hình configs/street.yaml)

Kết quả: data/outputs/street_reference_sensitivity.json
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saferoad.config import Config  # noqa: E402
from saferoad.evaluation.real_conflicts import reference_events  # noqa: E402

BAND_65M = [[0, 562], [1920, 799], [1920, 1080], [0, 1080]]


def draw_region(cfg: Config, out: Path, frame_idx: int = 150) -> None:
    """Ảnh minh hoạ: vùng phân tích (vàng), dải ≤ 65 m (xanh), điểm hiệu chuẩn (đỏ)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(ROOT / "data/processed/street/street_labeled_0_300s.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    over = img.copy()
    cv2.fillPoly(over, [np.array(cfg.conflict.analysis_region_px, np.int32)], (0, 200, 255))
    img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
    cv2.polylines(img, [np.array(BAND_65M, np.int32)], True, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array(cfg.conflict.analysis_region_px, np.int32)], True,
                  (0, 200, 255), 3, cv2.LINE_AA)
    for x, y in cfg.homography.image_points:
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    base = Config.load(ROOT / "configs" / "street.yaml")
    bundle = pickle.loads((ROOT / "data/processed/street/street_groundtruth.pkl").read_bytes())
    seq, fps = bundle["sequence"], float(bundle.get("fps", 30.0))
    hours = seq.n_frames / fps / 3600.0

    def variant(ped_pairs: bool, region, stationary: bool, lane_gate: bool = False) -> Config:
        cfg = copy.deepcopy(base)
        cfg.conflict.same_lane_gate = lane_gate
        cfg.conflict.ignore_pedestrian_pairs = not ped_pairs
        cfg.conflict.analysis_region_px = region
        cfg.tracking.stationary_speed = base.tracking.stationary_speed if stationary else 0.0
        return cfg

    steps = [
        ("A", "Toàn khung hình, cấu hình cũ", variant(True, [], False)),
        ("B", "+ bỏ cặp hai người đi bộ", variant(False, [], False)),
        ("C", "+ chỉ xét mặt đường ≤ 65 m", variant(False, BAND_65M, False)),
        ("D", "+ kẹp vận tốc xe đứng yên", variant(False, BAND_65M, True)),
        ("E", "+ chỉ lòng đường đã hiệu chuẩn",
         variant(False, base.conflict.analysis_region_px, True)),
        ("F", "+ cổng cùng làn (cấu hình dùng)",
         variant(False, base.conflict.analysis_region_px, True, True)),
    ]
    rows = []
    for key, label, cfg in steps:
        t0 = time.perf_counter()
        events, funnel, tracks = reference_events(cfg, seq, fps)
        pairs = Counter("-".join(sorted((tracks[e.track_a].cls.value,
                                         tracks[e.track_b].cls.value))) for e in events)
        rows.append({
            "step": key, "label": label, "n_events": len(events),
            "per_hour": round(len(events) / hours, 1),
            "by_contact": dict(Counter(e.contact for e in events)),
            "by_type": dict(Counter(e.conflict_type.value for e in events)),
            "by_pair": dict(pairs),
            "funnel": funnel,
            "seconds": round(time.perf_counter() - t0, 1),
        })
        print(f"  {key}. {label}: {len(events)} cảnh báo ({rows[-1]['per_hour']}/giờ)")
    draw_region(base, ROOT / "docs" / "assets" / "street_analysis_region.jpg")
    out = ROOT / "data" / "outputs" / "street_reference_sensitivity.json"
    out.write_text(json.dumps({
        "video": "data/processed/street/street_labeled_0_300s.mp4",
        "duration_s": round(seq.n_frames / fps, 1),
        "note": ("Cảnh báo tham chiếu = tầng xung đột chạy trên quỹ đạo người gán nhãn. "
                 "Không phải nhãn xung đột; dùng để thấy phần nào của số cảnh báo đến "
                 "từ sai số hình học thay vì từ tình huống giao thông."),
        "steps": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
