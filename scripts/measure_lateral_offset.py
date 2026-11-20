"""Đo độ lệch ngang của từng cặp xe tại thời điểm hệ thống phát cảnh báo.

    python scripts/measure_lateral_offset.py \
        --results data/outputs/archive_before_lane_gate/street_heldout_results.json \
        --video data/processed/street/street_heldout_300_741s.mp4 \
        --out data/outputs/lateral_offset_street_heldout.json

Với mỗi cảnh báo, chạy lại pipeline (tắt cổng cùng làn, tức đúng luật cũ) trên
cửa sổ 8 s trước đến 2 s sau khung hình cảnh báo, rồi ghi độ lệch ngang
(vuông góc với hướng di chuyển trung bình) của cặp xe nằm đúng vị trí cảnh báo.
Đây là bước kiểm chứng giả thuyết "hai xe song song ở làn kề bị gọi là đâm đuôi"
trước khi sửa luật.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import saferoad.conflict.detector as det
from saferoad.config import Config
from saferoad.conflict.indicators import approach_angle, lateral_offset
from saferoad.pipeline.runner import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.json của lần chạy luật cũ")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/street.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = json.loads(Path(args.results).read_text(encoding="utf-8"))["events"]
    cur: dict = {}
    orig = det.classify_conflict

    def spy(a, b, ca, cb, *x, **k):
        typ = orig(a, b, ca, cb, *x, **k)
        px = (0.5 * (a.anchor[0] + b.anchor[0]), 0.5 * (a.anchor[1] + b.anchor[1]))
        d = math.hypot(px[0] - cur["px"][0], px[1] - cur["px"][1])
        if abs(a.frame_idx - cur["frame"]) <= 2 and d < 15:
            cur["hits"].append({"frame": a.frame_idx, "lateral_m": round(lateral_offset(a, b), 2),
                                "angle_deg": round(approach_angle(a, b), 1), "px_dist": round(d, 1)})
        return typ

    det.classify_conflict = spy
    rows = []
    for e in events:
        cfg = Config.load(args.config)
        cfg.conflict.same_lane_gate = False
        cfg.video.source = args.video
        cfg.video.start_frame = max(0, e["frame_idx"] - 240)
        cfg.video.max_frames = 300
        cfg.video.write_overlay = False
        cur = {"frame": e["frame_idx"], "px": e["location_px"], "hits": []}
        Pipeline(cfg, write_db=False).run(progress_every=10**9)
        best = (min(cur["hits"], key=lambda h: (abs(h["frame"] - e["frame_idx"]), h["px_dist"]))
                if cur["hits"] else None)
        rows.append({"event_id": e["event_id"], "t": e["t"], "frame_idx": e["frame_idx"],
                     "conflict_type": e["conflict_type"], "approach_angle": e["approach_angle"],
                     "gap_m": e["gap"], "lateral_m": best["lateral_m"] if best else None})
        print(rows[-1], flush=True)
    Path(args.out).write_text(json.dumps({"video": args.video, "events": rows},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
