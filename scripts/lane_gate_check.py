"""So sánh trước/sau khi thêm cổng cùng làn, ghi data/outputs/lane_gate_check.json.

    python scripts/lane_gate_check.py            # chỉ ghi JSON
    python scripts/lane_gate_check.py --figure   # + ảnh docs/assets/heldout_lane_gate.jpg (cần video)

Đầu vào:
  - data/outputs/archive_before_lane_gate/*: kết quả của bản trước cổng (giữ nguyên)
  - data/outputs/*: kết quả chạy lại với cổng (cùng lệnh, cùng dữ liệu)
  - data/outputs/lateral_offset_street_heldout.json (scripts/measure_lateral_offset.py)
  - nhãn người rà soát labels_street_heldout_*.csv (gán trên bản trước cổng)

Cổng chỉ loại bớt cặp xe, nên mỗi cảnh báo còn lại sau khi sửa phải trùng một
cảnh báo cũ đã có nhãn; cảnh báo mới (nếu có) được ghi riêng là "chưa rà soát".
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
BEFORE = OUT / "archive_before_lane_gate"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def heldout() -> dict:
    old = load(BEFORE / "street_heldout_results.json")
    new = load(OUT / "street_heldout" / "results.json")
    lat = {r["event_id"]: r["lateral_m"]
           for r in (load(OUT / "lateral_offset_street_heldout.json") or {"events": []})["events"]}
    cands = read_csv(OUT / "review_candidates_street_heldout.csv")
    by_event = {c["system_event"]: c["candidate_id"] for c in cands if c.get("system_event")}
    labels: dict[str, list[str]] = {}
    for f in sorted(OUT.glob("labels_street_heldout_*.csv")):
        for row in read_csv(f):
            if row.get("label"):
                labels.setdefault(row["candidate_id"], []).append(row["label"])

    def consensus(cid: str | None) -> str | None:
        ls = labels.get(cid or "", [])
        return ls[0] if len(ls) >= 2 and len(set(ls)) == 1 else None

    new_events = new["events"]
    hours = new["summary"]["duration_s"] / 3600.0
    rows, matched = [], set()
    for e in old["events"]:
        hit = next((n for n in new_events
                    if abs(n["frame_idx"] - e["frame_idx"]) <= 15 and n["event_id"] not in matched), None)
        if hit:
            matched.add(hit["event_id"])
        cid = by_event.get(e["event_id"])
        rows.append({"event_id": e["event_id"], "candidate_id": cid, "t": e["t"],
                     "conflict_type": e["conflict_type"], "approach_angle": e["approach_angle"],
                     "gap_m": e["gap"], "lateral_m": lat.get(e["event_id"]),
                     "human_consensus": consensus(cid),
                     "after": "kept" if hit else "removed"})
    unreviewed = [n["event_id"] for n in new_events if n["event_id"] not in matched]
    kept = [r for r in rows if r["after"] == "kept"]
    kept_conflict = sum(1 for r in kept if r["human_consensus"] in ("near_miss", "collision"))
    kept_normal = sum(1 for r in kept if r["human_consensus"] in ("normal", "overlap_artifact"))
    removed_conflict = sum(1 for r in rows if r["after"] == "removed"
                           and r["human_consensus"] in ("near_miss", "collision"))
    lats = [r["lateral_m"] for r in rows if r["lateral_m"] is not None]
    return {
        "video": "data/processed/street/street_heldout_300_741s.mp4",
        "duration_s": new["summary"]["duration_s"],
        "alerts_before": len(old["events"]),
        "alerts_after": len(new_events),
        "removed": sum(1 for r in rows if r["after"] == "removed"),
        "removed_that_humans_called_conflict": removed_conflict,
        "kept_judged_normal": kept_normal,
        "kept_judged_conflict": kept_conflict,
        "new_unreviewed": unreviewed,
        "false_alarms_per_hour_before": round(len(old["events"]) / hours, 1),
        "false_alarms_per_hour_after": round((kept_normal + len(unreviewed)) / hours, 1),
        "lateral_m_min": min(lats) if lats else None,
        "lateral_m_max": max(lats) if lats else None,
        "lateral_threshold_m": 1.75,
        "independent": False,
        "note": ("Cổng được thiết kế sau khi phân tích lỗi trên chính đoạn này, nên số 'sau' "
                 "không phải phép thử độc lập. Ngưỡng 1,75 m = nửa làn 3,5 m giả định khi hiệu "
                 "chuẩn, không dò; tại khung hình cảnh báo, lệch ngang nhỏ nhất của các cặp "
                 "là 2,2 m nên ngưỡng không nằm sát biên dữ liệu."),
        "events": rows,
    }


def alerts(name: str) -> dict | None:
    b, a = load(BEFORE / name), load(OUT / name)
    if not b or not a:
        return None

    def pick(x):
        al = x["alerts"]
        return {"system": al["reconciliation"]["rows"]["emitted"], "reference": al["n_reference"],
                "precision_vs_reference": al["metrics_vs_reference"]["precision"],
                "recall_vs_reference": al["metrics_vs_reference"]["recall"]}
    return {"before": pick(b), "after": pick(a)}


def simulation() -> dict | None:
    ev, old = load(OUT / "evaluation.json"), load(BEFORE / "evaluation.json")
    if not ev:
        return None
    loo = {r["key"]: r for r in ev["ablation"]["leave_one_out"]}
    full, off = loo.get("full"), loo.get("-lane_gate")
    keep = lambda r: r and {k: r[k] for k in ("precision", "recall", "f1")}  # noqa: E731
    sweep = []
    if old:
        for w, o in zip(ev.get("noise_sweep", []), old.get("noise_sweep", [])):
            if w["noise_px"] == o["noise_px"]:
                sweep.append({"noise_px": w["noise_px"], "with_gate": keep(w), "without_gate": keep(o)})
    grid = None
    gn, go = load(OUT / "evaluation_grid.json"), load(BEFORE / "evaluation_grid.json")
    if gn and go and len(gn["cells"]) == len(go["cells"]):
        tot = lambda cells, k: sum(c["full"][k]["num" if k == "recall" else "den"] for c in cells)  # noqa: E731
        grid = {"n_cells": len(gn["cells"]),
                "n_gt": sum(c["n_gt"] for c in gn["cells"]),
                "tp_with_gate": tot(gn["cells"], "recall"), "tp_without_gate": tot(go["cells"], "recall"),
                "scored_with_gate": tot(gn["cells"], "precision"),
                "scored_without_gate": tot(go["cells"], "precision"),
                "cells_changed": sum(1 for a, b in zip(gn["cells"], go["cells"])
                                     if a["full"]["recall"]["num"] != b["full"]["recall"]["num"]
                                     or a["full"]["precision"]["den"] != b["full"]["precision"]["den"])}
    return {"with_gate": keep(full), "without_gate": keep(off), "noise_sweep": sweep, "grid": grid,
            "published_before": old and {k: old["main"]["metrics"][k] for k in ("precision", "recall", "f1")}}


def figure(h: dict, video: Path, out: Path) -> None:
    """Lưới ảnh: mỗi cảnh báo cũ một ô, ghi lệch ngang, nhãn người và kết quả sau cổng."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    f1, f2 = ImageFont.truetype(font, 17), ImageFont.truetype(font, 15)
    old = {e["event_id"]: e for e in load(BEFORE / "street_heldout_results.json")["events"]}
    cap = cv2.VideoCapture(str(video))
    tiles = []
    vi = {"normal": "bình thường", "near_miss": "near-miss", "collision": "va chạm", None: "chưa thống nhất"}
    for r in h["events"]:
        e = old[r["event_id"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_idx"])
        ok, frame = cap.read()
        if not ok:
            continue
        x, y = map(int, e["location_px"])
        x0 = min(max(0, x - 240), frame.shape[1] - 480)
        y0 = min(max(0, y - 170), frame.shape[0] - 340)
        crop = frame[y0:y0 + 340, x0:x0 + 480].copy()
        kept = r["after"] == "kept"
        cv2.circle(crop, (x - x0, y - y0), 46, (0, 140, 255) if kept else (0, 230, 255), 3)
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 480, 56], fill=(0, 0, 0))
        d.text((8, 4), f'{r["event_id"]}  t = {r["t"]:.0f} s  {r["conflict_type"]}, góc {r["approach_angle"]:.0f}°',
               font=f1, fill=(255, 255, 255))
        lat = "-" if r["lateral_m"] is None else f'{r["lateral_m"]:.1f}'.replace(".", ",")
        d.text((8, 30), f'lệch ngang {lat} m · người: {vi.get(r["human_consensus"], r["human_consensus"])} · '
               + ("còn giữ" if kept else "bị cổng loại"), font=f2,
               fill=(255, 190, 90) if kept else (140, 255, 140))
        tiles.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    grid = cv2.resize(grid, (grid.shape[1] * 2 // 3, grid.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"✓ {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--video", default=str(ROOT / "data/processed/street/street_heldout_300_741s.mp4"))
    args = ap.parse_args()
    sets = {}
    for name in ("evaluation_street.json", "evaluation_street_pretrained_test.json",
                 "evaluation_street_finetuned_test.json", "evaluation_real.json",
                 "evaluation_mvti_pretrained_test.json", "evaluation_mvti_finetuned_test.json"):
        r = alerts(name)
        if r:
            sets[name.removeprefix("evaluation_").removesuffix(".json")] = r
    res = {"rule": ("Cặp có góc tiếp cận < angle_rear_end (30°) và không xe nào đổi hướng nhanh "
                    "hơn 45°/s chỉ được xét khi lệch ngang (vuông góc hướng đi trung bình) < 1,75 m"),
           "heldout": heldout(), "simulation": simulation(), "labeled_videos": sets}
    out = OUT / "lane_gate_check.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    h = res["heldout"]
    print(f"Đoạn 441 s: {h['alerts_before']} → {h['alerts_after']} cảnh báo; báo nhầm/giờ "
          f"{h['false_alarms_per_hour_before']} → {h['false_alarms_per_hour_after']}; "
          f"lệch ngang {h['lateral_m_min']}-{h['lateral_m_max']} m; chưa rà soát: {h['new_unreviewed']}")
    print(f"✓ {out.relative_to(ROOT)}")
    if args.figure:
        figure(h, Path(args.video), ROOT / "docs" / "assets" / "heldout_lane_gate.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
