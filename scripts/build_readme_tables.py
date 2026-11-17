"""Ghi lại các bảng số liệu trong README từ data/outputs/*.json.

    python scripts/build_readme_tables.py

Chỉ thay phần nằm giữa hai dấu ``<!-- AUTO:RESULTS:START -->`` và
``<!-- AUTO:RESULTS:END -->``. Nhờ vậy số trong README luôn khớp với tệp kết quả
và với báo cáo - hồ sơ cũ từng lệch vì số được gõ tay ở nhiều nơi.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"


def load(name: str):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def n(v, d=3):
    if v is None:
        return "-"
    return f"{v:.{d}f}".replace(".", ",")


def r(x, d=3):
    if isinstance(x, dict) and "den" in x:
        return f"{n(x['value'], d)} ({x['num']}/{x['den']})"
    return n(x, d)


def build() -> str:
    ev = load("evaluation.json")
    out: list[str] = []
    if ev and ev.get("version") == 2:
        m = ev["main"]
        mt, rc, pe, st = m["metrics"], m["reconciliation"], m["pet_errors"], ev["setup"]
        out += [
            f"**Mô phỏng có đáp án** ({st['duration_s']:.0f} s, seed {st['seed']}, "
            f"{st['n_ground_truth']} xung đột chuẩn = {st['n_ground_truth_near_miss']} near-miss + "
            f"{st['n_ground_truth_collision']} va chạm; detector mô phỏng nhiễu {n(st['noise_px'], 1)} px, "
            f"bỏ sót {st['miss_rate'] * 100:.0f}%, seed {st['replay_seed']}):",
            "",
            "| Chỉ số | Kết quả (tử/mẫu) |",
            "|---|---:|",
            f"| Precision | {r(mt['precision'])} |",
            f"| Recall | {r(mt['recall'])} |",
            f"| F1 | {n(mt['f1'])} |",
            f"| TTC MAE | {n(mt['ttc_mae'])} s |",
            f"| PET đo - MAE | {n((pe.get('measured') or {}).get('mae'))} s (n={(pe.get('measured') or {}).get('n', 0)}) |",
            f"| PET ngoại suy - MAE | {n((pe.get('predicted') or {}).get('mae'))} s (n={(pe.get('predicted') or {}).get('n', 0)}) |",
            f"| Recall near-miss / va chạm | {r(m['by_kind']['near_miss']['recall'])} / {r(m['by_kind']['collision']['recall'])} |",
            f"| Tracking IDF1 / MOTA / đổi ID | {n(m['tracking']['idf1'])} / {n(m['tracking']['mota'])} / {m['tracking']['id_switches']} |",
            f"| Quỹ đạo bị gắn cờ hành vi | {r(m['behavior']['flagged_tracks'])} |",
            "",
            f"Đối soát: {rc['rows']['emitted']} sự kiện phát ra = {rc['rows']['exported_to_results']} trên dashboard"
            f" = {rc['rows']['ignored_outside_eval_region']} ngoài vùng đánh giá + {rc['rows']['scored']} được chấm"
            f" ({rc['rows']['scored_tp']} TP + {rc['rows']['scored_mapped_fp'] + rc['rows']['scored_unmapped_track_fp']} FP). "
            f"Đẳng thức tự kiểm: {'đều đúng' if rc['all_checks_pass'] else 'CÓ LỆCH'}.",
            "",
            "Ablation bỏ từng khối (cùng mẫu, cùng seed):",
            "",
            "| Cấu hình | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ]
        for row in ev["ablation"]["leave_one_out"]:
            out.append(f"| {row['label']} | {r(row['precision'])} | {r(row['recall'])} | {n(row['f1'])} |")
        out.append("")

    reals = [("Đại lộ (Kaggle), 300 s, YOLO11n gốc", load("evaluation_street.json")),
             ("Đại lộ, test 220-300 s, YOLO11n gốc", load("evaluation_street_pretrained_test.json")),
             ("Đại lộ, test 220-300 s, YOLO11n tinh chỉnh", load("evaluation_street_finetuned_test.json")),
             ("MVTI, YOLO11n gốc", load("evaluation_real.json"))]
    reals = [(k, v) for k, v in reals if v and v.get("version") == 2]
    if reals:
        out += [
            "**Video thật - detection → tracking → cảnh báo trên cùng một video** "
            "(cảnh báo so với tham chiếu = cùng thuật toán chạy trên quỹ đạo người gán nhãn):",
            "",
            "| Dữ liệu / detector | mAP@0,5 | mAP@0,5 (bỏ hộp nhỏ) | Recall @ngưỡng chạy | IDF1 | Cảnh báo / tham chiếu | P so tham chiếu | R so tham chiếu | Báo nhầm/giờ | FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k, v in reals:
            a = v["alerts"]
            ph = a["reconciliation"]["per_hour"]
            big = v.get("detection_min_area") or {}
            out.append(
                f"| {k} | {n(v['detection']['mAP50'])} | {n(big.get('mAP50'))} | "
                f"{n(big.get('recall50_op', v['detection'].get('recall50_op')))} | {n(v['tracking']['idf1'])} | "
                f"{a['reconciliation']['rows']['emitted']} / {a['n_reference']} | "
                f"{r(a['metrics_vs_reference']['precision'])} | {r(a['metrics_vs_reference']['recall'])} | "
                f"{n(ph['false_alarms_per_hour'], 1)} | {n(v['runtime']['processing_fps'], 1)} |")
        out.append("")
        out.append("\"Bỏ hộp nhỏ\": hộp chuẩn nhỏ hơn `detection.min_box_area` (200 px² với đại lộ) "
                   "được coi là vùng bỏ qua theo quy ước COCO, vì pipeline cố ý không xử lý vật thể "
                   "xa như vậy. Người đi bộ trên video đại lộ có kích thước trung vị 8×23 px - ở "
                   "đầu vào 640 px chỉ còn khoảng 3×8 px nên YOLO11n gần như không thấy.")
        out.append("")

    sens = load("street_reference_sensitivity.json")
    if sens:
        out += [
            f"**Cảnh báo tham chiếu trên video đại lộ theo từng bước lọc** "
            f"({sens['duration_s']:.0f} s, quỹ đạo người gán nhãn):",
            "",
            "| Bước | Cấu hình | Cảnh báo | /giờ | Near-miss / chồng thân / nghi va chạm |",
            "|---|---|---:|---:|---:|",
        ]
        for row in sens["steps"]:
            c = row["by_contact"]
            out.append(f"| {row['step']} | {row['label']} | {row['n_events']} | {n(row['per_hour'], 1)} | "
                       f"{c.get('near_miss', 0)} / {c.get('body_overlap', 0)} / {c.get('collision_suspected', 0)} |")
        out.append("")
        out.append("Phần lớn cảnh báo ở bước A-D là sai số hình học (xe đỗ ở xa có bbox rung hơn 1 m; "
                   "lòng đường bên phải nằm ngoài vùng có điểm hiệu chuẩn), nên cảnh báo chỉ được phân "
                   "tích trong lòng đường đã hiệu chuẩn (vàng):")
        out.append("")
        out.append("![Vùng phân tích](docs/assets/street_analysis_region.jpg)")
        out.append("")

    ho = load("street_heldout/results.json")
    if ho:
        sm = ho.get("summary", {})
        dur = sm.get("duration_s") or 0
        nev = sm.get("total_events", len(ho.get("events", [])))
        out.append(f"**Chạy đầu-cuối trên đoạn chưa từng dùng** (đại lộ 300-741 s, không nhãn, detector "
                   f"tinh chỉnh): {nev} cảnh báo trong {dur:.0f} s "
                   f"({n((nev / dur * 3600) if dur else 0, 1)}/giờ), {n(sm.get('processing_fps'), 1)} FPS; "
                   f"toàn bộ cảnh báo nằm trong `data/outputs/review_candidates_street_heldout.csv` để người rà soát.")
        out.append("")
    cal = load("calibration_street.json")
    if cal:
        out.append(f"Hiệu chuẩn video đại lộ: {cal['n_points']} điểm mốc từ vạch kẻ, sai số fit "
                   f"{n(cal['fit']['rmse_m'])} m, leave-one-out {n(cal['leave_one_out']['rmse_m'])} m "
                   f"(tối đa {n(cal['leave_one_out']['max_m'])} m) - ước lượng, chưa đo thực địa.")
        out.append("")
    return "\n".join(out)


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    a, b = "<!-- AUTO:RESULTS:START -->", "<!-- AUTO:RESULTS:END -->"
    if a not in text or b not in text:
        raise SystemExit("README thiếu dấu AUTO:RESULTS")
    head, rest = text.split(a, 1)
    _old, tail = rest.split(b, 1)
    readme.write_text(head + a + "\n" + build() + "\n" + b + tail, encoding="utf-8")
    print("✓ README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
