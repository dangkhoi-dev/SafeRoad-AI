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
             ("MVTI, YOLO11n gốc", load("evaluation_real.json")),
             ("MVTI, test 57-81 s, YOLO11n gốc", load("evaluation_mvti_pretrained_test.json")),
             ("MVTI, test 57-81 s, YOLO11n tinh chỉnh", load("evaluation_mvti_finetuned_test.json"))]
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
        mp, mf = load("evaluation_mvti_pretrained_test.json"), load("evaluation_mvti_finetuned_test.json")
        if mp and mf and mp.get("detection_unseen_objects") and mf.get("detection_unseen_objects"):
            up, uf = mp["detection_unseen_objects"], mf["detection_unseen_objects"]
            out.append(f"MVTI chỉ có 19 phương tiện và phần lớn đứng trong khung suốt 81 s, nên trong "
                       f"đoạn test {uf['n_test_tracks'] - uf['n_unseen_tracks']}/{uf['n_test_tracks']} xe đã gặp lúc huấn luyện "
                       f"(model có thể đã nhớ chúng). Chấm riêng {uf['n_unseen_tracks']} xe "
                       f"chỉ xuất hiện sau phần huấn luyện ({uf['n_gt']} hộp): mAP@0,5 "
                       f"{n(up['mAP50'])} (gốc) → {n(uf['mAP50'])} (tinh chỉnh).")
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
    lg = load("lane_gate_check.json")
    if ho:
        sm = ho.get("summary", {})
        dur = sm.get("duration_s") or 0
        nev = sm.get("total_events", len(ho.get("events", [])))
        hb = (lg or {}).get("heldout")
        before = (f"Bản trước cổng cùng làn phát {hb['alerts_before']} cảnh báo "
                  f"({n(hb['false_alarms_per_hour_before'], 1)}/giờ), danh sách để người rà soát ở "
                  "`data/outputs/review_candidates_street_heldout.csv`; " if hb else "")
        out.append(f"**Chạy đầu-cuối trên đoạn chưa từng dùng** (đại lộ 300-741 s, không nhãn, detector "
                   f"tinh chỉnh, {n(sm.get('processing_fps'), 1)} FPS). {before}bản hiện tại phát {nev} cảnh báo "
                   f"trong {dur:.0f} s ({n((nev / dur * 3600) if dur else 0, 1)}/giờ).")
        out.append("")
    reviews = sorted(OUT.glob("review_score_*.json"))
    if reviews:
        out += [
            "**Cảnh báo so với nhãn xung đột do người rà soát** (`scripts/score_reviews.py`; nhãn gán "
            "trên bản trước cổng cùng làn). "
            "Kappa \"-\" khi cả hai người chỉ dùng một nhãn (không xác định), khi đó đọc cột trùng nhãn; "
            "bỏ sót/giờ \"-\" là tập chưa được xem lại toàn bộ video:",
            "",
            "| Tập | Ứng viên | Trùng nhãn | Kappa (5 lớp / có-không) | P | R | Báo nhầm/giờ | Bỏ sót/giờ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for f in reviews:
            x = json.loads(f.read_text(encoding="utf-8"))
            name = f.stem.replace("review_score_", "")
            ag = x.get("raw_agreement")
            out.append(f"| {name} | {x['n_candidates']} | {'-' if ag is None else f'{ag:.0%}'} | "
                       f"{n(x.get('kappa_5class'))} / "
                       f"{n(x.get('kappa_conflict_vs_not'))} | {r(x['precision'])} | "
                       f"{r(x['recall'])} | {n(x.get('false_alarms_per_hour'), 1)} | "
                       f"{n(x.get('misses_per_hour'), 1)} |")
        out.append("")
        h = load("review_score_street_heldout.json")
        if h and h["precision"]["den"] and h["precision"]["num"] == 0:
            out.append(f"Trên 441 s chưa từng dùng, cả {h['precision']['den']} cảnh báo của bản trước cổng đều được "
                       f"người rà soát đánh giá là giao thông bình thường ({n(h['false_alarms_per_hour'], 1)} "
                       f"báo nhầm/giờ); " + ("hai người xem lại toàn bộ video và không thấy xung đột thật nào, nên chưa đo được "
                       "Recall." if h.get("extra_missed") == 0 else "chưa xem lại toàn bộ video để đếm bỏ sót."
                       if h.get("extra_missed") is None else f"người xem ghi thêm {h.get('extra_missed')} "
                       "xung đột bị bỏ sót."))
            out.append("")
        hb = (lg or {}).get("heldout")
        sim = (lg or {}).get("simulation") or {}
        if hb:
            par = sum(1 for e in hb["events"] if e["after"] == "removed")
            kept = [e for e in hb["events"] if e["after"] == "kept"]
            kept_txt = ", ".join(f"{e['conflict_type']} {e['approach_angle']:.0f}°" for e in kept)
            simtxt = ""
            if sim.get("with_gate") and sim.get("without_gate"):
                w, o = sim["with_gate"], sim["without_gate"]
                simtxt = (f" Trên mô phỏng có đáp án, bỏ cổng này cho Precision {r(o['precision'])}, Recall "
                          f"{r(o['recall'])}; có cổng: {r(w['precision'])}, {r(w['recall'])}" +
                          (" - Recall không đổi" if w["recall"]["num"] >= o["recall"]["num"] else " - Recall giảm") +
                          (f", thêm {w['precision']['den'] - o['precision']['den']} báo nhầm."
                           if w["precision"]["den"] > o["precision"]["den"] else "."))
                gd = sim.get("grid")
                if gd:
                    simtxt += (f" Trên lưới {gd['n_cells']} ô ({gd['n_gt']} xung đột chuẩn): TP "
                               f"{gd['tp_without_gate']} → {gd['tp_with_gate']}, cảnh báo được chấm "
                               f"{gd['scored_without_gate']} → {gd['scored_with_gate']}.")
            out.append(f"**Phân tích lỗi và cổng cùng làn** (`scripts/measure_lateral_offset.py`, "
                       f"`scripts/lane_gate_check.py`). Đo lại từng cảnh báo: tại khung hình cảnh báo, hai xe lệch "
                       f"ngang {n(hb['lateral_m_min'], 1)}-{n(hb['lateral_m_max'], 1)} m, tức là đi cùng chiều ở "
                       "hai làn khác nhau; "
                       "luật cũ chỉ xét góc giữa hai vectơ vận tốc nên gọi đó là đâm đuôi. Cổng mới: cặp có góc "
                       "tiếp cận dưới 30° và không xe nào đang rẽ gấp (> 45°/s) chỉ được xét khi lệch ngang dưới "
                       f"1,75 m (nửa làn 3,5 m đã giả định khi hiệu chuẩn, không dò). Kết quả trên đoạn này: {hb['alerts_before']} → {hb['alerts_after']} "
                       f"cảnh báo, {n(hb['false_alarms_per_hour_before'], 1)} → "
                       f"{n(hb['false_alarms_per_hour_after'], 1)} báo nhầm/giờ; {par} cảnh báo bị loại đều đã "
                       f"được người đánh giá là bình thường; {len(kept)} cảnh báo còn lại ({kept_txt}: góc tại đỉnh "
                       "trên 30° nên cổng không áp dụng) cũng đã bị đánh giá là bình thường." + simtxt + " Vì cổng được rút ra từ chính đoạn 441 s "
                       "này nên con số sau sửa không phải phép thử độc lập; cần một đoạn video mới có nhãn người.")
            out.append("")
            out.append("![Mười một cảnh báo trên đoạn chưa từng dùng](docs/assets/heldout_lane_gate.jpg)")
            out.append("")
    cal = load("calibration_street.json")
    if cal:
        out.append(f"Hiệu chuẩn video đại lộ: {cal['n_points']} điểm mốc từ vạch kẻ, sai số fit "
                   f"{n(cal['fit']['rmse_m'])} m, leave-one-out {n(cal['leave_one_out']['rmse_m'])} m "
                   f"(tối đa {n(cal['leave_one_out']['max_m'])} m) - ước lượng, chưa đo thực địa.")
        out.append("")
    h = load("review_score_street_heldout.json")
    first = ("chưa có nhãn xung đột do người rà soát trên video thật (công cụ đã sẵn sàng)" if h is None else
             "hai video thật đã có nhãn người đều là giao thông bình thường, nên chưa đo được Recall trên "
             "video thật, và báo nhầm/giờ sau khi thêm cổng cùng làn mới được kiểm trên chính đoạn đã dùng "
             "để phân tích lỗi" if h["precision"]["den"] and h["precision"]["num"] == 0
             and not h.get("extra_missed") else
             "nhãn người rà soát mới có trên ít tình huống, khoảng tin cậy còn rộng")
    out.append(f"Giới hạn còn lại được báo cáo nguyên trạng: {first}; hiệu chuẩn video đại lộ là ước lượng "
               "từ vạch kẻ đường chứ chưa đo thực địa và chỉ phủ lòng đường bên trái, nên cảnh báo chỉ được "
               "phân tích trong vùng đó; người đi bộ trên video đại lộ quá nhỏ cho YOLO ở 640 px; hai bộ video "
               "thật không có xe máy; mô phỏng dày đặc va chạm hơn thực tế; \"ánh sáng\" trong lưới thực "
               "nghiệm là hồ sơ chất lượng detector.")
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
