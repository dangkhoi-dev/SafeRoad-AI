"""Chấm toàn bộ nhãn rà soát đã lưu, rồi cập nhật README và báo cáo.

    python scripts/score_reviews.py --missed street_heldout=2

Tìm mọi file data/outputs/labels_<tập>_<người>.csv do tab "Rà soát nhãn" lưu ra,
gom theo tập; tập nào có từ hai người trở lên thì chấm bằng `saferoad review-score`
(kappa, nhãn đồng thuận, Precision/Recall, báo nhầm/bỏ sót mỗi giờ) và ghi
data/outputs/review_score_<tập>.json. Thời lượng video lấy từ chính file kết quả
đã sinh ra danh sách ứng viên. `--missed tập=số` là số xung đột người xem thấy
trong video nhưng KHÔNG có trong danh sách (bỏ sót), sau khi hai người xem lại
toàn bộ video. Tập không có `--missed` thì để trống bỏ sót/giờ (chưa đếm), không ghi 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from saferoad.evaluation.real_conflicts import score_reviewed  # noqa: E402


def duration_of(name: str) -> float | None:
    """Thời lượng (giây) của đoạn video ứng với danh sách review_candidates_<name>.csv."""
    for path, keys in ((OUT / f"evaluation_{name}.json", ("runtime", "duration_s")),
                       (OUT / f"{name}" / "results.json", ("summary", "duration_s"))):
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            for k in keys:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if isinstance(d, (int, float)):
                return float(d)
    return None


def _frac(r: dict) -> str:
    return f"{r['num']}/{r['den']}" if r and r.get("den") else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missed", nargs="*", default=[],
                    help="số xung đột bị bỏ sót theo từng tập, dạng tập=số")
    ap.add_argument("--no-rebuild", action="store_true", help="không dựng lại README/báo cáo")
    args = ap.parse_args()
    missed = {k: int(v) for k, v in (m.split("=", 1) for m in args.missed)}

    sets = sorted(p.name[len("review_candidates_"):-4] for p in OUT.glob("review_candidates_*.csv"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(OUT.glob("labels_*.csv")):
        stem = f.stem[len("labels_"):]
        name = max((s for s in sets if stem.startswith(s + "_")), key=len, default=None)
        if name:
            groups[name].append(f)

    if not groups:
        print("Chưa có file labels_*.csv nào trong data/outputs - lưu nhãn ở tab Rà soát nhãn trước.")
        return 1
    done = 0
    for name, files in sorted(groups.items()):
        people = [f.stem[len("labels_" + name) + 1:] for f in files]
        if len(files) < 2:
            print(f"- {name}: mới có nhãn của {people[0]}, cần thêm một người nữa - bỏ qua")
            continue
        dur = duration_of(name)
        if dur is None:
            print(f"- {name}: không tìm được thời lượng video - bỏ qua")
            continue
        rep = score_reviewed(OUT / f"review_candidates_{name}.csv", files, dur,
                             missed.get(name))
        rep["reviewers"] = people
        (OUT / f"review_score_{name}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        k, ag = rep.get("kappa_5class"), rep.get("raw_agreement")
        kap = (f"κ = {k}" if k is not None else
               "κ không xác định (cả hai chỉ dùng một nhãn)" if ag is not None else "κ = -")
        warn = "  ⚠ κ < 0,6: ngồi lại thống nhất cách hiểu rồi gán lại" if rep.get("kappa_ok") is False else ""
        print(f"- {name}: {len(files)} người ({', '.join(people)}), {rep['n_candidates']} ứng viên, "
              f"trùng nhãn {'-' if ag is None else f'{ag:.0%}'}, {kap}, P = {_frac(rep['precision'])}, "
              f"R = {_frac(rep['recall'])}, bỏ sót thêm = {missed.get(name, 'chưa đếm')}{warn}")
        done += 1

    if done and not args.no_rebuild:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_readme_tables.py")], check=True)
        try:
            subprocess.run(["node", str(ROOT / "scripts" / "build_improvement_report.js")], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"  (chưa dựng lại được báo cáo PDF: {exc})")
    print(f"✓ Đã chấm {done} tập")
    return 0


if __name__ == "__main__":
    sys.exit(main())
