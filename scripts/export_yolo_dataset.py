#!/usr/bin/env python3
"""Xuất dataset MVTI sang định dạng YOLO để fine-tune YOLO11n.

Vì sao cần fine-tune
--------------------
Đo trực tiếp trên MVTI, YOLO11n với trọng số COCO gốc chỉ đạt recall@0.5 ≈ 0.19.
Nguyên nhân không phải model yếu mà là **lệch miền (domain gap)**: COCO chủ yếu
gồm ảnh chụp ngang tầm mắt, trong khi camera giao thông đặt cao 10-18 m nhìn
chếch xuống, đối tượng nhỏ và bị nén phối cảnh. Trong một khung hình thử,
YOLO11n chỉ phát hiện đúng một vật thể và gán nhầm nhãn "train" cho một chiếc ô tô.

MVTI có sẵn 14.488 bounding box trên 2.441 ảnh — quá đủ để fine-tune và đóng
phần lớn khoảng cách đó.

Cách dùng::

    python scripts/export_yolo_dataset.py \\
        --root data/raw/mvti --view Infrastructure --out data/yolo_mvti

Sau đó nén ``data/yolo_mvti`` và tải lên Colab; xem
``notebooks/01_finetune_yolo11n_colab.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# 5 lớp của SafeRoad AI, theo đúng thứ tự chỉ số dùng khi train.
CLASS_NAMES = ["pedestrian", "bicycle", "motorcycle", "car", "truck"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

#: category_id của MVTI → tên lớp SafeRoad.
MVTI_TO_NAME = {
    1: "pedestrian",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "truck",   # bus
    8: "truck",   # lorry / van
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Xuất MVTI sang định dạng YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", required=True, help="Thư mục chứa Infrastructure/, Drone/, *.json")
    ap.add_argument("--view", default="Infrastructure", choices=["Infrastructure", "Drone", "both"])
    ap.add_argument("--out", default="data/yolo_mvti", help="Thư mục kết quả")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true",
                    help="Tạo symlink thay vì sao chép ảnh (tiết kiệm dung lượng, Linux/macOS)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    views = ["Infrastructure", "Drone"] if args.view == "both" else [args.view]
    ann_files = {
        "Infrastructure": root / "infrastructure-mscoco.json",
        "Drone": root / "drone-mscoco.json",
    }

    # --- Gom ảnh + nhãn từ mọi view yêu cầu -------------------------------- #
    records: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    class_counts: Counter = Counter()
    missing = 0

    for view in views:
        ann_path = ann_files[view]
        if not ann_path.exists():
            print(f"Bỏ qua {view}: không có {ann_path.name}", file=sys.stderr)
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        images = {im["id"]: im for im in data["images"] if im["file_name"].startswith(view)}
        by_image: dict[int, list] = {}
        for a in data["annotations"]:
            if a["image_id"] in images:
                by_image.setdefault(a["image_id"], []).append(a)

        for image_id, im in images.items():
            src = root / im["file_name"]
            if not src.exists():
                missing += 1
                continue
            w, h = im["width"], im["height"]
            labels = []
            for a in by_image.get(image_id, []):
                name = MVTI_TO_NAME.get(a["category_id"])
                if name is None:
                    continue
                x, y, bw, bh = a["bbox"]
                # YOLO: cx, cy, w, h — tất cả chuẩn hoá về [0, 1].
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                if nw <= 0 or nh <= 0:
                    continue
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                nw, nh = min(nw, 1.0), min(nh, 1.0)
                labels.append((CLASS_INDEX[name], cx, cy, nw, nh))
                class_counts[name] += 1
            # Giữ cả ảnh không có nhãn: chúng là mẫu nền (negative), giúp giảm
            # cảnh báo giả — YOLO xử lý được file nhãn rỗng.
            records.append((src, labels))

    if not records:
        print("Lỗi: không tìm thấy ảnh nào. Kiểm tra lại --root.", file=sys.stderr)
        return 2
    if missing:
        print(f"Cảnh báo: {missing} ảnh có trong annotation nhưng thiếu trên đĩa")

    # --- Chia train/val theo THỜI GIAN, không xáo trộn ngẫu nhiên ---------- #
    # Các khung hình liên tiếp gần như giống hệt nhau; chia ngẫu nhiên sẽ khiến
    # ảnh gần như trùng lặp xuất hiện ở cả hai tập và làm điểm val cao giả tạo.
    # Cắt theo mốc thời gian cho ước lượng trung thực hơn nhiều.
    records.sort(key=lambda r: str(r[0]))
    split = int(len(records) * (1 - args.val_ratio))
    subsets = {"train": records[:split], "val": records[split:]}

    if out.exists():
        shutil.rmtree(out)
    for subset in subsets:
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for subset, items in subsets.items():
        for i, (src, labels) in enumerate(items):
            stem = f"{src.parent.name}_{src.stem}"
            dst_img = out / "images" / subset / f"{stem}{src.suffix}"
            if args.symlink:
                try:
                    dst_img.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst_img)
            else:
                shutil.copy2(src, dst_img)

            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels]
            (out / "labels" / subset / f"{stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            if (i + 1) % 500 == 0:
                print(f"  {subset}: {i + 1}/{len(items)}")

    # --- data.yaml -------------------------------------------------------- #
    yaml_text = (
        f"# Dataset MVTI xuất sang định dạng YOLO cho SafeRoad AI\n"
        f"# Nguồn: Multi-view Traffic Intersection Dataset (Møgelmose)\n"
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    stats = {
        "n_images": len(records),
        "n_train": len(subsets["train"]),
        "n_val": len(subsets["val"]),
        "views": views,
        "class_counts": dict(class_counts),
        "classes_present": sorted(class_counts),
        "classes_missing": [c for c in CLASS_NAMES if c not in class_counts],
        "split": "theo thời gian (không xáo trộn) để tránh rò rỉ giữa các khung hình liền kề",
    }
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✓ Đã xuất {len(records)} ảnh → {out}")
    print(f"  train={stats['n_train']}  val={stats['n_val']}")
    print(f"  Phân bố lớp: {stats['class_counts']}")
    if stats["classes_missing"]:
        print(f"  ⚠ Lớp KHÔNG có trong dataset này: {stats['classes_missing']}")
        print("    → sau khi fine-tune, các lớp đó sẽ không phát hiện được.")
        print("    → xem hướng dẫn giữ lại năng lực COCO trong notebook Colab.")
    print(f"\n  data.yaml: {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
