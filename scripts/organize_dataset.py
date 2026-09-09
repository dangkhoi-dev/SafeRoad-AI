#!/usr/bin/env python3
"""Sắp xếp dữ liệu thô đã tải về thành cấu trúc thư mục chuẩn của repo.

Chạy trên **máy của bạn** (nơi chứa dữ liệu), không phải trong container.

Cách dùng::

    # 1. Xem trước sẽ di chuyển những gì (KHÔNG đụng vào file)
    python scripts/organize_dataset.py --source D:\\SV7 --dry-run

    # 2. Thực hiện thật
    python scripts/organize_dataset.py --source D:\\SV7

Thiết kế
--------
* Mặc định **di chuyển** (``move``) chứ không sao chép: bộ dữ liệu ~11 GB, sao
  chép sẽ tốn thêm 11 GB và vài phút. Trên cùng một ổ đĩa, ``move`` gần như tức
  thời. Dùng ``--copy`` nếu muốn giữ nguyên bản gốc.
* **Không bao giờ ghi đè**: gặp file trùng tên thì bỏ qua và báo lại.
* ``--dry-run`` là mặc định an toàn cho lần chạy đầu — luôn xem trước.

Cấu trúc kết quả::

    data/raw/
      mvti/                    # Multi-view Traffic Intersection (giao lộ thật)
        Infrastructure/        #   camera hạ tầng
        Drone/                 #   camera drone
        *-mscoco.json          #   annotation COCO (có object_id = GT tracking)
      ucsd-highway/            # UCSD Highway Traffic Database
        video/                 #   254 clip .avi
        *.mat, info.txt ...
      own-footage/             # video tự quay của nhóm
      _archives/               # file .zip gốc (giữ lại để đối chiếu)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Ánh xạ: tên trong thư mục nguồn → đường dẫn đích tương đối trong data/raw.
LAYOUT: list[tuple[str, str, str]] = [
    # (tên nguồn, thư mục đích, mô tả)
    ("Infrastructure",              "mvti/Infrastructure",        "MVTI — camera hạ tầng"),
    ("Drone",                       "mvti/Drone",                 "MVTI — camera drone"),
    ("infrastructure-mscoco.json",  "mvti/infrastructure-mscoco.json", "MVTI — annotation COCO (hạ tầng)"),
    ("drone-mscoco.json",           "mvti/drone-mscoco.json",     "MVTI — annotation COCO (drone)"),
    ("merged_annotations.json",     "mvti/merged_annotations.json", "MVTI — annotation gộp"),

    ("video",                       "ucsd-highway/video",         "UCSD — 254 clip cao tốc"),
    ("README_TRAFFICDB",            "ucsd-highway/README_TRAFFICDB", "UCSD — readme gốc"),
    ("info.txt",                    "ucsd-highway/info.txt",      "UCSD — metadata từng clip"),
    ("ImageMaster",                 "ucsd-highway/ImageMaster",   "UCSD — chỉ mục lớp"),
    ("ImageMaster.mat",             "ucsd-highway/ImageMaster.mat", "UCSD — chỉ mục lớp (MATLAB)"),
    ("EvalSet.mat",                 "ucsd-highway/EvalSet.mat",   "UCSD — tập train/test"),
    ("EvalSet_train",               "ucsd-highway/EvalSet_train", "UCSD — tập train"),
    ("EvalSet_test",                "ucsd-highway/EvalSet_test",  "UCSD — tập test"),
    ("traffic_patches.mat",         "ucsd-highway/traffic_patches.mat", "UCSD — patch video"),
    ("traffic_patches_reg.mat",     "ucsd-highway/traffic_patches_reg.mat", "UCSD — patch đã căn chỉnh"),

    ("real_traffic",                "own-footage",                "Video giao thông tự quay / bổ sung"),
]

#: File giữ nguyên tại chỗ, không di chuyển.
KEEP_IN_PLACE = {"KeHoach_AI2026.xlsx", "SafeRoad-AI"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def move_item(src: Path, dst: Path, copy: bool, dry_run: bool) -> tuple[bool, str]:
    """Di chuyển (hoặc sao chép) một mục; trả về ``(thành_công, ghi_chú)``."""
    if not src.exists():
        return False, "không tồn tại"
    if dst.exists():
        return False, f"đích đã tồn tại ({dst})"

    if dry_run:
        return True, "sẽ " + ("sao chép" if copy else "di chuyển")

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if copy:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
    except Exception as exc:                      # noqa: BLE001
        return False, f"lỗi: {exc}"
    return True, "xong"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sắp xếp dữ liệu thô vào cấu trúc data/raw của SafeRoad AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source", required=True, help="Thư mục chứa dữ liệu đã tải (VD: D:\\SV7)")
    ap.add_argument("--repo", default=".", help="Thư mục gốc repo SafeRoad-AI")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ xem trước, không đụng file")
    ap.add_argument("--copy", action="store_true", help="Sao chép thay vì di chuyển")
    ap.add_argument("--archives", action="store_true",
                    help="Di chuyển luôn các file .zip gốc vào data/raw/_archives")
    args = ap.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    raw = Path(args.repo).expanduser().resolve() / "data" / "raw"
    if not source.is_dir():
        print(f"Lỗi: không tìm thấy thư mục nguồn {source}", file=sys.stderr)
        return 2

    mode = "XEM TRƯỚC (không đổi gì)" if args.dry_run else (
        "SAO CHÉP" if args.copy else "DI CHUYỂN")
    print(f"Nguồn : {source}")
    print(f"Đích  : {raw}")
    print(f"Chế độ: {mode}\n")

    ok_count = skip_count = 0
    moved_bytes = 0

    for name, target, desc in LAYOUT:
        src = source / name
        dst = raw / target
        if not src.exists():
            continue
        size = dir_size(src)
        ok, note = move_item(src, dst, args.copy, args.dry_run)
        status = "✓" if ok else "–"
        print(f"  {status} {name:32s} → data/raw/{target:38s} {human(size):>10s}  [{desc}]")
        if not ok:
            print(f"      ↳ {note}")
            skip_count += 1
        else:
            ok_count += 1
            moved_bytes += size

    if args.archives:
        for zip_path in sorted(source.glob("*.zip")):
            dst = raw / "_archives" / zip_path.name
            size = zip_path.stat().st_size
            ok, note = move_item(zip_path, dst, args.copy, args.dry_run)
            print(f"  {'✓' if ok else '–'} {zip_path.name:32s} → data/raw/_archives/"
                  f"{'':22s} {human(size):>10s}  [archive gốc]")
            if not ok:
                print(f"      ↳ {note}")

    # Các mục còn lại chưa được ánh xạ — báo để không bỏ sót gì.
    known = {n for n, _t, _d in LAYOUT} | KEEP_IN_PLACE
    leftovers = [
        p for p in sorted(source.iterdir())
        if p.name not in known and p.suffix.lower() != ".zip"
    ]
    if leftovers:
        print("\n  Chưa ánh xạ (giữ nguyên tại chỗ):")
        for p in leftovers:
            print(f"      {p.name}  {human(dir_size(p))}")

    print(f"\nTổng kết: {ok_count} mục{' sẽ được' if args.dry_run else ''} xử lý "
          f"({human(moved_bytes)}), {skip_count} bỏ qua.")
    if args.dry_run:
        print("\nĐây mới là xem trước. Chạy lại KHÔNG kèm --dry-run để thực hiện thật.")
    else:
        print("\nBước tiếp theo:")
        print("  python -m saferoad prepare-real --root data/raw/mvti --view Infrastructure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
