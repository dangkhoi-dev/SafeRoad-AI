#!/usr/bin/env python3
"""Tải các tài nguyên cần thiết (trọng số YOLO) về thư mục models/.

Tách riêng khỏi bước cài đặt để việc cài package không phụ thuộc mạng, và để
người dùng biết chính xác file nào được tải về từ đâu.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ASSETS = {
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "YOLO11n — trọng số COCO (5,6 MB)",
    ),
}


def download(name: str, url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  ✓ {name} đã có ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  ↓ Đang tải {name} …")
    try:
        with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as exc:                      # noqa: BLE001
        print(f"  ✗ Lỗi tải {name}: {exc}", file=sys.stderr)
        print(f"    Tải thủ công từ: {url}", file=sys.stderr)
        print(f"    Rồi đặt vào:     {dest}", file=sys.stderr)
        return False
    print(f"  ✓ {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tải trọng số model cho SafeRoad AI")
    ap.add_argument("--out", default="models", help="Thư mục lưu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Tải tài nguyên vào {out.resolve()}")
    ok = all(download(name, url, out / name) for name, (url, _d) in ASSETS.items())

    if ok:
        print("\n✓ Xong. Chạy thử:")
        print("    python -m saferoad simulate --duration 60")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
