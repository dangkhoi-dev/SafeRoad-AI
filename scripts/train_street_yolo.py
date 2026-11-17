"""Fine-tune YOLO11n trên tập đại lộ đã xuất bằng scripts/export_street_yolo.py.

    python scripts/train_street_yolo.py            # CPU, khoảng 35 phút

Cấu hình đã dùng cho models/saferoad_street_yolo11n.pt: 10 epoch, imgsz 640,
batch 4, tắt mosaic (máy 8 GB RAM), seed 0, deterministic. Tập "val" của
Ultralytics chính là các khung kiểm tra (≥ 6600) nên trọng số được lấy ở epoch
CUỐI, không chọn epoch theo tập kiểm tra (ở lần chạy này best = last vì mAP
tăng đều). Kết quả từng epoch: data/outputs/train_street_yolo11n_results.csv.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_street/data.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="models/saferoad_street_yolo11n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    model.train(data=args.data, imgsz=640, epochs=args.epochs, batch=args.batch,
                workers=0, mosaic=0.0, close_mosaic=0, device=args.device,
                project=str(ROOT / "runs"), name="street", exist_ok=True,
                patience=0, plots=False, seed=0, deterministic=True, amp=False,
                cache=False)
    run = ROOT / "runs" / "street"
    shutil.copy2(run / "weights" / "last.pt", ROOT / args.out)
    shutil.copy2(run / "results.csv",
                 ROOT / "data" / "outputs" / "train_street_yolo11n_results.csv")
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
