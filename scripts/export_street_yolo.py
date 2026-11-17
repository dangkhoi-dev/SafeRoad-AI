"""Xuất video đại lộ (Kaggle) + nhãn đã ghép thành tập YOLO để fine-tune.

    python scripts/export_street_yolo.py --video data/processed/street/street_labeled_0_300s.mp4 \\
        --ground-truth data/processed/street/street_groundtruth.pkl --out data/yolo_street

Chia theo THỜI GIAN, không chia ngẫu nhiên: khung hình liền nhau gần như giống
hệt nhau, chia ngẫu nhiên thì tập kiểm tra chứa "bản sao" của tập huấn luyện và
mAP bị thổi phồng. Mặc định: huấn luyện 0-210 s, bỏ trống 210-220 s, kiểm tra
220-300 s. Phần 300-741 s của video gốc không có nhãn, dùng cho chạy thử đầu-cuối.

Giữ nguyên chỉ số lớp COCO (person = 0, car = 2) với đủ 80 tên lớp, để model
fine-tune dùng thẳng được trong pipeline mà không phải đổi ánh xạ lớp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saferoad.types import VehicleClass  # noqa: E402
from saferoad.utils.bundle import load_bundle  # noqa: E402

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
TO_COCO = {VehicleClass.PEDESTRIAN: 0, VehicleClass.CAR: 2, VehicleClass.TRUCK: 7,
           VehicleClass.MOTORCYCLE: 3, VehicleClass.BICYCLE: 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="data/yolo_street")
    ap.add_argument("--train-end", type=int, default=6300, help="khung cuối (không gồm) của tập train")
    ap.add_argument("--test-start", type=int, default=6600)
    ap.add_argument("--train-step", type=int, default=10)
    ap.add_argument("--val-step", type=int, default=30)
    args = ap.parse_args()

    seq = load_bundle(args.ground_truth)["sequence"]
    out = Path(args.out)
    want: dict[int, str] = {}
    for f in range(0, args.train_end, args.train_step):
        want[f] = "train"
    for f in range(args.test_start, seq.n_frames, args.val_step):
        want[f] = "val"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    f = 0
    n = {"train": 0, "val": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        split = want.get(f)
        if split is not None:
            h, w = frame.shape[:2]
            name = f"street_{f:06d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for det in seq.detections.get(f, []):
                x1, y1, x2, y2 = det.bbox
                cid = TO_COCO.get(det.cls)
                if cid is None or x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n[split] += 1
        f += 1
    cap.release()
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 80\nnames: {COCO_NAMES}\n", encoding="utf-8")
    print(f"✓ {n['train']} ảnh train (khung 0-{args.train_end - 1}), {n['val']} ảnh val "
          f"(khung {args.test_start}-{seq.n_frames - 1}) → {yaml_path}")
    print("Fine-tune (GPU/Colab khuyến nghị; CPU vẫn chạy được, chậm):")
    print("  python scripts/train_street_yolo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
