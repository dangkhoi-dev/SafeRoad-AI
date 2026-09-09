"""Bộ nạp **Multi-view Traffic Intersection Dataset** (MVTI).

Nguồn: Andreas Møgelmose, "Multi-view Traffic Intersection Dataset",
https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset

Vì sao dataset này quan trọng với SafeRoad AI
--------------------------------------------
Đây là dữ liệu **thật** của một giao lộ, quay từ camera hạ tầng cố định trên cao
— đúng cấu hình mà đề bài mô tả. Quan trọng hơn, mỗi annotation mang trường
``object_id`` **bền vững qua các khung hình**, nghĩa là dataset có sẵn *ground
truth cho tracking*. Nhờ đó ta đo được IDF1/MOTA và mAP trên dữ liệu thật, chứ
không chỉ trên mô phỏng.

Phân công vai trò giữa hai nguồn dữ liệu (nêu rõ trong báo cáo):

===================  ==========================  ==============================
Nguồn                Đo được gì                  Không đo được gì
===================  ==========================  ==============================
MVTI (thật)          mAP detection, IDF1/MOTA    Precision/Recall của near-miss
                     tracking, FPS thực tế       (dataset không có nhãn xung đột)
Mô phỏng (synthetic) Precision/Recall near-miss, Độ khó thị giác của ảnh thật
                     TTC MAE, ablation
===================  ==========================  ==============================

Ánh xạ lớp: dataset dùng bộ nhãn kiểu COCO nhưng chỉ xuất hiện 4 lớp —
car(3), lorry/truck(8), bicycle(2), bus(6). Không có người đi bộ và xe máy, nên
khi báo cáo phải nói rõ hai lớp đó chỉ được đánh giá trên tập mô phỏng.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import Detection, VehicleClass

log = logging.getLogger(__name__)

#: Ánh xạ category_id của MVTI sang lớp của SafeRoad.
MVTI_TO_SAFEROAD: dict[int, VehicleClass] = {
    1: VehicleClass.PEDESTRIAN,   # person
    2: VehicleClass.BICYCLE,      # bicycle / cyclist
    3: VehicleClass.CAR,          # car
    4: VehicleClass.MOTORCYCLE,   # motorbike
    6: VehicleClass.TRUCK,        # bus
    8: VehicleClass.TRUCK,        # lorry / truck / van
}


@dataclass
class GtTrack:
    """Quỹ đạo chuẩn của một đối tượng, lấy từ ``object_id`` của dataset."""

    object_id: int
    cls: VehicleClass
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    def box_at(self, frame_idx: int) -> tuple[float, float, float, float] | None:
        try:
            return self.boxes[self.frames.index(frame_idx)]
        except ValueError:
            return None

    @property
    def anchors(self) -> np.ndarray:
        """Điểm tiếp đất (đáy giữa bbox) của từng khung hình, dạng (N, 2)."""
        return np.array(
            [((b[0] + b[2]) / 2.0, b[3]) for b in self.boxes], dtype=float
        )


@dataclass
class MvtiSequence:
    """Một chuỗi ảnh đã nạp cùng annotation của nó."""

    name: str
    root: Path
    frame_files: list[Path]
    #: detections[frame_idx] = [Detection, ...]
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    #: id_map[frame_idx][chỉ_số_trong_danh_sách] = object_id
    id_map: dict[int, dict[int, int]] = field(default_factory=dict)
    tracks: dict[int, GtTrack] = field(default_factory=dict)
    width: int = 0
    height: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frame_files)

    def summary(self) -> dict:
        from collections import Counter

        cls_counts = Counter(t.cls.value for t in self.tracks.values())
        lengths = [len(t.frames) for t in self.tracks.values()]
        return {
            "name": self.name,
            "n_frames": self.n_frames,
            "n_annotated_frames": len(self.detections),
            "n_tracks": len(self.tracks),
            "n_boxes": sum(len(d) for d in self.detections.values()),
            "resolution": f"{self.width}x{self.height}",
            "classes": dict(cls_counts),
            "track_length_median": int(np.median(lengths)) if lengths else 0,
        }


def load_coco(
    annotation_path: str | Path,
    dataset_root: str | Path,
    view: str = "Infrastructure",
    min_box_area: float = 0.0,
) -> MvtiSequence:
    """Nạp annotation COCO của MVTI và dựng lại các quỹ đạo chuẩn.

    Tham số:
        annotation_path: file ``infrastructure-mscoco.json`` hoặc ``drone-mscoco.json``.
        dataset_root: thư mục gốc chứa ``Infrastructure/`` và ``Drone/``.
        view: ``"Infrastructure"`` hoặc ``"Drone"`` — lọc theo tiền tố ``file_name``.
        min_box_area: bỏ qua bbox nhỏ hơn ngưỡng (pixel²).

    Trả về ``MvtiSequence`` với ``detections``, ``id_map`` và ``tracks`` đã điền.
    Chỉ số frame được đánh **liên tục theo thứ tự tên file**, để khớp với video
    dựng bởi :func:`build_video`.
    """
    annotation_path = Path(annotation_path)
    dataset_root = Path(dataset_root)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    images = [im for im in data["images"] if im["file_name"].startswith(view)]
    if not images:
        raise ValueError(
            f"Không có ảnh nào thuộc view '{view}' trong {annotation_path.name}"
        )
    images.sort(key=lambda im: im["file_name"])

    # Đánh lại chỉ số frame liên tục theo thứ tự tên file.
    order = {im["id"]: i for i, im in enumerate(images)}
    by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in order:
            by_image[ann["image_id"]].append(ann)

    seq = MvtiSequence(
        name=f"{view.lower()}-{annotation_path.stem}",
        root=dataset_root,
        frame_files=[dataset_root / im["file_name"] for im in images],
        width=images[0]["width"],
        height=images[0]["height"],
    )

    skipped_cls: set[int] = set()
    for im in images:
        idx = order[im["id"]]
        dets: list[Detection] = []
        ids: dict[int, int] = {}

        for ann in by_image.get(im["id"], []):
            vclass = MVTI_TO_SAFEROAD.get(ann["category_id"])
            if vclass is None:
                skipped_cls.add(ann["category_id"])
                continue
            # COCO bbox là [x, y, w, h] → đổi sang (x1, y1, x2, y2).
            x, y, w, h = ann["bbox"]
            if w * h < min_box_area:
                continue
            box = (float(x), float(y), float(x + w), float(y + h))

            ids[len(dets)] = ann.get("object_id", -1)
            dets.append(Detection(bbox=box, score=1.0, cls=vclass))

            oid = ann.get("object_id")
            if oid is not None and oid >= 0:
                track = seq.tracks.get(oid)
                if track is None:
                    track = GtTrack(object_id=oid, cls=vclass)
                    seq.tracks[oid] = track
                track.frames.append(idx)
                track.boxes.append(box)

        if dets:
            seq.detections[idx] = dets
            seq.id_map[idx] = ids

    if skipped_cls:
        log.info("Bỏ qua các category không thuộc 5 lớp SafeRoad: %s", sorted(skipped_cls))
    log.info(
        "Đã nạp %s: %d frame, %d annotated, %d track",
        seq.name, seq.n_frames, len(seq.detections), len(seq.tracks),
    )
    return seq


def build_video(
    seq: MvtiSequence,
    out_path: str | Path,
    fps: int = 30,
    max_frames: int = 0,
    progress_every: int = 300,
) -> Path:
    """Ghép chuỗi ảnh thành một video MP4 để pipeline đọc như camera thật.

    Pipeline làm việc trên luồng video; ghép sẵn thành MP4 vừa cho phép tua/xem
    lại khi demo, vừa tránh chi phí mở hàng nghìn file JPEG lẻ mỗi lần chạy.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = seq.frame_files[:max_frames] if max_frames else seq.frame_files
    if not files:
        raise ValueError("Chuỗi không có khung hình nào")

    first = cv2.imread(str(files[0]))
    if first is None:
        raise FileNotFoundError(f"Không đọc được ảnh đầu tiên: {files[0]}")
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho {out_path}")

    written = 0
    try:
        for i, path in enumerate(files):
            img = cv2.imread(str(path))
            if img is None:
                log.warning("Bỏ qua ảnh hỏng: %s", path)
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"  ghép {written}/{len(files)} khung hình...", flush=True)
    finally:
        writer.release()

    log.info("Đã ghép %d khung hình → %s", written, out_path)
    return out_path


def discover(root: str | Path) -> dict[str, Path]:
    """Tìm các file annotation MVTI trong một thư mục.

    Trả về map ``{view: đường dẫn json}`` cho các view tìm được.
    """
    root = Path(root)
    found: dict[str, Path] = {}
    for name, view in (
        ("infrastructure-mscoco.json", "Infrastructure"),
        ("drone-mscoco.json", "Drone"),
        ("merged_annotations.json", "merged"),
    ):
        path = root / name
        if path.exists():
            found[view] = path
    return found
