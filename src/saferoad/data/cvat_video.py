"""Bộ nạp nhãn video xuất từ CVAT dạng COCO, có ghép lại quỹ đạo giữa các đoạn.

Bộ dữ liệu đường phố tải từ Kaggle (``data/raw/own-footage/input-001.MOV`` +
``merged_annotations.json``) được gán nhãn trên CVAT theo từng **đoạn 30 khung
hình, chồng nhau 5 khung**. Mỗi đoạn đánh lại ``track_id`` từ đầu, nên nếu đọc
thẳng thì một chiếc xe đi qua khung hình trong 20 giây sẽ thành ~24 "đối tượng"
khác nhau - IDF1/MOTA tính trên đó vô nghĩa, và không dựng được quỹ đạo để đo
TTC/PET tham chiếu.

Module này ghép lại: ở 5 khung chồng nhau giữa đoạn ``c`` và ``c+1``, hai
quỹ đạo có IoU trung bình cao nhất (và ≥ ``min_iou``) được coi là cùng một đối
tượng và nhận chung một ID toàn cục. Khung chồng nhau chỉ giữ bản nhãn của đoạn
sau để mỗi khung có đúng một bộ nhãn.

Ánh xạ lớp: ``Car`` → ô tô (bao gồm cả xe tải nhẹ, xe van - bộ nhãn gốc không
tách), ``People`` → người đi bộ. Bộ nhãn **không có** xe máy và xe đạp.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Detection, VehicleClass, iou
from .mvti import GtTrack, MvtiSequence

log = logging.getLogger(__name__)

CVAT_TO_SAFEROAD: dict[str, VehicleClass] = {
    "car": VehicleClass.CAR,
    "people": VehicleClass.PEDESTRIAN,
    "person": VehicleClass.PEDESTRIAN,
    "pedestrian": VehicleClass.PEDESTRIAN,
    "truck": VehicleClass.TRUCK,
    "bus": VehicleClass.TRUCK,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "bicycle": VehicleClass.BICYCLE,
}


def _frame_index(file_name: str) -> int:
    stem = Path(file_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def load_cvat_video(
    annotation_path: str | Path,
    name: str = "street",
    min_iou: float = 0.3,
    min_box_area: float = 0.0,
    max_frames: int = 0,
) -> tuple[MvtiSequence, dict[str, Any]]:
    """Nạp nhãn và ghép quỹ đạo. Trả về ``(sequence, thống_kê_ghép)``."""
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"].strip().lower() for c in data["categories"]}

    images = sorted(data["images"], key=lambda im: im["id"])
    width = int(images[0]["width"]) if images else 0
    height = int(images[0]["height"]) if images else 0

    # --- Tách đoạn: chỉ số khung hình giảm/lặp lại ⇒ bắt đầu một đoạn mới ---- #
    chunk_of: dict[int, int] = {}
    frame_of: dict[int, int] = {}
    chunk = 0
    prev = -1
    for im in images:
        f = _frame_index(im["file_name"])
        if f <= prev:
            chunk += 1
        prev = f
        chunk_of[im["id"]] = chunk
        frame_of[im["id"]] = f
    n_chunks = chunk + 1

    # boxes[(chunk, local_tid)] = {frame: (bbox, cls)}
    boxes: dict[tuple[int, int], dict[int, tuple[tuple[float, float, float, float], VehicleClass]]] = defaultdict(dict)
    skipped = 0
    for ann in data["annotations"]:
        cname = cats.get(ann["category_id"], "")
        vcls = CVAT_TO_SAFEROAD.get(cname)
        if vcls is None:
            skipped += 1
            continue
        x, y, w, h = ann["bbox"]
        if w * h < min_box_area:
            continue
        tid = (ann.get("attributes") or {}).get("track_id")
        if tid is None:
            tid = -ann["id"]          # hộp lẻ không thuộc track nào
        img = ann["image_id"]
        key = (chunk_of[img], int(tid))
        boxes[key][frame_of[img]] = ((float(x), float(y), float(x + w), float(y + h)), vcls)

    by_chunk: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in boxes:
        by_chunk[key[0]].append(key)

    # --- Ghép quỹ đạo giữa hai đoạn liên tiếp ---------------------------- #
    global_id: dict[tuple[int, int], int] = {}
    next_gid = 0
    links = 0
    for key in sorted(by_chunk.get(0, [])):
        global_id[key] = next_gid
        next_gid += 1

    for c in range(1, n_chunks):
        prev_keys = by_chunk.get(c - 1, [])
        cur_keys = by_chunk.get(c, [])
        pairs: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
        for kc in cur_keys:
            fc = boxes[kc]
            for kp in prev_keys:
                fp = boxes[kp]
                shared = set(fc) & set(fp)
                if not shared:
                    continue
                if fc[min(shared)][1] != fp[min(shared)][1]:
                    continue
                s = float(np.mean([iou(fc[f][0], fp[f][0]) for f in shared]))
                if s < min_iou:
                    # Hộp nhỏ (người đi bộ ở xa) có IoU giữa hai lượt gán nhãn
                    # độc lập rất thấp dù cùng một vật. Chấp nhận thêm khi tâm
                    # hai hộp lệch nhau dưới một nửa kích thước hộp.
                    dists = []
                    for f in shared:
                        a, b = fc[f][0], fp[f][0]
                        size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
                        dc = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
                        dists.append(dc / max(size, 1.0))
                    if float(np.mean(dists)) > 0.5:
                        continue
                    s = 0.5 * s + 0.5 * (1.0 - float(np.mean(dists)))
                pairs.append((s, kp, kc))
        pairs.sort(key=lambda p: -p[0])
        used_p: set = set()
        used_c: set = set()
        for _s, kp, kc in pairs:
            if kp in used_p or kc in used_c:
                continue
            used_p.add(kp)
            used_c.add(kc)
            global_id[kc] = global_id[kp]
            links += 1
        for kc in cur_keys:
            if kc not in global_id:
                global_id[kc] = next_gid
                next_gid += 1

    # --- Mỗi khung chỉ giữ nhãn của đoạn có chỉ số lớn nhất chứa nó -------- #
    owner: dict[int, int] = {}
    for key, frames in boxes.items():
        for f in frames:
            owner[f] = max(owner.get(f, -1), key[0])

    n_frames = (max(owner) + 1) if owner else 0
    if max_frames:
        n_frames = min(n_frames, max_frames)

    seq = MvtiSequence(name=name, root=Path(annotation_path).parent, frame_files=[],
                       width=width, height=height, n_frames_override=n_frames)
    per_frame: dict[int, list[tuple[int, tuple, VehicleClass]]] = defaultdict(list)
    for key, frames in boxes.items():
        gid = global_id[key]
        for f, (box, vcls) in frames.items():
            if f >= n_frames or owner[f] != key[0]:
                continue
            per_frame[f].append((gid, box, vcls))

    for f in sorted(per_frame):
        dets: list[Detection] = []
        ids: dict[int, int] = {}
        for gid, box, vcls in per_frame[f]:
            ids[len(dets)] = gid
            dets.append(Detection(bbox=box, score=1.0, cls=vcls))
            tr = seq.tracks.get(gid)
            if tr is None:
                tr = GtTrack(object_id=gid, cls=vcls)
                seq.tracks[gid] = tr
            tr.frames.append(f)
            tr.boxes.append(box)
        seq.detections[f] = dets
        seq.id_map[f] = ids

    stats = {
        "n_chunks": n_chunks,
        "n_local_tracks": len(boxes),
        "n_links": links,
        "n_global_tracks": len(seq.tracks),
        "skipped_annotations": skipped,
        "min_iou": min_iou,
    }
    log.info("CVAT video %s: %d đoạn, %d track cục bộ → %d track toàn cục",
             name, n_chunks, len(boxes), len(seq.tracks))
    return seq, stats
