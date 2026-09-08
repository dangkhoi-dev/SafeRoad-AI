"""Dựng video từ kịch bản mô phỏng, kèm bbox chuẩn cho từng frame.

Cảnh được vẽ qua một camera pinhole thật (:mod:`saferoad.simulation.camera`),
phương tiện là khối hộp 3D chứ không phải hình chiếu bóng xuống đất. Nhờ vậy
bounding box phản ánh đúng thứ camera thật nhìn thấy — đặc biệt quan trọng với
người đi bộ, vốn chỉ chiếm 0.5 × 0.5 m mặt đất nhưng cao 1.7 m.

Homography mà pipeline dùng được **suy ra từ chính ma trận camera** này, nên
toàn bộ chuỗi *mặt đất → ảnh → mặt đất* khép kín và ta cô lập được sai số của
từng khối:

* ``ReplayDetector`` với ``noise_px = 0`` ⇒ đo sai số thuần của tracking + TTC/PET;
* tăng dần ``noise_px`` ⇒ đo độ nhạy của hệ thống với chất lượng detector.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..config import HomographyConfig
from ..types import Detection, VehicleClass
from .camera import PinholeCamera, build_default_camera
from .scenario import CENTER, ROAD_HALF_WIDTH, WORLD_BOUNDS, VehicleSpec

#: Màu thân xe (BGR).
CLASS_COLOR: dict[VehicleClass, tuple[int, int, int]] = {
    VehicleClass.MOTORCYCLE: (72, 172, 244),   # cam
    VehicleClass.CAR: (214, 158, 74),          # xanh dương
    VehicleClass.TRUCK: (104, 122, 214),       # đỏ gạch
    VehicleClass.BICYCLE: (120, 206, 140),     # xanh lá
    VehicleClass.PEDESTRIAN: (196, 138, 226),  # tím
}

#: Bốn điểm mốc trên mặt đất dùng làm cặp calibration cho pipeline.
CALIBRATION_POINTS = [
    (-12.0, 16.0), (12.0, 16.0), (12.0, 44.0), (-12.0, 44.0)
]


def camera_to_homography(camera: PinholeCamera) -> HomographyConfig:
    """Sinh ``HomographyConfig`` khớp chính xác với camera đã dựng cảnh."""
    image_points, world_points = camera.ground_homography_points(CALIBRATION_POINTS)
    return HomographyConfig(image_points=image_points, world_points=world_points)


class SceneRenderer:
    """Vẽ kịch bản ra khung hình và sinh bbox chuẩn."""

    def __init__(self, frame_w: int = 1280, frame_h: int = 720,
                 camera: PinholeCamera | None = None):
        self.w = frame_w
        self.h = frame_h
        self.cam = camera or build_default_camera(frame_w, frame_h, centre=CENTER)
        self.camera = camera_to_homography(self.cam)
        self._background = self._draw_background()

    # ------------------------------------------------------------------ #
    def _draw_background(self) -> np.ndarray:
        """Vẽ nền tĩnh: mặt đường, vạch kẻ, vạch dừng — chỉ tính một lần."""
        img = np.full((self.h, self.w, 3), (58, 78, 62), dtype=np.uint8)  # nền cỏ
        x_min, y_min, x_max, y_max = WORLD_BOUNDS
        cx, cy = CENTER

        def fill(world_pts, color):
            pts = self.cam.project_ground(np.asarray(world_pts, dtype=float))
            cv2.fillPoly(img, [pts.astype(np.int32)], color)

        road = (76, 80, 84)
        fill([(cx - ROAD_HALF_WIDTH, y_min), (cx + ROAD_HALF_WIDTH, y_min),
              (cx + ROAD_HALF_WIDTH, y_max), (cx - ROAD_HALF_WIDTH, y_max)], road)
        fill([(x_min, cy - ROAD_HALF_WIDTH), (x_max, cy - ROAD_HALF_WIDTH),
              (x_max, cy + ROAD_HALF_WIDTH), (x_min, cy + ROAD_HALF_WIDTH)], road)

        # Vạch tim đường (nét đứt), bỏ qua vùng giao lộ.
        def dashes(start, end, skip_axis):
            n = 44
            for i in range(n):
                u0, u1 = i / n, (i + 0.5) / n
                p0 = (start[0] + (end[0] - start[0]) * u0,
                      start[1] + (end[1] - start[1]) * u0)
                p1 = (start[0] + (end[0] - start[0]) * u1,
                      start[1] + (end[1] - start[1]) * u1)
                if skip_axis == "v" and abs(p0[1] - cy) < ROAD_HALF_WIDTH + 1.5:
                    continue
                if skip_axis == "h" and abs(p0[0] - cx) < ROAD_HALF_WIDTH + 1.5:
                    continue
                a, b = self.cam.project_ground(np.array([p0, p1]))
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                         (205, 205, 205), 2, cv2.LINE_AA)

        dashes((cx, y_min), (cx, y_max), "v")
        dashes((x_min, cy), (x_max, cy), "h")

        # Vạch dừng ở 4 nhánh.
        stop = ROAD_HALF_WIDTH + 1.5
        for a, b in [
            ((cx - ROAD_HALF_WIDTH, cy + stop), (cx, cy + stop)),
            ((cx, cy - stop), (cx + ROAD_HALF_WIDTH, cy - stop)),
            ((cx + stop, cy + ROAD_HALF_WIDTH), (cx + stop, cy)),
            ((cx - stop, cy - ROAD_HALF_WIDTH), (cx - stop, cy)),
        ]:
            pa, pb = self.cam.project_ground(np.array([a, b]))
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     (238, 238, 238), 4, cv2.LINE_AA)

        # Vạch bộ hành (kẻ sọc ngựa vằn) — chỉ trải hết bề rộng mặt đường.
        span = 2 * ROAD_HALF_WIDTH + 1.0
        n_bands = 9
        pitch = span / n_bands
        for base, along, across in [
            ((-ROAD_HALF_WIDTH - 0.5, 41.0), (1.0, 0.0), (0.0, 1.0)),
            ((-ROAD_HALF_WIDTH - 0.5, 19.0), (1.0, 0.0), (0.0, 1.0)),
            ((11.0, CENTER[1] - ROAD_HALF_WIDTH - 0.5), (0.0, 1.0), (1.0, 0.0)),
            ((-11.0, CENTER[1] - ROAD_HALF_WIDTH - 0.5), (0.0, 1.0), (1.0, 0.0)),
        ]:
            for k in range(n_bands):
                s0 = k * pitch
                s1 = s0 + pitch * 0.55
                quad = [
                    (base[0] + along[0] * s0 - across[0] * 1.2,
                     base[1] + along[1] * s0 - across[1] * 1.2),
                    (base[0] + along[0] * s1 - across[0] * 1.2,
                     base[1] + along[1] * s1 - across[1] * 1.2),
                    (base[0] + along[0] * s1 + across[0] * 1.2,
                     base[1] + along[1] * s1 + across[1] * 1.2),
                    (base[0] + along[0] * s0 + across[0] * 1.2,
                     base[1] + along[1] * s0 + across[1] * 1.2),
                ]
                fill(quad, (222, 222, 222))

        return img

    # ------------------------------------------------------------------ #
    def _draw_vehicle(self, img: np.ndarray, veh: VehicleSpec,
                      pos: tuple[float, float], heading: float) -> np.ndarray | None:
        """Vẽ một phương tiện dạng khối hộp 3D; trả về bbox ảnh (x1,y1,x2,y2)."""
        length, width = veh.cls.footprint
        corners3d = self.cam.cuboid_corners(pos, heading, length, width, veh.cls.height)
        pts = self.cam.project(corners3d)
        if not np.isfinite(pts).all() or pts.min() < -5e4:
            return None

        # Thứ tự đỉnh: (đáy, nóc) cho 4 góc theo chiều kim đồng hồ.
        bottom = pts[0::2].astype(np.int32)
        top = pts[1::2].astype(np.int32)
        colour = CLASS_COLOR[veh.cls]
        dark = tuple(int(c * 0.62) for c in colour)

        # Mặt bên (nối đáy với nóc) — vẽ trước để nóc đè lên.
        for i in range(4):
            j = (i + 1) % 4
            side = np.array([bottom[i], bottom[j], top[j], top[i]], dtype=np.int32)
            cv2.fillConvexPoly(img, side, dark)
        cv2.fillConvexPoly(img, top, colour)
        cv2.polylines(img, [top], True, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.polylines(img, [bottom], True, (35, 35, 35), 1, cv2.LINE_AA)

        # Vạch trắng chỉ hướng đầu xe.
        front = ((top[0] + top[1]) / 2).astype(int)
        centre = top.mean(axis=0).astype(int)
        cv2.line(img, tuple(centre), tuple(front), (250, 250, 250), 1, cv2.LINE_AA)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return np.array([x1, y1, x2, y2], dtype=float)

    def render_frame(
        self, vehicles: list[VehicleSpec], t: float
    ) -> tuple[np.ndarray, list[tuple[int, Detection]]]:
        """Vẽ một khung hình, trả về ``(ảnh, [(vehicle_id, Detection), ...])``."""
        img = self._background.copy()
        out: list[tuple[int, Detection]] = []

        drawable = []
        for veh in vehicles:
            pose = veh.pose_at(t)
            if pose is None:
                continue
            pos, vel = pose
            heading = math.atan2(vel[1], vel[0]) if math.hypot(*vel) > 1e-6 else 0.0
            drawable.append((veh, pos, heading))
        # Vẽ xe ở xa trước để xe gần đè lên (thứ tự theo chiều sâu).
        drawable.sort(key=lambda item: -item[1][1])

        for veh, pos, heading in drawable:
            bbox = self._draw_vehicle(img, veh, pos, heading)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x2 < 0 or y2 < 0 or x1 > self.w or y1 > self.h:
                continue
            out.append((
                veh.vid,
                Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                          score=1.0, cls=veh.cls),
            ))

        return img, out


def render_video(
    vehicles: list[VehicleSpec],
    duration: float,
    out_path: str | Path,
    fps: int = 30,
    frame_w: int = 1280,
    frame_h: int = 720,
    progress: bool = True,
) -> tuple[dict[int, list[Detection]], dict[int, dict[int, int]], SceneRenderer]:
    """Xuất video mô phỏng và trả về bbox chuẩn theo frame.

    Trả về ``(detections_by_frame, id_map_by_frame, renderer)``, trong đó
    ``id_map_by_frame[frame][chỉ_số_trong_danh_sách] = vehicle_id`` để đối chiếu
    track do hệ thống sinh ra với xe thật khi chấm điểm.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = SceneRenderer(frame_w, frame_h)
    n_frames = int(duration * fps)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho {out_path}")

    dets_by_frame: dict[int, list[Detection]] = {}
    ids_by_frame: dict[int, dict[int, int]] = {}

    for idx in range(n_frames):
        t = idx / fps
        img, pairs = renderer.render_frame(vehicles, t)

        cv2.putText(img, f"SIM  t={t:6.2f}s  frame={idx:05d}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 238, 238), 1, cv2.LINE_AA)
        writer.write(img)

        dets_by_frame[idx] = [d for _vid, d in pairs]
        ids_by_frame[idx] = {i: vid for i, (vid, _d) in enumerate(pairs)}

        if progress and idx % (fps * 10) == 0:
            print(f"  render {idx}/{n_frames} frames ({t:.0f}s)", flush=True)

    writer.release()
    return dets_by_frame, ids_by_frame, renderer
