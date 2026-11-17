"""Hiệu chuẩn camera thật: tính homography từ điểm mốc và **tự kiểm sai số**.

Một homography tính từ đúng 4 điểm luôn khớp hoàn hảo 4 điểm đó (sai số 0) - con
số "sai số tái chiếu" khi ấy không nói gì về chất lượng hiệu chuẩn. Vì vậy công
cụ này yêu cầu **nhiều hơn 4 điểm** và báo cáo ba loại sai số khác nhau:

* ``fit`` - sai số trên chính các điểm dùng để tính (lạc quan nhất);
* ``leave_one_out`` - bỏ lần lượt từng điểm (hoặc từng nhóm điểm), tính lại
  homography từ phần còn lại rồi đo sai số ở điểm bị bỏ ra. Đây là ước lượng
  trung thực cho sai số ở một điểm *bất kỳ* trên mặt đường;
* ``check`` - sai số trên các điểm kiểm tra độc lập (nếu có), không tham gia
  tính toán.

Tệp hiệu chuẩn (YAML)::

    source: "Mô tả nguồn đo: đo thực địa / ước lượng từ vạch kẻ..."
    assumptions: ["bề rộng làn 3,5 m", ...]
    frame_size: [1920, 1080]
    image_points: [[u, v], ...]          # ≥ 4, nên ≥ 6
    world_points: [[x, y], ...]          # mét, cùng thứ tự
    groups: [0, 0, 1, 1, ...]            # tuỳ chọn: bỏ cả nhóm khi leave-one-out
    check_image_points: [[u, v], ...]    # tuỳ chọn
    check_world_points: [[x, y], ...]

Đầu ra: khối ``homography`` dán thẳng vào file cấu hình pipeline, kèm báo cáo
sai số (JSON) và ảnh lưới mét vẽ chồng lên khung hình để kiểm bằng mắt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class CalibrationInput:
    image_points: np.ndarray
    world_points: np.ndarray
    groups: list[int] | None = None
    check_image_points: np.ndarray | None = None
    check_world_points: np.ndarray | None = None
    source: str = ""
    assumptions: list[str] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationInput":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        ip = np.asarray(raw["image_points"], dtype=float)
        wp = np.asarray(raw["world_points"], dtype=float)
        if ip.shape != wp.shape or ip.shape[0] < 4:
            raise ValueError("Cần ít nhất 4 cặp điểm, image_points và world_points cùng số lượng")
        cip = raw.get("check_image_points")
        cwp = raw.get("check_world_points")
        return cls(
            image_points=ip, world_points=wp, groups=raw.get("groups"),
            check_image_points=np.asarray(cip, dtype=float) if cip else None,
            check_world_points=np.asarray(cwp, dtype=float) if cwp else None,
            source=str(raw.get("source", "")),
            assumptions=list(raw.get("assumptions", [])),
            frame_size=tuple(raw["frame_size"]) if raw.get("frame_size") else None,
        )


def fit_homography(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """Homography ảnh → mặt đất bằng bình phương tối thiểu trên mọi điểm."""
    H, _ = cv2.findHomography(image_points.astype(np.float64),
                              world_points.astype(np.float64), 0)
    if H is None:
        raise ValueError("Không tính được homography - các điểm có thể thẳng hàng")
    return H / H[2, 2]


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def _stats(err: np.ndarray) -> dict[str, Any]:
    if err.size == 0:
        return {"n": 0}
    return {
        "n": int(err.size),
        "rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mean_m": round(float(err.mean()), 3),
        "max_m": round(float(err.max()), 3),
    }


def evaluate_calibration(inp: CalibrationInput) -> tuple[np.ndarray, dict[str, Any]]:
    """Tính homography và ba loại sai số (m trên mặt đất)."""
    H = fit_homography(inp.image_points, inp.world_points)
    fit_err = np.linalg.norm(_project(H, inp.image_points) - inp.world_points, axis=1)

    groups = inp.groups if inp.groups else list(range(len(inp.image_points)))
    loo: list[float] = []
    for gid in sorted(set(groups)):
        train = [i for i, g in enumerate(groups) if g != gid]
        test = [i for i, g in enumerate(groups) if g == gid]
        if len(train) < 4:
            continue
        try:
            Hg = fit_homography(inp.image_points[train], inp.world_points[train])
        except ValueError:
            continue
        loo.extend(np.linalg.norm(
            _project(Hg, inp.image_points[test]) - inp.world_points[test], axis=1).tolist())

    report: dict[str, Any] = {
        "n_points": int(len(inp.image_points)),
        "fit": _stats(fit_err),
        "leave_one_out": _stats(np.asarray(loo)),
        "source": inp.source,
        "assumptions": inp.assumptions,
    }
    if inp.check_image_points is not None and inp.check_world_points is not None:
        chk = np.linalg.norm(_project(H, inp.check_image_points) - inp.check_world_points, axis=1)
        report["check"] = _stats(chk)
    if len(inp.image_points) == 4:
        report["warning"] = ("Chỉ 4 điểm: homography khớp tuyệt đối 4 điểm này, sai số "
                             "fit = 0 không có ý nghĩa. Cần thêm điểm hoặc điểm kiểm tra.")
    return H, report


def homography_block(inp: CalibrationInput) -> dict[str, Any]:
    """Khối ``homography`` cho file cấu hình pipeline (dùng toàn bộ điểm)."""
    return {
        "image_points": inp.image_points.round(2).tolist(),
        "world_points": inp.world_points.round(4).tolist(),
    }


def draw_metric_grid(frame: np.ndarray, H: np.ndarray, x_range: tuple[float, float],
                     y_range: tuple[float, float], step_x: float, step_y: float,
                     points: np.ndarray | None = None) -> np.ndarray:
    """Vẽ lưới mét (theo homography) lên khung hình để kiểm tra bằng mắt."""
    out = frame.copy()
    h, w = out.shape[:2]
    Hi = np.linalg.inv(H)

    def poly(pts_w: np.ndarray, colour) -> None:
        p = _project(Hi, pts_w)
        ok = (p[:, 0] > -50) & (p[:, 0] < w + 50) & (p[:, 1] > 0) & (p[:, 1] < h + 50)
        p = p[ok]
        if len(p) > 1:
            cv2.polylines(out, [p.astype(np.int32)], False, colour, 2, cv2.LINE_AA)

    ys = np.arange(y_range[0], y_range[1] + 1e-6, 0.5)
    for x in np.arange(x_range[0], x_range[1] + 1e-6, step_x):
        poly(np.column_stack([np.full_like(ys, x), ys]), (0, 220, 255))
    xs = np.arange(x_range[0], x_range[1] + 1e-6, 0.25)
    for y in np.arange(y_range[0], y_range[1] + 1e-6, step_y):
        poly(np.column_stack([xs, np.full_like(xs, y)]), (255, 0, 255))
    if points is not None:
        for u, v in points:
            cv2.circle(out, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def run_calibration(calib_path: str | Path, frame_path: str | Path | None,
                    out_json: str | Path, out_image: str | Path | None = None,
                    grid: dict[str, Any] | None = None) -> dict[str, Any]:
    inp = CalibrationInput.load(calib_path)
    H, report = evaluate_calibration(inp)
    report["homography"] = homography_block(inp)
    report["H_image_to_ground"] = H.round(8).tolist()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if frame_path and out_image:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            cap = cv2.VideoCapture(str(frame_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise FileNotFoundError(f"Không đọc được khung hình từ {frame_path}")
        g = grid or {}
        wp = inp.world_points
        img = draw_metric_grid(
            frame, H,
            tuple(g.get("x_range", (float(wp[:, 0].min()) - 7, float(wp[:, 0].max()) + 7))),
            tuple(g.get("y_range", (float(wp[:, 1].min()) - 10, float(wp[:, 1].max()) + 30))),
            float(g.get("step_x", 3.5)), float(g.get("step_y", 6.0)),
            points=inp.image_points,
        )
        Path(out_image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image), img)
    return report


def speed_check(seq, homography_cfg, fps: float, min_anchor_y_frac: float = 0.6,
                min_frames: int = 60, min_speed: float = 2.0) -> dict[str, Any]:
    """Kiểm tra độc lập tỉ lệ hiệu chuẩn bằng vận tốc ô tô trên quỹ đạo người gán nhãn.

    Chỉ dùng phần quỹ đạo ở nửa dưới khung hình (gần camera, sai số chiếu nhỏ)
    và hồi quy tuyến tính vị trí theo thời gian. Nếu tỉ lệ sai lệch nhiều, phân
    bố vận tốc sẽ lệch khỏi giới hạn tốc độ của con đường một cách dễ thấy.
    """
    from ..geometry.homography import GroundPlane
    from ..types import VehicleClass

    gp = GroundPlane(homography_cfg, (seq.width, seq.height))
    speeds = []
    for tr in seq.tracks.values():
        if tr.cls != VehicleClass.CAR or len(tr.frames) < min_frames:
            continue
        boxes = np.asarray(tr.boxes, dtype=float)
        sel = boxes[:, 3] > min_anchor_y_frac * seq.height
        if sel.sum() < min_frames:
            continue
        anc = np.column_stack([(boxes[sel, 0] + boxes[sel, 2]) / 2, boxes[sel, 3]]).astype(np.float32)
        g = gp.to_ground_batch(anc)
        t = np.asarray(tr.frames, dtype=float)[sel] / fps
        vx = np.polyfit(t, g[:, 0], 1)[0]
        vy = np.polyfit(t, g[:, 1], 1)[0]
        sp = float(np.hypot(vx, vy))
        if sp > min_speed:
            speeds.append(sp * 3.6)
    if not speeds:
        return {"n_cars": 0}
    a = np.asarray(speeds)
    return {"n_cars": int(a.size), "median_kmh": round(float(np.median(a)), 1),
            "p25_kmh": round(float(np.percentile(a, 25)), 1),
            "p75_kmh": round(float(np.percentile(a, 75)), 1),
            "p90_kmh": round(float(np.percentile(a, 90)), 1)}
