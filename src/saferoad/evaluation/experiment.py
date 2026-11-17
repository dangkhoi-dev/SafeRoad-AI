"""Khung chạy thực nghiệm trên mô phỏng: một hàm cho mọi lần đo.

Mọi bảng trong báo cáo (kết quả chính, ablation, quét nhiễu, lưới mật độ/góc
camera/điều kiện ánh sáng) đều đi qua :func:`run_case`, nên chúng dùng **cùng**
một cách đếm sự kiện, cùng vùng đánh giá và cùng hàm chấm. Không còn chuyện mỗi
bảng tự đếm một kiểu.

Mỗi lần chạy trả về một bản ghi có đủ: cấu hình đã dùng (kèm mã băm), seed,
bảng đối soát, chỉ số kèm mẫu số. Ghi các bản ghi này ra JSON là có ngay "log
cấu hình tái hiện" cho hội đồng.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import Config
from ..detection.detector import ReplayDetector
from ..pipeline.runner import Pipeline, PipelineResult
from ..types import ConflictEvent, GroundTruthConflict, Track, TrackState
from .metrics import build_gt_points, build_id_mapping
from .scoring import (
    by_kind,
    contact_confusion,
    pet_error_table,
    reconciliation,
    score_events,
    summary_metrics,
    ttc_bands,
)


# --------------------------------------------------------------------------- #
@dataclass
class SimCase:
    """Một bộ dữ liệu mô phỏng đã sẵn sàng để chạy pipeline."""

    vehicles: list
    ground_truth: list[GroundTruthConflict]
    detections: dict
    id_map: dict
    homography: Any                 # HomographyConfig khớp camera
    fps: float
    duration: float
    frame_size: tuple[int, int]
    region: tuple[float, float, float, float]
    scenario_report: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    camera: Any = None              # PinholeCamera để lọc vùng nhìn thấy

    def visible(self, ev: ConflictEvent) -> bool:
        """Điểm xung đột có nằm trong khung hình (không chỉ trong vùng chữ nhật)."""
        if self.camera is None:
            return True
        u, v = self.camera.project_ground(np.array([ev.location]))[0]
        w, h = self.frame_size
        return 0 <= u <= w and 0.05 * h <= v <= h


def build_case(
    duration: float = 180.0,
    seed: int = 42,
    arrival_rate: float = 1.1,
    camera_name: str = "cao_18m",
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> SimCase:
    """Sinh kịch bản → mô phỏng → nhãn chuẩn → bbox (không vẽ video)."""
    from ..simulation.camera import COVERAGE
    from ..simulation.render import project_detections
    from ..simulation.scenario import build_scenario, compute_ground_truth, simulate

    cam = make_camera(camera_name, width, height)
    report: dict = {}
    vehicles = simulate(
        build_scenario(duration=duration, seed=seed, arrival_rate=arrival_rate),
        duration, report=report,
    )
    gt = compute_ground_truth(vehicles, region=COVERAGE)
    dets, ids, renderer = project_detections(vehicles, duration, fps, width, height, camera=cam)
    case = SimCase(
        vehicles=vehicles, ground_truth=gt, detections=dets, id_map=ids,
        homography=renderer.camera, fps=float(fps), duration=duration,
        frame_size=(width, height), region=COVERAGE, scenario_report=report,
        params={"duration": duration, "seed": seed, "arrival_rate": arrival_rate,
                "camera": camera_name, "fps": fps},
        camera=cam,
    )
    # Nhãn nằm ngoài khung hình của camera này không được tính vào mẫu số.
    case.ground_truth = [g for g in gt if _gt_visible(case, g)]
    case.scenario_report["gt_outside_camera_view"] = len(gt) - len(case.ground_truth)
    return case


def _gt_visible(case: SimCase, g: GroundTruthConflict) -> bool:
    by_id = {v.vid: v for v in case.vehicles}
    va, vb = by_id.get(g.track_a), by_id.get(g.track_b)
    if va is None or vb is None:
        return False
    ia = int(np.argmin(np.abs(va.times - g.t)))
    ib = int(np.argmin(np.abs(vb.times - g.t)))
    mid = 0.5 * (va.positions[ia] + vb.positions[ib])
    ev = ConflictEvent.__new__(ConflictEvent)
    ev.location = (float(mid[0]), float(mid[1]))
    return case.visible(ev)


def case_from_bundle(bundle: dict) -> SimCase:
    """Dựng ``SimCase`` từ bundle do ``saferoad simulate`` ghi ra."""
    from ..simulation.camera import COVERAGE

    return SimCase(
        vehicles=bundle["vehicles"], ground_truth=bundle["ground_truth"],
        detections=bundle["detections"], id_map=bundle["id_map"],
        homography=bundle["camera"], fps=float(bundle["fps"]),
        duration=float(bundle["duration"]),
        frame_size=tuple(bundle.get("frame_size", (1280, 720))),
        region=COVERAGE, scenario_report=bundle.get("scenario_report", {}),
        params={"duration": bundle["duration"], "seed": bundle["seed"],
                "arrival_rate": bundle.get("arrival_rate"), "camera": "cao_18m",
                "fps": bundle["fps"]},
        camera=bundle.get("pinhole"),
    )


# --------------------------------------------------------------------------- #
def make_camera(name: str, width: int = 1280, height: int = 720):
    """Ba vị trí camera dùng trong lưới thực nghiệm.

    * ``cao_18m`` - mặc định: cao 18 m, lùi 35 m, nhìn dọc trục Bắc-Nam.
    * ``thap_10m`` - cột thấp: cao 10 m, lùi 26 m - góc nhìn thoải, xe che nhau
      nhiều hơn, sai số chiếu điểm tiếp đất lớn hơn.
    * ``xien_45`` - đặt ở góc giao lộ, nhìn chéo 45° - mọi hướng đi đều chéo
      với trục ảnh.
    """
    from ..simulation.camera import PinholeCamera, build_default_camera
    from ..simulation.scenario import CENTER

    cx, cy = CENTER
    if name == "cao_18m":
        return build_default_camera(width, height, centre=CENTER)
    if name == "thap_10m":
        return build_default_camera(width, height, centre=CENTER,
                                    camera_height=10.0, setback=26.0, fov_deg=62.0)
    if name == "xien_45":
        d = 32.0 / math.sqrt(2.0)
        return PinholeCamera(width, height, position=(cx + d, cy - d, 18.0),
                             look_at=(cx, cy, 0.0), fov_deg=58.0)
    raise ValueError(f"Camera không hợp lệ: {name}")


#: Hồ sơ chất lượng detector dùng làm **đại diện** cho điều kiện ánh sáng. Trình
#: mô phỏng vẽ cảnh bằng khối hộp nên không mô phỏng được ánh sáng thật; thứ ánh
#: sáng thay đổi đối với pipeline là chất lượng bbox (rung, bỏ sót). Các mức dưới
#: đây là giả định, ghi rõ trong báo cáo, KHÔNG phải số đo trên video ban đêm.
LIGHT_PROFILES: dict[str, tuple[float, float]] = {
    "ngay": (1.5, 0.03),
    "chang_vang": (3.0, 0.10),
    "dem_mua": (5.0, 0.25),
}


# --------------------------------------------------------------------------- #
def run_case(
    cfg: Config,
    case: SimCase,
    noise_px: float | None = None,
    miss_rate: float | None = None,
    seed: int | None = None,
    mutate: Callable[[Config], Any] | None = None,
    with_behavior: bool = False,
) -> dict[str, Any]:
    """Chạy pipeline (không đọc video) trên một ``SimCase`` và chấm điểm đầy đủ."""
    cfg = copy.deepcopy(cfg)
    cfg.video.write_overlay = False
    cfg.homography.image_points = case.homography.image_points
    cfg.homography.world_points = case.homography.world_points
    if mutate is not None:
        mutate(cfg)
    noise = cfg.detection.replay_noise_px if noise_px is None else noise_px
    miss = cfg.detection.replay_miss_rate if miss_rate is None else miss_rate
    rseed = cfg.detection.replay_seed if seed is None else seed

    detector = ReplayDetector(case.detections, noise_px=noise, miss_rate=miss, seed=rseed)
    n_frames = int(round(case.duration * case.fps))
    pipe = Pipeline(cfg, detector=detector, write_db=False,
                    frameless=(n_frames, case.fps, *case.frame_size))
    t0 = time.perf_counter()
    result = pipe.run(progress_every=0)
    wall = time.perf_counter() - t0

    gt_points = build_gt_points(case.detections, case.id_map, case.vehicles, case.fps,
                                min_box_area=cfg.detection.min_box_area)
    mapping, tracking = build_id_mapping(result.tracks, gt_points)
    detail = score_events(result.events, case.ground_truth, mapping,
                          region=case.region, visible=case.visible)

    rec = {
        "metrics": summary_metrics(detail),
        "reconciliation": reconciliation(detail, result.funnel, len(result.events),
                                         case.duration),
        "by_kind": by_kind(detail, case.ground_truth),
        "ttc_bands": ttc_bands(detail, case.ground_truth),
        "pet_errors": pet_error_table(detail, cfg.conflict.pet_threshold),
        "contact_confusion": contact_confusion(detail),
        "tracking": tracking.to_dict(),
        "runtime": {
            "processing_fps": round(result.processing_fps, 1),
            "wall_seconds": round(wall, 1),
            "frame_latency_median_ms": round(result.frame_latency_median_ms, 2),
        },
        "detector": {"noise_px": noise, "miss_rate": miss, "seed": rseed},
        "config_sha256": config_hash(cfg),
        "case": case.params,
    }
    if with_behavior:
        rec["behavior"] = behavior_report(cfg, result, case, mapping)
    rec["_result"] = result
    rec["_detail"] = detail
    return rec


def strip_private(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
def behavior_report(cfg: Config, result: PipelineResult, case: SimCase,
                    mapping: dict[int, int]) -> dict[str, Any]:
    """Cờ hành vi: tỉ lệ gắn cờ và độ khớp với nhãn tính từ quỹ đạo giải tích.

    Nhãn chuẩn cho mỗi xe mô phỏng được tính bằng **cùng bộ luật** nhưng trên
    quỹ đạo giải tích 60 Hz (không nhiễu bbox, không sai số theo vết), kèm thông
    tin kịch bản (xe được lập trình phanh gấp chủ động). Chênh lệch giữa hai bên
    là phần do nhiễu đo gây ra - đúng thứ cần biết khi bản cũ gắn cờ 305/307.
    """
    from ..behavior.classifier import BehaviorClassifier

    clf = BehaviorClassifier(cfg.behavior)
    clf.model = None                       # nhãn chuẩn: chỉ luật, không ML
    by_id = {v.vid: v for v in case.vehicles}
    truth: dict[int, str] = {}
    for vid, veh in by_id.items():
        tr = Track(track_id=vid, cls=veh.cls, confirmed=True)
        step = max(1, int(round(len(veh.times) / max(veh.times[-1] - veh.times[0], 1e-6)
                                / case.fps)))
        for i in range(0, len(veh.times), step):
            tr.history.append(TrackState(
                frame_idx=i, t=float(veh.times[i]), bbox=(0, 0, 1, 1), anchor=(0, 0),
                ground=(float(veh.positions[i, 0]), float(veh.positions[i, 1])),
                velocity=(float(veh.velocities[i, 0]), float(veh.velocities[i, 1])),
            ))
        truth[vid] = clf.classify(tr).label

    preds = result.behaviors
    n_tracks = len(preds)
    flagged = sum(1 for b in preds.values() if b.is_risky)
    tp = fp = fn = tn = 0
    exact = 0
    scored = 0
    for tid, b in preds.items():
        vid = mapping.get(tid)
        if vid is None or vid not in truth:
            continue
        scored += 1
        t_risky = truth[vid] != "binh_thuong"
        if b.is_risky and t_risky:
            tp += 1
        elif b.is_risky:
            fp += 1
        elif t_risky:
            fn += 1
        else:
            tn += 1
        exact += int(b.label == truth[vid])
    forced = [v.vid for v in case.vehicles if v.forced_brake]
    forced_hit = sum(
        1 for tid, b in preds.items()
        if mapping.get(tid) in forced and b.label == "phanh_gap"
    )
    forced_tracks = sum(1 for tid in preds if mapping.get(tid) in forced)

    def r(a, b):
        return {"value": round(a / b, 4) if b else None, "num": a, "den": b}

    return {
        "flagged_tracks": r(flagged, n_tracks),
        "truth_flagged_vehicles": r(sum(1 for v in truth.values() if v != "binh_thuong"),
                                    len(truth)),
        "risky_precision": r(tp, tp + fp),
        "risky_recall": r(tp, tp + fn),
        "label_agreement": r(exact, scored),
        "forced_brake_detected": r(forced_hit, forced_tracks),
        "ml_layer": "on" if clf.model is not None else "off",
    }


# --------------------------------------------------------------------------- #
def config_hash(cfg: Config) -> str:
    raw = json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def file_hash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).resolve().parents[3])
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5,
                               cwd=Path(__file__).resolve().parents[3])
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None
    return None


def environment() -> dict[str, str]:
    import cv2
    import numpy

    env = {
        "python": platform.python_version(),
        "os": platform.system(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import sklearn

        env["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return env


def reproducibility(cfg: Config, extra: dict | None = None) -> dict[str, Any]:
    """Mọi thứ cần để chạy lại đúng con số: lệnh, cấu hình đầy đủ, mã băm, phiên bản."""
    return {
        "command": " ".join(["saferoad", *sys.argv[1:]]) if sys.argv else "",
        "git_commit": git_commit(),
        "config_sha256": config_hash(cfg),
        "config": cfg.to_dict(),
        "environment": environment(),
        **(extra or {}),
    }
