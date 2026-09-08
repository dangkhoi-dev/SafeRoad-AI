"""Pipeline end-to-end: video → detection → tracking → trajectory → conflict → risk.

Đây là khối "AI Processing" trong sơ đồ kiến trúc, ghép toàn bộ các module con
thành một luồng chạy liền mạch và đo thời gian từng bước để báo cáo latency.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..behavior.classifier import BehaviorClassifier, BehaviorResult
from ..config import Config
from ..conflict.detector import ConflictDetector
from ..detection.detector import BaseDetector, build_detector
from ..geometry.homography import GroundPlane, make_default_homography
from ..geometry.trajectory import TrajectoryProcessor
from ..privacy.anonymize import Anonymizer
from ..risk.riskmap import RiskMap
from ..risk.scoring import RiskScorer
from ..storage.db import EventStore
from ..tracking.bytetrack import ByteTracker
from ..types import ConflictEvent, Track
from .overlay import OverlayRenderer

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Kết quả sau khi chạy hết một video."""

    events: list[ConflictEvent] = field(default_factory=list)
    tracks: dict[int, Track] = field(default_factory=dict)
    behaviors: dict[int, BehaviorResult] = field(default_factory=dict)
    n_frames: int = 0
    fps_source: float = 30.0
    duration: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)
    processing_fps: float = 0.0
    risk_map: RiskMap | None = None
    session_id: int | None = None
    overlay_path: str | None = None
    homography_error: float | None = None

    @property
    def severe_events(self) -> list[ConflictEvent]:
        return [e for e in self.events if e.ttc is not None and e.ttc < 1.5]

    @property
    def avg_risk(self) -> float:
        if not self.events:
            return 0.0
        return round(sum(e.risk_score for e in self.events) / len(self.events), 1)

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        for e in self.events:
            by_type[e.conflict_type.value] = by_type.get(e.conflict_type.value, 0) + 1
        return {
            "n_frames": self.n_frames,
            "duration_s": round(self.duration, 2),
            "fps_source": round(self.fps_source, 2),
            "processing_fps": round(self.processing_fps, 2),
            "total_events": len(self.events),
            "severe_events": len(self.severe_events),
            "avg_risk_score": self.avg_risk,
            "n_tracks": len(self.tracks),
            "by_type": by_type,
            "latency_ms": {k: round(v, 2) for k, v in self.latency.items()},
            "total_latency_ms": round(sum(self.latency.values()), 2),
            "homography_error_m": self.homography_error,
        }


class Pipeline:
    """Bộ điều phối toàn bộ các khối xử lý.

    Cách dùng::

        pipe = Pipeline(cfg)
        result = pipe.run()
    """

    def __init__(
        self,
        cfg: Config,
        detector: BaseDetector | None = None,
        store: EventStore | None = None,
        write_db: bool = True,
    ):
        self.cfg = cfg
        self.detector = detector
        self.store = store
        self.write_db = write_db
        self._timings: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ #
    def _record(self, stage: str, dt: float) -> None:
        self._timings.setdefault(stage, []).append(dt * 1000.0)

    def run(self, progress_every: int = 150) -> PipelineResult:
        cfg = self.cfg
        video_path = cfg.video.source
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Không mở được video: {video_path}\n"
                f"Kiểm tra lại 'video.source' trong file cấu hình."
            )

        fps = cfg.video.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("Video %s: %dx%d @ %.2f FPS, %d frames", video_path, width, height, fps, n_total)

        # --- Khởi tạo các khối ------------------------------------------ #
        if self.detector is None:
            self.detector = build_detector(cfg.detection)

        homo_cfg = cfg.homography
        if not homo_cfg.image_points:
            log.info("Chưa có calibration — sinh homography xấp xỉ từ kích thước khung hình")
            homo_cfg = make_default_homography(
                width, height, camera_height_m=cfg.site.camera_height_m
            )
        ground = GroundPlane(homo_cfg, (width, height))
        traj = TrajectoryProcessor(ground, homo_cfg, cfg.tracking)
        tracker = ByteTracker(cfg.tracking)
        scorer = RiskScorer(cfg.risk)
        conflicts = ConflictDetector(cfg.conflict)
        behavior = BehaviorClassifier(cfg.behavior)
        anonymizer = Anonymizer(cfg.privacy)

        risk_map = RiskMap(cfg.risk, bounds=_ground_bounds(ground, width, height))

        overlay: OverlayRenderer | None = None
        writer: cv2.VideoWriter | None = None
        if cfg.video.write_overlay:
            Path(cfg.video.overlay_path).parent.mkdir(parents=True, exist_ok=True)
            overlay = OverlayRenderer(ground, cfg)
            writer = cv2.VideoWriter(
                cfg.video.overlay_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / max(1, cfg.video.stride),
                (width, height),
            )

        store = self.store
        session_id = None
        if self.write_db:
            if store is None:
                store = EventStore(cfg.db_path)
            session_id = store.start_session(
                source=video_path, site=cfg.site.name, fps=fps, config=cfg.to_dict()
            )

        # --- Vòng lặp chính --------------------------------------------- #
        all_events: list[ConflictEvent] = []
        frame_idx = 0
        processed = 0
        wall_start = time.perf_counter()

        if cfg.video.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.video.start_frame)
            frame_idx = cfg.video.start_frame

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if cfg.video.max_frames and processed >= cfg.video.max_frames:
                    break
                if cfg.video.stride > 1 and (frame_idx % cfg.video.stride):
                    frame_idx += 1
                    continue

                t = frame_idx / fps

                # 1. Detection
                t0 = time.perf_counter()
                detections = self.detector.detect(frame, frame_idx)
                # Lọc bbox quá nhỏ ở TẦNG NÀY (không chỉ trong YoloDetector) để
                # mọi detector đều chịu cùng một giới hạn. Đối tượng nhỏ hơn vài
                # trăm pixel vuông nằm quá xa camera: vị trí của chúng nhiễu tới
                # mức vận tốc ước lượng vô nghĩa, và TTC tính từ đó chỉ tạo báo
                # động giả.
                min_area = cfg.detection.min_box_area
                if min_area > 0:
                    detections = [d for d in detections if d.area >= min_area]
                self._record("detection", time.perf_counter() - t0)

                # 2. Ẩn danh (nếu bật) — làm ngay sau detection để mọi thứ ghi
                #    ra đĩa từ đây trở đi đều đã được ẩn danh.
                if cfg.privacy.enabled:
                    t0 = time.perf_counter()
                    frame = anonymizer.apply(frame, detections)
                    self._record("privacy", time.perf_counter() - t0)

                # 3. Tracking
                t0 = time.perf_counter()
                tracks = tracker.update(detections, frame_idx, t)
                self._record("tracking", time.perf_counter() - t0)

                # 4. Trajectory + homography
                t0 = time.perf_counter()
                for tr in tracks:
                    traj.process(tr)
                self._record("trajectory", time.perf_counter() - t0)

                # 5. Conflict + risk
                t0 = time.perf_counter()
                events = conflicts.update(tracks, frame_idx, t, risk_scorer=scorer)
                self._record("conflict", time.perf_counter() - t0)

                if events:
                    all_events.extend(events)
                    risk_map.extend(events)
                    if store is not None:
                        store.add_events(events, session_id)

                # 6. Overlay
                if overlay is not None and writer is not None:
                    t0 = time.perf_counter()
                    canvas = overlay.draw(
                        frame, tracks, events, conflicts, frame_idx, t, risk_map
                    )
                    writer.write(canvas)
                    self._record("overlay", time.perf_counter() - t0)

                if store is not None:
                    lat = sum(v[-1] for v in self._timings.values() if v)
                    store.add_frame_stat(frame_idx, t, len(tracks), len(events), lat)

                frame_idx += 1
                processed += 1
                if progress_every and processed % progress_every == 0:
                    elapsed = time.perf_counter() - wall_start
                    print(
                        f"  {processed} frames | {len(all_events)} near-miss "
                        f"| {processed / max(elapsed, 1e-6):.1f} FPS xử lý",
                        flush=True,
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # Đóng nốt các episode xung đột còn dở khi video kết thúc — nếu không,
        # mọi tình huống đang diễn ra ở những giây cuối sẽ bị mất.
        tail_events = conflicts.flush(risk_scorer=scorer)
        if tail_events:
            all_events.extend(tail_events)
            risk_map.extend(tail_events)
            if store is not None:
                store.add_events(tail_events, session_id)

        wall = time.perf_counter() - wall_start

        # --- Hành vi (tính một lần trên toàn bộ track) ------------------- #
        behaviors = behavior.classify_all(list(tracker.all_tracks.values()))

        if store is not None:
            store.finish_session(processed)

        result = PipelineResult(
            events=all_events,
            tracks=dict(tracker.all_tracks),
            behaviors=behaviors,
            n_frames=processed,
            fps_source=fps,
            duration=processed * max(1, cfg.video.stride) / fps,
            latency={k: float(np.mean(v)) for k, v in self._timings.items() if v},
            processing_fps=processed / max(wall, 1e-6),
            risk_map=risk_map,
            session_id=session_id,
            overlay_path=cfg.video.overlay_path if cfg.video.write_overlay else None,
            homography_error=ground.reprojection_error(),
        )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def export_json(result: PipelineResult, path: str | Path, cfg: Config | None = None) -> None:
        """Xuất kết quả ra JSON cho dashboard tĩnh và cho bước đánh giá."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": result.summary(),
            "site": cfg.site.name if cfg else "",
            "events": [e.to_dict() for e in result.events],
            "behaviors": [b.to_dict() for b in result.behaviors.values() if b.is_risky],
            "risk_map": result.risk_map.to_payload() if result.risk_map else None,
            "timeline": _timeline(result),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def _ground_bounds(
    ground: GroundPlane,
    width: int,
    height: int,
    max_extent: float = 120.0,
    skip_top: float = 0.30,
) -> tuple[float, float, float, float]:
    """Biên vùng mặt đất hợp lý cho Risk Map, suy từ khung hình camera.

    Hai điều chỉnh cần thiết
    ------------------------
    * **Bỏ phần trên khung hình** — các điểm ảnh gần đường chân trời chiếu xuống
      mặt đất ở khoảng cách gần như vô hạn (tia nhìn gần song song mặt đất). Lấy
      nguyên bốn góc ảnh sẽ cho ra một vùng rộng hàng trăm mét mà 99% là trống,
      khiến bản đồ nhiệt co lại thành một chấm nhỏ ở góc.
    * **Chặn kích thước tối đa** — thêm một lớp bảo vệ cho trường hợp camera đặt
      rất thoải hoặc homography chưa chuẩn.
    """
    ys = np.linspace(height * skip_top, height - 1, 6)
    xs = np.linspace(0, width - 1, 6)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    pts = ground.to_ground_batch(grid)

    # Bỏ các điểm chiếu ra quá xa (tia gần song song mặt đất).
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) == 0:
        return (-40.0, 0.0, 40.0, 80.0)

    cx, cy = float(np.median(finite[:, 0])), float(np.median(finite[:, 1]))
    keep = finite[
        (np.abs(finite[:, 0] - cx) <= max_extent)
        & (np.abs(finite[:, 1] - cy) <= max_extent)
    ]
    if len(keep) < 4:
        keep = finite

    pad = 4.0
    return (
        float(keep[:, 0].min()) - pad, float(keep[:, 1].min()) - pad,
        float(keep[:, 0].max()) + pad, float(keep[:, 1].max()) + pad,
    )


def _timeline(result: PipelineResult, bins: int = 48) -> list[dict]:
    """Chuỗi số near-miss theo thời gian, cho biểu đồ trên dashboard."""
    if not result.events or result.duration <= 0:
        return []
    width = result.duration / bins
    counts = [0] * bins
    severe = [0] * bins
    for e in result.events:
        b = min(bins - 1, int(e.t / width))
        counts[b] += 1
        if e.ttc is not None and e.ttc < 1.5:
            severe[b] += 1
    return [
        {"t": round(i * width, 1), "count": counts[i], "severe": severe[i]}
        for i in range(bins)
    ]
