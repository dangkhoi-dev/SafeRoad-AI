"""Chạy pipeline thật trên video người dùng tải lên - phục vụ bản demo trực tuyến.

Hội đồng nhận xét bản trực tuyến cũ "chỉ là dashboard tính sẵn". Module này cho
phép bất kỳ ai mở trang demo tải một đoạn video lên (hoặc bấm chạy video mẫu)
và xem **chính pipeline YOLO → ByteTrack → TTC/PET → Risk** chạy trên đó: tiến
độ theo từng khung hình, video overlay, danh sách cảnh báo và bảng đối soát.

Giới hạn để chạy được trên máy chủ CPU miễn phí: tối đa ``max_seconds`` giây
video, mỗi lúc một job, file tối đa ``max_mb`` MB.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from ..config import Config

log = logging.getLogger(__name__)

#: Cấu hình chọn được khi chạy thử. Video tải lên không có hiệu chuẩn thì
#: homography là xấp xỉ - giao diện phải nói rõ điều đó.
PRESETS: dict[str, dict[str, Any]] = {
    "uncalibrated": {
        "label": "Video bất kỳ (chưa hiệu chuẩn - TTC/PET chỉ mang tính tương đối)",
        "config": "configs/default.yaml",
        "calibrated": False,
    },
    "street": {
        "label": "Video đại lộ Kaggle (đã hiệu chuẩn từ vạch kẻ đường)",
        "config": "configs/street.yaml",
        "calibrated": True,
    },
    "mvti": {
        "label": "Giao lộ MVTI (camera hạ tầng, chưa hiệu chuẩn thực địa)",
        "config": "configs/mvti.yaml",
        "calibrated": False,
    },
}


@dataclass
class Job:
    job_id: str
    source: str
    preset: str
    created: float = field(default_factory=time.time)
    state: str = "queued"          # queued | running | done | error
    message: str = "Đang chờ"
    processed: int = 0
    total: int = 0
    fps: float = 0.0
    out_dir: str = ""
    error: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "state": self.state, "message": self.message,
            "processed": self.processed, "total": self.total,
            "progress": round(self.processed / self.total, 3) if self.total else 0.0,
            "fps": round(self.fps, 1), "preset": self.preset,
            "preset_label": PRESETS.get(self.preset, {}).get("label", self.preset),
            "calibrated": PRESETS.get(self.preset, {}).get("calibrated", False),
            "error": self.error, "summary": self.summary,
        }


class JobManager:
    """Hàng đợi một-luồng: mỗi lúc chỉ chạy một video để không quá tải CPU."""

    def __init__(self, work_dir: str | Path = "data/jobs", max_seconds: float = 30.0,
                 max_mb: float = 150.0, root: str | Path = "."):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_seconds = max_seconds
        self.max_mb = max_mb
        self.root = Path(root)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._busy = False

    # ------------------------------------------------------------------ #
    def submit(self, video_path: str | Path, preset: str = "uncalibrated") -> Job:
        if preset not in PRESETS:
            raise ValueError(f"Cấu hình không hợp lệ: {preset}")
        with self._lock:
            if self._busy:
                raise RuntimeError("Đang có một video khác được xử lý, thử lại sau ít phút.")
            self._busy = True
        jid = uuid.uuid4().hex[:10]
        job = Job(job_id=jid, source=str(video_path), preset=preset,
                  out_dir=str(self.work_dir / jid))
        Path(job.out_dir).mkdir(parents=True, exist_ok=True)
        self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------------ #
    def _run(self, job: Job) -> None:
        from ..pipeline.runner import Pipeline

        try:
            job.state = "running"
            cap = cv2.VideoCapture(job.source)
            if not cap.isOpened():
                raise RuntimeError("Không đọc được video - hãy dùng MP4 (H.264) hoặc MOV")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            max_frames = int(self.max_seconds * fps)
            job.total = min(n, max_frames) if n > 0 else max_frames

            cfg_path = self.root / PRESETS[job.preset]["config"]
            cfg = Config.load(cfg_path)
            cfg.video.source = job.source
            cfg.video.max_frames = job.total
            cfg.video.write_overlay = True
            cfg.video.overlay_path = str(Path(job.out_dir) / "overlay_raw.mp4")
            cfg.output_dir = job.out_dir
            cfg.db_path = str(Path(job.out_dir) / "saferoad.db")
            weights = self.root / cfg.detection.weights
            if not weights.exists():
                # Trọng số tinh chỉnh chưa được tải về - dùng YOLO11n gốc.
                weights = self.root / "models" / "yolo11n.pt"
            cfg.detection.weights = str(weights)
            cfg.behavior.model_path = str(self.root / cfg.behavior.model_path)
            # Video tải lên có độ phân giải khác video hiệu chuẩn thì homography
            # theo pixel không còn đúng - quay về xấp xỉ và báo rõ.
            if PRESETS[job.preset]["calibrated"]:
                ref_w = 1920 if job.preset == "street" else w
                if w != ref_w:
                    cfg.homography.image_points = []
                    cfg.homography.world_points = []
                    cfg.conflict.analysis_region_px = []
                    job.preset = "uncalibrated"
            job.message = "Đang chạy YOLO → theo vết → TTC/PET"

            pipe = Pipeline(cfg, write_db=False)
            t0 = time.perf_counter()

            def progress(done: int, total: int) -> None:
                job.processed = done
                el = time.perf_counter() - t0
                job.fps = done / el if el > 0 else 0.0

            pipe.on_progress = progress
            result = pipe.run(progress_every=0)
            Pipeline.export_json(result, Path(job.out_dir) / "results.json", cfg)
            job.message = "Đang chuyển mã video cho trình duyệt"
            self._transcode(Path(cfg.video.overlay_path), Path(job.out_dir) / "overlay.mp4")
            s = result.summary()
            job.summary = {
                "n_frames": s["n_frames"], "n_tracks": s["n_tracks"],
                "total_events": s["total_events"], "processing_fps": s["processing_fps"],
                "event_funnel": s.get("event_funnel", {}), "detector": s.get("detector"),
            }
            job.processed = job.total
            job.state = "done"
            job.message = "Xong"
        except Exception as exc:  # pragma: no cover - phụ thuộc video người dùng
            log.exception("Job %s lỗi", job.job_id)
            job.state = "error"
            job.error = str(exc)
            job.message = "Lỗi"
        finally:
            with self._lock:
                self._busy = False

    @staticmethod
    def _transcode(src: Path, dst: Path) -> None:
        """mp4v (OpenCV) → H.264 để thẻ <video> của trình duyệt phát được."""
        if not src.exists():
            return
        ff = shutil.which("ffmpeg")
        if ff is None:
            shutil.copy(src, dst)
            return
        subprocess.run([ff, "-v", "error", "-y", "-i", str(src), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=False, timeout=600)
        if not dst.exists():
            shutil.copy(src, dst)

    def results_path(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        if job is None or job.state != "done":
            return None
        return Path(job.out_dir) / "results.json"

    def video_path(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        p = Path(job.out_dir) / "overlay.mp4"
        return p if p.exists() else None

    @staticmethod
    def presets() -> list[dict[str, Any]]:
        return [{"key": k, "label": v["label"], "calibrated": v["calibrated"]}
                for k, v in PRESETS.items()]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
