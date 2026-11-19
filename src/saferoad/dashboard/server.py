"""Dashboard web - FastAPI backend cho SafeRoad AI.

Phục vụ hai nguồn dữ liệu:

* ``results.json`` do pipeline xuất ra (chế độ phát lại - dùng cho demo và cho
  hồ sơ dự thi, không cần chạy lại pipeline);
* SQLite ``saferoad.db`` (chế độ trực tiếp - dashboard đọc trong khi pipeline
  vẫn đang ghi, nhờ SQLite ở chế độ WAL).

Frontend là **một file HTML tĩnh** duy nhất, không build tool, không CDN - mở
được cả khi máy không có mạng, điều quan trọng khi demo tại hội trường thi.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..storage.db import EventStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class DataSource:
    """Nguồn dữ liệu cho dashboard, ưu tiên JSON rồi mới tới database."""

    def __init__(self, results_path: str | None, db_path: str | None):
        self.results_path = Path(results_path) if results_path else None
        self.db_path = Path(db_path) if db_path else None
        self._cache: dict[str, Any] | None = None
        self._mtime: float = 0.0
        # Các endpoint chạy song song trong threadpool: khoá để file chỉ nạp một lần.
        self._lock = threading.Lock()

    def payload(self) -> dict[str, Any]:
        """Trả về toàn bộ dữ liệu, có cache theo thời gian sửa file."""
        if self.results_path and self.results_path.exists():
            with self._lock:
                mtime = self.results_path.stat().st_mtime
                if self._cache is None or mtime != self._mtime:
                    self._cache = json.loads(self.results_path.read_text(encoding="utf-8"))
                    self._mtime = mtime
                    log.info("Đã nạp %s", self.results_path)
            return self._cache

        if self.db_path and self.db_path.exists():
            store = EventStore(self.db_path)
            session = store.latest_session()
            sid = session["id"] if session else None
            events = store.load_events(sid)
            summary = store.summary(sid)
            return {
                "summary": {
                    "total_events": summary["total_events"],
                    "severe_events": summary["severe_events"],
                    "avg_risk_score": summary["avg_risk_score"],
                    "by_type": summary["by_type"],
                    "n_frames": session["total_frames"] if session else 0,
                    "duration_s": 0,
                    "processing_fps": 0,
                    "latency_ms": {},
                    "total_latency_ms": 0,
                    "n_tracks": 0,
                },
                "site": session["site"] if session else "",
                "events": [e.to_dict() for e in events],
                "behaviors": [],
                "risk_map": None,
                "timeline": [],
            }

        raise HTTPException(
            status_code=404,
            detail="Chưa có dữ liệu. Chạy 'saferoad run' trước khi mở dashboard.",
        )


def create_app(
    cfg: Config,
    results_path: str | None = None,
    video_path: str | None = None,
    demo: bool = False,
    sample_video: str | None = None,
) -> FastAPI:
    """Tạo ứng dụng FastAPI phục vụ dashboard.

    ``demo=True`` bật chức năng tải video lên và chạy pipeline thật (bản demo
    trực tuyến). ``sample_video`` là video mẫu cho nút "Chạy video mẫu".
    """
    import csv
    import os
    import tempfile

    from .jobs import JobManager

    app = FastAPI(title="SafeRoad AI Dashboard", version="2.0.0")
    default_source = DataSource(results_path, cfg.db_path)
    video = Path(video_path) if video_path else None
    out_dir = Path(cfg.output_dir)
    jobs = JobManager(work_dir=os.environ.get("SAFEROAD_JOBS_DIR", "data/jobs"),
                      max_seconds=float(os.environ.get("SAFEROAD_MAX_SECONDS", "30")))
    sources: dict[str, DataSource] = {}
    sample = sample_video or os.environ.get("SAFEROAD_SAMPLE_VIDEO")
    sample_preset = os.environ.get("SAFEROAD_SAMPLE_PRESET", "street")

    def src(job: str = "") -> DataSource:
        if not job:
            return default_source
        path = jobs.results_path(job)
        if path is None:
            raise HTTPException(status_code=404, detail="Job chưa xong hoặc không tồn tại")
        if job not in sources:
            sources[job] = DataSource(str(path), None)
        return sources[job]

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = STATIC_DIR / "index.html"
        if not html.exists():
            raise HTTPException(status_code=500, detail="Thiếu file giao diện index.html")
        text = html.read_text(encoding="utf-8")
        flags = {"demo": demo, "sample": bool(sample and Path(sample).exists())}
        return text.replace("<script>", f"<script>window.SAFEROAD_FLAGS={json.dumps(flags)};</script>\n<script>", 1)

    @app.get("/api/summary")
    def api_summary(job: str = "") -> JSONResponse:
        data = src(job).payload()
        has_video = (jobs.video_path(job) is not None) if job else bool(video and video.exists())
        return JSONResponse({
            "summary": data.get("summary", {}),
            "site": data.get("site", cfg.site.name),
            "site_short": cfg.site.short_name,
            "thresholds": {
                "ttc": cfg.conflict.ttc_threshold,
                "pet": cfg.conflict.pet_threshold,
                "severe_ttc": 1.5,
            },
            "has_video": has_video,
            "job": job or None,
        })

    @app.get("/api/events")
    def api_events(limit: int = 500, min_risk: float = 0.0, kind: str = "",
                   job: str = "") -> JSONResponse:
        """Danh sách sự kiện, mới nhất trước; lọc theo mức rủi ro và kiểu."""
        events = src(job).payload().get("events", [])
        if min_risk > 0:
            events = [e for e in events if e.get("risk_score", 0) >= min_risk]
        if kind:
            events = [e for e in events if e.get("conflict_type") == kind]
        events = sorted(events, key=lambda e: e.get("t", 0), reverse=True)[:limit]
        return JSONResponse({"events": events, "count": len(events)})

    @app.get("/api/event/{event_id}")
    def api_event(event_id: str, job: str = "") -> JSONResponse:
        for e in src(job).payload().get("events", []):
            if e.get("event_id") == event_id:
                return JSONResponse(e)
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sự kiện {event_id}")

    @app.get("/api/riskmap")
    def api_riskmap(job: str = "") -> JSONResponse:
        rm = src(job).payload().get("risk_map")
        if not rm:
            return JSONResponse({"cells": [], "hotspots": [], "bounds": None})
        return JSONResponse(rm)

    @app.get("/api/timeline")
    def api_timeline(job: str = "") -> JSONResponse:
        return JSONResponse({"timeline": src(job).payload().get("timeline", [])})

    @app.get("/api/behaviors")
    def api_behaviors(job: str = "") -> JSONResponse:
        return JSONResponse({"behaviors": src(job).payload().get("behaviors", [])})

    @app.get("/api/evaluation")
    def api_evaluation() -> JSONResponse:
        """Kết quả đánh giá: mô phỏng, lưới điều kiện, dữ liệu thật, hiệu chuẩn."""
        def read(name: str):
            path = out_dir / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        main = read("evaluation.json")
        if main is None:
            return JSONResponse({"available": False})
        return JSONResponse({
            "available": True, **main,
            "grid": read("evaluation_grid.json"),
            "real": {k: v for k, v in (
                ("street", read("evaluation_street.json")),
                ("street_pretrained_test", read("evaluation_street_pretrained_test.json")),
                ("street_finetuned_test", read("evaluation_street_finetuned_test.json")),
                ("mvti", read("evaluation_real.json")),
                ("mvti_pretrained_test", read("evaluation_mvti_pretrained_test.json")),
                ("mvti_finetuned_test", read("evaluation_mvti_finetuned_test.json")),
            ) if v is not None},
            "calibration_street": read("calibration_street.json"),
        })

    @app.get("/api/video")
    def api_video(job: str = ""):
        path = jobs.video_path(job) if job else video
        if not path or not Path(path).exists():
            raise HTTPException(status_code=404, detail="Chưa có video overlay")
        kind = {".webm": "video/webm", ".mp4": "video/mp4",
                ".mov": "video/quicktime"}.get(Path(path).suffix.lower(), "video/mp4")
        return FileResponse(path, media_type=kind)

    # ---------------------- Chạy thử trên video tải lên ----------------- #
    @app.get("/api/presets")
    def api_presets() -> JSONResponse:
        return JSONResponse({"presets": JobManager.presets(), "demo": demo,
                             "max_seconds": jobs.max_seconds,
                             "sample": bool(sample and Path(sample).exists())})

    @app.post("/api/jobs")
    async def api_jobs(file: UploadFile = File(...), preset: str = Form("uncalibrated")):
        if not demo:
            raise HTTPException(status_code=403, detail="Chức năng chạy thử chỉ bật khi chạy với --demo")
        suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
        if suffix not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            raise HTTPException(status_code=400, detail="Chỉ nhận video MP4/MOV/AVI/MKV/WEBM")
        tmp = Path(tempfile.mkdtemp(prefix="saferoad_")) / f"upload{suffix}"
        size = 0
        with open(tmp, "wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > jobs.max_mb * (1 << 20):
                    raise HTTPException(status_code=413, detail=f"File quá {jobs.max_mb:.0f} MB")
                fh.write(chunk)
        try:
            job = jobs.submit(tmp, preset)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return JSONResponse(job.to_dict())

    @app.post("/api/jobs/sample")
    def api_jobs_sample(preset: str = Form("")):
        if not demo:
            raise HTTPException(status_code=403, detail="Chức năng chạy thử chỉ bật khi chạy với --demo")
        if not sample or not Path(sample).exists():
            raise HTTPException(status_code=404, detail="Máy chủ chưa có video mẫu")
        try:
            job = jobs.submit(sample, preset or sample_preset)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return JSONResponse(job.to_dict())

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không có job này")
        return JSONResponse(job.to_dict())

    # ----------------------------- Rà soát nhãn ------------------------- #
    @app.get("/api/review")
    def api_review_list() -> JSONResponse:
        # Bỏ các danh sách rỗng (đoạn video không có cảnh báo nào) - không có gì để gán.
        names = sorted(p.stem.replace("review_candidates_", "")
                       for p in out_dir.glob("review_candidates_*.csv")
                       if len(p.read_text(encoding="utf-8-sig").strip().splitlines()) > 1)
        return JSONResponse({"datasets": names})

    @app.get("/api/review/{name}")
    def api_review(name: str, job: str = "") -> JSONResponse:
        if name == "job":
            events = src(job).payload().get("events", [])
            rows = [{"candidate_id": e["event_id"], "source": "system", "t_s": e["t"],
                     "track_a": e["track_a"], "track_b": e["track_b"],
                     "ttc_s": e.get("ttc"), "pet_s": e.get("pet"),
                     "pet_status": e.get("pet_status", ""), "contact": e.get("contact", ""),
                     "type": e.get("conflict_type")} for e in sorted(events, key=lambda e: e["t"])]
            return JSONResponse({"rows": rows, "video": f"/api/video?job={job}"})
        path = out_dir / f"review_candidates_{name}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Không có danh sách rà soát này")
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        vid = rows[0].get("video") if rows else ""
        return JSONResponse({"rows": rows,
                             "video": f"/api/review/{name}/video" if vid and Path(vid).exists() else ""})

    @app.get("/api/review/{name}/video")
    def api_review_video(name: str):
        path = out_dir / f"review_candidates_{name}.csv"
        if not path.exists():
            raise HTTPException(status_code=404)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            first = next(csv.DictReader(fh), None)
        vid = Path(first["video"]) if first and first.get("video") else None
        if vid is None or not vid.exists():
            raise HTTPException(status_code=404, detail="Video gốc không có trên máy chủ này")
        return FileResponse(vid, media_type="video/mp4")

    @app.post("/api/review/{name}")
    async def api_review_save(name: str, payload: dict) -> JSONResponse:
        reviewer = "".join(ch for ch in str(payload.get("reviewer", "")) if ch.isalnum() or ch in "-_")
        if not reviewer:
            raise HTTPException(status_code=400, detail="Cần tên người rà soát")
        labels = payload.get("labels", {})
        path = out_dir / f"labels_{name}_{reviewer}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["candidate_id", "label", "reviewer", "note"])
            for cid, lab in sorted(labels.items()):
                w.writerow([cid, lab.get("label", ""), reviewer, lab.get("note", "")])
        return JSONResponse({"saved": str(path), "n": len(labels)})

    return app
