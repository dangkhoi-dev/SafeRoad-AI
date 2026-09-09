"""Dashboard web — FastAPI backend cho SafeRoad AI.

Phục vụ hai nguồn dữ liệu:

* ``results.json`` do pipeline xuất ra (chế độ phát lại — dùng cho demo và cho
  hồ sơ dự thi, không cần chạy lại pipeline);
* SQLite ``saferoad.db`` (chế độ trực tiếp — dashboard đọc trong khi pipeline
  vẫn đang ghi, nhờ SQLite ở chế độ WAL).

Frontend là **một file HTML tĩnh** duy nhất, không build tool, không CDN — mở
được cả khi máy không có mạng, điều quan trọng khi demo tại hội trường thi.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
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

    def payload(self) -> dict[str, Any]:
        """Trả về toàn bộ dữ liệu, có cache theo thời gian sửa file."""
        if self.results_path and self.results_path.exists():
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
) -> FastAPI:
    """Tạo ứng dụng FastAPI phục vụ dashboard."""
    app = FastAPI(title="SafeRoad AI Dashboard", version="1.0.0")
    source = DataSource(results_path, cfg.db_path)
    video = Path(video_path) if video_path else None

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = STATIC_DIR / "index.html"
        if not html.exists():
            raise HTTPException(status_code=500, detail="Thiếu file giao diện index.html")
        return html.read_text(encoding="utf-8")

    @app.get("/api/summary")
    def api_summary() -> JSONResponse:
        data = source.payload()
        return JSONResponse({
            "summary": data.get("summary", {}),
            "site": data.get("site", cfg.site.name),
            "site_short": cfg.site.short_name,
            "thresholds": {
                "ttc": cfg.conflict.ttc_threshold,
                "pet": cfg.conflict.pet_threshold,
                "severe_ttc": 1.5,
            },
            "has_video": bool(video and video.exists()),
        })

    @app.get("/api/events")
    def api_events(limit: int = 500, min_risk: float = 0.0, kind: str = "") -> JSONResponse:
        """Danh sách sự kiện, mới nhất trước; lọc theo mức rủi ro và kiểu."""
        events = source.payload().get("events", [])
        if min_risk > 0:
            events = [e for e in events if e.get("risk_score", 0) >= min_risk]
        if kind:
            events = [e for e in events if e.get("conflict_type") == kind]
        events = sorted(events, key=lambda e: e.get("t", 0), reverse=True)[:limit]
        return JSONResponse({"events": events, "count": len(events)})

    @app.get("/api/event/{event_id}")
    def api_event(event_id: str) -> JSONResponse:
        for e in source.payload().get("events", []):
            if e.get("event_id") == event_id:
                return JSONResponse(e)
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sự kiện {event_id}")

    @app.get("/api/riskmap")
    def api_riskmap() -> JSONResponse:
        data = source.payload()
        rm = data.get("risk_map")
        if not rm:
            return JSONResponse({"cells": [], "hotspots": [], "bounds": None})
        return JSONResponse(rm)

    @app.get("/api/timeline")
    def api_timeline() -> JSONResponse:
        return JSONResponse({"timeline": source.payload().get("timeline", [])})

    @app.get("/api/behaviors")
    def api_behaviors() -> JSONResponse:
        return JSONResponse({"behaviors": source.payload().get("behaviors", [])})

    @app.get("/api/evaluation")
    def api_evaluation() -> JSONResponse:
        """Kết quả đánh giá (ablation, quét nhiễu) nếu đã chạy ``saferoad evaluate``."""
        path = Path(cfg.output_dir) / "evaluation.json"
        if not path.exists():
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, **json.loads(path.read_text(encoding="utf-8"))})

    @app.get("/api/video")
    def api_video():
        if not video or not video.exists():
            raise HTTPException(status_code=404, detail="Chưa có video overlay")
        return FileResponse(video, media_type="video/mp4")

    return app
