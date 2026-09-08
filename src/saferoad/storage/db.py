"""Lưu trữ sự kiện bằng SQLite — khối "Database (Time-series)" trong kiến trúc.

Vì sao SQLite: hệ thống chạy trên Jetson Orin Nano ở biên, không có server
database. SQLite cho phép ghi bền vững, truy vấn SQL đầy đủ, và cả file chỉ là
một file duy nhất — dễ đóng gói vào hồ sơ dự thi và dễ đồng bộ lên trung tâm.

Mọi truy vấn của dashboard đều đi qua module này, nên đổi sang PostgreSQL/
TimescaleDB sau này chỉ cần thay lớp cài đặt mà không đụng tầng trên.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..types import ConflictEvent, ConflictType, RiskLevel, VehicleClass

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    source        TEXT NOT NULL,
    site          TEXT,
    fps           REAL,
    total_frames  INTEGER DEFAULT 0,
    config_json   TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    event_id       TEXT NOT NULL,
    frame_idx      INTEGER NOT NULL,
    t              REAL NOT NULL,
    track_a        INTEGER NOT NULL,
    track_b        INTEGER NOT NULL,
    cls_a          TEXT NOT NULL,
    cls_b          TEXT NOT NULL,
    ttc            REAL,
    pet            REAL,
    conflict_type  TEXT NOT NULL,
    risk_score     REAL NOT NULL,
    risk_level     TEXT NOT NULL,
    loc_x          REAL, loc_y REAL,
    loc_px         REAL, loc_py REAL,
    approach_angle REAL,
    rel_speed      REAL,
    gap            REAL,
    reasons_json   TEXT,
    terms_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_t       ON events(session_id, t);
CREATE INDEX IF NOT EXISTS idx_events_risk    ON events(session_id, risk_score DESC);

CREATE TABLE IF NOT EXISTS frame_stats (
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    frame_idx   INTEGER NOT NULL,
    t           REAL NOT NULL,
    n_tracks    INTEGER NOT NULL,
    n_events    INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    PRIMARY KEY (session_id, frame_idx)
);
"""


class EventStore:
    """Lớp truy cập database sự kiện."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id: int | None = None
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        # WAL cho phép dashboard đọc trong khi pipeline đang ghi.
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    def start_session(
        self, source: str, site: str = "", fps: float = 30.0, config: dict | None = None
    ) -> int:
        from datetime import datetime

        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO sessions (started_at, source, site, fps, config_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    source, site, fps,
                    json.dumps(config or {}, ensure_ascii=False, default=str),
                ),
            )
            self._session_id = int(cur.lastrowid)
        return self._session_id

    def finish_session(self, total_frames: int) -> None:
        if self._session_id is None:
            return
        with self._conn() as con:
            con.execute(
                "UPDATE sessions SET total_frames = ? WHERE id = ?",
                (total_frames, self._session_id),
            )

    # ------------------------------------------------------------------ #
    def add_events(self, events: list[ConflictEvent], session_id: int | None = None) -> None:
        sid = session_id or self._session_id
        if sid is None or not events:
            return
        rows = [
            (
                sid, e.event_id, e.frame_idx, e.t, e.track_a, e.track_b,
                e.cls_a.value, e.cls_b.value, e.ttc, e.pet,
                e.conflict_type.value, e.risk_score, e.risk_level.value,
                e.location[0], e.location[1], e.location_px[0], e.location_px[1],
                e.approach_angle, e.rel_speed, e.gap,
                json.dumps(e.reasons, ensure_ascii=False),
                json.dumps(e.severity_terms, ensure_ascii=False),
            )
            for e in events
        ]
        with self._conn() as con:
            con.executemany(
                "INSERT INTO events (session_id, event_id, frame_idx, t, track_a, track_b,"
                " cls_a, cls_b, ttc, pet, conflict_type, risk_score, risk_level,"
                " loc_x, loc_y, loc_px, loc_py, approach_angle, rel_speed, gap,"
                " reasons_json, terms_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def add_frame_stat(
        self, frame_idx: int, t: float, n_tracks: int, n_events: int, latency_ms: float
    ) -> None:
        if self._session_id is None:
            return
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO frame_stats VALUES (?,?,?,?,?,?)",
                (self._session_id, frame_idx, t, n_tracks, n_events, latency_ms),
            )

    # ------------------------------------------------------------------ #
    def latest_session(self) -> dict[str, Any] | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def load_events(self, session_id: int | None = None, limit: int = 0) -> list[ConflictEvent]:
        """Đọc sự kiện về dạng ``ConflictEvent``."""
        sql = "SELECT * FROM events"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY t"
        if limit:
            sql += f" LIMIT {int(limit)}"

        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()

        out: list[ConflictEvent] = []
        for r in rows:
            out.append(
                ConflictEvent(
                    event_id=r["event_id"], frame_idx=r["frame_idx"], t=r["t"],
                    track_a=r["track_a"], track_b=r["track_b"],
                    cls_a=VehicleClass(r["cls_a"]), cls_b=VehicleClass(r["cls_b"]),
                    ttc=r["ttc"], pet=r["pet"],
                    conflict_type=ConflictType(r["conflict_type"]),
                    risk_score=r["risk_score"], risk_level=RiskLevel(r["risk_level"]),
                    location=(r["loc_x"], r["loc_y"]),
                    location_px=(r["loc_px"], r["loc_py"]),
                    approach_angle=r["approach_angle"], rel_speed=r["rel_speed"],
                    gap=r["gap"],
                    reasons=json.loads(r["reasons_json"] or "[]"),
                    severity_terms=json.loads(r["terms_json"] or "{}"),
                )
            )
        return out

    def summary(self, session_id: int | None = None) -> dict[str, Any]:
        """Số liệu tổng hợp cho các thẻ KPI trên dashboard."""
        where = "WHERE session_id = ?" if session_id is not None else ""
        params = [session_id] if session_id is not None else []

        with self._conn() as con:
            total = con.execute(f"SELECT COUNT(*) c FROM events {where}", params).fetchone()["c"]
            severe = con.execute(
                f"SELECT COUNT(*) c FROM events {where} "
                f"{'AND' if where else 'WHERE'} ttc IS NOT NULL AND ttc < 1.5",
                params,
            ).fetchone()["c"]
            avg = con.execute(
                f"SELECT AVG(risk_score) a FROM events {where}", params
            ).fetchone()["a"]
            by_type = con.execute(
                f"SELECT conflict_type, COUNT(*) c FROM events {where} GROUP BY conflict_type",
                params,
            ).fetchall()
            by_level = con.execute(
                f"SELECT risk_level, COUNT(*) c FROM events {where} GROUP BY risk_level",
                params,
            ).fetchall()

        return {
            "total_events": total,
            "severe_events": severe,
            "avg_risk_score": round(avg or 0.0, 1),
            "by_type": {r["conflict_type"]: r["c"] for r in by_type},
            "by_level": {r["risk_level"]: r["c"] for r in by_level},
        }
