"""Vẽ overlay lên video: bbox, ID, quỹ đạo, cảnh báo TTC/RISK, mini risk map.

Đây là hình ảnh xuất hiện trong panel "LIVE CAMERA" của dashboard và trong video
demo, nên phải đọc được ngay trong 1-2 giây: dùng màu theo mức rủi ro, chỉ vẽ
nhãn cho đối tượng đang có xung đột, và luôn hiện TTC của cặp nguy hiểm nhất.

Ghi chú kỹ thuật: OpenCV không vẽ được tiếng Việt có dấu bằng ``cv2.putText``
(font Hershey chỉ có ASCII). Các nhãn trên video vì vậy dùng nhãn không dấu; bản
tiếng Việt đầy đủ hiển thị trên dashboard HTML.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..geometry.homography import GroundPlane
from ..risk.riskmap import RiskMap
from ..types import ConflictEvent, RiskLevel, Track, VehicleClass

#: Bảng màu theo mức rủi ro (BGR).
LEVEL_COLOR = {
    RiskLevel.NONE: (150, 190, 120),
    RiskLevel.LOW: (120, 220, 240),
    RiskLevel.MEDIUM: (80, 165, 250),
    RiskLevel.HIGH: (70, 70, 245),
}

#: Nhãn ASCII cho từng lớp (OpenCV không vẽ được dấu tiếng Việt).
CLASS_LABEL = {
    VehicleClass.PEDESTRIAN: "PED",
    VehicleClass.BICYCLE: "BIKE",
    VehicleClass.MOTORCYCLE: "MOTO",
    VehicleClass.CAR: "CAR",
    VehicleClass.TRUCK: "TRUCK",
}

CLASS_COLOR = {
    VehicleClass.PEDESTRIAN: (208, 160, 255),
    VehicleClass.BICYCLE: (140, 232, 168),
    VehicleClass.MOTORCYCLE: (96, 196, 255),
    VehicleClass.CAR: (232, 190, 108),
    VehicleClass.TRUCK: (128, 158, 232),
}


class OverlayRenderer:
    """Vẽ lớp thông tin lên trên khung hình gốc."""

    def __init__(self, ground: GroundPlane, cfg: Config, trail_length: int = 45):
        self.ground = ground
        self.cfg = cfg
        self.trail_length = trail_length
        #: Cặp track đang trong trạng thái cảnh báo → (mức, TTC, số frame còn hiện).
        self._active_alerts: dict[tuple[int, int], list] = {}

    # ------------------------------------------------------------------ #
    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        events: list[ConflictEvent],
        conflict_detector,
        frame_idx: int,
        t: float,
        risk_map: RiskMap | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Cảnh báo hiển thị lấy từ các episode ĐANG diễn ra, không phải từ các
        # sự kiện đã chốt: sự kiện chỉ được chốt sau khi tình huống kết thúc, nên
        # nếu vẽ theo nó thì khung hình luôn hiện cảnh báo trễ mất 1-2 giây.
        live = conflict_detector.active_alerts(frame_idx) if conflict_detector else {}
        self._active_alerts = {
            key: [ep.live_level,
                  None if ep.current_ttc == float("inf") else ep.current_ttc,
                  0]
            for key, ep in live.items()
        }

        alert_ids: dict[int, RiskLevel] = {}
        for (a, b), (level, _ttc, _n) in self._active_alerts.items():
            for tid in (a, b):
                prev = alert_ids.get(tid)
                if prev is None or level.level > prev.level:
                    alert_ids[tid] = level

        # --- 1. Quỹ đạo ------------------------------------------------- #
        for tr in tracks:
            pts = [s.anchor for s in tr.history[-self.trail_length:]]
            if len(pts) < 2:
                continue
            colour = CLASS_COLOR.get(tr.cls, (200, 200, 200))
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p1 = (int(pts[i][0]), int(pts[i][1]))
                shade = tuple(int(c * (0.35 + 0.65 * alpha)) for c in colour)
                cv2.line(canvas, p0, p1, shade, 2, cv2.LINE_AA)

        # --- 2. Bounding box + nhãn -------------------------------------- #
        for tr in tracks:
            st = tr.last
            x1, y1, x2, y2 = (int(v) for v in st.bbox)
            level = alert_ids.get(tr.track_id)
            if level is not None:
                colour = LEVEL_COLOR[level]
                thickness = 3
            else:
                colour = CLASS_COLOR.get(tr.cls, (200, 200, 200))
                thickness = 1

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

            label = f"{CLASS_LABEL.get(tr.cls, '?')} #{tr.track_id}"
            if st.speed > 0.5:
                label += f" {st.speed * 3.6:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(canvas, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

        # --- 3. Đường nối cặp xung đột + nhãn TTC ------------------------- #
        track_by_id = {tr.track_id: tr for tr in tracks}
        for (a, b), (level, ttc, _n) in self._active_alerts.items():
            tr_a, tr_b = track_by_id.get(a), track_by_id.get(b)
            if tr_a is None or tr_b is None:
                continue
            pa = tuple(int(v) for v in tr_a.last.anchor)
            pb = tuple(int(v) for v in tr_b.last.anchor)
            colour = LEVEL_COLOR[level]
            cv2.line(canvas, pa, pb, colour, 2, cv2.LINE_AA)

            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            text = f"TTC {ttc:.1f}s" if ttc is not None else "PET"
            text += f"  {level.value.upper()}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(canvas, (mid[0] - 4, mid[1] - th - 8),
                          (mid[0] + tw + 6, mid[1] + 4), colour, -1)
            cv2.putText(canvas, text, (mid[0], mid[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # --- 4. Thanh trạng thái ----------------------------------------- #
        self._draw_hud(canvas, conflict_detector, frame_idx, t, len(tracks))

        # --- 5. Mini risk map góc dưới phải ------------------------------- #
        if risk_map is not None and risk_map.events:
            self._draw_mini_map(canvas, risk_map, t)

        return canvas

    # ------------------------------------------------------------------ #
    def _draw_hud(self, canvas, detector, frame_idx: int, t: float, n_tracks: int) -> None:
        h, w = canvas.shape[:2]
        bar_h = 42
        strip = canvas[0:bar_h, 0:w].copy()
        cv2.rectangle(strip, (0, 0), (w, bar_h), (24, 26, 30), -1)
        cv2.addWeighted(strip, 0.78, canvas[0:bar_h, 0:w], 0.22, 0, canvas[0:bar_h, 0:w])

        total = detector.total_events if detector else 0
        severe = len(detector.severe_events()) if detector else 0
        text = (
            f"SafeRoad AI | t={t:6.2f}s  f={frame_idx:05d} | "
            f"tracks={n_tracks:2d} | near-miss={total:3d} | severe={severe:2d}"
        )
        cv2.putText(canvas, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (235, 240, 245), 1, cv2.LINE_AA)

    def _draw_mini_map(self, canvas, risk_map: RiskMap, t: float, size: int = 168) -> None:
        h, w = canvas.shape[:2]
        grid = risk_map.render(now_t=t, normalize=True)
        if grid.max() <= 0:
            return

        heat = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # Lật trục Y để "xa camera" nằm ở phía trên như bản đồ thật.
        heat = cv2.flip(heat, 0)
        heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)

        margin = 14
        x0, y0 = w - size - margin, h - size - margin
        roi = canvas[y0:y0 + size, x0:x0 + size]
        cv2.addWeighted(heat, 0.72, roi, 0.28, 0, roi)
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + size, y0 + size), (200, 205, 210), 1)
        cv2.putText(canvas, "RISK MAP", (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (225, 230, 235), 1, cv2.LINE_AA)
