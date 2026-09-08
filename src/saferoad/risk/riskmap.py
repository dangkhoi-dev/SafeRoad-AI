"""Risk Map — bản đồ nhiệt rủi ro theo không gian và thời gian.

Mỗi ``ConflictEvent`` được "rải" lên một lưới ô vuông trên mặt đất bằng nhân
Gauss: sự kiện không chỉ ảnh hưởng đúng một ô mà lan sang các ô lân cận, vì
vị trí xung đột luôn có sai số và vì rủi ro thực tế mang tính vùng chứ không
phải điểm.

Trọng số mỗi sự kiện suy giảm theo thời gian::

    w(t) = risk_score · exp(-(t_now - t_event) / τ)

Nhờ đó bản đồ phản ánh tình hình **gần đây**: một điểm nóng của 3 tiếng trước
không che lấp một điểm nóng vừa hình thành.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import RiskConfig
from ..types import ConflictEvent


@dataclass
class Hotspot:
    """Một điểm nóng rủi ro trên bản đồ."""

    rank: int
    x: float                  # mét
    y: float                  # mét
    score: float              # 0-100, đã chuẩn hoá
    event_count: int
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "score": round(self.score, 1),
            "event_count": self.event_count,
            "label": self.label,
        }


class RiskMap:
    """Lưới rủi ro tích luỹ trên mặt đất.

    Cách dùng::

        rmap = RiskMap(cfg.risk, bounds=(-25, 0, 25, 60))
        for ev in events:
            rmap.add(ev)
        grid = rmap.render(now_t=120.0)
        top = rmap.hotspots(now_t=120.0)
    """

    def __init__(
        self,
        cfg: RiskConfig,
        bounds: tuple[float, float, float, float] = (-30.0, 0.0, 30.0, 80.0),
    ):
        self.cfg = cfg
        self.bounds = bounds  # (x_min, y_min, x_max, y_max) tính bằng mét
        x_min, y_min, x_max, y_max = bounds
        self.nx = max(1, int(math.ceil((x_max - x_min) / cfg.grid_size)))
        self.ny = max(1, int(math.ceil((y_max - y_min) / cfg.grid_size)))
        self.events: list[ConflictEvent] = []
        #: Nhân Gauss dựng sẵn — tránh tính lại cho từng sự kiện.
        self._kernel = self._build_kernel()

    # ------------------------------------------------------------------ #
    def _build_kernel(self) -> np.ndarray:
        """Nhân Gauss 2D có bán kính ``kernel_radius`` mét."""
        radius_cells = max(1, int(round(self.cfg.kernel_radius / self.cfg.grid_size)))
        size = 2 * radius_cells + 1
        sigma = radius_cells / 2.0
        ax = np.arange(size, dtype=float) - radius_cells
        gx, gy = np.meshgrid(ax, ax)
        kernel = np.exp(-(gx**2 + gy**2) / (2.0 * sigma**2))
        return kernel / kernel.max()

    def _to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        x_min, y_min, x_max, y_max = self.bounds
        if not (x_min <= x < x_max and y_min <= y < y_max):
            return None
        cx = int((x - x_min) / self.cfg.grid_size)
        cy = int((y - y_min) / self.cfg.grid_size)
        return (min(cx, self.nx - 1), min(cy, self.ny - 1))

    def cell_center(self, cx: int, cy: int) -> tuple[float, float]:
        x_min, y_min, _, _ = self.bounds
        g = self.cfg.grid_size
        return (x_min + (cx + 0.5) * g, y_min + (cy + 0.5) * g)

    # ------------------------------------------------------------------ #
    def add(self, event: ConflictEvent) -> None:
        """Đưa một sự kiện vào bản đồ."""
        self.events.append(event)

    def extend(self, events: list[ConflictEvent]) -> None:
        self.events.extend(events)

    def render(self, now_t: float | None = None, normalize: bool = True) -> np.ndarray:
        """Dựng lưới nhiệt (ny, nx). Giá trị đã chuẩn hoá về [0, 1] nếu ``normalize``."""
        grid = np.zeros((self.ny, self.nx), dtype=float)
        if not self.events:
            return grid

        if now_t is None:
            now_t = max(e.t for e in self.events)

        k = self._kernel
        kr = k.shape[0] // 2

        for ev in self.events:
            cell = self._to_cell(*ev.location)
            if cell is None:
                continue
            cx, cy = cell
            age = max(0.0, now_t - ev.t)
            weight = ev.risk_score * math.exp(-age / self.cfg.decay_tau)
            if weight <= 1e-6:
                continue

            # Cửa sổ dán nhân, có cắt biên.
            y0, y1 = max(0, cy - kr), min(self.ny, cy + kr + 1)
            x0, x1 = max(0, cx - kr), min(self.nx, cx + kr + 1)
            ky0, ky1 = y0 - (cy - kr), k.shape[0] - ((cy + kr + 1) - y1)
            kx0, kx1 = x0 - (cx - kr), k.shape[1] - ((cx + kr + 1) - x1)
            grid[y0:y1, x0:x1] += weight * k[ky0:ky1, kx0:kx1]

        if normalize and grid.max() > 0:
            grid /= grid.max()
        return grid

    # ------------------------------------------------------------------ #
    def hotspots(self, now_t: float | None = None, labels: list[str] | None = None) -> list[Hotspot]:
        """Trả về ``top_k_hotspots`` điểm nóng, đã tách nhau ra về không gian.

        Sau khi lấy một đỉnh, ta **xoá vùng lân cận** rồi mới tìm đỉnh tiếp theo
        (non-maximum suppression). Không có bước này thì 5 điểm nóng sẽ chỉ là 5
        ô cạnh nhau của cùng một cụm.
        """
        grid = self.render(now_t=now_t, normalize=True)
        if grid.max() <= 0:
            return []

        work = grid.copy()
        suppress = max(1, int(round(self.cfg.kernel_radius / self.cfg.grid_size)))
        out: list[Hotspot] = []

        for rank in range(1, self.cfg.top_k_hotspots + 1):
            idx = int(np.argmax(work))
            cy, cx = divmod(idx, self.nx)
            peak = float(work[cy, cx])
            if peak <= 0.02:  # còn lại toàn nhiễu
                break

            x, y = self.cell_center(cx, cy)
            count = sum(
                1 for ev in self.events
                if math.dist(ev.location, (x, y)) <= self.cfg.kernel_radius
            )
            label = ""
            if labels and rank - 1 < len(labels):
                label = labels[rank - 1]
            out.append(
                Hotspot(rank=rank, x=x, y=y, score=peak * 100.0, event_count=count, label=label)
            )

            # Non-maximum suppression quanh đỉnh vừa lấy.
            y0, y1 = max(0, cy - suppress), min(self.ny, cy + suppress + 1)
            x0, x1 = max(0, cx - suppress), min(self.nx, cx + suppress + 1)
            work[y0:y1, x0:x1] = 0.0

        return out

    # ------------------------------------------------------------------ #
    def to_payload(self, now_t: float | None = None, max_cells: int = 1200) -> dict:
        """Đóng gói cho dashboard: chỉ gửi các ô có giá trị đáng kể.

        Lưới đầy đủ có thể tới hàng chục nghìn ô; gửi hết qua JSON mỗi giây là
        lãng phí. Ta chỉ gửi các ô > 2% giá trị đỉnh.
        """
        grid = self.render(now_t=now_t, normalize=True)
        ys, xs = np.nonzero(grid > 0.02)
        order = np.argsort(grid[ys, xs])[::-1][:max_cells]

        cells = []
        for i in order:
            cy, cx = int(ys[i]), int(xs[i])
            x, y = self.cell_center(cx, cy)
            cells.append({"x": round(x, 2), "y": round(y, 2), "v": round(float(grid[cy, cx]), 4)})

        return {
            "bounds": {
                "x_min": self.bounds[0], "y_min": self.bounds[1],
                "x_max": self.bounds[2], "y_max": self.bounds[3],
            },
            "grid_size": self.cfg.grid_size,
            "nx": self.nx,
            "ny": self.ny,
            "cells": cells,
            "hotspots": [h.to_dict() for h in self.hotspots(now_t=now_t)],
        }

    def coverage_ratio(self, threshold: float = 0.5) -> float:
        """Tỉ lệ diện tích có mức rủi ro vượt ngưỡng — chỉ số "khu vực rủi ro cao"."""
        grid = self.render()
        if grid.size == 0:
            return 0.0
        return float((grid >= threshold).sum() / grid.size)
