"""Xử lý quỹ đạo trên mặt đất: làm mượt vị trí và ước lượng vận tốc.

Vận tốc là đầu vào trực tiếp của TTC, nên chất lượng của nó quyết định chất
lượng toàn hệ thống. Lấy hiệu hai vị trí liên tiếp (finite difference) sẽ khuếch
đại nhiễu bbox lên rất mạnh: bbox rung ±3 px ở 30 FPS tương đương nhiễu vận tốc
tới vài m/s. Module này khử nhiễu đó bằng hai bước:

1. **Làm mượt vị trí** — trung bình trượt có trọng số trên cửa sổ ngắn.
2. **Ước lượng vận tốc bằng hồi quy tuyến tính** trên cửa sổ trượt, thay vì lấy
   hiệu 2 điểm. Hệ số góc của đường hồi quy chính là vận tốc, và nó dùng toàn bộ
   các điểm trong cửa sổ nên nhiễu bị triệt tiêu theo :math:`1/\\sqrt{n}`.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import HomographyConfig, TrackingConfig
from ..types import Track, TrackState
from .homography import GroundPlane


def smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
    """Làm mượt chuỗi vị trí (N, 2) bằng cửa sổ tam giác.

    Cửa sổ tam giác (thay vì hộp) cho đáp ứng tần số mượt hơn, tránh hiện tượng
    "gợn" khi đối tượng đi thẳng đều.
    """
    n = len(points)
    if n < 3 or window < 3:
        return points.copy()
    window = min(window, n if n % 2 else n - 1)
    if window < 3:
        return points.copy()
    if window % 2 == 0:
        window -= 1

    half = window // 2
    kernel = np.array([half + 1 - abs(i - half) for i in range(window)], dtype=float)
    kernel /= kernel.sum()

    out = np.empty_like(points)
    for dim in range(points.shape[1]):
        padded = np.pad(points[:, dim], (half, half), mode="edge")
        out[:, dim] = np.convolve(padded, kernel, mode="valid")
    return out


def velocity_from_window(times: np.ndarray, values: np.ndarray) -> float:
    """Hệ số góc của hồi quy tuyến tính ``values ~ times`` — chính là vận tốc.

    Trả về 0 khi cửa sổ quá ngắn hoặc mọi mốc thời gian trùng nhau.
    """
    n = len(times)
    if n < 2:
        return 0.0
    t_mean = times.mean()
    v_mean = values.mean()
    denom = float(((times - t_mean) ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((times - t_mean) * (values - v_mean)).sum() / denom)


class TrajectoryProcessor:
    """Điền toạ độ mặt đất và vận tốc vào lịch sử của các track.

    Gọi ``process(track)`` sau mỗi frame; hàm chỉ tính lại phần đuôi lịch sử nên
    chi phí là O(window) chứ không phải O(độ dài track).
    """

    def __init__(
        self,
        ground: GroundPlane,
        homo_cfg: HomographyConfig,
        track_cfg: TrackingConfig,
    ):
        self.ground = ground
        self.smooth_window = homo_cfg.smooth_window
        self.min_states = track_cfg.min_states_for_velocity
        self.alpha = track_cfg.velocity_alpha
        self._view_cache: dict[tuple[int, int], tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    def view_direction(self, anchor: tuple[float, float]) -> tuple[float, float]:
        """Vector đơn vị trên mặt đất chỉ hướng **ra xa camera** tại điểm ảnh này.

        Suy ra bằng cách chiếu hai điểm ảnh cách nhau theo chiều dọc: đi lên trên
        trong ảnh tương ứng với đi ra xa camera trên mặt đất.
        """
        key = (int(anchor[0]) // 16, int(anchor[1]) // 16)   # cache theo ô 16 px
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached

        p0 = np.asarray(self.ground.to_ground(anchor), dtype=float)
        p1 = np.asarray(
            self.ground.to_ground((anchor[0], anchor[1] - 20.0)), dtype=float
        )
        d = p1 - p0
        norm = float(np.linalg.norm(d))
        out = (float(d[0] / norm), float(d[1] / norm)) if norm > 1e-9 else (0.0, 1.0)
        self._view_cache[key] = out
        return out

    def project_state(self, state: TrackState, cls, heading: float | None) -> None:
        """Điền ``state.ground`` từ điểm tiếp đất của bbox, có bù sai lệch hình học.

        Vì sao cần bù
        -------------
        ``anchor`` là **đáy giữa** của bounding box, tức là điểm của thân xe nằm
        gần camera nhất, chứ không phải tâm xe. Chiếu thẳng điểm đó xuống mặt
        đất sẽ đặt chiếc xe lệch về phía camera một khoảng bằng nửa bề dài biểu
        kiến của nó — đo được trên tập synthetic là 0.5 m với xe máy và tới
        2.3 m với xe tải. Sai lệch này *có hệ thống* và phụ thuộc loại xe, nên
        nó bóp méo cả khoảng cách giữa các xe lẫn TTC suy ra từ đó.

        Cách bù: dịch điểm chiếu ra xa camera một nửa bề dài biểu kiến, tính từ
        kích thước thật của loại xe và góc giữa trục xe với hướng nhìn.
        """
        base = self.ground.to_ground(state.anchor)
        if cls is None:
            state.ground = base
            return

        length, width = cls.footprint
        ux, uy = self.view_direction(state.anchor)

        if heading is None:
            # Chưa biết hướng xe: lấy trung bình giữa nhìn ngang và nhìn dọc.
            extent = 0.5 * (length + width)
        else:
            # Bề dài biểu kiến của hình chữ nhật theo hướng nhìn.
            ca = math.cos(heading) * ux + math.sin(heading) * uy   # cos(góc lệch)
            sa = math.sqrt(max(0.0, 1.0 - ca * ca))
            extent = abs(length * ca) + abs(width * sa)

        shift = 0.5 * extent
        state.ground = (base[0] + ux * shift, base[1] + uy * shift)

    def process(self, track: Track) -> None:
        """Cập nhật toạ độ mặt đất + vận tốc cho state mới nhất của track."""
        if not track.history:
            return

        last = track.history[-1]
        # Hướng xe lấy từ vận tốc của frame TRƯỚC — vận tốc frame này chưa tính
        # được vì nó phụ thuộc chính vị trí đang cần bù.
        heading = None
        if len(track.history) >= 2:
            prev_state = track.history[-2]
            if prev_state.speed > 0.5:
                heading = prev_state.heading
        self.project_state(last, track.cls, heading)

        n = len(track.history)
        if n < self.min_states:
            last.velocity = (0.0, 0.0)
            return

        # Chỉ lấy phần đuôi để làm mượt — đủ cho ước lượng vận tốc tức thời.
        win = max(self.min_states, min(self.smooth_window * 2 + 1, n))
        tail = track.history[-win:]

        pts = np.array([s.ground for s in tail], dtype=float)
        times = np.array([s.t for s in tail], dtype=float)
        smoothed = smooth_positions(pts, self.smooth_window)

        vx = velocity_from_window(times, smoothed[:, 0])
        vy = velocity_from_window(times, smoothed[:, 1])

        # Làm mượt thêm một lớp EMA so với vận tốc frame trước, để vector vận tốc
        # không nhảy giật khi detector rung.
        prev = track.history[-2].velocity if n >= 2 else (0.0, 0.0)
        a = self.alpha
        last.velocity = (a * vx + (1 - a) * prev[0], a * vy + (1 - a) * prev[1])

        # Ghi lại vị trí đã làm mượt để quỹ đạo vẽ ra không răng cưa.
        last.ground = (float(smoothed[-1, 0]), float(smoothed[-1, 1]))


def acceleration(track: Track, window: int = 10) -> float:
    """Gia tốc dọc theo hướng di chuyển (m/s²) trên cửa sổ gần nhất.

    Dương = tăng tốc, âm = phanh. Dùng cho module Behavior để bắt phanh gấp.
    """
    hist = track.history[-window:]
    if len(hist) < 3:
        return 0.0
    times = np.array([s.t for s in hist], dtype=float)
    speeds = np.array([s.speed for s in hist], dtype=float)
    return velocity_from_window(times, speeds)


def heading_change_rate(track: Track, window: int = 15, min_speed: float = 1.5) -> float:
    """Tốc độ đổi hướng (độ/giây) — dùng để phát hiện tạt đầu / chuyển làn gấp.

    Chỉ dùng các trạng thái có tốc độ trên ``min_speed``: hướng đi được suy ra
    từ vector vận tốc, nên khi xe gần như đứng yên thì hướng chỉ là nhiễu chia
    cho nhiễu. Nếu không lọc, mọi xe đang bò trong hàng chờ đều bị coi là "đổi
    hướng gấp" và toàn bộ xung đột bị gán nhầm nhãn "chuyển hướng".

    Ước lượng bằng hồi quy tuyến tính trên hướng đã unwrap thay vì lấy hiệu hai
    mẫu, để một mẫu nhiễu không kéo lệch kết quả.
    """
    hist = [s for s in track.history[-window:] if s.speed > min_speed]
    if len(hist) < 5:
        return 0.0
    times = np.array([s.t for s in hist], dtype=float)
    headings = np.unwrap(np.array([s.heading for s in hist], dtype=float))
    return math.degrees(velocity_from_window(times, headings))


def predict_position(state: TrackState, dt: float) -> tuple[float, float]:
    """Ngoại suy vị trí sau ``dt`` giây theo mô hình vận tốc không đổi."""
    x, y = state.ground
    vx, vy = state.velocity
    return (x + vx * dt, y + vy * dt)
