"""Vi mô phỏng giao lộ có ground truth chính xác.

Vì sao cần simulator
--------------------
Poster SafeRoad AI cam kết Conflict Precision ≥ 0.75, Recall ≥ 0.80 và
TTC MAE ≤ 0.30 s. Muốn **đo** được ba con số đó phải có nhãn near-miss chuẩn,
trong khi mọi dataset giao thông công khai đều chỉ có nhãn bounding box, không
có nhãn xung đột; còn gán nhãn tay thì tốn công và phụ thuộc chủ quan người gán.

Simulator giải quyết vấn đề: ta biết **chính xác** quỹ đạo từng xe nên tính được
TTC/PET thật ở độ phân giải thời gian tuỳ ý. Hệ thống thực chỉ nhìn thấy bbox
nhiễu ở 30 FPS và phải tái tạo lại các con số đó — đúng bài toán cần đo.

Mô hình hành vi
---------------
Xe **không** chạy theo quỹ đạo định sẵn một cách máy móc — nếu vậy chúng sẽ đi
xuyên qua nhau và sinh ra hàng trăm "near-miss" vô nghĩa. Thay vào đó ta chạy vi
mô phỏng với ba cơ chế của giao thông thật:

1. **Đèn tín hiệu** — giao lộ có chu kỳ đèn Bắc-Nam / Đông-Tây luân phiên, đúng
   như Ngã tư Hàng Xanh. Đây là thứ giữ cho hai dòng cắt nhau không cùng lúc
   chiếm giao lộ. (Cơ chế "ai cũng nhường ai" không dùng được: nó gây tắc nghẽn
   chết — mọi xe cùng dừng và không xe nào đi tiếp.)
2. **Car-following (mô hình IDM)** — Treiber, Hennecke & Helbing (2000). Xe giữ
   khoảng cách an toàn với xe phía trước và với vạch dừng khi đèn đỏ.
3. **Phanh tránh khẩn cấp** — khi TTC tụt xuống dưới ~1.2 s, người lái phanh gấp.
   Đây là lý do phần lớn tình huống nguy hiểm kết thúc bằng "suýt" chứ không
   phải va chạm thật.

Near-miss sinh ra từ đúng những nguyên nhân ngoài đời: xe **vượt đèn đỏ** (tỉ lệ
``reckless_ratio``), xe **rẽ cắt** dòng đi thẳng ngược chiều, và **người đi bộ**
băng qua đường. Không có tình huống nào được "dàn dựng" thủ công.

Nguyên tắc trung thực: nhãn chuẩn (oracle) tính bằng quy trình **khác** với
thuật toán online — lấy cực tiểu toàn cục trên chuỗi thời gian mịn, dùng vận tốc
giải tích thay vì vận tốc ước lượng từ bbox nhiễu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..conflict.geometry import series_ttc_and_gap
from ..types import ConflictType, GroundTruthConflict, VehicleClass

#: Vùng mô phỏng (mét): x ∈ [-30, 30], y ∈ [0, 60].
WORLD_BOUNDS = (-30.0, 0.0, 30.0, 60.0)
#: Tâm giao lộ.
CENTER = (0.0, 30.0)
#: Nửa bề rộng mỗi nhánh đường (mét).
ROAD_HALF_WIDTH = 7.5
#: Bán kính vùng giao lộ.
JUNCTION_RADIUS = 11.0
#: Bán kính vạch dừng — xe dừng đèn đỏ tại đây.
STOP_LINE_RADIUS = 9.0
#: Khoảng trống (giây) mà xe rẽ đòi hỏi trước khi cắt qua dòng ngược chiều.
TURN_GAP = 2.6
#: Sau chừng này giây chờ, người lái mất kiên nhẫn và cứ thế cắt qua —
#: đây là một trong những nguồn xung đột "chuyển hướng" thực tế nhất.
TURN_PATIENCE = 6.0

# --------------------------------------------------------------------------- #
# Tham số mô hình IDM (Intelligent Driver Model)
# --------------------------------------------------------------------------- #
#: Gia tốc tối đa (m/s²) theo lớp xe.
IDM_A_MAX = {
    VehicleClass.MOTORCYCLE: 2.6,
    VehicleClass.CAR: 2.0,
    VehicleClass.TRUCK: 1.2,
    VehicleClass.BICYCLE: 1.0,
    VehicleClass.PEDESTRIAN: 0.8,
}
#: Giảm tốc thoải mái (m/s²).
IDM_B_COMFORT = {
    VehicleClass.MOTORCYCLE: 3.2,
    VehicleClass.CAR: 2.8,
    VehicleClass.TRUCK: 2.0,
    VehicleClass.BICYCLE: 1.6,
    VehicleClass.PEDESTRIAN: 1.2,
}
#: Khoảng cách tối thiểu khi dừng (m) — xe máy Việt Nam bám rất sát.
IDM_S0 = {
    VehicleClass.MOTORCYCLE: 0.9,
    VehicleClass.CAR: 2.0,
    VehicleClass.TRUCK: 2.8,
    VehicleClass.BICYCLE: 0.8,
    VehicleClass.PEDESTRIAN: 0.5,
}
#: Thời gian giãn cách mong muốn (s).
IDM_T = {
    VehicleClass.MOTORCYCLE: 0.8,
    VehicleClass.CAR: 1.3,
    VehicleClass.TRUCK: 1.7,
    VehicleClass.BICYCLE: 1.0,
    VehicleClass.PEDESTRIAN: 1.0,
}


# --------------------------------------------------------------------------- #
# Định nghĩa phương tiện
# --------------------------------------------------------------------------- #
class TrafficSignal:
    """Đèn tín hiệu hai pha: Bắc-Nam ↔ Đông-Tây.

    Chu kỳ: ``green → yellow → all-red`` cho pha 1, rồi tương tự cho pha 2.
    Khoảng **all-red** (đèn đỏ cả hai chiều) là thứ tồn tại trong thực tế để
    giải phóng giao lộ; xe vượt đèn đỏ trong khoảng này chính là nguồn xung đột
    cắt ngang nguy hiểm nhất.
    """

    def __init__(self, green: float = 30.0, yellow: float = 3.0, all_red: float = 2.0,
                 offset: float = 0.0):
        self.green = green
        self.yellow = yellow
        self.all_red = all_red
        self.offset = offset
        self.half = green + yellow + all_red
        self.cycle = 2 * self.half

    def phase(self, t: float, group: str) -> str:
        """Trạng thái đèn (``green`` | ``yellow`` | ``red``) cho nhóm hướng."""
        u = (t + self.offset) % self.cycle
        # Nửa đầu chu kỳ dành cho nhóm NS, nửa sau cho nhóm EW.
        if group == "NS":
            local, mine = u, u < self.half
        else:
            local, mine = u - self.half, u >= self.half
        if not mine:
            return "red"
        if local < self.green:
            return "green"
        if local < self.green + self.yellow:
            return "yellow"
        return "red"

    def time_to_red(self, t: float, group: str) -> float:
        """Số giây còn lại trước khi nhóm này chuyển sang đỏ (dùng cho vùng lưỡng lự)."""
        u = (t + self.offset) % self.cycle
        start = 0.0 if group == "NS" else self.half
        end_green = start + self.green + self.yellow
        if start <= u < end_green:
            return end_green - u
        return -1.0


@dataclass
class VehicleSpec:
    """Một phương tiện trong kịch bản, kèm quỹ đạo đã mô phỏng.

    Trước khi chạy ``simulate()`` chỉ có ``waypoints`` (tuyến đường dự định).
    Sau khi mô phỏng, ``times``/``positions``/``velocities`` chứa quỹ đạo thực
    tế đã tính đến đèn tín hiệu, giữ khoảng cách và phanh tránh.
    """

    vid: int
    cls: VehicleClass
    waypoints: list[tuple[float, float]]
    t_start: float
    desired_speed: float
    approach: str = ""
    #: Nhóm pha đèn: "NS", "EW", hoặc "PED" (người đi bộ, không theo đèn xe).
    signal_group: str = "NS"
    #: Có vượt đèn đỏ hay không.
    runs_red: bool = False
    #: Có pha phanh gấp chủ động (dùng cho nhãn hành vi).
    forced_brake: tuple[float, float] | None = None
    #: Quãng đường (m) dọc tuyến tới vạch dừng. Tính trong ``__post_init__``.
    stop_line_s: float = field(default=0.0, init=False)

    # -- Kết quả mô phỏng ------------------------------------------------ #
    times: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), repr=False)
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), repr=False)

    _cum: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        cum = [0.0]
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            cum.append(cum[-1] + math.dist(a, b))
        self._cum = cum
        self.stop_line_s = self._find_stop_line()

    @property
    def path_length(self) -> float:
        return self._cum[-1]

    @property
    def is_turning(self) -> bool:
        """Xe rẽ (tuyến có đoạn bo cung) — cần nhường dòng cắt trước khi vào giao lộ."""
        return len(self.waypoints) > 2

    def _find_stop_line(self) -> float:
        """Quãng đường tới vạch dừng — nơi tuyến đường chạm mép vùng giao lộ."""
        total = self._cum[-1]
        if total < 1e-6:
            return 0.0
        steps = 200
        for i in range(steps + 1):
            s = total * i / steps
            pos, _ = self.point_at(s)
            if math.dist(pos, CENTER) <= STOP_LINE_RADIUS:
                return s
        return total  # tuyến không đi qua giao lộ

    # -- Hình học tuyến đường -------------------------------------------- #
    def point_at(self, s: float) -> tuple[tuple[float, float], float]:
        """Vị trí và hướng tại quãng đường ``s`` dọc tuyến."""
        s = max(0.0, min(s, self._cum[-1]))
        idx = int(np.searchsorted(self._cum, s, side="right")) - 1
        idx = max(0, min(idx, len(self.waypoints) - 2))
        p0, p1 = self.waypoints[idx], self.waypoints[idx + 1]
        seg = self._cum[idx + 1] - self._cum[idx]
        frac = 0.0 if seg < 1e-9 else (s - self._cum[idx]) / seg
        pos = (p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac)
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return pos, heading

    # -- Truy vấn quỹ đạo sau mô phỏng ----------------------------------- #
    def pose_at(self, t: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """``((x, y), (vx, vy))`` tại thời điểm ``t``, hoặc None nếu chưa/đã hết."""
        if self.times.size == 0:
            return None
        if t < self.times[0] or t > self.times[-1]:
            return None
        i = int(np.searchsorted(self.times, t))
        i = max(0, min(i, len(self.times) - 1))
        return (
            (float(self.positions[i, 0]), float(self.positions[i, 1])),
            (float(self.velocities[i, 0]), float(self.velocities[i, 1])),
        )


# --------------------------------------------------------------------------- #
# Sinh kịch bản
# --------------------------------------------------------------------------- #
#: Bốn nhánh: tên → (điểm vào, điểm ra khi đi thẳng, nhóm pha đèn).
_APPROACHES = {
    "Hướng 1 (Bắc→Nam)": ((-4.5, 60.0), (-4.5, 0.0), "NS"),
    "Hướng 2 (Nam→Bắc)": ((4.5, 0.0), (4.5, 60.0), "NS"),
    "Hướng 3 (Đông→Tây)": ((30.0, 34.5), (-30.0, 34.5), "EW"),
    "Hướng 4 (Tây→Đông)": ((-30.0, 25.5), (30.0, 25.5), "EW"),
}

#: Bốn lối đi bộ qua đường, ngay sát mép giao lộ.
_CROSSWALKS = {
    "Vạch bộ hành Bắc": ((-10.5, 41.0), (10.5, 41.0)),
    "Vạch bộ hành Nam": ((10.5, 19.0), (-10.5, 19.0)),
    "Vạch bộ hành Đông": ((11.0, 40.5), (11.0, 19.5)),
    "Vạch bộ hành Tây": ((-11.0, 19.5), (-11.0, 40.5)),
}

#: Phân bố lớp phương tiện tại giao lộ Việt Nam — xe máy áp đảo.
#: Người đi bộ không nằm trong bảng này vì họ đi trên vạch bộ hành riêng.
_CLASS_WEIGHTS = [
    (VehicleClass.MOTORCYCLE, 0.68),
    (VehicleClass.CAR, 0.22),
    (VehicleClass.TRUCK, 0.05),
    (VehicleClass.BICYCLE, 0.05),
]

#: Tốc độ mong muốn (m/s) theo lớp.
_SPEED_RANGE = {
    VehicleClass.MOTORCYCLE: (7.5, 12.0),
    VehicleClass.CAR: (7.0, 11.0),
    VehicleClass.TRUCK: (5.5, 8.5),
    VehicleClass.BICYCLE: (3.2, 5.0),
    VehicleClass.PEDESTRIAN: (1.1, 1.7),
}

#: Số làn mỗi chiều — dùng để rải xe theo làn thay vì xếp một hàng.
_LANE_OFFSETS = (-1.7, 0.0, 1.7)


def _turn_path(entry: tuple[float, float], exit_pt: tuple[float, float]) -> list[tuple[float, float]]:
    """Quỹ đạo rẽ: thẳng vào giao lộ → bo cung Bézier → thẳng ra."""
    cx, cy = CENTER

    def toward(origin, target, dist):
        vx, vy = target[0] - origin[0], target[1] - origin[1]
        norm = math.hypot(vx, vy) or 1.0
        return (origin[0] + vx / norm * dist, origin[1] + vy / norm * dist)

    start_curve = toward(CENTER, entry, 11.0)
    end_curve = toward(CENTER, exit_pt, 11.0)

    pts = [entry, start_curve]
    for i in range(1, 8):
        u = i / 8.0
        x = (1 - u) ** 2 * start_curve[0] + 2 * u * (1 - u) * cx + u**2 * end_curve[0]
        y = (1 - u) ** 2 * start_curve[1] + 2 * u * (1 - u) * cy + u**2 * end_curve[1]
        pts.append((x, y))
    pts.extend([end_curve, exit_pt])
    return pts


def _lane_shift(
    entry: tuple[float, float], exit_pt: tuple[float, float], offset: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Dịch tuyến đường sang ngang ``offset`` mét (đổi làn)."""
    dx, dy = exit_pt[0] - entry[0], exit_pt[1] - entry[1]
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm   # vector pháp tuyến
    return (
        (entry[0] + nx * offset, entry[1] + ny * offset),
        (exit_pt[0] + nx * offset, exit_pt[1] + ny * offset),
    )


def _pick_vehicle_class(rng: np.random.Generator) -> VehicleClass:
    """Chọn lớp phương tiện (không gồm người đi bộ — họ đi trên vạch riêng)."""
    r = rng.random()
    acc = 0.0
    for cls, w in _CLASS_WEIGHTS:
        acc += w
        if r <= acc:
            return cls
    return VehicleClass.MOTORCYCLE


def build_scenario(
    duration: float = 120.0,
    seed: int = 42,
    arrival_rate: float = 1.1,
    reckless_ratio: float = 0.08,
    turn_ratio: float = 0.26,
    pedestrian_rate: float = 0.10,
) -> list[VehicleSpec]:
    """Sinh danh sách phương tiện cho một kịch bản giao lộ.

    Tham số:
        duration: độ dài kịch bản (giây).
        seed: hạt giống ngẫu nhiên — cùng seed cho kịch bản y hệt (tái lập được).
        arrival_rate: số xe/giây (tiến trình Poisson).
        reckless_ratio: tỉ lệ người lái **vượt đèn đỏ**. Đây là tham số điều
            khiển chính tần suất xung đột cắt ngang. 0.08 tương ứng một giao lộ
            đông, ý thức chấp hành trung bình.
        turn_ratio: tỉ lệ xe rẽ sang nhánh khác (rẽ cắt dòng ngược chiều).
        pedestrian_rate: số người đi bộ băng qua đường mỗi giây.
    """
    rng = np.random.default_rng(seed)
    vehicles: list[VehicleSpec] = []
    vid = 1
    names = list(_APPROACHES)

    # --- Dòng phương tiện ------------------------------------------------ #
    t = 0.0
    while t < duration:
        t += float(rng.exponential(1.0 / arrival_rate))
        if t >= duration:
            break

        approach = names[int(rng.integers(len(names)))]
        entry, straight_exit, group = _APPROACHES[approach]
        cls = _pick_vehicle_class(rng)
        lo, hi = _SPEED_RANGE[cls]
        speed = float(rng.uniform(lo, hi))

        # Chọn làn — rải xe ngang thay vì xếp một hàng dọc.
        offset = float(_LANE_OFFSETS[int(rng.integers(len(_LANE_OFFSETS)))])
        offset += float(rng.uniform(-0.25, 0.25))
        e, x_out = _lane_shift(entry, straight_exit, offset)

        if rng.random() < turn_ratio:
            others = [n for n in names if n != approach]
            other = others[int(rng.integers(len(others)))]
            _, other_exit, _ = _APPROACHES[other]
            waypoints = _turn_path(e, other_exit)
        else:
            waypoints = [e, x_out]

        forced_brake = None
        if rng.random() < 0.08:
            t_brake = t + float(rng.uniform(2.5, 5.0))
            forced_brake = (t_brake, t_brake + 1.4)

        vehicles.append(
            VehicleSpec(
                vid=vid, cls=cls, waypoints=waypoints, t_start=t,
                desired_speed=speed, approach=approach, signal_group=group,
                runs_red=bool(rng.random() < reckless_ratio),
                forced_brake=forced_brake,
            )
        )
        vid += 1

    # --- Người đi bộ băng qua đường -------------------------------------- #
    walk_names = list(_CROSSWALKS)
    t = 0.0
    while t < duration:
        t += float(rng.exponential(1.0 / max(pedestrian_rate, 1e-6)))
        if t >= duration:
            break
        name = walk_names[int(rng.integers(len(walk_names)))]
        a, b = _CROSSWALKS[name]
        if rng.random() < 0.5:
            a, b = b, a
        jitter = float(rng.uniform(-1.2, 1.2))
        a = (a[0] + jitter, a[1] + jitter * 0.2)
        b = (b[0] + jitter, b[1] + jitter * 0.2)
        lo, hi = _SPEED_RANGE[VehicleClass.PEDESTRIAN]

        vehicles.append(
            VehicleSpec(
                vid=vid, cls=VehicleClass.PEDESTRIAN, waypoints=[a, b], t_start=t,
                desired_speed=float(rng.uniform(lo, hi)), approach=name,
                signal_group="PED",
                # Người đi bộ băng ẩu là nguồn xung đột với người đi bộ.
                runs_red=bool(rng.random() < 0.35),
            )
        )
        vid += 1

    vehicles.sort(key=lambda v: v.t_start)
    return vehicles


# --------------------------------------------------------------------------- #
# Vi mô phỏng
# --------------------------------------------------------------------------- #
def simulate(
    vehicles: list[VehicleSpec],
    duration: float,
    dt: float = 1.0 / 60.0,
    signal: TrafficSignal | None = None,
    emergency_ttc: float = 1.2,
) -> list[VehicleSpec]:
    """Chạy vi mô phỏng, điền quỹ đạo thực tế vào từng ``VehicleSpec``.

    Mỗi bước thời gian, gia tốc của mỗi xe là **giá trị nhỏ nhất** trong bốn
    thành phần (tức là ràng buộc chặt nhất thắng):

    1. **IDM tự do + bám xe trước** — giữ tốc độ mong muốn, giảm tốc khi tới gần
       xe phía trước cùng làn.
    2. **Đèn tín hiệu** — khi đèn đỏ/vàng, vạch dừng đóng vai trò một "xe đứng
       yên" trong công thức IDM, nên xe dừng lại mượt mà. Xe ``runs_red`` bỏ qua
       ràng buộc này.
    3. **Phanh tránh khẩn cấp** — TTC với bất kỳ xe nào tụt dưới ``emergency_ttc``
       thì phanh gấp. Chỉ kích hoạt khi thật sự sắp va chạm nên không gây tắc.
    4. **Phanh gấp chủ động** — tạo nhãn cho module Behavior.

    Toàn bộ phép tính theo cặp được vector hoá bằng numpy: với ~40 xe hoạt động
    và 7200 bước thời gian, vòng lặp Python thuần sẽ mất hàng phút.

    Trả về danh sách các xe thực sự có quỹ đạo (đã cập nhật tại chỗ).
    """
    signal = signal or TrafficSignal()
    n_steps = int(round(duration / dt)) + 1

    state = {
        v.vid: {"s": 0.0, "v": 0.0, "active": False, "done": False, "wait": 0.0,
                "log_t": [], "log_p": [], "log_v": []}
        for v in vehicles
    }
    pending = sorted(vehicles, key=lambda v: v.t_start)
    next_spawn = 0
    #: Xe đã tới giờ nhưng chưa có chỗ trống ở điểm vào — chờ ở đây.
    waiting: list[VehicleSpec] = []

    for step in range(n_steps):
        now = step * dt

        while next_spawn < len(pending) and pending[next_spawn].t_start <= now:
            waiting.append(pending[next_spawn])
            next_spawn += 1

        active = [v for v in vehicles if state[v.vid]["active"] and not state[v.vid]["done"]]

        # --- Kết nạp xe đang chờ, nếu điểm vào đủ trống ----------------- #
        # Không có bước này thì hai xe cùng làn có thể xuất hiện chồng lên nhau
        # và khoá cứng lối vào — cả hàng phía sau đứng im vĩnh viễn.
        if waiting:
            occupied = [
                (veh, state[veh.vid]["s"]) for veh in active
            ]
            still_waiting: list[VehicleSpec] = []
            for veh in waiting:
                entry = veh.waypoints[0]
                need = veh.cls.footprint[0] + 4.0
                clear = True
                for other, _s in occupied:
                    op, _h = other.point_at(state[other.vid]["s"])
                    if math.dist(op, entry) < max(need, other.cls.footprint[0] + 4.0):
                        clear = False
                        break
                if clear:
                    state[veh.vid]["active"] = True
                    state[veh.vid]["v"] = veh.desired_speed * 0.9
                    active.append(veh)
                    occupied.append((veh, 0.0))
                elif now - veh.t_start < 25.0:
                    still_waiting.append(veh)   # thử lại ở bước sau
                # Quá 25 s không vào được thì bỏ — tránh dồn ứ vô hạn.
            waiting = still_waiting

        n = len(active)
        if n == 0:
            continue

        # --- Ảnh chụp trạng thái, dạng mảng ---------------------------- #
        pos = np.empty((n, 2))
        heading = np.empty(n)
        speed = np.empty(n)
        half_len = np.empty(n)
        half_wid = np.empty(n)
        radius = np.empty(n)
        for i, veh in enumerate(active):
            st = state[veh.vid]
            p, h = veh.point_at(st["s"])
            pos[i] = p
            heading[i] = h
            speed[i] = st["v"]
            length, width = veh.cls.footprint
            half_len[i] = length / 2.0
            half_wid[i] = width / 2.0
            radius[i] = veh.cls.radius

        dir_vec = np.stack([np.cos(heading), np.sin(heading)], axis=1)
        vel = dir_vec * speed[:, None]

        # --- Ma trận quan hệ theo cặp ---------------------------------- #
        # delta[i, j] = vị trí của j so với i.
        delta = pos[None, :, :] - pos[:, None, :]
        longitudinal = np.einsum("ijk,ik->ij", delta, dir_vec)
        lateral = np.abs(
            delta[:, :, 1] * dir_vec[:, None, 0] - delta[:, :, 0] * dir_vec[:, None, 1]
        )
        np.fill_diagonal(longitudinal, -1.0)

        # Góc giữa hai hướng đi: cos ≈ 1 ⇒ cùng chiều; ≈ 0 ⇒ cắt ngang; < 0 ⇒ ngược chiều.
        cos_dir = dir_vec @ dir_vec.T

        # --- 1. Bám xe trước ------------------------------------------- #
        # Điều kiện ``cos_dir > 0.7`` (lệch hướng dưới ~45°) là bắt buộc: nếu
        # thiếu nó, một xe đang cắt ngang trước mũi xe ta sẽ bị coi là "xe phía
        # trước cùng làn" và IDM sẽ hãm ta lại phía sau nó. Hai xe cắt nhau
        # trong giao lộ khi đó cùng dừng và cùng chờ nhau — nút giao khoá chết
        # vĩnh viễn. Xe cắt ngang không phải xe dẫn đầu; đó là xung đột, và do
        # cơ chế phanh tránh khẩn cấp bên dưới xử lý.
        lane_width = half_wid[:, None] + half_wid[None, :] + 0.35
        is_leader = (
            (longitudinal > 0)
            & (longitudinal < 45.0)
            & (lateral <= lane_width)
            & (cos_dir > 0.7)
        )
        gaps = np.where(
            is_leader, longitudinal - (half_len[:, None] + half_len[None, :]), np.inf
        )
        gaps = np.maximum(gaps, 0.30)
        lead_idx = np.argmin(gaps, axis=1)
        lead_gap = gaps[np.arange(n), lead_idx]
        lead_speed = np.where(np.isfinite(lead_gap), speed[lead_idx], 0.0)

        # --- 3. TTC theo cặp cho phanh tránh --------------------------- #
        # CHỈ xét các cặp **không** cùng làn. Xe phía trước cùng làn đã do IDM
        # xử lý ở trên; nếu tính cả nhóm đó vào đây thì khoảng cách bám đuôi
        # bình thường (~0.9 m với xe máy) sẽ luôn nhỏ hơn bán kính an toàn và
        # mọi xe trong hàng chờ sẽ phanh gấp lẫn nhau đến mức đứng im.
        dv = vel[None, :, :] - vel[:, None, :]
        aa = np.einsum("ijk,ijk->ij", dv, dv)
        bb = 2.0 * np.einsum("ijk,ijk->ij", delta, dv)
        dist_sq = np.einsum("ijk,ijk->ij", delta, delta)
        rr = (radius[:, None] + radius[None, :]) ** 2
        cc = dist_sq - rr
        disc = bb**2 - 4.0 * aa * cc

        with np.errstate(invalid="ignore", divide="ignore"):
            root = (-bb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * aa)
        closing = np.sqrt(aa)
        ok = (
            (~is_leader)                    # bỏ qua xe cùng làn phía trước
            & (aa > 1e-6)
            & (bb < 0)
            & (disc >= 0)
            & (root >= 0)
            & (root <= emergency_ttc)
            & (closing > 1.0)               # phải có tốc độ tiếp cận thật
        )
        np.fill_diagonal(ok, False)
        ttc_pair = np.where(ok, root, np.inf)
        min_ttc = ttc_pair.min(axis=1)

        # --- Trạng thái vùng giao lộ ----------------------------------- #
        dist_centre = np.linalg.norm(pos - np.asarray(CENTER), axis=1)
        in_junction = dist_centre < JUNCTION_RADIUS
        # ETA tới tâm giao lộ, dùng cho luật nhường của xe rẽ.
        eta_centre = dist_centre / np.maximum(speed, 0.5)

        # --- Gộp các ràng buộc và tích phân ---------------------------- #
        for i, veh in enumerate(active):
            st = state[veh.vid]
            cls = veh.cls
            a_max = IDM_A_MAX[cls]
            b_comf = IDM_B_COMFORT[cls]
            s0 = IDM_S0[cls]
            headway = IDM_T[cls]
            v_now = speed[i]

            free = 1.0 - (v_now / max(veh.desired_speed, 0.1)) ** 4
            accel = a_max * free

            def idm_brake(gap: float, lead_v: float) -> float:
                """Số hạng tương tác IDM với một vật cản phía trước."""
                delta_v = v_now - lead_v
                s_star = s0 + max(
                    0.0,
                    v_now * headway + v_now * delta_v / (2.0 * math.sqrt(a_max * b_comf)),
                )
                return a_max * (free - (s_star / max(gap, 0.30)) ** 2)

            # 1. Xe phía trước.
            if np.isfinite(lead_gap[i]):
                accel = min(accel, idm_brake(float(lead_gap[i]), float(lead_speed[i])))

            # 2. Đèn tín hiệu — vạch dừng như một vật cản đứng yên.
            if veh.signal_group in ("NS", "EW") and not veh.runs_red:
                phase = signal.phase(now, veh.signal_group)
                dist_to_stop = veh.stop_line_s - st["s"]
                if phase != "green" and 0.0 < dist_to_stop < 60.0:
                    # Vùng lưỡng lự: quá gần vạch khi đèn vàng thì đi tiếp cho an toàn.
                    braking_distance = v_now**2 / (2.0 * b_comf)
                    if not (phase == "yellow" and dist_to_stop < braking_distance):
                        accel = min(accel, idm_brake(max(dist_to_stop, 0.30), 0.0))
            elif veh.signal_group == "PED" and not veh.runs_red:
                # Người đi bộ chờ khi dòng xe cắt ngang đang được đi.
                # Vạch bộ hành Bắc/Nam cắt qua trục Bắc-Nam ⇒ xung đột với nhóm NS.
                cross_group = "NS" if ("Bắc" in veh.approach or "Nam" in veh.approach) else "EW"
                dist_to_stop = veh.stop_line_s - st["s"]
                if signal.phase(now, cross_group) == "green" and 0.0 < dist_to_stop < 25.0:
                    accel = min(accel, idm_brake(max(dist_to_stop, 0.30), 0.0))

            # 3. Xe rẽ nhường dòng cắt — quyết định TRƯỚC vạch dừng.
            # Người lái rẽ chờ ở vạch dừng cho tới khi thấy khoảng trống, rồi
            # mới "cắm đầu" đi qua. Nếu để họ dừng giữa giao lộ thì cả nút bị
            # khoá và dòng phía sau tắc dây chuyền — đó chính là lỗi phải tránh.
            if veh.is_turning and not in_junction[i]:
                dist_to_stop = veh.stop_line_s - st["s"]
                if 0.0 < dist_to_stop < 22.0 and st["wait"] < TURN_PATIENCE:
                    blocked = False
                    for j in range(n):
                        if j == i:
                            continue
                        # Chỉ quan tâm dòng cắt ngang hoặc ngược chiều.
                        if cos_dir[i, j] > 0.5:
                            continue
                        if speed[j] < 1.0:
                            continue
                        if in_junction[j] or eta_centre[j] < TURN_GAP:
                            blocked = True
                            break
                    if blocked:
                        accel = min(accel, idm_brake(max(dist_to_stop, 0.30), 0.0))
                        st["wait"] += dt
                    else:
                        st["wait"] = 0.0

            # 4. Phanh tránh khẩn cấp.
            # Bên trong giao lộ, giới hạn lực phanh: người lái đã lỡ vào thì cố
            # thoát ra chứ không đứng chết giữa nút.
            if np.isfinite(min_ttc[i]):
                cap = 4.0 if in_junction[i] else 8.0
                accel = min(accel, -min(cap, 3.0 / max(float(min_ttc[i]), 0.25)))

            # 5. Phanh gấp chủ động (nhãn hành vi).
            if veh.forced_brake and veh.forced_brake[0] <= now <= veh.forced_brake[1]:
                accel = min(accel, -4.2)

            accel = float(np.clip(accel, -8.5, a_max))

            new_v = max(0.0, v_now + accel * dt)
            st["v"] = new_v
            st["s"] += new_v * dt

            if st["s"] >= veh.path_length:
                st["done"] = True
                continue

            p, h = veh.point_at(st["s"])
            st["log_t"].append(now)
            st["log_p"].append(p)
            st["log_v"].append((new_v * math.cos(h), new_v * math.sin(h)))

    out: list[VehicleSpec] = []
    for veh in vehicles:
        st = state[veh.vid]
        if len(st["log_t"]) < 5:
            continue
        veh.times = np.asarray(st["log_t"], dtype=float)
        veh.positions = np.asarray(st["log_p"], dtype=float)
        veh.velocities = np.asarray(st["log_v"], dtype=float)
        out.append(veh)
    return out


# --------------------------------------------------------------------------- #
# Sinh nhãn chuẩn (oracle)
# --------------------------------------------------------------------------- #
def compute_ground_truth(
    vehicles: list[VehicleSpec],
    ttc_threshold: float = 3.0,
    pet_threshold: float = 1.5,
    proximity: float = 2.0,
    min_closing_speed: float = 2.5,
    min_speed: float = 0.8,
    horizon: float = 10.0,
    region: tuple[float, float, float, float] | None = None,
) -> list[GroundTruthConflict]:
    """Tính nhãn near-miss chuẩn từ quỹ đạo đã mô phỏng.

    Định nghĩa near-miss (bám theo Traffic Conflict Technique)
    ---------------------------------------------------------
    Một cặp ``(a, b)`` là near-miss khi thoả **đồng thời** cả ba điều kiện:

    1. **Thực sự đến gần nhau** — khoảng cách mặt-tới-mặt nhỏ nhất trong cả quá
       trình < ``proximity`` mét.
    2. **Có nguy cơ va chạm** — cực tiểu TTC < ``ttc_threshold``, hoặc PET tại
       điểm gặp < ``pet_threshold``.
    3. **Có tốc độ tiếp cận thật** — vận tốc tương đối tại thời điểm căng thẳng
       nhất > ``min_closing_speed`` m/s.

    Điều kiện 1 và 3 là thứ phân biệt near-miss với **bám đuôi bình thường**.
    Chỉ dùng TTC là không đủ: hai xe nối đuôi nhau cách 1 m ở cùng tốc độ có TTC
    rất nhỏ theo mô hình hình tròn, nhưng đó là dòng xe bình thường chứ không
    phải tình huống nguy hiểm. Ngược lại, một xe phanh gấp phía sau xe khác với
    chênh lệch tốc độ 4 m/s thì đúng là xung đột.

    Khác biệt then chốt so với thuật toán online:

    * lấy mẫu ở tần số mô phỏng (60 Hz) thay vì tần số video (30 Hz);
    * dùng vận tốc **giải tích** thay vì ước lượng từ chuỗi bbox nhiễu;
    * lấy **cực tiểu toàn cục** trên cả quãng gặp nhau, thay vì quyết định tại
      từng frame riêng lẻ.

    Nhờ đó đây là nhãn độc lập, không phải thuật toán tự chấm điểm cho chính nó.

    ``region`` giới hạn nhãn trong vùng camera thực sự bao phủ, dạng
    ``(x_min, y_min, x_max, y_max)``. Xung đột xảy ra ngoài khung hình không thể
    quy trách nhiệm cho hệ thống, nên đưa chúng vào mẫu số của Recall sẽ cho ra
    một con số vô nghĩa. Giới hạn phạm vi đánh giá theo vùng cảm biến là thực
    hành chuẩn trong các benchmark thị giác.
    """
    results: list[GroundTruthConflict] = []
    usable = [v for v in vehicles if v.times.size > 5]

    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            va, vb = usable[i], usable[j]

            # Khoảng thời gian cùng xuất hiện.
            t0 = max(va.times[0], vb.times[0])
            t1 = min(va.times[-1], vb.times[-1])
            if t1 - t0 < 0.15:
                continue

            # Lấy mẫu chung trên lưới thời gian của xe a.
            mask = (va.times >= t0) & (va.times <= t1)
            ts = va.times[mask]
            if ts.size < 5:
                continue

            pa = va.positions[mask]
            wa = va.velocities[mask]
            idx_b = np.searchsorted(vb.times, ts)
            idx_b = np.clip(idx_b, 0, len(vb.times) - 1)
            pb = vb.positions[idx_b]
            wb = vb.velocities[idx_b]

            # Loại các cặp mà cả hai gần như đứng yên.
            speeds = np.maximum(
                np.linalg.norm(wa, axis=1), np.linalg.norm(wb, axis=1)
            )
            if speeds.max() < min_speed:
                continue

            dv = wb - wa
            # TTC và khoảng cách theo mô hình đa hình tròn, vector hoá trên
            # toàn chuỗi thời gian.
            ttc_series, gaps = series_ttc_and_gap(
                pa, wa, va.cls, pb, wb, vb.cls, horizon=horizon
            )

            best_ttc = float(ttc_series.min())
            best_idx = int(np.argmin(ttc_series))
            min_gap = float(gaps.min())
            gap_idx = int(np.argmin(gaps))

            # --- Điều kiện 1: thực sự đến gần nhau ---------------------- #
            if min_gap > proximity:
                continue

            # PET thật tại điểm hai xe đi gần nhau nhất.
            #
            # Chỉ có nghĩa với hai quỹ đạo **thực sự cắt nhau**: với hai xe đi
            # gần song song (bám đuôi, cùng làn), "điểm hai xe gần nhau nhất"
            # nằm dọc theo cả đoạn đường và |t_a − t_b| tại đó chỉ phản ánh
            # khoảng cách bám đuôi, không phải mức nguy hiểm. Áp dụng cùng
            # ngưỡng góc mà thuật toán online dùng, để hai bên đo cùng một đại
            # lượng và sai số PET so sánh được.
            sa = float(np.linalg.norm(wa[gap_idx]))
            sb = float(np.linalg.norm(wb[gap_idx]))
            pet_val = None
            if sa > 1e-3 and sb > 1e-3:
                cos_t = float(np.dot(wa[gap_idx], wb[gap_idx])) / (sa * sb)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))
                if angle >= 20.0:
                    mid = 0.5 * (pa[gap_idx] + pb[gap_idx])
                    ia = int(np.argmin(np.linalg.norm(va.positions - mid, axis=1)))
                    ib = int(np.argmin(np.linalg.norm(vb.positions - mid, axis=1)))
                    pet_val = abs(float(va.times[ia]) - float(vb.times[ib]))

            # --- Điều kiện 2: có nguy cơ va chạm ------------------------ #
            hit_ttc = best_ttc < ttc_threshold
            hit_pet = pet_val is not None and pet_val < pet_threshold
            if not (hit_ttc or hit_pet):
                continue

            idx = best_idx if np.isfinite(best_ttc) else gap_idx

            # --- Điều kiện 3: có tốc độ tiếp cận thật ------------------- #
            # Loại bỏ bám đuôi bình thường: cùng tốc độ, khoảng cách nhỏ nhưng
            # ổn định thì không phải xung đột.
            closing = float(np.linalg.norm(dv[idx]))
            if closing < min_closing_speed:
                continue

            # --- Điều kiện 4: nằm trong vùng camera bao phủ -------------- #
            location = 0.5 * (pa[idx] + pb[idx])
            if region is not None:
                x_min, y_min, x_max, y_max = region
                if not (x_min <= location[0] <= x_max and y_min <= location[1] <= y_max):
                    continue
            results.append(
                GroundTruthConflict(
                    t=float(ts[idx]),
                    track_a=va.vid,
                    track_b=vb.vid,
                    ttc=round(best_ttc, 3) if np.isfinite(best_ttc) else float("inf"),
                    pet=None if pet_val is None else round(pet_val, 3),
                    conflict_type=_gt_type(va, vb, wa[idx], wb[idx]),
                    source="synthetic-oracle",
                )
            )

    results.sort(key=lambda g: g.t)
    return results


def _gt_type(va, vb, vel_a, vel_b) -> ConflictType:
    """Kiểu xung đột chuẩn, suy từ vận tốc giải tích tại thời điểm căng nhất."""
    if VehicleClass.PEDESTRIAN in (va.cls, vb.cls):
        return ConflictType.PEDESTRIAN

    sa = float(np.linalg.norm(vel_a))
    sb = float(np.linalg.norm(vel_b))
    if sa < 1e-3 or sb < 1e-3:
        return ConflictType.CROSSING

    cos_t = float(np.dot(vel_a, vel_b)) / (sa * sb)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

    if angle >= 150:
        return ConflictType.HEAD_ON
    if 60 <= angle <= 120:
        return ConflictType.CROSSING
    if angle < 30:
        return ConflictType.REAR_END
    return ConflictType.LANE_CHANGE


def build_and_simulate(
    duration: float = 120.0, seed: int = 42, dt: float = 1.0 / 60.0, **kwargs
) -> tuple[list[VehicleSpec], list[GroundTruthConflict]]:
    """Tiện ích: sinh kịch bản → mô phỏng → tính nhãn chuẩn, trong một lệnh."""
    vehicles = build_scenario(duration=duration, seed=seed, **kwargs)
    vehicles = simulate(vehicles, duration, dt=dt)
    gt = compute_ground_truth(vehicles)
    return vehicles, gt
