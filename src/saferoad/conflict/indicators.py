"""Chỉ số xung đột giao thông: TTC và PET.

Đây là hai chỉ số thay thế (surrogate safety measures) được dùng phổ biến nhất
trong nghiên cứu an toàn giao thông, cho phép đo mức nguy hiểm mà **không cần
chờ tai nạn xảy ra**.

TTC — Time To Collision
-----------------------
Thời gian còn lại đến va chạm nếu cả hai đối tượng giữ nguyên vận tốc hiện tại.
Ta xấp xỉ mỗi đối tượng bằng một hình tròn bán kính :math:`r` (suy ra từ kích
thước xe), nên va chạm xảy ra khi khoảng cách tâm bằng :math:`R = r_a + r_b`.

Đặt :math:`\\Delta p = p_b - p_a` và :math:`\\Delta v = v_b - v_a`, điều kiện va
chạm tại thời điểm :math:`t` là:

.. math::
    \\|\\Delta p + t\\,\\Delta v\\| = R

Bình phương hai vế cho phương trình bậc hai theo :math:`t`:

.. math::
    \\|\\Delta v\\|^2 t^2 + 2(\\Delta p \\cdot \\Delta v)\\,t + (\\|\\Delta p\\|^2 - R^2) = 0

TTC là **nghiệm dương nhỏ nhất**. Không có nghiệm dương ⇒ hai đối tượng đang
tách xa nhau hoặc sẽ đi lướt qua nhau ⇒ không có xung đột.

PET — Post-Encroachment Time
----------------------------
Khoảng thời gian giữa lúc đối tượng thứ nhất **rời khỏi** điểm xung đột và lúc
đối tượng thứ hai **đi tới** chính điểm đó. Khác với TTC, PET vẫn đo được cả khi
hai xe không bao giờ có nguy cơ va chạm tức thời — nó bắt được tình huống "suýt"
mà TTC bỏ sót (ví dụ: xe A vừa qua, 0.8 s sau xe B mới tới đúng chỗ đó).

Tham khảo:
* Hayward (1972) — khái niệm TTC.
* Allen, Shin & Cooper (1978) — PET và conflict analysis.
* Zheng, Ismail & Meng (2014), "Traffic conflict techniques for road safety
  analysis", Canadian Journal of Civil Engineering.
"""

from __future__ import annotations

import math

from ..types import ConflictType, TrackState, VehicleClass
from .geometry import NO_COLLISION, vehicle_gap, vehicle_ttc


def time_to_collision(
    state_a: TrackState,
    state_b: TrackState,
    cls_a: VehicleClass,
    cls_b: VehicleClass,
    max_horizon: float = 10.0,
) -> float:
    """TTC (giây) giữa hai đối tượng, hoặc ``inf`` nếu không hội tụ.

    Dùng mô hình **đa hình tròn** (xem :mod:`saferoad.conflict.geometry`): thân
    xe được phủ bằng 1-3 hình tròn nhỏ dọc trục thay vì một hình tròn ngoại tiếp.
    Nếu dùng hình tròn ngoại tiếp, một ô tô 1.8 m ngang sẽ bị mô hình hoá như
    vật thể rộng 4.8 m và hai làn ngược chiều cạnh nhau sẽ luôn bị báo "đối đầu".

    Trả về ``0.0`` khi hai thân xe đã chồng lấn tại thời điểm hiện tại.
    """
    return vehicle_ttc(
        state_a.ground, state_a.velocity, cls_a,
        state_b.ground, state_b.velocity, cls_b,
        horizon=max_horizon,
    )


def conflict_point(
    state_a: TrackState, state_b: TrackState
) -> tuple[tuple[float, float], float, float] | None:
    """Giao điểm của hai tia quỹ đạo, kèm thời gian tới của mỗi đối tượng.

    Trả về ``((x, y), t_a, t_b)`` với ``t_a``, ``t_b`` là thời gian (giây) để A và
    B đi tới giao điểm. Trả về ``None`` khi hai quỹ đạo song song hoặc giao điểm
    nằm phía sau (đã đi qua rồi).

    Giải hệ::

        p_a + t_a * v_a = p_b + t_b * v_b
    """
    ax, ay = state_a.ground
    bx, by = state_b.ground
    avx, avy = state_a.velocity
    bvx, bvy = state_b.velocity

    # Định thức của hệ 2x2. Gần 0 ⇒ hai hướng song song.
    det = avx * (-bvy) - avy * (-bvx)
    if abs(det) < 1e-9:
        return None

    dx = bx - ax
    dy = by - ay
    t_a = (dx * (-bvy) - dy * (-bvx)) / det
    t_b = (avx * dy - avy * dx) / det

    # Giao điểm phải nằm phía TRƯỚC cả hai (thời gian không âm).
    if t_a < 0 or t_b < 0:
        return None

    return ((ax + avx * t_a, ay + avy * t_a), t_a, t_b)


def post_encroachment_time(
    state_a: TrackState,
    state_b: TrackState,
    cls_a: VehicleClass,
    cls_b: VehicleClass,
    max_horizon: float = 10.0,
    min_crossing_angle: float = 20.0,
    max_conflict_distance: float = 40.0,
) -> tuple[float, tuple[float, float]] | None:
    """PET (giây) và toạ độ điểm xung đột, hoặc ``None`` nếu không xác định.

    PET = ``|t_a - t_b|`` tại giao điểm quỹ đạo. Giá trị nhỏ nghĩa là hai đối
    tượng đi qua cùng một điểm cách nhau rất ngắn — đúng định nghĩa "suýt va chạm".

    Ta trừ đi thời gian mỗi xe cần để **giải phóng** điểm xung đột (chiều dài xe
    chia cho tốc độ), vì PET đo từ lúc đuôi xe trước rời điểm đó.

    Hai điều kiện bảo vệ quan trọng
    -------------------------------
    * ``min_crossing_angle`` — PET chỉ có nghĩa với hai quỹ đạo **thực sự cắt
      nhau**. Với hai xe đi gần song song (ví dụ xe máy vượt nhau cùng làn), giao
      điểm của hai tia nằm rất xa và cực kỳ nhạy với nhiễu: lệch hướng 1° đã dời
      giao điểm hàng chục mét, sinh ra một giá trị PET nhỏ hoàn toàn giả. Trường
      hợp đó phải để TTC xử lý, không phải PET.
    * ``max_conflict_distance`` — giao điểm nằm quá xa thì tình huống chưa xảy
      ra, và ngoại suy tuyến tính tới đó không còn đáng tin.
    """
    if approach_angle(state_a, state_b) < min_crossing_angle:
        return None

    result = conflict_point(state_a, state_b)
    if result is None:
        return None

    point, t_a, t_b = result
    if t_a > max_horizon or t_b > max_horizon:
        return None
    if (
        math.dist(point, state_a.ground) > max_conflict_distance
        or math.dist(point, state_b.ground) > max_conflict_distance
    ):
        return None

    speed_a = state_a.speed
    speed_b = state_b.speed
    if speed_a < 1e-3 or speed_b < 1e-3:
        return None

    # Thời gian chiếm dụng điểm xung đột = chiều dài xe / tốc độ.
    occupancy_a = cls_a.footprint[0] / speed_a
    occupancy_b = cls_b.footprint[0] / speed_b

    gap = abs(t_a - t_b)
    # Trừ đi phần chiếm dụng của xe tới trước — nếu âm nghĩa là hai xe chồng
    # lấn thời gian tại điểm đó (PET = 0, cực kỳ nguy hiểm).
    lead_occupancy = occupancy_a if t_a < t_b else occupancy_b
    pet = max(0.0, gap - lead_occupancy)
    return (pet, point)


def relative_speed(state_a: TrackState, state_b: TrackState) -> float:
    """Độ lớn vận tốc tương đối (m/s)."""
    return math.hypot(
        state_b.velocity[0] - state_a.velocity[0],
        state_b.velocity[1] - state_a.velocity[1],
    )


def approach_angle(state_a: TrackState, state_b: TrackState) -> float:
    """Góc (độ, trong [0, 180]) giữa hai vector vận tốc.

    0° = cùng hướng, 90° = cắt vuông góc, 180° = đối đầu.
    """
    if state_a.speed < 1e-3 or state_b.speed < 1e-3:
        return 0.0
    dot = (
        state_a.velocity[0] * state_b.velocity[0]
        + state_a.velocity[1] * state_b.velocity[1]
    )
    cos_theta = dot / (state_a.speed * state_b.speed)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def surface_gap(
    state_a: TrackState, state_b: TrackState, cls_a: VehicleClass, cls_b: VehicleClass
) -> float:
    """Khoảng cách mặt-tới-mặt (m) giữa hai thân xe; 0 nếu chồng lấn."""
    return vehicle_gap(
        state_a.ground, state_a.heading, cls_a,
        state_b.ground, state_b.heading, cls_b,
    )


def classify_conflict(
    state_a: TrackState,
    state_b: TrackState,
    cls_a: VehicleClass,
    cls_b: VehicleClass,
    heading_rate_a: float = 0.0,
    heading_rate_b: float = 0.0,
    angle_rear_end: float = 30.0,
    angle_crossing_min: float = 60.0,
    angle_crossing_max: float = 120.0,
    angle_head_on: float = 150.0,
    sharp_turn_rate: float = 45.0,
) -> ConflictType:
    """Phân loại kiểu xung đột dựa trên góc tiếp cận và hành vi rẽ.

    Thứ tự ưu tiên có chủ đích: xung đột liên quan **người đi bộ** luôn được gán
    nhãn riêng bất kể hình học, vì nhóm này dễ tổn thương nhất và cần tách riêng
    trong thống kê an toàn.
    """
    if VehicleClass.PEDESTRIAN in (cls_a, cls_b):
        return ConflictType.PEDESTRIAN

    angle = approach_angle(state_a, state_b)

    # Một trong hai đang đổi hướng gấp ⇒ xung đột do chuyển hướng.
    if max(abs(heading_rate_a), abs(heading_rate_b)) > sharp_turn_rate:
        return ConflictType.TURNING

    if angle >= angle_head_on:
        return ConflictType.HEAD_ON
    if angle_crossing_min <= angle <= angle_crossing_max:
        return ConflictType.CROSSING
    if angle < angle_rear_end:
        return ConflictType.REAR_END
    # Vùng góc trung gian (30-60° và 120-150°): tạt đầu / chuyển làn.
    return ConflictType.LANE_CHANGE
