"""Kiểm thử các chỉ số xung đột — TTC, PET, phân loại.

Đây là phần **toán lõi** của toàn hệ thống: mọi con số trên dashboard và trong
báo cáo đều bắt nguồn từ đây, nên nó được kiểm bằng các tình huống có đáp án
tính tay được, chứ không phải bằng ảnh chụp kết quả hiện tại.
"""

from __future__ import annotations

import math

import pytest

from saferoad.conflict.geometry import NO_COLLISION, vehicle_gap, vehicle_ttc
from saferoad.conflict.indicators import (
    approach_angle,
    classify_conflict,
    conflict_point,
    post_encroachment_time,
    relative_speed,
    time_to_collision,
)
from saferoad.types import ConflictType, TrackState, VehicleClass


def state(x, y, vx, vy, t=0.0, frame=0) -> TrackState:
    """Tạo nhanh một TrackState tại vị trí mặt đất cho trước."""
    return TrackState(
        frame_idx=frame, t=t, bbox=(0, 0, 10, 10), anchor=(0, 0),
        ground=(x, y), velocity=(vx, vy),
    )


# --------------------------------------------------------------------------- #
# TTC
# --------------------------------------------------------------------------- #
class TestTimeToCollision:
    def test_head_on_approach(self):
        """Hai xe đối đầu, cách 20 m, tiếp cận 10 m/s ⇒ TTC ≈ (20 - kích thước)/10."""
        a = state(0.0, 0.0, 5.0, 0.0)
        b = state(20.0, 0.0, -5.0, 0.0)
        ttc = time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR)
        # Hai ô tô dài 4.4 m ⇒ va chạm khi tâm cách nhau ~4.4 m, không phải 0.
        assert 1.4 < ttc < 1.7, f"TTC = {ttc}"

    def test_diverging_returns_infinity(self):
        """Hai xe đang tách xa nhau thì không có va chạm."""
        a = state(0.0, 0.0, -5.0, 0.0)
        b = state(20.0, 0.0, 5.0, 0.0)
        assert time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR) == NO_COLLISION

    def test_parallel_same_speed_no_collision(self):
        """Đi song song cùng tốc độ: vận tốc tương đối bằng 0 ⇒ không bao giờ gặp."""
        a = state(0.0, 0.0, 10.0, 0.0)
        b = state(0.0, 5.0, 10.0, 0.0)
        assert time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR) == NO_COLLISION

    def test_perpendicular_crossing(self):
        """Cắt vuông góc, cả hai tới gốc toạ độ cùng lúc ⇒ TTC gần 2 s."""
        a = state(-20.0, 0.0, 10.0, 0.0)
        b = state(0.0, -20.0, 0.0, 10.0)
        ttc = time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR)
        assert 1.5 < ttc < 2.0, f"TTC = {ttc}"

    def test_overlapping_gives_zero(self):
        a = state(0.0, 0.0, 5.0, 0.0)
        b = state(0.3, 0.0, -5.0, 0.0)
        assert time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR) == 0.0

    def test_adjacent_lanes_are_not_a_conflict(self):
        """Hai ô tô đi ngược chiều ở hai làn cách nhau 4 m KHÔNG phải xung đột.

        Đây chính là lỗi mà mô hình một-hình-tròn mắc phải: hình tròn ngoại tiếp
        một ô tô có bán kính 2.38 m, nên hai xe cách 4 m bị coi là đang chạm nhau.
        Mô hình đa hình tròn phải cho ra ``NO_COLLISION``.
        """
        a = state(0.0, 0.0, 10.0, 0.0)
        b = state(30.0, 4.0, -10.0, 0.0)
        assert time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR) == NO_COLLISION

    def test_horizon_is_respected(self):
        """Va chạm ngoài tầm dự báo thì không tính."""
        a = state(0.0, 0.0, 1.0, 0.0)
        b = state(200.0, 0.0, 0.0, 0.0)
        assert time_to_collision(a, b, VehicleClass.CAR, VehicleClass.CAR, 5.0) == NO_COLLISION


# --------------------------------------------------------------------------- #
# Hình học đa hình tròn
# --------------------------------------------------------------------------- #
class TestVehicleGeometry:
    def test_multi_circle_covers_length(self):
        """Ô tô phải được phủ hết chiều dài 4.4 m bằng nhiều hình tròn."""
        circles = VehicleClass.CAR.circles
        assert len(circles) >= 2
        offsets = [o for o, _r in circles]
        radius = circles[0][1]
        assert math.isclose(radius, 0.9, abs_tol=1e-6)      # nửa bề rộng 1.8 m
        assert max(offsets) + radius == pytest.approx(2.2)  # nửa chiều dài 4.4 m

    def test_pedestrian_single_circle(self):
        assert len(VehicleClass.PEDESTRIAN.circles) == 1

    def test_gap_side_by_side(self):
        """Hai xe máy song song cách nhau 2 m ⇒ khoảng hở ≈ 2 - 0.35 - 0.35."""
        gap = vehicle_gap(
            (0.0, 0.0), 0.0, VehicleClass.MOTORCYCLE,
            (0.0, 2.0), 0.0, VehicleClass.MOTORCYCLE,
        )
        assert gap == pytest.approx(1.3, abs=0.05)

    def test_gap_nose_to_tail(self):
        """Hai ô tô nối đuôi, tâm cách 6 m ⇒ khoảng hở ≈ 6 - 4.4 = 1.6 m."""
        gap = vehicle_gap(
            (0.0, 0.0), 0.0, VehicleClass.CAR,
            (6.0, 0.0), 0.0, VehicleClass.CAR,
        )
        assert gap == pytest.approx(1.6, abs=0.1)

    def test_gap_zero_when_overlapping(self):
        gap = vehicle_gap(
            (0.0, 0.0), 0.0, VehicleClass.CAR,
            (1.0, 0.0), 0.0, VehicleClass.CAR,
        )
        assert gap == 0.0


# --------------------------------------------------------------------------- #
# PET
# --------------------------------------------------------------------------- #
class TestPostEncroachmentTime:
    def test_crossing_paths_have_conflict_point(self):
        a = state(-20.0, 0.0, 10.0, 0.0)
        b = state(0.0, -30.0, 0.0, 10.0)
        result = conflict_point(a, b)
        assert result is not None
        point, t_a, t_b = result
        assert point[0] == pytest.approx(0.0, abs=0.01)
        assert point[1] == pytest.approx(0.0, abs=0.01)
        assert t_a == pytest.approx(2.0, abs=0.01)
        assert t_b == pytest.approx(3.0, abs=0.01)

    def test_parallel_paths_rejected(self):
        """Quỹ đạo gần song song ⇒ PET vô nghĩa, phải trả về None.

        Nếu không chặn, giao điểm của hai tia gần song song nằm rất xa và cực
        nhạy với nhiễu — lệch hướng 1° đã dời điểm đó hàng chục mét.
        """
        a = state(0.0, 0.0, 10.0, 0.0)
        b = state(0.0, 3.0, 10.0, 0.1)
        assert post_encroachment_time(a, b, VehicleClass.CAR, VehicleClass.CAR) is None

    def test_far_conflict_point_rejected(self):
        a = state(0.0, 0.0, 10.0, 0.0)
        b = state(0.0, -200.0, 0.1, 10.0)
        assert post_encroachment_time(a, b, VehicleClass.CAR, VehicleClass.CAR) is None

    def test_pet_small_when_arrivals_close(self):
        """Hai xe tới điểm giao lệch nhau ~0.5 s ⇒ PET nhỏ."""
        a = state(-20.0, 0.0, 10.0, 0.0)     # tới (0,0) sau 2.0 s
        b = state(0.0, -25.0, 0.0, 10.0)     # tới (0,0) sau 2.5 s
        result = post_encroachment_time(a, b, VehicleClass.CAR, VehicleClass.CAR)
        assert result is not None
        pet, point = result
        assert pet < 0.6
        assert point[0] == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Góc & phân loại
# --------------------------------------------------------------------------- #
class TestClassification:
    def test_angle_same_direction(self):
        assert approach_angle(state(0, 0, 10, 0), state(5, 0, 10, 0)) == pytest.approx(0, abs=1e-6)

    def test_angle_perpendicular(self):
        assert approach_angle(state(0, 0, 10, 0), state(5, 0, 0, 10)) == pytest.approx(90, abs=1e-6)

    def test_angle_head_on(self):
        assert approach_angle(state(0, 0, 10, 0), state(5, 0, -10, 0)) == pytest.approx(180, abs=1e-6)

    def test_pedestrian_always_wins(self):
        """Xung đột có người đi bộ luôn được gán nhãn riêng, bất kể hình học."""
        kind = classify_conflict(
            state(0, 0, 10, 0), state(5, 0, 10, 0),
            VehicleClass.CAR, VehicleClass.PEDESTRIAN,
        )
        assert kind is ConflictType.PEDESTRIAN

    def test_crossing(self):
        kind = classify_conflict(
            state(0, 0, 10, 0), state(5, 0, 0, 10),
            VehicleClass.CAR, VehicleClass.CAR,
        )
        assert kind is ConflictType.CROSSING

    def test_rear_end(self):
        kind = classify_conflict(
            state(0, 0, 10, 0), state(5, 0, 6, 0),
            VehicleClass.CAR, VehicleClass.CAR,
        )
        assert kind is ConflictType.REAR_END

    def test_turning_detected_from_heading_rate(self):
        kind = classify_conflict(
            state(0, 0, 10, 0), state(5, 0, 9, 1),
            VehicleClass.CAR, VehicleClass.CAR,
            heading_rate_a=60.0,
        )
        assert kind is ConflictType.TURNING


def test_relative_speed():
    assert relative_speed(state(0, 0, 5, 0), state(10, 0, -5, 0)) == pytest.approx(10.0)
