"""Ca tái hiện cho từng lỗi PET / near-miss mà hội đồng chỉ ra, và bản sửa.

Mỗi test dựng một tình huống tính tay được, chạy đúng hàm của hệ thống, rồi so
với đáp án giải tích. Tên test nói rõ lỗi cũ là gì.
"""

from __future__ import annotations

import numpy as np
import pytest

from saferoad.config import ConflictConfig
from saferoad.conflict.detector import ConflictDetector
from saferoad.conflict.encroachment import (
    CONTACT_COLLISION,
    CONTACT_NONE,
    CONTACT_OVERLAP,
    PET_CONTACT,
    PET_INCOMPLETE,
    PET_MEASURED,
    PET_NO_ZONE,
    PET_PARALLEL,
    classify_contact,
    measure_pet,
)
from saferoad.conflict.indicators import post_encroachment_time
from saferoad.types import Track, TrackState, VehicleClass as V

FPS = 30.0


def _line(t, start, vel):
    return np.stack([start[0] + vel[0] * t, start[1] + vel[1] * t], axis=1), np.tile(vel, (len(t), 1))


def _crossing(delay: float, fps: float = FPS):
    """A đi Đông 10 m/s qua gốc lúc t=2 s; B đi Bắc 8 m/s qua gốc lúc t=2+delay."""
    t = np.arange(0.0, 6.0, 1.0 / fps)
    pa, va = _line(t, (-20.0, 0.0), (10.0, 0.0))
    pb, vb = _line(t, (0.0, -8.0 * (2.0 + delay)), (0.0, 8.0))
    return t, pa, va, pb, vb


def _analytic_pet(delay: float) -> float:
    """PET theo mô hình đa hình tròn: ô tô 4,4 × 1,8 m ⇒ vùng chiếm 3,1 m mỗi phía."""
    a_exit = 2.0 + 3.1 / 10.0
    b_entry = 2.0 + delay - 3.1 / 8.0
    return max(0.0, b_entry - a_exit)


class TestMeasuredPet:
    @pytest.mark.parametrize("delay", [0.9, 1.2, 1.5, 2.5])
    def test_matches_analytic_value(self, delay):
        t, pa, va, pb, vb = _crossing(delay)
        r = measure_pet(t, pa, va, V.CAR, t, pb, vb, V.CAR)
        assert r.status == PET_MEASURED
        assert r.first == "a"
        # Sai số chỉ còn do lấy mẫu 30 Hz (nội suy điểm giữa ⇒ ≤ 1 bước).
        assert r.pet == pytest.approx(_analytic_pet(delay), abs=1.0 / FPS + 1e-6)

    def test_keeps_all_four_timestamps(self):
        t, pa, va, pb, vb = _crossing(1.5)
        r = measure_pet(t, pa, va, V.CAR, t, pb, vb, V.CAR)
        assert r.t_a_in < r.t_a_out < r.t_b_in < r.t_b_out
        assert r.pet == pytest.approx(r.t_b_in - r.t_a_out)

    def test_contact_when_bodies_meet(self):
        t, pa, va, pb, vb = _crossing(0.0)
        r = measure_pet(t, pa, va, V.CAR, t, pb, vb, V.CAR)
        assert r.status == PET_CONTACT
        assert r.pet == 0.0

    def test_parallel_lanes_have_no_pet(self):
        """Lỗi cũ: hai xe song song cho ra một PET nhỏ giả. Nay: không xác định."""
        t = np.arange(0.0, 4.0, 1.0 / FPS)
        pa, va = _line(t, (-20.0, 0.0), (10.0, 0.0))
        pb, vb = _line(t, (-24.0, 0.4), (10.5, 0.0))   # cùng làn, vượt sát
        assert measure_pet(t, pa, va, V.MOTORCYCLE, t, pb, vb, V.MOTORCYCLE).status == PET_PARALLEL
        pc, vc = _line(t, (-24.0, 3.5), (10.5, 0.0))   # làn bên cạnh
        assert measure_pet(t, pa, va, V.CAR, t, pc, vc, V.CAR).status == PET_NO_ZONE

    def test_incomplete_when_follower_has_not_arrived(self):
        t, pa, va, pb, vb = _crossing(1.5)
        cut = t <= 2.6                                 # B chưa tới vùng xung đột
        r = measure_pet(t[cut], pa[cut], va[cut], V.CAR, t[cut], pb[cut], vb[cut], V.CAR,
                        b_continues=True)
        assert r.status in (PET_INCOMPLETE, PET_NO_ZONE)


class TestPredictedPet:
    def _state(self, x, y, vx, vy):
        return TrackState(frame_idx=0, t=0.0, bbox=(0, 0, 1, 1), anchor=(0, 0),
                          ground=(x, y), velocity=(vx, vy))

    def test_still_defined_after_leader_passed(self):
        """Lỗi cũ: xe A vừa qua giao điểm thì PET ngoại suy trả về None - đúng lúc
        xe B đang lao tới. Nay xe trước được phép đã qua giao điểm."""
        a = self._state(3.0, 0.0, 10.0, 0.0)       # đã qua gốc 0,3 s
        b = self._state(0.0, -6.0, 0.0, 8.0)       # còn 0,75 s nữa tới gốc
        r = post_encroachment_time(a, b, V.CAR, V.CAR)
        assert r is not None
        pet, point = r
        assert point == pytest.approx((0.0, 0.0), abs=1e-6)
        # Cùng định nghĩa vùng với PET đo: (0,75 - 3,1/8) - (-0,3 + 3,1/10) ≈ 0,353 s.
        assert pet == pytest.approx(0.75 - 3.1 / 8 - (-0.3 + 3.1 / 10), abs=0.08)

    def test_none_when_both_passed(self):
        a = self._state(5.0, 0.0, 10.0, 0.0)
        b = self._state(0.0, 5.0, 0.0, 8.0)
        assert post_encroachment_time(a, b, V.CAR, V.CAR) is None


class TestContactClass:
    def test_gap_positive_is_near_miss(self):
        assert classify_contact(0.4, (8.0, 6.0), (8.0, 6.0)) == CONTACT_NONE

    def test_overlap_without_stopping_is_measurement_artifact(self):
        """Thân xe chồng nhau mà cả hai vẫn giữ tốc độ: vật lý không cho phép ⇒
        sai số đo, KHÔNG được gán là va chạm."""
        assert classify_contact(0.0, (8.0, 6.0), (7.5, 6.2)) == CONTACT_OVERLAP

    def test_overlap_then_stop_is_collision_suspect(self):
        assert classify_contact(0.0, (8.0, 6.0), (0.3, 1.0)) == CONTACT_COLLISION


def _feed(det: ConflictDetector, tracks: list[Track], n: int):
    full = [list(tr.history) for tr in tracks]
    events = []
    for f in range(n):
        for tr, h in zip(tracks, full):
            tr.history = h[: f + 1]
        events += det.update(tracks, f, f / FPS)
    events += det.flush(now=(n - 1) / FPS)
    return events


def _track(tid, cls, pos, vel):
    tr = Track(track_id=tid, cls=cls, confirmed=True)
    for i, (p, v) in enumerate(zip(pos, vel)):
        tr.history.append(TrackState(frame_idx=i, t=i / FPS, bbox=(0, 0, 1, 1), anchor=(0, 0),
                                     ground=(float(p[0]), float(p[1])),
                                     velocity=(float(v[0]), float(v[1]))))
    return tr


class TestDetectorUsesMeasuredPet:
    def test_event_carries_both_pets_and_timestamps(self):
        t, pa, va, pb, vb = _crossing(1.2)
        det = ConflictDetector(ConflictConfig(proximity_gate=6.0))
        events = _feed(det, [_track(1, V.CAR, pa, va), _track(2, V.CAR, pb, vb)], len(t))
        assert len(events) == 1
        ev = events[0]
        assert ev.pet_status == PET_MEASURED
        assert ev.pet_source == "measured"
        assert ev.pet == pytest.approx(_analytic_pet(1.2), abs=0.07)
        assert {"a_in", "a_out", "b_in", "b_out"} <= set(ev.pet_times)
        assert ev.contact == CONTACT_NONE

    def test_predicted_only_episode_rejected_when_measured_pet_is_large(self):
        """Lỗi cũ: episode mở chỉ vì PET ngoại suy < 1,5 s được giữ lại dù xe sau
        đã phanh và thực tế đi qua muộn. Nay PET đo ≥ ngưỡng ⇒ bị loại, có ghi lý do."""
        t = np.arange(0.0, 8.0, 1.0 / FPS)
        pa, va = _line(t, (-20.0, 0.0), (10.0, 0.0))
        # B lao tới để qua giao điểm ~0,3 s sau khi A ra khỏi vùng (PET ngoại suy
        # nhỏ, TTC vô cùng vì hai xe lướt qua nhau), rồi phanh từ t = 0,5 s và đi
        # chậm 2,5 m/s - thực tế qua vùng xung đột muộn hơn nhiều giây.
        vb_y = np.where(t < 0.5, 8.0, np.maximum(2.5, 8.0 - 7.0 * (t - 0.5)))
        pb_y = -24.0 + np.concatenate([[0.0], np.cumsum(vb_y[:-1] / FPS)])
        pb = np.stack([np.zeros_like(t), pb_y], axis=1)
        vb = np.stack([np.zeros_like(t), vb_y], axis=1)
        cfg = ConflictConfig(proximity_gate=20.0, min_rel_speed=1.0)
        det = ConflictDetector(cfg)
        events = _feed(det, [_track(1, V.CAR, pa, va), _track(2, V.CAR, pb, vb)], len(t))
        assert events == []
        assert det.funnel.counts.get("rejected_pet_measured_above_threshold", 0) >= 1

    def test_funnel_balances(self):
        t, pa, va, pb, vb = _crossing(1.2)
        det = ConflictDetector(ConflictConfig(proximity_gate=6.0))
        _feed(det, [_track(1, V.CAR, pa, va), _track(2, V.CAR, pb, vb)], len(t))
        c = det.funnel.counts
        rejected = sum(v for k, v in c.items() if k.startswith("rejected_"))
        assert c["episodes_opened"] == rejected + c.get("merged_into_previous", 0) + c["emitted"]

    def test_lower_id_is_track_a_and_classes_follow(self):
        """Lỗi cũ: nếu track id lớn hơn đứng trước trong danh sách, lớp đối tượng
        của sự kiện bị đảo (cls_a là lớp của track_b)."""
        t, pa, va, pb, vb = _crossing(1.2)
        det = ConflictDetector(ConflictConfig(proximity_gate=6.0))
        events = _feed(det, [_track(9, V.TRUCK, pa, va), _track(3, V.CAR, pb, vb)], len(t))
        assert events[0].track_a == 3 and events[0].cls_a == V.CAR
        assert events[0].track_b == 9 and events[0].cls_b == V.TRUCK
