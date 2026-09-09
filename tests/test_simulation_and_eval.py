"""Kiểm thử trình mô phỏng và bộ chấm điểm.

Hai thứ này quyết định **tính đáng tin của mọi con số trong báo cáo**, nên chúng
phải được kiểm riêng: một simulator sai sẽ tạo ra nhãn chuẩn sai, và một bộ chấm
điểm sai sẽ cho ra Precision/Recall đẹp mà vô nghĩa.
"""

from __future__ import annotations

import numpy as np
import pytest

from saferoad.conflict.geometry import series_ttc_and_gap
from saferoad.evaluation.detection import evaluate_detection
from saferoad.evaluation.metrics import ConflictMetrics, evaluate_conflicts
from saferoad.simulation.camera import COVERAGE, PinholeCamera, build_default_camera
from saferoad.simulation.scenario import (
    TrafficSignal, build_scenario, compute_ground_truth, simulate,
)
from saferoad.types import (
    ConflictEvent, ConflictType, Detection, GroundTruthConflict, RiskLevel, VehicleClass,
)


# --------------------------------------------------------------------------- #
class TestTrafficSignal:
    def test_phases_are_mutually_exclusive(self):
        """Hai nhóm hướng KHÔNG BAO GIỜ được cùng đèn xanh — đó là toàn bộ lý do
        đèn tín hiệu tồn tại trong mô phỏng này."""
        sig = TrafficSignal()
        for i in range(1000):
            t = i * sig.cycle / 1000.0
            ns, ew = sig.phase(t, "NS"), sig.phase(t, "EW")
            assert not (ns == "green" and ew == "green"), f"t={t}"

    def test_each_group_gets_green(self):
        sig = TrafficSignal()
        ts = np.linspace(0, sig.cycle, 500)
        assert any(sig.phase(t, "NS") == "green" for t in ts)
        assert any(sig.phase(t, "EW") == "green" for t in ts)

    def test_all_red_interval_exists(self):
        sig = TrafficSignal(green=20.0, yellow=3.0, all_red=2.0)
        ts = np.linspace(0, sig.cycle, 2000)
        assert any(
            sig.phase(t, "NS") == "red" and sig.phase(t, "EW") == "red" for t in ts
        )


# --------------------------------------------------------------------------- #
class TestCamera:
    def test_ground_projection_round_trip(self):
        cam = build_default_camera()
        pts = np.array([[0.0, 30.0], [-10.0, 25.0], [12.0, 40.0]])
        back = cam.image_to_ground(cam.project_ground(pts))
        assert np.allclose(back, pts, atol=1e-6)

    def test_taller_object_has_taller_bbox(self):
        """Chiều cao thật phải thể hiện trên bbox — đây là lý do phải dùng camera 3D.

        Nếu chỉ chiếu bóng xuống đất, người đi bộ (0.5 × 0.5 m) cho ra bbox tí
        hon và không bao giờ được phát hiện.
        """
        cam = build_default_camera()
        flat = cam.project(cam.cuboid_corners((0.0, 30.0), 0.0, 0.5, 0.5, 0.01))
        tall = cam.project(cam.cuboid_corners((0.0, 30.0), 0.0, 0.5, 0.5, 1.7))
        flat_h = flat[:, 1].max() - flat[:, 1].min()
        tall_h = tall[:, 1].max() - tall[:, 1].min()
        assert tall_h > flat_h * 3

    def test_coverage_region_is_visible(self):
        """Mọi góc của vùng COVERAGE phải nằm trong khung hình."""
        cam = build_default_camera(1280, 720)
        x0, y0, x1, y1 = COVERAGE
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        uv = cam.project_ground(corners)
        assert (uv[:, 0] > -80).all() and (uv[:, 0] < 1360).all()
        assert (uv[:, 1] > -80).all() and (uv[:, 1] < 800).all()

    def test_derived_homography_matches_camera(self):
        """Homography suy ra phải khớp CHÍNH XÁC phép chiếu của camera.

        Nếu lệch, sai số TTC đo được sẽ lẫn cả sai lệch hình học giữa bộ sinh dữ
        liệu và bộ xử lý, chứ không còn là sai số thuật toán.
        """
        from saferoad.geometry.homography import GroundPlane
        from saferoad.simulation.render import camera_to_homography

        cam = build_default_camera()
        gp = GroundPlane(camera_to_homography(cam))
        assert gp.calibrated
        assert gp.reprojection_error() == pytest.approx(0.0, abs=1e-3)

        for world in [(0.0, 30.0), (-8.0, 22.0), (10.0, 40.0)]:
            expected = cam.project_ground(np.array([world]))[0]
            actual = gp.to_image(world)
            assert actual[0] == pytest.approx(expected[0], abs=0.5)
            assert actual[1] == pytest.approx(expected[1], abs=0.5)


# --------------------------------------------------------------------------- #
class TestSimulation:
    @pytest.fixture(scope="class")
    def sim(self):
        vehicles = build_scenario(duration=45.0, seed=11)
        vehicles = simulate(vehicles, 45.0)
        return vehicles

    def test_vehicles_actually_move(self, sim):
        """Kiểm tra chống tắc nghẽn: mô phỏng phải THÔNG, không đứng im.

        Từng có lỗi khiến xe cắt ngang bị coi là 'xe phía trước cùng làn' và cả
        nút giao khoá chết; test này chặn lỗi đó quay lại.
        """
        speeds = np.concatenate([np.linalg.norm(v.velocities, axis=1) for v in sim])
        assert speeds.mean() > 1.0, f"tốc độ trung bình quá thấp: {speeds.mean():.2f} m/s"
        assert speeds.max() > 6.0
        assert (speeds > 2.0).mean() > 0.10

    def test_vehicles_complete_their_route(self, sim):
        displacement = [
            float(np.linalg.norm(v.positions[-1] - v.positions[0])) for v in sim
        ]
        completed = sum(1 for d in displacement if d > 40.0)
        assert completed >= len(sim) * 0.25, f"chỉ {completed}/{len(sim)} xe đi hết tuyến"

    def test_bodies_rarely_overlap(self, sim):
        """Xe không được đi xuyên qua nhau (cho phép một tỉ lệ nhỏ ở nút giao)."""
        overlaps = total = 0
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                a, b = sim[i], sim[j]
                t0 = max(a.times[0], b.times[0])
                t1 = min(a.times[-1], b.times[-1])
                if t1 - t0 < 0.2:
                    continue
                mask = (a.times >= t0) & (a.times <= t1)
                ts = a.times[mask]
                if ts.size < 5:
                    continue
                ib = np.clip(np.searchsorted(b.times, ts), 0, len(b.times) - 1)
                _ttc, gap = series_ttc_and_gap(
                    a.positions[mask], a.velocities[mask], a.cls,
                    b.positions[ib], b.velocities[ib], b.cls,
                )
                total += 1
                if gap.min() <= 0.0:
                    overlaps += 1
        assert total > 0
        assert overlaps / total < 0.10, f"tỉ lệ chồng lấn {overlaps / total:.2%} quá cao"

    def test_reproducible(self):
        """Cùng seed phải cho kịch bản y hệt — điều kiện để kết quả kiểm chứng được."""
        a = simulate(build_scenario(duration=20.0, seed=5), 20.0)
        b = simulate(build_scenario(duration=20.0, seed=5), 20.0)
        assert len(a) == len(b)
        assert np.allclose(a[0].positions, b[0].positions)

    def test_ground_truth_is_plausible(self, sim):
        gt = compute_ground_truth(sim, region=COVERAGE)
        assert len(gt) > 0, "kịch bản không sinh được near-miss nào"
        # Mật độ hợp lý: không phải mỗi giây vài chục vụ.
        assert len(gt) / 45.0 < 3.0, f"quá nhiều near-miss: {len(gt)} trong 45 s"
        for g in gt:
            assert g.track_a != g.track_b
            assert g.ttc >= 0

    def test_ground_truth_respects_region(self, sim):
        """Nhãn chuẩn phải nằm gọn trong vùng camera bao phủ."""
        gt = compute_ground_truth(sim, region=COVERAGE)
        by_id = {v.vid: v for v in sim}
        x0, y0, x1, y1 = COVERAGE
        for g in gt[:40]:
            va = by_id[g.track_a]
            idx = int(np.argmin(np.abs(va.times - g.t)))
            pos = va.positions[idx]
            assert x0 - 6 <= pos[0] <= x1 + 6
            assert y0 - 6 <= pos[1] <= y1 + 6

    def test_normal_following_is_not_a_conflict(self, sim):
        """Bám đuôi bình thường (cùng tốc độ, khoảng cách ổn định) KHÔNG phải near-miss."""
        gt = compute_ground_truth(sim, region=COVERAGE)
        by_id = {v.vid: v for v in sim}
        for g in gt:
            va, vb = by_id[g.track_a], by_id[g.track_b]
            ia = int(np.argmin(np.abs(va.times - g.t)))
            ib = int(np.argmin(np.abs(vb.times - g.t)))
            closing = float(np.linalg.norm(vb.velocities[ib] - va.velocities[ia]))
            assert closing > 1.0, "một cặp bám đuôi tốc độ đều đã lọt vào nhãn chuẩn"


# --------------------------------------------------------------------------- #
class TestEvaluationMetrics:
    def _event(self, ta, tb, t, ttc=1.0, kind=ConflictType.CROSSING):
        return ConflictEvent(
            event_id=f"E{ta}{tb}", frame_idx=int(t * 30), t=t, track_a=ta, track_b=tb,
            cls_a=VehicleClass.CAR, cls_b=VehicleClass.CAR, ttc=ttc, pet=0.8,
            conflict_type=kind, risk_score=70.0, risk_level=RiskLevel.HIGH,
            location=(0.0, 30.0), location_px=(640.0, 400.0),
            approach_angle=90.0, rel_speed=8.0, gap=1.0,
        )

    def test_perfect_match(self):
        preds = [self._event(1, 2, 10.0, ttc=1.0)]
        gt = [GroundTruthConflict(t=10.1, track_a=101, track_b=102, ttc=1.05)]
        m = evaluate_conflicts(preds, gt, {1: 101, 2: 102})
        assert m.tp == 1 and m.fp == 0 and m.fn == 0
        assert m.precision == 1.0 and m.recall == 1.0
        assert m.ttc_mae == pytest.approx(0.05, abs=1e-6)

    def test_time_tolerance_enforced(self):
        preds = [self._event(1, 2, 10.0)]
        gt = [GroundTruthConflict(t=30.0, track_a=101, track_b=102, ttc=1.0)]
        m = evaluate_conflicts(preds, gt, {1: 101, 2: 102}, time_tol=2.0)
        assert m.tp == 0 and m.fp == 1 and m.fn == 1

    def test_one_prediction_cannot_match_two_labels(self):
        """Ghép phải là một-một, nếu không Recall bị thổi phồng."""
        preds = [self._event(1, 2, 10.0)]
        gt = [
            GroundTruthConflict(t=10.0, track_a=101, track_b=102, ttc=1.0),
            GroundTruthConflict(t=10.5, track_a=101, track_b=102, ttc=1.2),
        ]
        m = evaluate_conflicts(preds, gt, {1: 101, 2: 102})
        assert m.tp == 1 and m.fn == 1

    def test_unmapped_track_counts_as_false_positive(self):
        preds = [self._event(1, 2, 10.0)]
        gt = [GroundTruthConflict(t=10.0, track_a=101, track_b=102, ttc=1.0)]
        m = evaluate_conflicts(preds, gt, {1: 101})   # track 2 không map được
        assert m.tp == 0 and m.fp == 1

    def test_region_filter_ignores_outside_predictions(self):
        """Dự đoán ngoài vùng đánh giá bị BỎ QUA, không tính là FP."""
        outside = self._event(1, 2, 10.0)
        outside.location = (500.0, 500.0)
        m = evaluate_conflicts([outside], [], {1: 101, 2: 102}, region=(-20, 0, 20, 60))
        assert m.fp == 0

    def test_empty_inputs(self):
        m = evaluate_conflicts([], [], {})
        assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


class TestDetectionMetrics:
    def test_perfect_detection(self):
        gt = {0: [Detection((10, 10, 50, 60), 1.0, VehicleClass.CAR)]}
        pred = {0: [Detection((10, 10, 50, 60), 0.95, VehicleClass.CAR)]}
        m = evaluate_detection(pred, gt)
        assert m.map50 == pytest.approx(1.0)
        assert m.recall50 == pytest.approx(1.0)

    def test_missed_detection(self):
        gt = {0: [Detection((10, 10, 50, 60), 1.0, VehicleClass.CAR)]}
        m = evaluate_detection({}, gt)
        assert m.map50 == 0.0
        assert m.recall50 == 0.0

    def test_wrong_class_is_not_a_match(self):
        gt = {0: [Detection((10, 10, 50, 60), 1.0, VehicleClass.CAR)]}
        pred = {0: [Detection((10, 10, 50, 60), 0.9, VehicleClass.TRUCK)]}
        m = evaluate_detection(pred, gt)
        assert m.ap50_per_class.get("car", 0.0) == 0.0
