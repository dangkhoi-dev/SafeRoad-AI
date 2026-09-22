"""Kiểm thử các khối còn lại: homography, quỹ đạo, tracking, risk, ẩn danh, DB."""

from __future__ import annotations

import math

import numpy as np
import pytest

from saferoad.behavior.classifier import extract_features
from saferoad.config import (
    BehaviorConfig, ConflictConfig, HomographyConfig, PrivacyConfig,
    RiskConfig, TrackingConfig,
)
from saferoad.conflict.detector import ConflictDetector
from saferoad.geometry.homography import GroundPlane, make_default_homography
from saferoad.geometry.trajectory import (
    heading_change_rate, smooth_positions, velocity_from_window,
)
from saferoad.privacy.anonymize import Anonymizer
from saferoad.risk.riskmap import RiskMap
from saferoad.risk.scoring import RiskScorer
from saferoad.storage.db import EventStore
from saferoad.tracking.bytetrack import ByteTracker
from saferoad.tracking.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah
from saferoad.types import (
    ConflictEvent, ConflictType, Detection, RiskLevel, Track, TrackState, VehicleClass,
)


# --------------------------------------------------------------------------- #
class TestHomography:
    def test_identity_round_trip(self):
        """Chiếu xuôi rồi chiếu ngược phải trả về đúng điểm ban đầu."""
        cfg = HomographyConfig(
            image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
            world_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        gp = GroundPlane(cfg)
        assert gp.calibrated
        for pt in [(50.0, 50.0), (10.0, 90.0), (99.0, 1.0)]:
            back = gp.to_image(gp.to_ground(pt))
            assert back[0] == pytest.approx(pt[0], abs=1e-6)
            assert back[1] == pytest.approx(pt[1], abs=1e-6)

    def test_scale_is_correct(self):
        cfg = HomographyConfig(
            image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
            world_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        gp = GroundPlane(cfg)
        assert gp.to_ground((50.0, 0.0))[0] == pytest.approx(5.0, abs=1e-6)

    def test_reprojection_error_zero_for_exact_points(self):
        cfg = HomographyConfig(
            image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
            world_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        assert GroundPlane(cfg).reprojection_error() == pytest.approx(0.0, abs=1e-9)

    def test_fallback_when_uncalibrated(self):
        gp = GroundPlane(HomographyConfig(fallback_scale=0.05))
        assert not gp.calibrated
        assert gp.to_ground((100.0, 200.0)) == (5.0, 10.0)

    def test_default_homography_is_usable(self):
        cfg = make_default_homography(1280, 720, camera_height_m=12.0)
        gp = GroundPlane(cfg)
        assert gp.calibrated
        near = gp.to_ground((640.0, 700.0))
        far = gp.to_ground((640.0, 300.0))
        # Điểm ở phía trên khung hình phải xa camera hơn.
        assert far[1] > near[1]


# --------------------------------------------------------------------------- #
class TestTrajectory:
    def test_velocity_from_perfect_line(self):
        """Hồi quy trên chuỗi thẳng đều phải cho đúng hệ số góc."""
        t = np.arange(10, dtype=float) / 30.0
        x = 5.0 * t + 2.0
        assert velocity_from_window(t, x) == pytest.approx(5.0, abs=1e-9)

    def test_velocity_robust_to_noise(self):
        """Hồi quy phải ổn định hơn nhiều so với lấy hiệu hai điểm."""
        rng = np.random.default_rng(0)
        t = np.arange(15, dtype=float) / 30.0
        clean = 8.0 * t
        noisy = clean + rng.normal(0, 0.05, size=len(t))

        regression = velocity_from_window(t, noisy)
        finite_difference = (noisy[-1] - noisy[-2]) / (t[-1] - t[-2])

        assert abs(regression - 8.0) < abs(finite_difference - 8.0)
        assert abs(regression - 8.0) < 1.0

    def test_smoothing_preserves_length_and_reduces_variance(self):
        rng = np.random.default_rng(1)
        pts = np.stack([np.arange(20.0), np.zeros(20)], axis=1)
        noisy = pts + rng.normal(0, 0.3, size=pts.shape)
        smoothed = smooth_positions(noisy, 5)
        assert smoothed.shape == noisy.shape
        assert np.std(smoothed[:, 1]) < np.std(noisy[:, 1])

    def test_heading_rate_ignores_slow_states(self):
        """Xe gần như đứng yên không được coi là 'đổi hướng gấp'.

        Hướng suy từ vector vận tốc; khi tốc độ ~0 thì hướng chỉ là nhiễu chia
        cho nhiễu, và nếu không lọc thì mọi xe trong hàng chờ đèn đỏ đều bị gán
        nhầm nhãn.
        """
        track = Track(track_id=1, cls=VehicleClass.CAR)
        rng = np.random.default_rng(2)
        for i in range(15):
            angle = rng.uniform(-math.pi, math.pi)
            track.history.append(
                TrackState(
                    frame_idx=i, t=i / 30.0, bbox=(0, 0, 1, 1), anchor=(0, 0),
                    ground=(0.0, 0.0),
                    velocity=(0.05 * math.cos(angle), 0.05 * math.sin(angle)),
                )
            )
        assert heading_change_rate(track) == 0.0


# --------------------------------------------------------------------------- #
class TestKalman:
    def test_bbox_conversion_round_trip(self):
        box = (10.0, 20.0, 50.0, 80.0)
        assert xyah_to_xyxy(xyxy_to_xyah(box)) == pytest.approx(box)

    def test_filter_tracks_constant_velocity(self):
        """Sau vài lần cập nhật, filter phải bám sát vật thể chuyển động đều."""
        kf = KalmanBoxFilter()
        mean, cov = kf.initiate(xyxy_to_xyah((0.0, 0.0, 20.0, 40.0)))
        for step in range(1, 12):
            mean, cov = kf.predict(mean, cov)
            box = (step * 5.0, 0.0, step * 5.0 + 20.0, 40.0)
            mean, cov = kf.update(mean, cov, xyxy_to_xyah(box))
        predicted = xyah_to_xyxy(kf.predict(mean, cov)[0])
        assert predicted[0] == pytest.approx(12 * 5.0, abs=3.0)


# --------------------------------------------------------------------------- #
class TestTracker:
    def test_keeps_single_id_for_steady_object(self):
        """Một vật thể đi thẳng đều phải giữ nguyên MỘT ID."""
        tracker = ByteTracker(TrackingConfig())
        ids = set()
        for frame in range(40):
            x = 100.0 + frame * 6.0
            det = Detection((x, 200.0, x + 60.0, 260.0), 0.9, VehicleClass.CAR)
            for track in tracker.update([det], frame, frame / 30.0):
                ids.add(track.track_id)
        assert len(ids) == 1

    def test_survives_short_occlusion(self):
        """Mất dấu 4 frame rồi xuất hiện lại: ID phải được giữ nguyên."""
        tracker = ByteTracker(TrackingConfig())
        ids = set()
        for frame in range(40):
            x = 100.0 + frame * 6.0
            dets = []
            if not (18 <= frame < 22):
                dets = [Detection((x, 200.0, x + 60.0, 260.0), 0.9, VehicleClass.CAR)]
            for track in tracker.update(dets, frame, frame / 30.0):
                ids.add(track.track_id)
        assert len(ids) == 1

    def test_does_not_merge_different_classes(self):
        tracker = ByteTracker(TrackingConfig())
        classes = {}
        for frame in range(20):
            x = 100.0 + frame * 5.0
            dets = [
                Detection((x, 200.0, x + 60.0, 260.0), 0.9, VehicleClass.CAR),
                Detection((x, 400.0, x + 30.0, 430.0), 0.9, VehicleClass.MOTORCYCLE),
            ]
            for track in tracker.update(dets, frame, frame / 30.0):
                classes[track.track_id] = track.cls
        assert set(classes.values()) == {VehicleClass.CAR, VehicleClass.MOTORCYCLE}


# --------------------------------------------------------------------------- #
def make_event(ttc=1.0, pet=0.8, rel_speed=8.0,
               kind=ConflictType.CROSSING, cls_a=VehicleClass.CAR,
               cls_b=VehicleClass.MOTORCYCLE, gap=1.0, t=10.0) -> ConflictEvent:
    return ConflictEvent(
        event_id="CF00001", frame_idx=300, t=t, track_a=1, track_b=2,
        cls_a=cls_a, cls_b=cls_b, ttc=ttc, pet=pet, conflict_type=kind,
        risk_score=0.0, risk_level=RiskLevel.NONE,
        location=(0.0, 30.0), location_px=(640.0, 400.0),
        approach_angle=90.0, rel_speed=rel_speed, gap=gap,
    )


class TestRiskScoring:
    def test_score_in_range(self):
        scorer = RiskScorer(RiskConfig())
        assert 0.0 <= scorer.score_event(make_event()) <= 100.0

    def test_lower_ttc_is_riskier(self):
        scorer = RiskScorer(RiskConfig())
        dangerous = scorer.score_event(make_event(ttc=0.4))
        mild = scorer.score_event(make_event(ttc=2.8))
        assert dangerous > mild

    def test_pedestrian_raises_score(self):
        scorer = RiskScorer(RiskConfig())
        with_ped = scorer.score_event(
            make_event(kind=ConflictType.PEDESTRIAN, cls_b=VehicleClass.PEDESTRIAN)
        )
        vehicles = scorer.score_event(make_event(kind=ConflictType.REAR_END))
        assert with_ped > vehicles

    def test_reasons_are_generated(self):
        scorer = RiskScorer(RiskConfig())
        event = make_event(ttc=0.6)
        scorer.score_event(event)
        assert event.reasons
        assert any("TTC" in r for r in event.reasons)

    def test_level_thresholds(self):
        assert RiskLevel.from_score(85) is RiskLevel.HIGH
        assert RiskLevel.from_score(55) is RiskLevel.MEDIUM
        assert RiskLevel.from_score(20) is RiskLevel.LOW


class TestRiskMap:
    def test_hotspots_are_spatially_separated(self):
        """Non-maximum suppression phải tách các điểm nóng ra, không dồn một cụm."""
        cfg = RiskConfig(grid_size=1.0, kernel_radius=4.0, top_k_hotspots=3)
        rmap = RiskMap(cfg, bounds=(-30.0, 0.0, 30.0, 60.0))
        scorer = RiskScorer(RiskConfig())
        for pos in [(-20.0, 10.0), (0.0, 30.0), (20.0, 50.0)]:
            for _ in range(5):
                ev = make_event()
                ev.location = pos
                scorer.score_event(ev)
                rmap.add(ev)

        hotspots = rmap.hotspots()
        assert len(hotspots) == 3
        for i in range(len(hotspots)):
            for j in range(i + 1, len(hotspots)):
                dist = math.dist(
                    (hotspots[i].x, hotspots[i].y), (hotspots[j].x, hotspots[j].y)
                )
                assert dist > cfg.kernel_radius

    def test_empty_map(self):
        rmap = RiskMap(RiskConfig())
        assert rmap.hotspots() == []
        assert rmap.render().max() == 0.0

    def test_time_decay(self):
        """Sự kiện cũ phải đóng góp ít hơn sự kiện mới."""
        cfg = RiskConfig(decay_tau=10.0, grid_size=2.0)
        rmap = RiskMap(cfg, bounds=(-10.0, 0.0, 10.0, 20.0))
        scorer = RiskScorer(RiskConfig())
        old = make_event(t=0.0)
        old.location = (0.0, 10.0)
        scorer.score_event(old)
        rmap.add(old)
        fresh_peak = rmap.render(now_t=0.0).max()
        decayed_peak = rmap.render(now_t=100.0, normalize=False).max()
        assert fresh_peak > 0
        assert decayed_peak < 1e-2


# --------------------------------------------------------------------------- #
class TestConflictDetector:
    def _track(self, tid, cls, positions, fps=30.0):
        track = Track(track_id=tid, cls=cls, confirmed=True)
        for i, (x, y, vx, vy) in enumerate(positions):
            track.history.append(
                TrackState(
                    frame_idx=i, t=i / fps, bbox=(0, 0, 10, 10), anchor=(0.0, 0.0),
                    ground=(x, y), velocity=(vx, vy),
                )
            )
        return track

    def test_no_event_when_far_apart(self):
        det = ConflictDetector(ConflictConfig())
        a = self._track(1, VehicleClass.CAR, [(i * 0.3, 0.0, 9.0, 0.0) for i in range(60)])
        b = self._track(2, VehicleClass.CAR, [(i * 0.3, 50.0, 9.0, 0.0) for i in range(60)])
        events = []
        for f in range(60):
            a.history = a.history[: f + 1]
            b.history = b.history[: f + 1]
            events += det.update([a, b], f, f / 30.0)
        events += det.flush()
        assert events == []

    def test_emits_single_event_per_episode(self):
        """Một tình huống kéo dài phải sinh ĐÚNG một sự kiện, không phải mỗi frame một cái."""
        cfg = ConflictConfig()
        det = ConflictDetector(cfg)
        scorer = RiskScorer(RiskConfig())

        n = 90
        a_pos, b_pos = [], []
        for i in range(n):
            t = i / 30.0
            a_pos.append((-20.0 + 10.0 * t, 0.0, 10.0, 0.0))
            b_pos.append((0.0, -20.0 + 10.0 * t, 0.0, 10.0))
        a = self._track(1, VehicleClass.CAR, a_pos)
        b = self._track(2, VehicleClass.CAR, b_pos)

        full_a, full_b = list(a.history), list(b.history)
        events = []
        for f in range(n):
            a.history = full_a[: f + 1]
            b.history = full_b[: f + 1]
            events += det.update([a, b], f, f / 30.0, risk_scorer=scorer)
        events += det.flush(risk_scorer=scorer)

        assert len(events) == 1, f"kỳ vọng 1 sự kiện, nhận {len(events)}"
        assert events[0].ttc is not None
        assert 0.0 <= events[0].risk_score <= 100.0

    def test_active_alerts_available_during_episode(self):
        """Cảnh báo thời gian thực phải xuất hiện NGAY khi tình huống đang diễn ra."""
        det = ConflictDetector(ConflictConfig())
        n = 60
        a = self._track(1, VehicleClass.CAR,
                        [(-15.0 + 10.0 * i / 30.0, 0.0, 10.0, 0.0) for i in range(n)])
        b = self._track(2, VehicleClass.CAR,
                        [(0.0, -15.0 + 10.0 * i / 30.0, 0.0, 10.0) for i in range(n)])
        full_a, full_b = list(a.history), list(b.history)

        saw_alert = False
        for f in range(n):
            a.history = full_a[: f + 1]
            b.history = full_b[: f + 1]
            det.update([a, b], f, f / 30.0)
            if det.active_alerts(f):
                saw_alert = True
        assert saw_alert


# --------------------------------------------------------------------------- #
class TestPrivacy:
    def test_disabled_returns_same_frame(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        anon = Anonymizer(PrivacyConfig(enabled=False))
        assert np.array_equal(anon.apply(frame, []), frame)

    def test_blurs_pedestrian_head_region(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        det = Detection((50.0, 50.0, 90.0, 150.0), 0.9, VehicleClass.PEDESTRIAN)
        out = Anonymizer(PrivacyConfig(enabled=True)).apply(frame, [det])
        head = out[50:78, 56:84]
        original = frame[50:78, 56:84]
        assert head.std() < original.std()
        # Vùng ngoài bbox phải giữ nguyên.
        assert np.array_equal(out[180:, 180:], frame[180:, 180:])

    def test_stats_counted(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        anon = Anonymizer(PrivacyConfig(enabled=True))
        anon.apply(frame, [
            Detection((10.0, 10.0, 50.0, 110.0), 0.9, VehicleClass.PEDESTRIAN),
            Detection((60.0, 60.0, 140.0, 100.0), 0.9, VehicleClass.CAR),
        ])
        assert anon.stats["faces_blurred"] >= 1
        assert anon.stats["plates_blurred"] >= 1


# --------------------------------------------------------------------------- #
class TestBehaviorFeatures:
    def test_features_complete_and_finite(self):
        track = Track(track_id=1, cls=VehicleClass.CAR)
        for i in range(20):
            track.history.append(
                TrackState(
                    frame_idx=i, t=i / 30.0, bbox=(0, 0, 10, 10), anchor=(0, 0),
                    ground=(i * 0.3, 0.0), velocity=(9.0, 0.0),
                )
            )
        feats = extract_features(track, BehaviorConfig().window)
        assert len(feats) == 10
        assert all(np.isfinite(v) for v in feats.values())

    def test_short_track_returns_zeros(self):
        track = Track(track_id=1, cls=VehicleClass.CAR)
        assert all(v == 0.0 for v in extract_features(track).values())


# --------------------------------------------------------------------------- #
class TestStorage:
    def test_round_trip(self, tmp_path):
        store = EventStore(tmp_path / "test.db")
        sid = store.start_session(source="x.mp4", site="Test", fps=30.0)

        scorer = RiskScorer(RiskConfig())
        events = []
        for i in range(5):
            ev = make_event(ttc=0.5 + i * 0.4, t=float(i))
            ev.event_id = f"CF{i:05d}"
            scorer.score_event(ev)
            events.append(ev)
        store.add_events(events, sid)
        store.finish_session(150)

        loaded = store.load_events(sid)
        assert len(loaded) == 5
        assert loaded[0].event_id == "CF00000"
        assert loaded[0].reasons

        summary = store.summary(sid)
        assert summary["total_events"] == 5
        assert summary["severe_events"] >= 1


class TestThroughputMetric:
    """Chỉ số FPS phải phản ánh tốc độ xử lý, không phản ánh máy có ngủ hay không.

    Trước đây FPS = số frame / tổng thời gian chạy. Chỉ cần máy sleep vài phút
    giữa lúc đo là con số tụt xuống dưới 1 FPS trong khi pipeline không hề chậm
    đi. Trung vị độ trễ từng frame không có nhược điểm đó.
    """

    @staticmethod
    def _fps(latencies_ms):
        import numpy as np

        return 1000.0 / float(np.median(latencies_ms))

    def test_matches_wall_clock_when_machine_behaves(self):
        lat = [20.0] * 500
        assert abs(self._fps(lat) - 50.0) < 1e-6

    def test_survives_a_long_stall(self):
        # 500 frame ở 20 ms, rồi một frame "nuốt" 3 tiếng vì máy ngủ.
        lat = [20.0] * 500 + [3 * 3600 * 1000.0]
        assert abs(self._fps(lat) - 50.0) < 0.2
        # Cách đo cũ sẽ cho ra con số vô nghĩa:
        naive = len(lat) / (sum(lat) / 1000.0)
        assert naive < 1.0

    def test_reports_tail_not_just_centre(self):
        """Trung vị giấu mất đuôi phân phối, nên phải báo cáo kèm p95.

        Với 10% frame chậm gấp 4 lần, trung vị vẫn 20 ms — đúng nhưng không đủ:
        người đọc cần biết trường hợp xấu. p95 mới cho thấy điều đó.
        """
        import numpy as np

        lat = [20.0] * 90 + [80.0] * 10
        assert abs(float(np.median(lat)) - 20.0) < 1e-6
        assert float(np.percentile(lat, 95)) > 60.0


class TestTiledDetector:
    """Detector chia ô — thứ quyết định có nhìn thấy xe ở xa hay không.

    Ba tính chất phải đúng, và cả ba đều đã từng sai trong lúc phát triển:
    ô phải phủ kín khung hình, phải chồng lấn nhau, và toạ độ trả về phải nằm
    trong hệ của khung hình gốc chứ không phải của ô.
    """

    @staticmethod
    def _det():
        from saferoad.config import DetectionConfig
        from saferoad.detection.detector import TiledYoloDetector
        cfg = DetectionConfig(tile_rows=2, tile_cols=3, tile_overlap=0.25)
        obj = TiledYoloDetector.__new__(TiledYoloDetector)  # bỏ qua việc nạp YOLO
        obj.cfg = cfg
        obj.rows, obj.cols, obj.overlap = 2, 3, 0.25
        return obj

    def test_tiles_cover_whole_frame(self):
        import numpy as np

        d = self._det()
        h, w = 640, 1024
        mask = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in d._tiles(h, w):
            mask[y1:y2, x1:x2] = True
        assert mask.all(), "có vùng khung hình không ô nào phủ tới"

    def test_tiles_overlap_each_other(self):
        d = self._det()
        tiles = d._tiles(640, 1024)
        assert len(tiles) == 6
        # hai ô đầu cùng hàng phải giao nhau, nếu không thì vật nằm trên đường
        # cắt sẽ bị xẻ đôi
        a, b = tiles[0], tiles[1]
        assert a[2] > b[0], "hai ô kề nhau không chồng lấn"

    def test_merge_drops_duplicates_but_keeps_other_classes(self):
        from saferoad.detection.detector import _merge
        from saferoad.types import Detection, VehicleClass

        box = (10.0, 10.0, 50.0, 50.0)
        near = (12.0, 11.0, 52.0, 51.0)      # cùng vật, nhìn từ ô kề bên
        far = (200.0, 200.0, 240.0, 240.0)   # vật khác hẳn
        dets = [
            Detection(bbox=box, score=0.9, cls=VehicleClass.CAR),
            Detection(bbox=near, score=0.7, cls=VehicleClass.CAR),
            Detection(bbox=box, score=0.8, cls=VehicleClass.PEDESTRIAN),
            Detection(bbox=far, score=0.6, cls=VehicleClass.CAR),
        ]
        out = _merge(dets, iou_thr=0.55)
        cars = [d for d in out if d.cls is VehicleClass.CAR]
        peds = [d for d in out if d.cls is VehicleClass.PEDESTRIAN]
        assert len(cars) == 2, "hai hộp trùng của cùng một xe phải gộp còn một"
        assert max(c.score for c in cars) == 0.9, "phải giữ hộp có điểm cao hơn"
        assert len(peds) == 1, "người đi bộ đứng cạnh xe không được gộp vào xe"

    def test_boxes_are_mapped_back_to_frame_coordinates(self):
        import numpy as np
        from saferoad.types import Detection, VehicleClass

        d = self._det()
        d.cfg.tile_full_frame = False
        d.cfg.tile_merge_iou = 0.55

        class _Stub:
            """Luôn báo một hộp ở góc trên trái của ô được đưa vào."""

            def detect(self, tile, frame_idx=0):
                return [Detection(bbox=(0.0, 0.0, 4.0, 4.0), score=0.9,
                                  cls=VehicleClass.CAR)]

        d.inner = _Stub()
        frame = np.zeros((640, 1024, 3), dtype=np.uint8)
        out = d.detect(frame)
        xs = sorted(round(o.bbox[0]) for o in out)
        assert xs[0] == 0, "ô đầu tiên phải cho hộp ở gốc toạ độ khung hình"
        assert max(xs) > 100, "các ô bên phải phải được dịch sang phải, không nằm chồng ở gốc"
