"""Ca tái hiện cho ba nguồn cảnh báo giả tìm thấy trên video đại lộ thật.

1. Cặp hai người đi bộ: nhãn chuẩn (protocol v2) bỏ, thuật toán thì không.
2. Vùng phân tích: đối tượng ngoài vùng homography được kiểm chứng không sinh
   cảnh báo.
3. Xe đứng yên có bbox rung: vận tốc ước lượng ra vài m/s và hai xe trong hàng
   chờ trông như đang lao vào nhau.
Và cách chấm mAP bỏ qua hộp chuẩn nhỏ hơn ngưỡng lọc của pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from saferoad.config import ConflictConfig, HomographyConfig, TrackingConfig
from saferoad.conflict.detector import ConflictDetector
from saferoad.evaluation.detection import evaluate_detection
from saferoad.geometry.homography import GroundPlane
from saferoad.geometry.trajectory import TrajectoryProcessor
from saferoad.types import Detection, Track, TrackState, VehicleClass as V

FPS = 30.0


def _crossing_tracks(cls_a, cls_b, anchor_a=(0, 0), anchor_b=(0, 0)):
    """A đi Đông, B đi Bắc, gặp nhau ở gốc toạ độ lúc t = 2 s."""
    t = np.arange(0.0, 5.0, 1.0 / FPS)
    tracks = []
    for tid, cls, start, vel, anchor in ((1, cls_a, (-4.0, 0.0), (2.0, 0.0), anchor_a),
                                         (2, cls_b, (0.0, -4.0), (0.0, 2.0), anchor_b)):
        tr = Track(track_id=tid, cls=cls, confirmed=True)
        for i, ti in enumerate(t):
            p = (start[0] + vel[0] * ti, start[1] + vel[1] * ti)
            tr.history.append(TrackState(frame_idx=i, t=ti, bbox=(0, 0, 1, 1), anchor=anchor,
                                         ground=p, velocity=vel))
        tracks.append(tr)
    return tracks, len(t)


def _feed(det, tracks, n):
    full = [list(tr.history) for tr in tracks]
    events = []
    for f in range(n):
        for tr, h in zip(tracks, full):
            tr.history = h[: f + 1]
        events += det.update(tracks, f, f / FPS)
    return events + det.flush(now=(n - 1) / FPS)


CFG = dict(proximity_gate=6.0, min_rel_speed=1.0, min_abs_speed=0.5)


class TestPedestrianPairs:
    def test_two_pedestrians_are_not_a_conflict(self):
        tracks, n = _crossing_tracks(V.PEDESTRIAN, V.PEDESTRIAN)
        assert _feed(ConflictDetector(ConflictConfig(**CFG)), tracks, n) == []

    def test_switch_restores_old_behaviour(self):
        tracks, n = _crossing_tracks(V.PEDESTRIAN, V.PEDESTRIAN)
        det = ConflictDetector(ConflictConfig(ignore_pedestrian_pairs=False, **CFG))
        assert len(_feed(det, tracks, n)) == 1

    def test_car_and_pedestrian_still_checked(self):
        tracks, n = _crossing_tracks(V.CAR, V.PEDESTRIAN)
        assert len(_feed(ConflictDetector(ConflictConfig(**CFG)), tracks, n)) == 1


class TestAnalysisRegion:
    REGION = [[0, 0], [100, 0], [100, 100], [0, 100]]

    def test_inside_region_is_analysed(self):
        tracks, n = _crossing_tracks(V.CAR, V.CAR, (50, 50), (60, 60))
        det = ConflictDetector(ConflictConfig(analysis_region_px=self.REGION, **CFG))
        assert len(_feed(det, tracks, n)) == 1

    def test_outside_region_is_ignored(self):
        tracks, n = _crossing_tracks(V.CAR, V.CAR, (50, 50), (500, 60))
        det = ConflictDetector(ConflictConfig(analysis_region_px=self.REGION, **CFG))
        assert _feed(det, tracks, n) == []


def _processor(stationary_speed: float) -> TrajectoryProcessor:
    homo = HomographyConfig(image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
                            world_points=[[0, 0], [10, 0], [10, 10], [0, 10]], smooth_window=5)
    ground = GroundPlane(homo, (100, 100))
    return TrajectoryProcessor(ground, homo, TrackingConfig(stationary_speed=stationary_speed,
                                                            stationary_window_s=2.0))


def _run(proc, tid, anchors):
    tr = Track(track_id=tid, cls=None, confirmed=True)
    for i, a in enumerate(anchors):
        tr.history.append(TrackState(frame_idx=i, t=i / FPS, bbox=(0, 0, 1, 1), anchor=a,
                                     ground=(0.0, 0.0)))
        proc.process(tr)
    return tr


def _parked_jitter(n=150, step=25, jump=8.0, seed=0):
    """Xe đỗ được gán nhãn theo từng đoạn: bbox đứng yên rồi nhảy vài pixel."""
    rng = np.random.default_rng(seed)
    out, off = [], np.zeros(2)
    for i in range(n):
        if i % step == 0:
            off = rng.normal(0, jump / 2, 2)
        out.append((50 + off[0], 50 + off[1]))
    return out


class TestStationaryClamp:
    def test_jitter_without_clamp_produces_fake_speed(self):
        tr = _run(_processor(0.0), 1, _parked_jitter())
        assert max(s.speed for s in tr.history[60:]) > 1.0

    def test_jitter_with_clamp_is_stationary(self):
        tr = _run(_processor(1.0), 1, _parked_jitter())
        assert max(s.speed for s in tr.history[70:]) < 1e-3

    def test_moving_car_is_not_clamped(self):
        anchors = [(10 + 0.5 * i, 50) for i in range(120)]       # 0,05 m/khung = 1,5 m/s
        tr = _run(_processor(1.0), 1, anchors)
        assert tr.history[-1].speed == pytest.approx(1.5, abs=0.2)

    def test_heading_kept_after_stopping(self):
        moving = [(10, 10 + 1.0 * i) for i in range(60)]          # đi theo +y, 3 m/s
        parked = [moving[-1]] * 120
        tr = _run(_processor(1.0), 1, moving + parked)
        last = tr.history[-1]
        assert last.speed < 1e-3
        assert math.cos(last.heading - math.pi / 2) > 0.95


class TestDetectionIgnoreSmall:
    def _det(self, box, score=None, cls=V.CAR):
        return Detection(bbox=box, score=1.0 if score is None else score, cls=cls)

    def test_small_gt_not_counted_as_miss(self):
        gt = {0: [self._det((0, 0, 40, 40)), self._det((100, 100, 105, 105))]}   # 1600 và 25 px²
        pred = {0: [self._det((0, 0, 40, 40), 0.9)]}
        strict = evaluate_detection(pred, gt, operating_conf=0.5)
        lenient = evaluate_detection(pred, gt, operating_conf=0.5, min_area=200)
        assert strict.recall50 == pytest.approx(0.5)
        assert lenient.recall50 == pytest.approx(1.0)
        assert lenient.n_gt_ignored == 1

    def test_prediction_on_small_gt_is_not_false_positive(self):
        gt = {0: [self._det((0, 0, 40, 40)), self._det((100, 100, 108, 108))]}
        pred = {0: [self._det((0, 0, 40, 40), 0.9), self._det((100, 100, 108, 108), 0.8)]}
        lenient = evaluate_detection(pred, gt, operating_conf=0.5, min_area=200)
        assert lenient.precision50 == pytest.approx(1.0)

    def test_ignore_gt_scores_only_unseen_objects(self):
        # Khung 0: xe đã gặp lúc train (bỏ qua) và xe mới.
        seen, new = self._det((0, 0, 40, 40)), self._det((200, 200, 240, 240))
        gt = {0: [seen, new]}
        skip = lambda f, d: d.bbox == seen.bbox  # noqa: E731
        hit_seen = evaluate_detection({0: [self._det((0, 0, 40, 40), 0.9)]}, gt,
                                      operating_conf=0.5, ignore_gt=skip)
        assert hit_seen.n_gt == 1 and hit_seen.n_gt_ignored == 1
        assert hit_seen.recall50 == pytest.approx(0.0)
        # Bắt cả hai: dự đoán rơi vào xe bị bỏ qua không tính là báo nhầm.
        hit_both = evaluate_detection({0: [self._det((0, 0, 40, 40), 0.9),
                                           self._det((200, 200, 240, 240), 0.8)]}, gt,
                                      operating_conf=0.5, ignore_gt=skip)
        assert hit_both.recall50 == pytest.approx(1.0)
        assert hit_both.precision50 == pytest.approx(1.0)
