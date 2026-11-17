"""Kiểm thử cho phần đánh giá nhất quán, hiệu chuẩn, dữ liệu thật và rà soát."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from saferoad.behavior.classifier import BehaviorClassifier, extract_features
from saferoad.config import BehaviorConfig
from saferoad.evaluation.real_conflicts import cohen_kappa, score_reviewed
from saferoad.evaluation.scoring import reconciliation, score_events, summary_metrics
from saferoad.geometry.calibration import CalibrationInput, evaluate_calibration
from saferoad.simulation.scenario import (
    STOP_LINE_RADIUS, VehicleSpec, build_scenario, simulate,
)
from saferoad.types import (
    ConflictEvent, ConflictType, GroundTruthConflict, RiskLevel, Track, TrackState,
    VehicleClass as V,
)


def _ev(eid, a, b, t, loc=(0.0, 30.0)):
    return ConflictEvent(event_id=eid, frame_idx=int(t * 30), t=t, track_a=a, track_b=b,
                         cls_a=V.CAR, cls_b=V.CAR, ttc=1.0, pet=None,
                         conflict_type=ConflictType.CROSSING, risk_score=50.0,
                         risk_level=RiskLevel.MEDIUM, location=loc, location_px=(0, 0),
                         approach_angle=90.0, rel_speed=5.0, gap=1.0)


class TestReconciliation:
    def test_every_event_lands_in_exactly_one_bucket(self):
        events = [_ev("E1", 1, 2, 10.0), _ev("E2", 3, 4, 20.0),
                  _ev("E3", 5, 6, 30.0, loc=(100.0, 100.0)),   # ngoài vùng
                  _ev("E4", 7, 8, 40.0)]                       # track không ánh xạ
        gt = [GroundTruthConflict(t=10.2, track_a=11, track_b=12, ttc=1.0),
              GroundTruthConflict(t=50.0, track_a=13, track_b=14, ttc=1.0)]
        mapping = {1: 11, 2: 12, 3: 13, 4: 15}
        d = score_events(events, gt, mapping, region=(-20, 0, 20, 60))
        rec = reconciliation(d, {"episodes_opened": 6, "rejected_proximity_gate": 2,
                                 "emitted": 4}, len(events), 3600.0)
        assert rec["all_checks_pass"], rec["checks"]
        assert rec["rows"]["ignored_outside_eval_region"] == 1
        assert rec["rows"]["scored_unmapped_track_fp"] == 1
        assert rec["rows"]["scored_tp"] == 1 and rec["rows"]["gt_missed_fn"] == 1
        m = summary_metrics(d)
        assert (m["precision"]["num"], m["precision"]["den"]) == (1, 3)
        assert (m["recall"]["num"], m["recall"]["den"]) == (1, 2)
        assert rec["per_hour"]["false_alarms_per_hour"] == 2.0

    def test_mismatch_is_detected(self):
        d = score_events([_ev("E1", 1, 2, 1.0)], [], {1: 1, 2: 2})
        rec = reconciliation(d, {"episodes_opened": 1, "emitted": 1}, 2, 60.0)
        assert not rec["checks"]["emitted = exported"]


class TestBehaviorFlags:
    def _track(self, speeds, heading_rate_deg=0.0, fps=30.0):
        tr = Track(track_id=1, cls=V.CAR, confirmed=True)
        x = y = 0.0
        h = 0.0
        rng = np.random.default_rng(0)
        for i, v in enumerate(speeds):
            h += np.radians(heading_rate_deg) / fps
            vx, vy = v * np.cos(h), v * np.sin(h)
            x += vx / fps
            y += vy / fps
            # Nhiễu vận tốc cỡ đầu ra của bộ theo vết.
            nvx, nvy = vx + rng.normal(0, 0.25), vy + rng.normal(0, 0.25)
            tr.history.append(TrackState(frame_idx=i, t=i / fps, bbox=(0, 0, 1, 1),
                                         anchor=(0, 0), ground=(x, y), velocity=(nvx, nvy)))
        return tr

    def test_steady_noisy_track_is_not_flagged(self):
        """Lỗi cũ: nhiễu vận tốc ±0,25 m/s giữa hai khung hình = ±7,5 m/s² ⇒ gần
        như mọi xe bị gắn cờ 'phanh gấp'/'tăng tốc đột ngột' (305/307)."""
        clf = BehaviorClassifier(BehaviorConfig(enabled=False))
        assert clf.classify(self._track([9.0] * 150)).label == "binh_thuong"

    def test_real_hard_brake_is_flagged(self):
        clf = BehaviorClassifier(BehaviorConfig(enabled=False))
        speeds = [10.0] * 40 + [max(1.0, 10.0 - 5.0 * i / 30) for i in range(60)] + [1.0] * 40
        assert clf.classify(self._track(speeds)).label == "phanh_gap"

    def test_features_finite(self):
        f = extract_features(self._track([8.0] * 60))
        assert all(math.isfinite(v) for v in f.values())


class TestCalibration:
    def _input(self, noise=0.0):
        H = np.array([[0.02, 0.001, -10.0], [0.0005, 0.05, -20.0], [0.0, 0.0008, 1.0]])
        rng = np.random.default_rng(1)
        img = np.array([[x, y] for x in (200, 600, 1000, 1400) for y in (500, 700, 900)], float)
        hom = np.column_stack([img, np.ones(len(img))]) @ H.T
        world = hom[:, :2] / hom[:, 2:3]
        return CalibrationInput(image_points=img + rng.normal(0, noise, img.shape),
                                world_points=world)

    def test_exact_points_give_zero_error(self):
        _H, rep = evaluate_calibration(self._input())
        assert rep["fit"]["rmse_m"] < 1e-6
        assert rep["leave_one_out"]["rmse_m"] < 1e-6

    def test_leave_one_out_exposes_noise(self):
        _H, rep = evaluate_calibration(self._input(noise=3.0))
        assert rep["leave_one_out"]["rmse_m"] >= rep["fit"]["rmse_m"]

    def test_four_points_warns(self):
        inp = self._input()
        inp.image_points, inp.world_points = inp.image_points[:4], inp.world_points[:4]
        _H, rep = evaluate_calibration(inp)
        assert "warning" in rep


class TestReview:
    def test_kappa_perfect_and_chance(self):
        assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)

    def test_score_reviewed(self, tmp_path):
        cands = tmp_path / "c.csv"
        with open(cands, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["candidate_id", "source", "t_s"])
            w.writeheader()
            for cid, src in (("C1", "system"), ("C2", "system"), ("C3", "reference"),
                             ("C4", "both")):
                w.writerow({"candidate_id": cid, "source": src, "t_s": 1})
        for name, labs in (("a", {"C1": "near_miss", "C2": "normal", "C3": "near_miss", "C4": "collision"}),
                           ("b", {"C1": "near_miss", "C2": "normal", "C3": "near_miss", "C4": "collision"})):
            with open(tmp_path / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["candidate_id", "label"])
                for k, v in labs.items():
                    w.writerow([k, v])
        rep = score_reviewed(cands, [tmp_path / "a.csv", tmp_path / "b.csv"], 3600.0)
        assert rep["kappa_5class"] == 1.0
        assert rep["precision"] == {"value": round(2 / 3, 4), "num": 2, "den": 3}
        assert rep["recall"] == {"value": round(2 / 3, 4), "num": 2, "den": 3}
        assert rep["false_alarms_per_hour"] == 1.0


class TestSimulatorScenario:
    def test_pedestrians_have_a_waiting_point(self):
        """Lỗi kịch bản cũ: lối đi bộ không bao giờ chạm vòng vạch dừng 9 m, nên
        không người đi bộ nào chờ đèn."""
        ped = VehicleSpec(vid=1, cls=V.PEDESTRIAN, waypoints=[(-10.5, 41.0), (10.5, 41.0)],
                          t_start=0.0, desired_speed=1.4, approach="Vạch bộ hành Bắc",
                          signal_group="PED")
        assert ped.stop_line_s < ped.path_length
        assert STOP_LINE_RADIUS > 12.2      # vạch dừng nằm ngoài vạch bộ hành

    def test_no_pass_through_without_crash(self):
        """Hai thân xe chồng nhau thì cả hai phải được ghi là va chạm (dừng lại),
        không được đi xuyên qua nhau rồi giữ tốc độ như bản cũ."""
        from saferoad.conflict.geometry import series_ttc_and_gap

        report: dict = {}
        sim = simulate(build_scenario(duration=60.0, seed=3), 60.0, report=report)
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                a, b = sim[i], sim[j]
                if V.PEDESTRIAN == a.cls == b.cls:
                    continue
                t0, t1 = max(a.times[0], b.times[0]), min(a.times[-1], b.times[-1])
                if t1 - t0 < 0.2:
                    continue
                m = (a.times >= t0) & (a.times <= t1)
                ib = np.clip(np.searchsorted(b.times, a.times[m]), 0, len(b.times) - 1)
                _ttc, gap = series_ttc_and_gap(a.positions[m], a.velocities[m], a.cls,
                                               b.positions[ib], b.velocities[ib], b.cls)
                if gap.min() <= 0.0:
                    assert a.crash_t is not None and b.crash_t is not None
        assert {"n_crashes", "n_dropped_at_entry", "n_completed_route"} <= set(report)


class TestCvatStitching:
    def test_tracks_are_linked_across_chunks(self, tmp_path):
        """Nhãn CVAT đánh lại track_id ở mỗi đoạn 30 khung hình, chồng 5 khung.
        Một xe đi qua 3 đoạn phải thành MỘT quỹ đạo, không phải ba."""
        from saferoad.data.cvat_video import load_cvat_video

        images, anns = [], []
        img_id = ann_id = 0
        for chunk, start in enumerate((0, 25, 50)):
            for f in range(start, start + 30):
                img_id += 1
                images.append({"id": img_id, "file_name": f"frame_{f:06d}.PNG",
                               "width": 1920, "height": 1080})
                for local_tid, (x0, speed) in enumerate(((100.0, 5.0), (900.0, -4.0))):
                    ann_id += 1
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                 "bbox": [x0 + speed * f, 500.0, 60.0, 40.0],
                                 "attributes": {"track_id": local_tid + 10 * chunk}})
        path = tmp_path / "ann.json"
        path.write_text(json.dumps({"images": images, "annotations": anns,
                                    "categories": [{"id": 1, "name": "Car"}]}))
        seq, stats = load_cvat_video(path)
        assert stats["n_local_tracks"] == 6
        assert len(seq.tracks) == 2
        assert sorted(len(t.frames) for t in seq.tracks.values()) == [80, 80]
        assert all(len(d) == 2 for d in seq.detections.values())
