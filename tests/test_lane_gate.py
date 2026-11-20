"""Cổng cùng làn: xe song song ở làn kề không được gán là đâm đuôi."""
from saferoad.conflict.indicators import approach_angle, lateral_offset
from saferoad.types import TrackState


def _st(x, y, vx, vy):
    return TrackState(frame_idx=0, t=0.0, bbox=(0, 0, 1, 1), anchor=(0, 0),
                      ground=(x, y), velocity=(vx, vy))


def test_parallel_adjacent_lane_offset_is_lane_width():
    a = _st(0.0, 0.0, 0.0, 12.0)
    b = _st(3.5, 5.0, 0.0, 8.0)          # làn kề, chạy cùng hướng
    assert approach_angle(a, b) < 1e-6   # góc tiếp cận ≈ 0: luật cũ gọi là đâm đuôi
    assert abs(lateral_offset(a, b) - 3.5) < 1e-6


def test_same_lane_offset_is_small():
    a = _st(0.0, 0.0, 0.0, 12.0)
    b = _st(0.4, 6.0, 0.0, 9.0)
    assert lateral_offset(a, b) < 0.5


def test_lateral_offset_invariant_to_heading():
    # Hướng đường chéo: độ lệch vẫn đo vuông góc với hướng di chuyển.
    a = _st(0.0, 0.0, 6.0, 6.0)
    k = 3.5 / 2 ** 0.5
    b = _st(-k, k, 6.0, 6.0)             # lệch vuông góc (-1,1)/√2 · 3.5 m
    assert abs(lateral_offset(a, b) - 3.5) < 1e-3


def test_rerun_does_not_overwrite_labeled_review_list(tmp_path):
    from saferoad.cli import _review_target

    p = tmp_path / "review_candidates_demo.csv"
    assert _review_target(p) == p                     # chưa có file: ghi thẳng
    p.write_text("candidate_id\n", encoding="utf-8")
    assert _review_target(p) == p                     # có file nhưng chưa ai gán nhãn
    (tmp_path / "labels_demo_khoi.csv").write_text("candidate_id,label\n", encoding="utf-8")
    assert _review_target(p) == tmp_path / "reruns" / p.name
