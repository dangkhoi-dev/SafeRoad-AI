"""ByteTrack — multi-object tracking hai vòng ghép.

Ý tưởng cốt lõi của ByteTrack (Zhang et al., ECCV 2022): thay vì vứt bỏ mọi
detection có điểm thấp, hãy dùng chúng ở **vòng ghép thứ hai** để nối lại các
track đang bị che khuất. Với giao lộ Việt Nam — xe máy chen dày, che nhau liên
tục — điều này giữ ID ổn định hơn hẳn so với SORT thuần.

Cài đặt tự viết (không phụ thuộc thư viện ngoài) để repo tự chứa và để có thể
gắn thêm phần quy đổi mặt đất ngay trong track.

Tài liệu: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", ECCV 2022. https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import math

import numpy as np

from ..config import TrackingConfig
from ..types import Detection, Track, TrackState, VehicleClass, iou
from .kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah


class _Strack:
    """Track nội bộ của ByteTrack, kèm trạng thái Kalman."""

    __slots__ = (
        "track_id", "cls", "mean", "cov", "score", "state",
        "hits", "age", "time_since_update", "start_frame", "history",
    )

    def __init__(self, track_id: int, det: Detection, kf: KalmanBoxFilter, frame_idx: int):
        self.track_id = track_id
        self.cls = det.cls
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.bbox))
        self.score = det.score
        self.state = "tentative"     # tentative | confirmed | lost | removed
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_idx
        self.history: list[TrackState] = []

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return xyah_to_xyxy(self.mean)

    def predict(self, kf: KalmanBoxFilter) -> None:
        # Với track đã mất dấu, đóng băng vận tốc chiều cao để bbox không "nở".
        if self.state != "confirmed" or self.time_since_update > 0:
            self.mean[7] = 0.0
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanBoxFilter, det: Detection) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det.bbox))
        self.score = det.score
        self.hits += 1
        self.time_since_update = 0


def _expand(box: tuple[float, float, float, float], extra: float = 0.0) -> tuple[float, float, float, float]:
    """Nới rộng bbox một biên tỉ lệ với kích thước của chính nó.

    Vì sao cần: IoU rất nhạy với nhiễu khi bbox nhỏ. Một xe máy ở xa camera chỉ
    chiếm ~12 × 10 px; detector lệch 1.5 px mỗi cạnh đã đủ kéo IoU giữa hai
    frame liên tiếp từ 0.93 xuống dưới 0.2, khiến tracker cắt track thành hàng
    chục mảnh. Nới cả hai bbox thêm một biên nhỏ trước khi tính IoU làm phép đo
    ổn định ở mọi kích thước, mà vẫn không lẫn hai xe khác làn vì các làn cách
    nhau xa hơn biên này nhiều.
    """
    x1, y1, x2, y2 = box
    margin = 2.0 + 0.10 * math.sqrt(max((x2 - x1) * (y2 - y1), 1.0)) + extra
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _iou_cost(tracks: list[_Strack], dets: list[Detection]) -> np.ndarray:
    """Ma trận chi phí ``1 - IoU`` (trên bbox đã nới biên) giữa track và detection."""
    if not tracks or not dets:
        return np.empty((len(tracks), len(dets)), dtype=float)

    det_boxes = [_expand(d.bbox) for d in dets]
    cost = np.ones((len(tracks), len(dets)), dtype=float)
    for i, t in enumerate(tracks):
        # Track đã mất dấu lâu thì bbox dự đoán kém tin cậy — nới thêm biên
        # theo số frame mất dấu để còn cơ hội bắt lại.
        tb = _expand(t.bbox, extra=1.5 * min(t.time_since_update, 8))
        for j, d in enumerate(dets):
            # Không ghép khác lớp: xe máy không thể "biến" thành ô tô.
            if t.cls is not d.cls:
                continue
            cost[i, j] = 1.0 - iou(tb, det_boxes[j])
    return cost


def _greedy_match(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Ghép tham lam theo chi phí tăng dần.

    Dùng thuật toán tham lam thay vì Hungarian: với số đối tượng mỗi frame ở
    giao lộ (thường < 60) chênh lệch chất lượng không đáng kể, nhưng tránh được
    phụ thuộc scipy.optimize và nhanh hơn đáng kể.
    """
    n_t, n_d = cost.shape if cost.size else (cost.shape[0], cost.shape[1])
    matches: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()

    if cost.size:
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for i, j in order:
            i, j = int(i), int(j)
            if i in used_t or j in used_d:
                continue
            if cost[i, j] > thresh:
                break
            matches.append((i, j))
            used_t.add(i)
            used_d.add(j)

    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_d = [j for j in range(n_d) if j not in used_d]
    return matches, unmatched_t, unmatched_d


class ByteTracker:
    """Multi-object tracker theo sơ đồ ByteTrack.

    Cách dùng::

        tracker = ByteTracker(cfg)
        for frame_idx, dets in enumerate(stream):
            tracks = tracker.update(dets, frame_idx, t=frame_idx / fps)
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.kf = KalmanBoxFilter()
        self.tracked: list[_Strack] = []
        self.lost: list[_Strack] = []
        self._next_id = 1
        #: Lịch sử đầy đủ của mọi track từng xuất hiện, kể cả đã kết thúc.
        self.all_tracks: dict[int, Track] = {}

    # ------------------------------------------------------------------ #
    def update(self, detections: list[Detection], frame_idx: int, t: float) -> list[Track]:
        """Cập nhật tracker với detection của một frame.

        Trả về danh sách các ``Track`` đã confirmed và đang hoạt động.
        """
        cfg = self.cfg

        # Chia detection theo điểm tin cậy — đây là điểm mấu chốt của ByteTrack.
        high = [d for d in detections if d.score >= cfg.track_high_thresh]
        low = [
            d for d in detections
            if cfg.track_low_thresh <= d.score < cfg.track_high_thresh
        ]

        # Bước 1: dự đoán tất cả track hiện có.
        pool = self.tracked + self.lost
        for tr in pool:
            tr.predict(self.kf)

        # Bước 2: ghép vòng 1 với detection điểm cao.
        cost = _iou_cost(pool, high)
        matches, u_track, u_det = _greedy_match(cost, 1.0 - (1.0 - cfg.match_thresh))
        activated: list[_Strack] = []
        for ti, di in matches:
            tr = pool[ti]
            tr.update(self.kf, high[di])
            if tr.state == "tentative" and tr.hits >= cfg.min_hits:
                tr.state = "confirmed"
            elif tr.state == "lost":
                tr.state = "confirmed"
            activated.append(tr)

        # Bước 3: ghép vòng 2 — track còn lại với detection điểm THẤP.
        # Đây là bước cứu các track đang bị che khuất một phần.
        rest = [pool[i] for i in u_track if pool[i].state == "confirmed"]
        cost2 = _iou_cost(rest, low)
        matches2, u_track2, _ = _greedy_match(cost2, 0.5)
        for ti, di in matches2:
            tr = rest[ti]
            tr.update(self.kf, low[di])
            tr.state = "confirmed"
            activated.append(tr)

        # Bước 4: track không ghép được → chuyển sang lost hoặc xoá.
        matched_ids = {id(tr) for tr in activated}
        still_lost: list[_Strack] = []
        for tr in pool:
            if id(tr) in matched_ids:
                continue
            if tr.state == "tentative":
                continue  # track non chưa xác nhận thì bỏ luôn
            tr.state = "lost"
            if tr.time_since_update <= cfg.track_buffer:
                still_lost.append(tr)

        # Bước 5: khởi tạo track mới từ detection điểm cao chưa dùng.
        for di in u_det:
            det = high[di]
            if det.score < cfg.new_track_thresh:
                continue
            tr = _Strack(self._next_id, det, self.kf, frame_idx)
            self._next_id += 1
            activated.append(tr)

        self.tracked = [tr for tr in activated if tr.state in ("confirmed", "tentative")]
        self.lost = still_lost

        # Bước 6: ghi lịch sử và trả kết quả.
        out: list[Track] = []
        for tr in self.tracked:
            if tr.state != "confirmed":
                continue
            x1, y1, x2, y2 = tr.bbox
            state = TrackState(
                frame_idx=frame_idx,
                t=t,
                bbox=(x1, y1, x2, y2),
                anchor=(0.5 * (x1 + x2), y2),
                ground=(0.0, 0.0),   # geometry/homography sẽ điền sau
                score=tr.score,
            )
            record = self.all_tracks.get(tr.track_id)
            if record is None:
                record = Track(track_id=tr.track_id, cls=tr.cls)
                self.all_tracks[tr.track_id] = record
            record.history.append(state)
            record.hits = tr.hits
            record.age = tr.age
            record.time_since_update = tr.time_since_update
            record.confirmed = True
            out.append(record)
        return out

    def reset(self) -> None:
        self.tracked.clear()
        self.lost.clear()
        self.all_tracks.clear()
        self._next_id = 1
