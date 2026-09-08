"""Phát hiện xung đột (near-miss) theo **episode**.

Luồng xử lý mỗi frame::

    tracks → lọc sơ bộ → sinh cặp ứng viên → tính TTC/PET
           → mở/duy trì episode → đóng episode → ConflictEvent

Vì sao phải gom theo episode
----------------------------
Một tình huống suýt va chạm kéo dài 1-2 giây. Ở 30 FPS, nếu phát ra một sự kiện
mỗi khung hình thoả ngưỡng thì **một** tình huống sẽ biến thành 30-60 "sự kiện",
làm mọi thống kê vô nghĩa.

Quan trọng hơn: mức nguy hiểm và thời điểm đại diện của một tình huống chỉ xác
định được khi nó đã **kết thúc**. TTC nhỏ nhất thường rơi vào giữa episode, còn
khoảng cách nhỏ nhất rơi vào cuối. Phát sự kiện ngay lúc TTC vừa chạm ngưỡng thì
ta chưa biết hai xe rồi có thực sự tới gần nhau không (một bên có thể kịp phanh);
còn chờ tới lúc đã gần nhất mới phát thì nhãn thời gian lại trễ hơn khoảnh khắc
nguy hiểm thật.

Cách làm ở đây: **theo dõi cả episode**, đóng lại khi điều kiện xung đột chấm
dứt, rồi phát **một** sự kiện duy nhất gán nhãn tại **thời điểm TTC nhỏ nhất** —
đúng khoảnh khắc căng thẳng nhất của tình huống.

Cho hiển thị thời gian thực, :meth:`ConflictDetector.active_alerts` trả về các
cặp đang xung đột ngay lúc này; overlay và dashboard dùng nó, độc lập với luồng
sự kiện đã chốt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ConflictConfig
from ..geometry.trajectory import heading_change_rate
from ..types import (
    ConflictEvent,
    ConflictType,
    RiskLevel,
    Track,
    VehicleClass,
    pair_key,
)
from .indicators import (
    NO_COLLISION,
    approach_angle,
    classify_conflict,
    post_encroachment_time,
    relative_speed,
    surface_gap,
    time_to_collision,
)


@dataclass
class _Episode:
    """Một đợt xung đột đang diễn ra giữa hai track."""

    cls_a: VehicleClass
    cls_b: VehicleClass
    min_ttc: float = NO_COLLISION
    min_pet: float | None = None
    min_gap: float = float("inf")
    last_seen_frame: int = -1
    n_frames: int = 0

    # Thông tin tại thời điểm căng thẳng nhất (TTC nhỏ nhất).
    peak_frame: int = -1
    peak_t: float = 0.0
    peak_location: tuple[float, float] = (0.0, 0.0)
    peak_location_px: tuple[float, float] = (0.0, 0.0)
    peak_angle: float = 0.0
    peak_rel_speed: float = 0.0
    peak_type: ConflictType = ConflictType.CROSSING

    #: TTC ở frame gần nhất — dùng cho cảnh báo thời gian thực.
    current_ttc: float = NO_COLLISION
    #: Mức rủi ro tạm tính để overlay tô màu ngay khi đang diễn ra.
    live_level: RiskLevel = RiskLevel.LOW


class ConflictDetector:
    """Sinh ``ConflictEvent`` từ các track đang hoạt động.

    Cách dùng::

        detector = ConflictDetector(cfg.conflict)
        for frame_idx, tracks in stream:
            events = detector.update(tracks, frame_idx, t, risk_scorer=scorer)
        events += detector.flush(risk_scorer=scorer)   # đóng nốt episode dở
    """

    def __init__(self, cfg: ConflictConfig):
        self.cfg = cfg
        self.open: dict[tuple[int, int], _Episode] = {}
        self._event_seq = 0
        #: Mọi sự kiện đã chốt, phục vụ thống kê cuối phiên.
        self.events: list[ConflictEvent] = []

    # ------------------------------------------------------------------ #
    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        t: float,
        risk_scorer=None,
    ) -> list[ConflictEvent]:
        """Xử lý một frame; trả về các sự kiện được **chốt** ở frame này."""
        cfg = self.cfg

        active = [
            tr for tr in tracks
            if len(tr.history) >= 3 and tr.history[-1].frame_idx == frame_idx
        ]
        turn_rate = {tr.track_id: heading_change_rate(tr) for tr in active}
        seen: set[tuple[int, int]] = set()

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                tr_a, tr_b = active[i], active[j]
                st_a, st_b = tr_a.last, tr_b.last

                # Lọc nhanh theo khoảng cách để tránh O(n²) đắt đỏ.
                if math.dist(st_a.ground, st_b.ground) > cfg.max_pair_distance:
                    continue
                if max(st_a.speed, st_b.speed) < cfg.min_abs_speed:
                    continue

                key = pair_key(tr_a.track_id, tr_b.track_id)
                gap_now = surface_gap(st_a, st_b, tr_a.cls, tr_b.cls)

                # Episode đang mở thì luôn cập nhật khoảng cách nhỏ nhất, kể cả
                # ở những frame điều kiện xung đột đã tắt. Khoảnh khắc hai xe
                # gần nhau NHẤT thường rơi vào lúc TTC đã hết ý nghĩa (chúng
                # đang lướt qua nhau); nếu chỉ đo trong các frame "đang xung
                # đột" thì min_gap bị đánh giá cao hơn thực tế và cổng khoảng
                # cách sẽ loại oan chính những tình huống sát sườn nhất.
                open_ep = self.open.get(key)
                if open_ep is not None:
                    open_ep.min_gap = min(open_ep.min_gap, gap_now)

                rel_v = relative_speed(st_a, st_b)
                if rel_v < cfg.min_rel_speed:
                    continue

                ttc = time_to_collision(st_a, st_b, tr_a.cls, tr_b.cls, cfg.horizon)
                pet_result = post_encroachment_time(
                    st_a, st_b, tr_a.cls, tr_b.cls, cfg.horizon
                )
                pet = pet_result[0] if pet_result else None

                # Điều kiện conflict (theo poster): TTC < 3.0s HOẶC PET < 1.5s.
                in_conflict = (ttc < cfg.ttc_threshold) or (
                    pet is not None and pet < cfg.pet_threshold
                )
                if not in_conflict:
                    continue

                seen.add(key)
                ep = open_ep
                if ep is None:
                    ep = _Episode(cls_a=tr_a.cls, cls_b=tr_b.cls)
                    self.open[key] = ep

                ep.last_seen_frame = frame_idx
                ep.n_frames += 1
                ep.current_ttc = ttc
                ep.min_gap = min(ep.min_gap, gap_now)
                if pet is not None:
                    ep.min_pet = pet if ep.min_pet is None else min(ep.min_pet, pet)
                ep.live_level = (
                    RiskLevel.HIGH if ttc < 1.0
                    else RiskLevel.MEDIUM if ttc < 2.0
                    else RiskLevel.LOW
                )

                # Ghi lại "đỉnh" — thời điểm TTC nhỏ nhất trong cả episode.
                if ttc < ep.min_ttc or ep.peak_frame < 0:
                    ep.min_ttc = min(ep.min_ttc, ttc)
                    ep.peak_frame = frame_idx
                    ep.peak_t = t
                    ep.peak_angle = approach_angle(st_a, st_b)
                    ep.peak_rel_speed = rel_v
                    ep.peak_type = classify_conflict(
                        st_a, st_b, tr_a.cls, tr_b.cls,
                        turn_rate.get(tr_a.track_id, 0.0),
                        turn_rate.get(tr_b.track_id, 0.0),
                        cfg.angle_rear_end, cfg.angle_crossing_min,
                        cfg.angle_crossing_max, cfg.angle_head_on,
                    )
                    if pet_result is not None:
                        ep.peak_location = pet_result[1]
                    else:
                        ep.peak_location = (
                            0.5 * (st_a.ground[0] + st_b.ground[0]),
                            0.5 * (st_a.ground[1] + st_b.ground[1]),
                        )
                    ep.peak_location_px = (
                        0.5 * (st_a.anchor[0] + st_b.anchor[0]),
                        0.5 * (st_a.anchor[1] + st_b.anchor[1]),
                    )

        # --- Đóng các episode đã kết thúc ------------------------------ #
        # Cho phép "hụt" vài frame trước khi đóng: detector có thể trượt một
        # nhịp, và đóng sớm sẽ cắt một tình huống thành hai sự kiện.
        closed: list[ConflictEvent] = []
        for key in list(self.open):
            ep = self.open[key]
            if key in seen:
                continue
            if frame_idx - ep.last_seen_frame <= self.GRACE_FRAMES:
                continue
            event = self._close(key, ep, risk_scorer)
            del self.open[key]
            if event is not None:
                closed.append(event)

        return closed

    #: Số frame cho phép "mất dấu" trước khi coi episode đã kết thúc.
    GRACE_FRAMES = 5
    #: Hai episode của CÙNG một cặp cách nhau dưới ngần này giây được gộp làm
    #: một: chúng gần như chắc chắn là hai đoạn của cùng một tình huống bị cắt
    #: rời do detector trượt vài nhịp, và đếm thành hai sự kiện sẽ vừa thổi
    #: phồng số liệu vừa tạo báo động giả khi chấm điểm.
    MERGE_WINDOW_S = 8.0

    # ------------------------------------------------------------------ #
    def _close(self, key, ep: _Episode, risk_scorer=None) -> ConflictEvent | None:
        """Chốt một episode thành ``ConflictEvent``, hoặc bỏ nếu không đủ điều kiện."""
        cfg = self.cfg

        # Cổng khoảng cách: hai xe phải THỰC SỰ đến sát nhau. TTC nhỏ chỉ là
        # phép ngoại suy — nếu một bên kịp phanh và hai xe chưa bao giờ tới gần,
        # đó là tình huống được xử lý tốt, không phải sự cố cần ghi nhận.
        if ep.min_gap > cfg.proximity_gate:
            return None
        # Episode quá ngắn thường chỉ là nhiễu ước lượng vận tốc trong 1-2 frame.
        if ep.n_frames < 3:
            return None
        if ep.peak_frame < 0:
            return None

        ttc_out = None if ep.min_ttc == NO_COLLISION else round(ep.min_ttc, 3)

        # Gộp với sự kiện trước đó của cùng cặp nếu quá gần nhau về thời gian.
        for prev in reversed(self.events):
            if prev.t < ep.peak_t - self.MERGE_WINDOW_S:
                break
            if (prev.track_a, prev.track_b) != key:
                continue
            prev_ttc = float("inf") if prev.ttc is None else prev.ttc
            new_ttc = float("inf") if ttc_out is None else ttc_out
            if new_ttc < prev_ttc:
                # Đoạn mới nguy hiểm hơn — cập nhật bản ghi cũ tại chỗ.
                prev.ttc = ttc_out
                prev.t = round(ep.peak_t, 3)
                prev.frame_idx = ep.peak_frame
                prev.conflict_type = ep.peak_type
                prev.location = ep.peak_location
                prev.location_px = ep.peak_location_px
                prev.approach_angle = round(ep.peak_angle, 1)
                prev.rel_speed = round(ep.peak_rel_speed, 2)
            prev.gap = round(min(prev.gap, ep.min_gap), 2)
            if ep.min_pet is not None:
                prev.pet = (
                    round(ep.min_pet, 3) if prev.pet is None
                    else round(min(prev.pet, ep.min_pet), 3)
                )
            if risk_scorer is not None:
                risk_scorer.score_event(prev)
            return None

        self._event_seq += 1
        event = ConflictEvent(
            event_id=f"CF{self._event_seq:05d}",
            frame_idx=ep.peak_frame,
            t=round(ep.peak_t, 3),
            track_a=key[0],
            track_b=key[1],
            cls_a=ep.cls_a,
            cls_b=ep.cls_b,
            ttc=ttc_out,
            pet=None if ep.min_pet is None else round(ep.min_pet, 3),
            conflict_type=ep.peak_type,
            risk_score=0.0,
            risk_level=RiskLevel.NONE,
            location=ep.peak_location,
            location_px=ep.peak_location_px,
            approach_angle=round(ep.peak_angle, 1),
            rel_speed=round(ep.peak_rel_speed, 2),
            gap=round(ep.min_gap, 2),
        )
        if risk_scorer is not None:
            risk_scorer.score_event(event)
        self.events.append(event)
        return event

    def flush(self, risk_scorer=None) -> list[ConflictEvent]:
        """Đóng mọi episode còn dở khi video kết thúc."""
        out: list[ConflictEvent] = []
        for key in list(self.open):
            event = self._close(key, self.open[key], risk_scorer)
            del self.open[key]
            if event is not None:
                out.append(event)
        return out

    # ------------------------------------------------------------------ #
    def active_alerts(
        self, frame_idx: int, max_age: int = 3
    ) -> dict[tuple[int, int], _Episode]:
        """Các cặp đang xung đột — dùng cho overlay/dashboard thời gian thực.

        Tách khỏi luồng sự kiện đã chốt: cảnh báo trên màn hình phải xuất hiện
        **ngay lúc** tình huống đang diễn ra, còn bản ghi thống kê chỉ chốt sau
        khi tình huống kết thúc.
        """
        return {
            key: ep for key, ep in self.open.items()
            if frame_idx - ep.last_seen_frame <= max_age
        }

    @property
    def total_events(self) -> int:
        return len(self.events)

    def severe_events(self, ttc_threshold: float = 1.5) -> list[ConflictEvent]:
        """Near-miss nghiêm trọng — TTC < ngưỡng (mặc định 1.5 s theo poster)."""
        return [e for e in self.events if e.ttc is not None and e.ttc < ttc_threshold]
