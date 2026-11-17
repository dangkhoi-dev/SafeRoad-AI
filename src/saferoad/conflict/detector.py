"""Phát hiện xung đột (near-miss) theo **episode**.

Luồng xử lý mỗi frame::

    tracks → lọc sơ bộ → sinh cặp ứng viên → tính TTC/PET ngoại suy
           → mở/duy trì episode → chờ hai xe qua hẳn vùng xung đột
           → đo PET từ quỹ đạo → phân loại kết cục → ConflictEvent

Vì sao phải gom theo episode
----------------------------
Một tình huống suýt va chạm kéo dài 1-2 giây. Ở 30 FPS, nếu phát ra một sự kiện
mỗi khung hình thoả ngưỡng thì **một** tình huống sẽ biến thành 30-60 "sự kiện",
làm mọi thống kê vô nghĩa. Hệ thống theo dõi cả episode, đóng lại khi điều kiện
xung đột chấm dứt, rồi phát **một** sự kiện gán nhãn tại **thời điểm TTC nhỏ
nhất** - khoảnh khắc căng thẳng nhất.

Vì sao phải chờ thêm sau khi episode kết thúc
---------------------------------------------
PET đúng nghĩa (thời gian từ lúc xe trước rời vùng xung đột tới lúc xe sau đi
vào) chỉ **đo** được khi xe sau đã thật sự vào vùng. Lúc điều kiện TTC vừa tắt,
xe sau thường chưa tới đó. Vì vậy episode chuyển sang trạng thái *chờ* và chỉ
được chốt khi phép đo PET đã có kết quả, hoặc khi hết thời gian chờ tối đa. Cảnh
báo thời gian thực trên màn hình không bị ảnh hưởng (xem :meth:`active_alerts`).

Đối soát luồng sự kiện
----------------------
Mọi episode đều được đếm vào ``funnel``: mở bao nhiêu, bị loại ở cổng nào, gộp
bao nhiêu, phát ra bao nhiêu. Nhờ đó báo cáo đánh giá chỉ ra được chính xác vì
sao số sự kiện xuất ra khác số sự kiện được chấm điểm.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

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
from .encroachment import (
    CONTACT_NONE,
    PET_INCOMPLETE,
    Encroachment,
    classify_contact,
    measure_pet,
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

#: Khoảng cách mặt-tới-mặt (m) coi như hai thân xe đã chạm nhau.
CONTACT_EPS = 0.05


@dataclass
class _Episode:
    """Một đợt xung đột đang diễn ra giữa hai track."""

    cls_a: VehicleClass
    cls_b: VehicleClass
    track_a: Track | None = None
    track_b: Track | None = None
    start_t: float = 0.0
    min_ttc: float = NO_COLLISION
    #: PET ngoại suy nhỏ nhất (giây) trong suốt episode.
    min_pet_pred: float | None = None
    min_gap: float = float("inf")
    min_gap_t: float = 0.0
    last_seen_frame: int = -1
    n_frames: int = 0
    hit_ttc: bool = False
    hit_pet_pred: bool = False

    # Thông tin tại thời điểm căng thẳng nhất (TTC nhỏ nhất).
    peak_frame: int = -1
    peak_t: float = 0.0
    peak_location: tuple[float, float] = (0.0, 0.0)
    peak_location_px: tuple[float, float] = (0.0, 0.0)
    peak_angle: float = 0.0
    peak_rel_speed: float = 0.0
    peak_type: ConflictType = ConflictType.CROSSING

    #: TTC ở frame gần nhất - dùng cho cảnh báo thời gian thực.
    current_ttc: float = NO_COLLISION
    #: Mức rủi ro tạm tính để overlay tô màu ngay khi đang diễn ra.
    live_level: RiskLevel = RiskLevel.LOW

    #: Frame bắt đầu chờ đo PET (-1 = episode còn đang diễn ra).
    pending_since: int = -1
    pending_t: float = 0.0
    last_measure_frame: int = -10**9
    encroachment: Encroachment | None = None


@dataclass
class DecisionLog:
    """Lý do chốt hoặc loại của từng episode - dùng cho bảng đối soát."""

    counts: Counter = field(default_factory=Counter)

    def add(self, key: str) -> None:
        self.counts[key] += 1

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)


def _inside(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Điểm nằm trong đa giác (ray casting)."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


class ConflictDetector:
    """Sinh ``ConflictEvent`` từ các track đang hoạt động.

    Cách dùng::

        detector = ConflictDetector(cfg.conflict)
        for frame_idx, tracks in stream:
            events = detector.update(tracks, frame_idx, t, risk_scorer=scorer)
        events += detector.flush(risk_scorer=scorer)   # đóng nốt episode dở
    """

    #: Số frame cho phép "mất dấu" trước khi coi episode đã kết thúc.
    GRACE_FRAMES = 5
    #: Hai episode của CÙNG một cặp cách nhau dưới ngần này giây được gộp làm
    #: một: chúng gần như chắc chắn là hai đoạn của cùng một tình huống bị cắt
    #: rời do detector trượt vài nhịp.
    MERGE_WINDOW_S = 8.0

    def __init__(self, cfg: ConflictConfig):
        self.cfg = cfg
        self.open: dict[tuple[int, int], _Episode] = {}
        self._event_seq = 0
        #: Mọi sự kiện đã chốt, phục vụ thống kê cuối phiên.
        self.events: list[ConflictEvent] = []
        #: Bảng đối soát: mỗi episode rơi vào đúng một ô.
        self.funnel = DecisionLog()
        self._last_per_frame: dict[tuple[int, int], float] = {}
        region = cfg.analysis_region_px or []
        self._region = [(float(x), float(y)) for x, y in region] if len(region) >= 3 else None

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
            and (self._region is None or _inside(tr.history[-1].anchor, self._region))
        ]
        turn_rate = {tr.track_id: heading_change_rate(tr) for tr in active}
        seen: set[tuple[int, int]] = set()
        closed_now: list[ConflictEvent] = []

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                tr_a, tr_b = active[i], active[j]
                # Khoá cặp không thứ tự: luôn đặt track id nhỏ hơn làm "a".
                if tr_a.track_id > tr_b.track_id:
                    tr_a, tr_b = tr_b, tr_a
                st_a, st_b = tr_a.last, tr_b.last

                if (cfg.ignore_pedestrian_pairs
                        and tr_a.cls == VehicleClass.PEDESTRIAN
                        and tr_b.cls == VehicleClass.PEDESTRIAN):
                    continue
                if math.dist(st_a.ground, st_b.ground) > cfg.max_pair_distance:
                    continue
                if max(st_a.speed, st_b.speed) < cfg.min_abs_speed:
                    continue

                key = pair_key(tr_a.track_id, tr_b.track_id)
                gap_now = surface_gap(st_a, st_b, tr_a.cls, tr_b.cls)

                # Episode đang mở (kể cả đang chờ đo PET) thì luôn cập nhật
                # khoảng cách nhỏ nhất: khoảnh khắc hai xe gần nhau NHẤT thường
                # rơi vào lúc TTC đã hết ý nghĩa vì chúng đang lướt qua nhau.
                open_ep = self.open.get(key)
                if open_ep is not None and gap_now < open_ep.min_gap:
                    open_ep.min_gap = gap_now
                    open_ep.min_gap_t = t

                rel_v = relative_speed(st_a, st_b)
                if rel_v < cfg.min_rel_speed:
                    continue

                ttc = time_to_collision(st_a, st_b, tr_a.cls, tr_b.cls, cfg.horizon)
                pet_result = post_encroachment_time(
                    st_a, st_b, tr_a.cls, tr_b.cls, cfg.horizon
                )
                pet = pet_result[0] if pet_result else None

                hit_ttc = ttc < cfg.ttc_threshold
                hit_pet = pet is not None and pet < cfg.pet_threshold
                if not (hit_ttc or hit_pet):
                    continue

                if not cfg.episode_grouping:
                    # Ablation: phát một sự kiện mỗi khung hình thoả ngưỡng (chỉ
                    # chống trùng trong ``dedup_window``) - cách làm ngây thơ mà
                    # gom episode thay thế.
                    ev = self._per_frame_event(key, tr_a, tr_b, st_a, st_b, frame_idx, t,
                                               ttc, pet, rel_v, gap_now, risk_scorer)
                    if ev is not None:
                        closed_now.append(ev)
                    continue

                seen.add(key)
                ep = open_ep
                if ep is None:
                    ep = _Episode(cls_a=tr_a.cls, cls_b=tr_b.cls,
                                  track_a=tr_a, track_b=tr_b, start_t=t)
                    ep.min_gap = gap_now
                    ep.min_gap_t = t
                    self.open[key] = ep
                    self.funnel.add("episodes_opened")
                ep.pending_since = -1            # tình huống còn đang diễn ra
                ep.hit_ttc |= hit_ttc
                ep.hit_pet_pred |= hit_pet

                ep.last_seen_frame = frame_idx
                ep.n_frames += 1
                ep.current_ttc = ttc
                if pet is not None:
                    ep.min_pet_pred = pet if ep.min_pet_pred is None else min(ep.min_pet_pred, pet)
                ep.live_level = (
                    RiskLevel.HIGH if ttc < 1.0
                    else RiskLevel.MEDIUM if ttc < 2.0
                    else RiskLevel.LOW
                )

                # Ghi lại "đỉnh" - thời điểm TTC nhỏ nhất trong cả episode.
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

        # --- Episode đã kết thúc: chờ đo PET rồi mới chốt ----------------- #
        closed: list[ConflictEvent] = closed_now
        for key in list(self.open):
            ep = self.open[key]
            if key in seen:
                continue
            if frame_idx - ep.last_seen_frame <= self.GRACE_FRAMES:
                continue
            if ep.pending_since < 0:
                ep.pending_since = frame_idx
                ep.pending_t = t
            if not self._ready_to_close(ep, frame_idx, t):
                continue
            event = self._close(key, ep, risk_scorer, now=t)
            del self.open[key]
            if event is not None:
                closed.append(event)

        return closed

    # ------------------------------------------------------------------ #
    def _measure(self, ep: _Episode, now: float, final: bool = False) -> Encroachment:
        """Đo PET từ lịch sử hai track trong cửa sổ quanh episode."""
        cfg = self.cfg
        t0 = ep.start_t - cfg.pet_lookback_s
        arrays = []
        continues = []
        for tr in (ep.track_a, ep.track_b):
            hist = [s for s in tr.history if s.t >= t0] if tr is not None else []
            if len(hist) < 2:
                return Encroachment(status=PET_INCOMPLETE)
            arrays.append((
                np.array([s.t for s in hist]),
                np.array([s.ground for s in hist]),
                np.array([s.velocity for s in hist]),
            ))
            # Track còn được cập nhật gần đây ⇒ quỹ đạo có thể còn tiếp diễn.
            continues.append((not final) and (now - hist[-1].t) < 0.25)
        (ta, pa, va), (tb, pb, vb) = arrays
        return measure_pet(
            ta, pa, va, ep.cls_a, tb, pb, vb, ep.cls_b,
            min_angle=cfg.pet_min_angle,
            a_continues=continues[0], b_continues=continues[1],
        )

    def _ready_to_close(self, ep: _Episode, frame_idx: int, now: float) -> bool:
        cfg = self.cfg
        waited = now - ep.pending_t
        if waited >= cfg.pet_max_wait_s:
            return True
        # Episode chắc chắn bị loại ở cổng khoảng cách thì không cần đo PET: chốt
        # ngay khi hai xe đã tách xa nhau (hoặc một bên đã mất dấu).
        if ep.min_gap > cfg.proximity_gate:
            a, b = ep.track_a, ep.track_b
            if a is None or b is None or not a.history or not b.history:
                return True
            if a.history[-1].frame_idx != frame_idx or b.history[-1].frame_idx != frame_idx:
                return True
            gap_now = surface_gap(a.last, b.last, a.cls, b.cls)
            return gap_now > ep.min_gap + 1.0
        # Đo thưa (mỗi 6 frame) để chi phí không phụ thuộc số frame chờ.
        if frame_idx - ep.last_measure_frame < 6:
            return False
        ep.last_measure_frame = frame_idx
        enc = self._measure(ep, now)
        ep.encroachment = enc
        if enc.status == PET_INCOMPLETE:
            return False
        # Cần thêm một quãng sau lúc chạm/gần nhất để đọc được động học sau đó
        # (phân biệt va chạm thật với chồng thân xe do sai số).
        return now - ep.min_gap_t >= cfg.contact_settle_s

    def _contact_class(self, ep: _Episode) -> str:
        if ep.min_gap > CONTACT_EPS:
            return CONTACT_NONE
        before: list[float] = []
        after: list[float] = []
        tc = ep.min_gap_t
        for tr in (ep.track_a, ep.track_b):
            pre = [s.speed for s in tr.history if tc - 1.2 <= s.t <= tc - 0.2]
            post = [s.speed for s in tr.history if tc + 0.5 <= s.t <= tc + 2.5]
            before.append(float(np.median(pre)) if pre else 0.0)
            # Track mất hẳn ngay sau va chạm (xe dừng, bị che) không chứng minh
            # được là đã dừng - giữ tốc độ trước đó để không gán "va chạm" bừa.
            after.append(float(min(post)) if post else before[-1])
        return classify_contact(0.0, tuple(before), tuple(after))

    def _close(self, key, ep: _Episode, risk_scorer=None, now: float = 0.0,
               final: bool = False) -> ConflictEvent | None:
        """Chốt một episode thành ``ConflictEvent``, hoặc bỏ nếu không đủ điều kiện."""
        cfg = self.cfg

        # Cổng khoảng cách: hai xe phải THỰC SỰ đến sát nhau.
        if ep.min_gap > cfg.proximity_gate:
            self.funnel.add("rejected_proximity_gate")
            return None
        # Episode quá ngắn thường chỉ là nhiễu ước lượng vận tốc trong 1-2 frame.
        if ep.n_frames < 3:
            self.funnel.add("rejected_too_short")
            return None
        if ep.peak_frame < 0:
            self.funnel.add("rejected_no_peak")
            return None

        enc = ep.encroachment
        if enc is None or final or enc.status == PET_INCOMPLETE:
            enc = self._measure(ep, now, final=True)
        pet_meas = enc.pet if enc.is_measured else None

        # PET dùng để quyết định: ưu tiên PET đo; chỉ khi không đo được (song
        # song, mất dấu) mới dùng lại PET ngoại suy.
        if not cfg.use_measured_pet:
            # Chế độ cũ (giữ cho ablation): chỉ dùng PET ngoại suy.
            pet_final = ep.min_pet_pred
            pet_source = "predicted" if pet_final is not None else ""
        elif pet_meas is not None:
            pet_final, pet_source = pet_meas, "measured"
        elif ep.min_pet_pred is not None and enc.status not in ("parallel", "no_zone"):
            pet_final, pet_source = ep.min_pet_pred, "predicted"
        else:
            pet_final, pet_source = None, ""

        hit_ttc = ep.min_ttc < cfg.ttc_threshold
        hit_pet = pet_final is not None and pet_final < cfg.pet_threshold
        if not (hit_ttc or hit_pet):
            # Episode mở ra chỉ vì PET ngoại suy, nhưng PET đo được thì vượt
            # ngưỡng: một bên đã phanh/nhường - tình huống được xử lý tốt.
            self.funnel.add("rejected_pet_measured_above_threshold")
            return None

        contact = self._contact_class(ep)
        ttc_out = None if ep.min_ttc == NO_COLLISION else round(ep.min_ttc, 3)
        pet_times = {
            k: (None if v is None else round(v, 3))
            for k, v in (("a_in", enc.t_a_in), ("a_out", enc.t_a_out),
                         ("b_in", enc.t_b_in), ("b_out", enc.t_b_out))
        }
        if enc.first:
            pet_times["first"] = enc.first

        # Gộp với sự kiện trước đó của cùng cặp nếu quá gần nhau về thời gian.
        for prev in reversed(self.events):
            if prev.t < ep.peak_t - self.MERGE_WINDOW_S:
                break
            if (prev.track_a, prev.track_b) != key:
                continue
            prev_ttc = float("inf") if prev.ttc is None else prev.ttc
            new_ttc = float("inf") if ttc_out is None else ttc_out
            if new_ttc < prev_ttc:
                prev.ttc = ttc_out
                prev.t = round(ep.peak_t, 3)
                prev.frame_idx = ep.peak_frame
                prev.conflict_type = ep.peak_type
                prev.location = ep.peak_location
                prev.location_px = ep.peak_location_px
                prev.approach_angle = round(ep.peak_angle, 1)
                prev.rel_speed = round(ep.peak_rel_speed, 2)
            prev.gap = round(min(prev.gap, ep.min_gap), 2)
            if pet_final is not None and (prev.pet is None or pet_final < prev.pet):
                prev.pet = round(pet_final, 3)
                prev.pet_source = pet_source
                prev.pet_measured = None if pet_meas is None else round(pet_meas, 3)
                prev.pet_status = enc.status
                prev.pet_times = pet_times
            if ep.min_pet_pred is not None:
                prev.pet_pred = (
                    round(ep.min_pet_pred, 3) if prev.pet_pred is None
                    else round(min(prev.pet_pred, ep.min_pet_pred), 3)
                )
            if contact != CONTACT_NONE:
                prev.contact = contact
            if risk_scorer is not None:
                risk_scorer.score_event(prev)
            self.funnel.add("merged_into_previous")
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
            pet=None if pet_final is None else round(pet_final, 3),
            conflict_type=ep.peak_type,
            risk_score=0.0,
            risk_level=RiskLevel.NONE,
            location=ep.peak_location,
            location_px=ep.peak_location_px,
            approach_angle=round(ep.peak_angle, 1),
            rel_speed=round(ep.peak_rel_speed, 2),
            gap=round(ep.min_gap, 2),
            pet_pred=None if ep.min_pet_pred is None else round(ep.min_pet_pred, 3),
            pet_measured=None if pet_meas is None else round(pet_meas, 3),
            pet_status=enc.status,
            pet_source=pet_source,
            pet_times=pet_times,
            contact=contact,
        )
        if risk_scorer is not None:
            risk_scorer.score_event(event)
        self.events.append(event)
        self.funnel.add("emitted")
        return event

    def _per_frame_event(self, key, tr_a, tr_b, st_a, st_b, frame_idx, t, ttc, pet,
                         rel_v, gap_now, risk_scorer) -> ConflictEvent | None:
        last = self._last_per_frame.get(key)
        if last is not None and t - last < self.cfg.dedup_window:
            return None
        self._last_per_frame[key] = t
        self.funnel.add("episodes_opened")
        if gap_now > self.cfg.proximity_gate:
            self.funnel.add("rejected_proximity_gate")
            return None
        self._event_seq += 1
        ev = ConflictEvent(
            event_id=f"CF{self._event_seq:05d}", frame_idx=frame_idx, t=round(t, 3),
            track_a=key[0], track_b=key[1], cls_a=tr_a.cls, cls_b=tr_b.cls,
            ttc=None if ttc == NO_COLLISION else round(ttc, 3),
            pet=None if pet is None else round(pet, 3),
            conflict_type=classify_conflict(st_a, st_b, tr_a.cls, tr_b.cls),
            risk_score=0.0, risk_level=RiskLevel.NONE,
            location=(0.5 * (st_a.ground[0] + st_b.ground[0]),
                      0.5 * (st_a.ground[1] + st_b.ground[1])),
            location_px=(0.5 * (st_a.anchor[0] + st_b.anchor[0]),
                         0.5 * (st_a.anchor[1] + st_b.anchor[1])),
            approach_angle=round(approach_angle(st_a, st_b), 1),
            rel_speed=round(rel_v, 2), gap=round(gap_now, 2),
            pet_pred=None if pet is None else round(pet, 3),
            pet_source="predicted" if pet is not None else "",
        )
        if risk_scorer is not None:
            risk_scorer.score_event(ev)
        self.events.append(ev)
        self.funnel.add("emitted")
        return ev

    def flush(self, risk_scorer=None, now: float | None = None) -> list[ConflictEvent]:
        """Đóng mọi episode còn dở khi video kết thúc."""
        out: list[ConflictEvent] = []
        for key in list(self.open):
            ep = self.open[key]
            t_end = now if now is not None else ep.peak_t
            event = self._close(key, ep, risk_scorer, now=t_end, final=True)
            del self.open[key]
            if event is not None:
                out.append(event)
        return out

    # ------------------------------------------------------------------ #
    def active_alerts(
        self, frame_idx: int, max_age: int = 3
    ) -> dict[tuple[int, int], _Episode]:
        """Các cặp đang xung đột - dùng cho overlay/dashboard thời gian thực."""
        return {
            key: ep for key, ep in self.open.items()
            if frame_idx - ep.last_seen_frame <= max_age
        }

    @property
    def total_events(self) -> int:
        return len(self.events)

    def severe_events(self, ttc_threshold: float = 1.5) -> list[ConflictEvent]:
        """Near-miss nghiêm trọng - TTC < ngưỡng (mặc định 1.5 s theo poster)."""
        return [e for e in self.events if e.ttc is not None and e.ttc < ttc_threshold]
