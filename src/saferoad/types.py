"""Kiểu dữ liệu lõi dùng chung cho toàn bộ pipeline SafeRoad AI.

Mọi module (detection → tracking → geometry → conflict → risk) đều trao đổi
qua các dataclass trong file này, nên đổi một module không phá vỡ module khác.

Quy ước toạ độ
--------------
* ``(x, y)`` trong **pixel** — gốc toạ độ ở góc trên-trái khung hình.
* ``(X, Y)`` trong **mét trên mặt đất** (ground plane) — sau khi quy đổi bằng
  homography. Trục X hướng sang phải, trục Y hướng "vào trong" ảnh.
* Vận tốc luôn tính trên mặt đất, đơn vị m/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# Lớp đối tượng
# --------------------------------------------------------------------------- #
class VehicleClass(str, Enum):
    """5 lớp đối tượng theo đề bài SafeRoad AI.

    Ánh xạ từ COCO: person(0), bicycle(1), car(2), motorcycle(3), bus(5),
    truck(7). ``bus`` và ``truck`` gộp chung thành ``TRUCK`` vì cùng đặc tính
    động học (khối lượng lớn, bán kính quay rộng) trong bài toán xung đột.
    """

    PEDESTRIAN = "pedestrian"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"

    @property
    def vi(self) -> str:
        """Tên tiếng Việt để hiển thị trên dashboard."""
        return {
            "pedestrian": "Người đi bộ",
            "bicycle": "Xe đạp",
            "motorcycle": "Xe máy",
            "car": "Ô tô",
            "truck": "Xe tải/Bus",
        }[self.value]

    @property
    def footprint(self) -> tuple[float, float]:
        """Kích thước xấp xỉ (dài, rộng) tính bằng mét.

        Dùng để xấp xỉ đối tượng thành hình tròn khi tính TTC, và để ước lượng
        khoảng cách an toàn. Số liệu lấy theo xe phổ thông tại Việt Nam.
        """
        return {
            "pedestrian": (0.5, 0.5),
            "bicycle": (1.7, 0.6),
            "motorcycle": (1.9, 0.7),
            "car": (4.4, 1.8),
            "truck": (7.5, 2.5),
        }[self.value]

    @property
    def height(self) -> float:
        """Chiều cao (m) — quyết định kích thước bbox mà camera thật nhìn thấy.

        Người đi bộ chỉ chiếm 0.5 × 0.5 m mặt đất nhưng cao 1.7 m, nên trên ảnh
        họ là một hình chữ nhật đứng chứ không phải một chấm nhỏ. Bỏ qua chiều
        cao khi dựng cảnh sẽ khiến nhóm đối tượng dễ tổn thương nhất trở nên
        không thể phát hiện.
        """
        return {
            "pedestrian": 1.70,
            "bicycle": 1.65,
            "motorcycle": 1.60,
            "car": 1.50,
            "truck": 3.20,
        }[self.value]

    @property
    def radius(self) -> float:
        """Bán kính hình tròn **nội tiếp** theo chiều ngang (m) = nửa bề rộng.

        Dùng cho các phép ước lượng thô cần một con số duy nhất. Với tính toán
        va chạm hãy dùng :pyattr:`circles` — xem giải thích ở đó.
        """
        return self.footprint[1] / 2.0

    @property
    def circles(self) -> tuple[tuple[float, float], ...]:
        """Xấp xỉ đa hình tròn: ``((độ lệch dọc trục, bán kính), ...)``.

        Vì sao KHÔNG dùng một hình tròn duy nhất
        ----------------------------------------
        Hình tròn ngoại tiếp một ô tô 4.4 × 1.8 m có bán kính
        ``½·√(4.4² + 1.8²) ≈ 2.38`` m — tức là mô hình hoá chiếc xe như một vật
        thể **rộng 4.76 m**. Hậu quả: hai ô tô đi ngược chiều ở hai làn cách nhau
        4 m sẽ bị coi là "đang va chạm", và hệ thống sinh ra hàng loạt cảnh báo
        giả kiểu "đối đầu" cho dòng xe hoàn toàn bình thường.

        Cách chuẩn là phủ thân xe bằng **vài hình tròn nhỏ** đặt dọc trục, mỗi
        hình có bán kính bằng nửa bề rộng xe. Cách này bám sát hình chữ nhật
        thật cả theo chiều dọc lẫn chiều ngang, mà vẫn giữ được công thức TTC
        dạng đóng cho từng cặp hình tròn (chỉ cần lấy giá trị nhỏ nhất trên các
        cặp).
        """
        length, width = self.footprint
        r = width / 2.0
        if length <= width * 1.5:
            return ((0.0, r),)
        n = min(3, max(2, int(round(length / width))))
        span = length / 2.0 - r
        return tuple((-span + 2.0 * span * i / (n - 1), r) for i in range(n))

    @property
    def mass_factor(self) -> float:
        """Hệ số mức độ nghiêm trọng theo khối lượng, chuẩn hoá về [0, 1].

        Va chạm giữa xe tải và người đi bộ nghiêm trọng hơn nhiều so với va
        chạm giữa hai xe đạp, dù TTC bằng nhau — hệ số này đưa yếu tố đó vào
        Risk Score.
        """
        return {
            "pedestrian": 1.00,  # dễ tổn thương nhất
            "bicycle": 0.85,
            "motorcycle": 0.75,
            "car": 0.55,
            "truck": 0.95,  # gây hậu quả nặng cho đối tượng còn lại
        }[self.value]


#: Ánh xạ chỉ số lớp COCO -> VehicleClass của SafeRoad.
COCO_TO_SAFEROAD: dict[int, VehicleClass] = {
    0: VehicleClass.PEDESTRIAN,
    1: VehicleClass.BICYCLE,
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.TRUCK,
    7: VehicleClass.TRUCK,
}


class ConflictType(str, Enum):
    """Phân loại xung đột theo hình học quỹ đạo (dựa trên góc tiếp cận)."""

    CROSSING = "crossing"          # cắt ngang, góc lớn
    TURNING = "turning"            # chuyển hướng / rẽ cắt dòng
    REAR_END = "rear_end"          # đâm đuôi, cùng hướng
    HEAD_ON = "head_on"            # đối đầu
    LANE_CHANGE = "lane_change"    # tạt đầu / chuyển làn
    PEDESTRIAN = "pedestrian"      # xung đột với người đi bộ

    @property
    def vi(self) -> str:
        return {
            "crossing": "Cắt ngang",
            "turning": "Chuyển hướng",
            "rear_end": "Tạt đầu/đâm đuôi",
            "head_on": "Đối đầu",
            "lane_change": "Chuyển làn",
            "pedestrian": "Người đi bộ",
        }[self.value]


class RiskLevel(str, Enum):
    """Mức rủi ro Level 0-3 theo Explainable Risk trong kế hoạch."""

    NONE = "none"        # Level 0 — không có xung đột
    LOW = "low"          # Level 1 — Risk Score < 40
    MEDIUM = "medium"    # Level 2 — 40 <= Risk Score < 70
    HIGH = "high"        # Level 3 — Risk Score >= 70

    @property
    def vi(self) -> str:
        return {"none": "An toàn", "low": "Thấp", "medium": "Trung bình", "high": "Cao"}[
            self.value
        ]

    @property
    def level(self) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3}[self.value]

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Quy đổi Risk Score 0-100 sang mức, theo ngưỡng trong poster."""
        if score >= 70:
            return cls.HIGH
        if score >= 40:
            return cls.MEDIUM
        if score > 0:
            return cls.LOW
        return cls.NONE


# --------------------------------------------------------------------------- #
# Detection / Track
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Detection:
    """Một bounding box do detector sinh ra tại một khung hình."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel
    score: float
    cls: VehicleClass

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def anchor(self) -> tuple[float, float]:
        """Điểm tiếp đất — đáy giữa bbox.

        Đây là điểm DUY NHẤT được phép đưa qua homography: tâm bbox nằm lơ
        lửng trên không nên quy đổi sẽ sai vị trí mặt đất rất nhiều.
        """
        x1, _y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), y2)

    @property
    def wh(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

    @property
    def area(self) -> float:
        w, h = self.wh
        return max(0.0, w) * max(0.0, h)


@dataclass(slots=True)
class TrackState:
    """Trạng thái của một track tại một thời điểm cụ thể."""

    frame_idx: int
    t: float                                   # giây kể từ đầu video
    bbox: tuple[float, float, float, float]    # pixel
    anchor: tuple[float, float]                # pixel, điểm tiếp đất
    ground: tuple[float, float]                # mét, trên mặt đất
    velocity: tuple[float, float] = (0.0, 0.0)  # m/s trên mặt đất
    score: float = 0.0

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    @property
    def heading(self) -> float:
        """Hướng di chuyển (radian). 0 = trục X dương."""
        vx, vy = self.velocity
        return math.atan2(vy, vx)


@dataclass
class Track:
    """Một đối tượng được theo dõi liên tục qua nhiều khung hình."""

    track_id: int
    cls: VehicleClass
    history: list[TrackState] = field(default_factory=list)
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    @property
    def last(self) -> TrackState:
        return self.history[-1]

    @property
    def duration(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].t - self.history[0].t

    def state_at(self, frame_idx: int) -> TrackState | None:
        for st in reversed(self.history):
            if st.frame_idx == frame_idx:
                return st
        return None

    def recent(self, n: int) -> list[TrackState]:
        return self.history[-n:]

    def path_length(self) -> float:
        """Tổng quãng đường đi được trên mặt đất (m)."""
        total = 0.0
        for a, b in zip(self.history, self.history[1:]):
            total += math.dist(a.ground, b.ground)
        return total


# --------------------------------------------------------------------------- #
# Conflict / Risk
# --------------------------------------------------------------------------- #
@dataclass
class ConflictEvent:
    """Một sự kiện near-miss được ghi nhận.

    Đây là đơn vị dữ liệu chính chảy vào database, dashboard và bảng đánh giá.
    """

    event_id: str
    frame_idx: int
    t: float
    track_a: int
    track_b: int
    cls_a: VehicleClass
    cls_b: VehicleClass
    ttc: float | None                 # giây; None nếu không hội tụ
    pet: float | None                 # giây; None nếu chưa xác định
    conflict_type: ConflictType
    risk_score: float                 # 0-100
    risk_level: RiskLevel
    location: tuple[float, float]     # mét, điểm xung đột trên mặt đất
    location_px: tuple[float, float]  # pixel, để vẽ overlay
    approach_angle: float             # độ, góc giữa hai vector vận tốc
    rel_speed: float                  # m/s, tốc độ tiếp cận
    gap: float                        # m, khoảng cách mặt-tới-mặt
    reasons: list[str] = field(default_factory=list)  # Explainable Risk
    severity_terms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cls_a"] = self.cls_a.value
        d["cls_b"] = self.cls_b.value
        d["conflict_type"] = self.conflict_type.value
        d["risk_level"] = self.risk_level.value
        return d


@dataclass
class GroundTruthConflict:
    """Nhãn chuẩn của một near-miss, dùng để chấm Precision/Recall.

    Sinh ra từ simulator (biết chính xác quỹ đạo) hoặc từ người gán nhãn tay.
    """

    t: float
    track_a: int
    track_b: int
    ttc: float
    pet: float | None = None
    conflict_type: ConflictType = ConflictType.CROSSING
    source: str = "synthetic"

    def matches(self, ev: ConflictEvent, time_tol: float = 1.0) -> bool:
        """Khớp GT với sự kiện dự đoán: cùng cặp đối tượng và lệch thời gian nhỏ.

        Cặp ``(a, b)`` được coi là không có thứ tự.
        """
        pair_gt = {self.track_a, self.track_b}
        pair_ev = {ev.track_a, ev.track_b}
        return pair_gt == pair_ev and abs(self.t - ev.t) <= time_tol


@dataclass
class FrameResult:
    """Toàn bộ kết quả xử lý của một khung hình."""

    frame_idx: int
    t: float
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    events: list[ConflictEvent] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())


def pair_key(a: int, b: int) -> tuple[int, int]:
    """Khoá không thứ tự cho một cặp track."""
    return (a, b) if a <= b else (b, a)


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU giữa hai bbox dạng (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
