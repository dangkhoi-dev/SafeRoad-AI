"""Risk Score 0-100 và Explainable Risk.

Công thức (theo poster SafeRoad AI)::

    RS = 100 · σ( w₁/TTC + w₂/PET + w₃·v_r + w₄·Type + w₅·Time + b )

trong đó σ là hàm sigmoid. Vì sao dùng **nghịch đảo** TTC và PET: mức nguy hiểm
không tuyến tính theo thời gian còn lại. Chênh lệch giữa TTC 0.5 s và 1.0 s
nghiêm trọng hơn rất nhiều so với chênh lệch giữa 4.0 s và 4.5 s — nghịch đảo
phản ánh đúng điều đó, còn sigmoid giữ kết quả nằm gọn trong [0, 100].

Mỗi số hạng đều được ghi lại riêng trong ``severity_terms`` để dashboard giải
thích được **vì sao** một cảnh báo có mức rủi ro cao — đó chính là yêu cầu
"Explainable Risk (cảnh báo kèm lý do)" trong kế hoạch.
"""

from __future__ import annotations

import math

from ..config import RiskConfig
from ..types import ConflictEvent, ConflictType, RiskLevel, VehicleClass

#: Hệ số nghiêm trọng theo kiểu xung đột, chuẩn hoá về [0, 1].
#: Cắt ngang và người đi bộ nguy hiểm nhất vì góc va chạm lớn và ít khả năng
#: tránh né; đâm đuôi thường ở tốc độ tương đối thấp nên hậu quả nhẹ hơn.
CONFLICT_SEVERITY: dict[ConflictType, float] = {
    ConflictType.PEDESTRIAN: 1.00,
    ConflictType.HEAD_ON: 0.95,
    ConflictType.CROSSING: 0.85,
    ConflictType.TURNING: 0.70,
    ConflictType.LANE_CHANGE: 0.55,
    ConflictType.REAR_END: 0.45,
}


def sigmoid(x: float) -> float:
    """Sigmoid ổn định số học (không tràn với |x| lớn)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class RiskScorer:
    """Tính Risk Score và sinh lời giải thích cho từng sự kiện."""

    def __init__(self, cfg: RiskConfig, peak_hours: tuple[int, ...] = (7, 8, 11, 17, 18, 19)):
        self.cfg = cfg
        self.peak_hours = set(peak_hours)

    # ------------------------------------------------------------------ #
    def score_event(self, event: ConflictEvent, hour: int | None = None) -> float:
        """Tính Risk Score, gán ``risk_level``/``reasons`` vào event tại chỗ."""
        cfg = self.cfg

        # --- Số hạng 1: TTC ------------------------------------------------
        # Chặn dưới 0.2 s để tránh chia cho ~0 làm điểm số bão hoà vô nghĩa.
        if event.ttc is not None:
            ttc_term = cfg.w_ttc / max(event.ttc, 0.2)
        else:
            ttc_term = 0.0

        # --- Số hạng 2: PET ------------------------------------------------
        if event.pet is not None:
            pet_term = cfg.w_pet / max(event.pet, 0.2)
        else:
            pet_term = 0.0

        # --- Số hạng 3: tốc độ tiếp cận ------------------------------------
        speed_term = cfg.w_rel_speed * event.rel_speed

        # --- Số hạng 4: kiểu xung đột × mức dễ tổn thương -------------------
        type_severity = CONFLICT_SEVERITY.get(event.conflict_type, 0.5)
        # Lấy đối tượng dễ tổn thương nhất trong cặp làm hệ số khối lượng.
        vulnerability = max(event.cls_a.mass_factor, event.cls_b.mass_factor)
        type_term = cfg.w_type * type_severity * vulnerability

        # --- Số hạng 5: yếu tố thời gian (giờ cao điểm) ---------------------
        is_peak = hour is not None and hour in self.peak_hours
        time_term = cfg.w_time if is_peak else 0.0

        z = ttc_term + pet_term + speed_term + type_term + time_term + cfg.bias
        score = 100.0 * sigmoid(z)

        event.risk_score = round(score, 1)
        event.risk_level = RiskLevel.from_score(score)
        event.severity_terms = {
            "ttc": round(ttc_term, 3),
            "pet": round(pet_term, 3),
            "rel_speed": round(speed_term, 3),
            "type": round(type_term, 3),
            "time": round(time_term, 3),
            "bias": round(cfg.bias, 3),
            "z": round(z, 3),
        }
        event.reasons = self.explain(event, is_peak=is_peak)
        return event.risk_score

    # ------------------------------------------------------------------ #
    def explain(self, event: ConflictEvent, is_peak: bool = False) -> list[str]:
        """Sinh danh sách lý do bằng tiếng Việt, sắp theo mức đóng góp giảm dần.

        Đây là phần "Explainable" — người vận hành đọc cảnh báo phải hiểu ngay
        vì sao hệ thống cho là nguy hiểm, thay vì chỉ thấy một con số.
        """
        reasons: list[tuple[float, str]] = []
        terms = event.severity_terms

        if event.ttc is not None and event.ttc < 3.0:
            sharpness = "rất gấp" if event.ttc < 1.0 else "gấp" if event.ttc < 2.0 else "đáng chú ý"
            reasons.append((
                terms.get("ttc", 0.0),
                f"TTC chỉ còn {event.ttc:.2f}s — thời gian tới va chạm {sharpness} "
                f"(ngưỡng cảnh báo 3.0s)",
            ))

        if event.pet is not None and event.pet < 1.5:
            reasons.append((
                terms.get("pet", 0.0),
                f"PET {event.pet:.2f}s — hai đối tượng đi qua cùng một điểm cách nhau "
                f"quá ngắn (ngưỡng 1.5s)",
            ))

        if event.rel_speed > 5.0:
            reasons.append((
                terms.get("rel_speed", 0.0),
                f"Tốc độ tiếp cận cao {event.rel_speed:.1f} m/s "
                f"(~{event.rel_speed * 3.6:.0f} km/h)",
            ))

        reasons.append((
            terms.get("type", 0.0),
            f"Kiểu xung đột: {event.conflict_type.vi} giữa {event.cls_a.vi} và "
            f"{event.cls_b.vi} (góc tiếp cận {event.approach_angle:.0f}°)",
        ))

        if event.gap < 2.0:
            reasons.append((
                0.5,
                f"Khoảng cách giữa hai phương tiện chỉ {event.gap:.1f} m",
            ))

        if VehicleClass.PEDESTRIAN in (event.cls_a, event.cls_b):
            reasons.append((
                0.9,
                "Có người đi bộ tham gia — nhóm đối tượng dễ tổn thương nhất",
            ))

        if is_peak:
            reasons.append((
                terms.get("time", 0.0),
                "Xảy ra trong khung giờ cao điểm, mật độ phương tiện cao",
            ))

        reasons.sort(key=lambda r: r[0], reverse=True)
        return [text for _weight, text in reasons]

    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        events: list[ConflictEvent],
        target_median: float = 55.0,
        target_p90: float = 90.0,
    ) -> dict[str, float]:
        """Hiệu chỉnh lại trọng số và ``bias`` cho một hiện trường mới.

        Vì sao cần: thang Risk Score chỉ hữu ích khi nó **phân biệt** được các
        tình huống. Nếu 88% cảnh báo đều ≥ 70/100 thì con số không còn giúp ai
        ưu tiên việc gì. Hàm này giữ nguyên *hình dạng* công thức (tỉ lệ giữa
        các số hạng không đổi) nhưng co giãn toàn bộ và dịch sigmoid để trung vị
        và phân vị 90 rơi đúng chỗ mong muốn.

        Trả về bộ tham số mới; áp dụng bằng cách ghi vào file cấu hình.
        """
        if len(events) < 10:
            raise ValueError("Cần ít nhất 10 sự kiện để hiệu chỉnh")

        z_raw = [
            e.severity_terms["z"] - self.cfg.bias
            for e in events
            if "z" in e.severity_terms
        ]
        z_raw.sort()
        median = z_raw[len(z_raw) // 2]
        p90 = z_raw[int(0.9 * (len(z_raw) - 1))]
        spread = max(p90 - median, 1e-6)

        logit_med = math.log(target_median / (100.0 - target_median))
        logit_p90 = math.log(target_p90 / (100.0 - target_p90))
        scale = (logit_p90 - logit_med) / spread
        bias = logit_med - scale * median

        return {
            "w_ttc": round(self.cfg.w_ttc * scale, 4),
            "w_pet": round(self.cfg.w_pet * scale, 4),
            "w_rel_speed": round(self.cfg.w_rel_speed * scale, 4),
            "w_type": round(self.cfg.w_type * scale, 4),
            "w_time": round(self.cfg.w_time * scale, 4),
            "bias": round(bias, 4),
        }
