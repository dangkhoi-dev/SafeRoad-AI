"""Phân loại hành vi lái xe nguy hiểm — rule-based kết hợp ML classifier.

Kiến trúc lai có chủ đích:

* **Rule-based** bắt các hành vi có định nghĩa vật lý rõ ràng (phanh gấp, tăng
  tốc đột ngột, đổi hướng gấp). Ưu điểm: giải thích được ngay, không cần dữ
  liệu huấn luyện, không bao giờ "ảo giác".
* **ML classifier** (Gradient Boosting) học các tổ hợp đặc trưng tinh vi hơn mà
  luật cứng bỏ sót — ví dụ chuỗi "giảm tốc nhẹ → lệch làn → tăng tốc" đặc trưng
  cho hành vi lách ẩu.

Nhãn huấn luyện được sinh tự động từ simulator (biết chính xác xe nào có pha
phanh gấp / rẽ cắt dòng), nên không cần gán nhãn tay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import BehaviorConfig
from ..geometry.trajectory import acceleration, heading_change_rate
from ..types import Track

log = logging.getLogger(__name__)

#: Tên các đặc trưng, giữ đúng thứ tự khi train và khi suy luận.
FEATURE_NAMES = [
    "mean_speed",
    "max_speed",
    "speed_std",
    "min_accel",
    "max_accel",
    "accel_std",
    "max_turn_rate",
    "turn_rate_std",
    "path_curvature",
    "lateral_drift",
]

#: Nhãn hành vi.
BEHAVIOR_LABELS = ["binh_thuong", "phanh_gap", "tang_toc_dot_ngot", "doi_huong_gap"]

BEHAVIOR_VI = {
    "binh_thuong": "Bình thường",
    "phanh_gap": "Phanh gấp",
    "tang_toc_dot_ngot": "Tăng tốc đột ngột",
    "doi_huong_gap": "Đổi hướng gấp / lách ẩu",
}


@dataclass
class BehaviorResult:
    """Kết quả phân loại hành vi cho một track."""

    track_id: int
    label: str
    confidence: float
    source: str                    # "rule" | "ml"
    features: dict[str, float]

    @property
    def label_vi(self) -> str:
        return BEHAVIOR_VI.get(self.label, self.label)

    @property
    def is_risky(self) -> bool:
        return self.label != "binh_thuong"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "label_vi": self.label_vi,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "risky": self.is_risky,
        }


def extract_features(track: Track, window: int = 15) -> dict[str, float]:
    """Trích 10 đặc trưng động học từ phần đuôi quỹ đạo của một track."""
    hist = track.history[-window:] if len(track.history) > window else track.history
    if len(hist) < 3:
        return {name: 0.0 for name in FEATURE_NAMES}

    speeds = np.array([s.speed for s in hist], dtype=float)
    times = np.array([s.t for s in hist], dtype=float)

    # Gia tốc theo từng bước, dùng để lấy cực trị.
    accels: list[float] = []
    for i in range(1, len(hist)):
        dt = times[i] - times[i - 1]
        if dt > 1e-6:
            accels.append((speeds[i] - speeds[i - 1]) / dt)
    accel_arr = np.array(accels, dtype=float) if accels else np.zeros(1)

    # Tốc độ đổi hướng theo từng bước.
    turns: list[float] = []
    moving = [s for s in hist if s.speed > 0.3]
    for a, b in zip(moving, moving[1:]):
        dt = b.t - a.t
        if dt > 1e-6:
            dh = np.degrees(np.arctan2(np.sin(b.heading - a.heading),
                                       np.cos(b.heading - a.heading)))
            turns.append(dh / dt)
    turn_arr = np.array(turns, dtype=float) if turns else np.zeros(1)

    # Độ cong quỹ đạo: 1 - (đường chim bay / quãng đường đi thực tế).
    pts = np.array([s.ground for s in hist], dtype=float)
    path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    curvature = 0.0 if path_len < 1e-6 else max(0.0, 1.0 - straight / path_len)

    # Độ lệch ngang so với đường thẳng nối đầu-cuối — bắt hành vi lượn lách.
    lateral = 0.0
    if straight > 1e-6:
        direction = (pts[-1] - pts[0]) / straight
        normal = np.array([-direction[1], direction[0]])
        lateral = float(np.max(np.abs((pts - pts[0]) @ normal)))

    return {
        "mean_speed": float(speeds.mean()),
        "max_speed": float(speeds.max()),
        "speed_std": float(speeds.std()),
        "min_accel": float(accel_arr.min()),
        "max_accel": float(accel_arr.max()),
        "accel_std": float(accel_arr.std()),
        "max_turn_rate": float(np.abs(turn_arr).max()),
        "turn_rate_std": float(turn_arr.std()),
        "path_curvature": curvature,
        "lateral_drift": lateral,
    }


class BehaviorClassifier:
    """Phân loại hành vi: luật cứng trước, ML bổ sung khi luật không kích hoạt."""

    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.model = None
        self.scaler = None
        if cfg.enabled and Path(cfg.model_path).exists():
            self._load(cfg.model_path)

    def _load(self, path: str) -> None:
        """Nạp model đã lưu; nếu hỏng thì lùi về luật cứng chứ không chết pipeline.

        Nguyên nhân hỏng phổ biến nhất **không phải** file lỗi mà là lệch phiên
        bản scikit-learn: pickle của ``HistGradientBoostingClassifier`` tham
        chiếu các module nội bộ (``_loss``, ``_predictor``…) mà scikit-learn đổi
        tên giữa các minor release. Model train bằng 1.8 nạp trên 1.9 sẽ báo
        ``No module named '_loss'``. Không có cách vá phía đọc — phải train lại
        trên chính môi trường đang chạy, nên thông báo phải nói thẳng điều đó.
        """
        try:
            import joblib

            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.scaler = bundle.get("scaler")
            trained_with = bundle.get("sklearn_version")
            log.info("Đã nạp behavior classifier từ %s", path)
            if trained_with:
                import sklearn

                if sklearn.__version__ != trained_with:
                    log.warning(
                        "Behavior model train bằng scikit-learn %s, đang chạy %s "
                        "— nếu kết quả bất thường hãy chạy: saferoad train-behavior",
                        trained_with, sklearn.__version__,
                    )
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            hint = ""
            if "No module named" in str(exc) or "sklearn" in str(exc).lower():
                try:
                    import sklearn

                    hint = (
                        f" — lệch phiên bản scikit-learn (đang chạy "
                        f"{sklearn.__version__}); train lại bằng: saferoad train-behavior"
                    )
                except Exception:
                    hint = " — train lại bằng: saferoad train-behavior"
            log.warning(
                "Không nạp được behavior model (%s)%s. Tạm dùng luật cứng.", exc, hint
            )
            self.model = None

    # ------------------------------------------------------------------ #
    def classify(self, track: Track) -> BehaviorResult:
        """Phân loại hành vi của một track."""
        feats = extract_features(track, self.cfg.window)
        cfg = self.cfg

        # --- Tầng 1: luật cứng, ưu tiên cao nhất ------------------------ #
        if feats["min_accel"] <= cfg.harsh_brake:
            return BehaviorResult(
                track.track_id, "phanh_gap",
                min(1.0, abs(feats["min_accel"]) / abs(cfg.harsh_brake) * 0.85),
                "rule", feats,
            )
        if feats["max_accel"] >= cfg.harsh_accel:
            return BehaviorResult(
                track.track_id, "tang_toc_dot_ngot",
                min(1.0, feats["max_accel"] / cfg.harsh_accel * 0.85),
                "rule", feats,
            )
        if feats["max_turn_rate"] >= cfg.sharp_turn_rate:
            return BehaviorResult(
                track.track_id, "doi_huong_gap",
                min(1.0, feats["max_turn_rate"] / cfg.sharp_turn_rate * 0.85),
                "rule", feats,
            )

        # --- Tầng 2: ML classifier -------------------------------------- #
        if self.model is not None:
            x = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
            if self.scaler is not None:
                x = self.scaler.transform(x)
            proba = self.model.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            label = self.model.classes_[idx]
            conf = float(proba[idx])
            # Chỉ tin ML khi nó đủ chắc chắn, tránh cảnh báo giả.
            if label != "binh_thuong" and conf >= 0.60:
                return BehaviorResult(track.track_id, str(label), conf, "ml", feats)

        return BehaviorResult(track.track_id, "binh_thuong", 0.9, "rule", feats)

    def classify_all(self, tracks: list[Track]) -> dict[int, BehaviorResult]:
        return {tr.track_id: self.classify(tr) for tr in tracks}
