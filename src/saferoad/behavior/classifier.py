"""Phân loại hành vi lái xe nguy hiểm - rule-based kết hợp ML classifier.

Kiến trúc lai có chủ đích:

* **Rule-based** bắt các hành vi có định nghĩa vật lý rõ ràng (phanh gấp, tăng
  tốc đột ngột, đổi hướng gấp). Ưu điểm: giải thích được ngay, không cần dữ
  liệu huấn luyện, không bao giờ "ảo giác".
* **ML classifier** (Gradient Boosting) học các tổ hợp đặc trưng tinh vi hơn mà
  luật cứng bỏ sót - ví dụ chuỗi "giảm tốc nhẹ → lệch làn → tăng tốc" đặc trưng
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

#: Tăng khi cách tính đặc trưng thay đổi - model train bằng phiên bản khác bị bỏ qua.
FEATURE_VERSION = 2

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


def _sliding_slope(times: np.ndarray, values: np.ndarray, half: int) -> np.ndarray:
    """Hệ số góc hồi quy tuyến tính trên cửa sổ trượt ``2*half+1`` mẫu."""
    n = len(times)
    out = np.full(n, np.nan)
    w = 2 * half + 1
    if n < w:
        return out
    k = np.ones(w)
    t0 = times - times[0]                  # dịch gốc để tránh mất chính xác
    st = np.convolve(t0, k, "valid")
    sv = np.convolve(values, k, "valid")
    stt = np.convolve(t0 * t0, k, "valid")
    stv = np.convolve(t0 * values, k, "valid")
    den = w * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(np.abs(den) > 1e-12, (w * stv - st * sv) / den, np.nan)
    out[half: n - half] = slope
    return out


def _sustained(series: np.ndarray, n: int, mode: str) -> float:
    """Giá trị cực trị được DUY TRÌ ít nhất ``n`` mẫu liên tiếp.

    Lấy cực trị của trung vị trượt thay vì cực trị thô: một mẫu nhiễu đơn lẻ
    không thể kéo kết quả, chỉ một xu hướng kéo dài mới làm được.
    """
    s = series[np.isfinite(series)]
    if s.size == 0:
        return 0.0
    if s.size < n:
        return float(np.median(s))
    win = np.lib.stride_tricks.sliding_window_view(s, n)
    med = np.median(win, axis=1)
    return float(med.min() if mode == "min" else med.max())


def extract_features(
    track: Track,
    window: int = 15,
    trim_s: float = 0.4,
    sustain_s: float = 0.3,
    min_turn_speed: float = 3.0,
) -> dict[str, float]:
    """Trích 10 đặc trưng động học trên **toàn bộ** quỹ đạo của một track.

    Vì sao bản cũ gắn cờ 305/307 quỹ đạo
    ------------------------------------
    Bản cũ (i) chỉ nhìn ``window`` khung hình CUỐI của track - đúng lúc xe đang
    ra khỏi khung hình và bbox bị cắt mép, vị trí nhảy; (ii) lấy gia tốc bằng
    hiệu hai mẫu liên tiếp của tốc độ đã ước lượng từ bbox, rồi lấy cực trị
    thô. Sai số 0,1 m/s giữa hai khung hình cách nhau 1/30 s đã là 3 m/s² - vượt
    ngưỡng "phanh gấp". Kết quả là gần như xe nào cũng bị gắn cờ.

    Cách làm mới:

    * bỏ ``trim_s`` giây đầu và cuối track (xe vào/ra khung hình);
    * gia tốc = hệ số góc hồi quy tốc độ trên cửa sổ ``window`` khung hình;
    * tốc độ đổi hướng = hệ số góc hồi quy của hướng đi (đã unwrap), chỉ tính
      khi xe chạy trên ``min_turn_speed`` m/s;
    * ngưỡng phải được **duy trì** ít nhất ``sustain_s`` giây (trung vị trượt).
    """
    hist = track.history
    if len(hist) < max(6, window):
        return {name: 0.0 for name in FEATURE_NAMES}

    times = np.array([s.t for s in hist], dtype=float)
    keep = (times >= times[0] + trim_s) & (times <= times[-1] - trim_s)
    if keep.sum() < max(6, window // 2):
        keep = np.ones(len(hist), dtype=bool)
    sub = [s for s, k in zip(hist, keep) if k]
    times = times[keep]
    speeds = np.array([s.speed for s in sub], dtype=float)
    fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3) if len(times) > 1 else 30.0
    half = max(2, int(round(window / 2)))
    n_sustain = max(2, int(round(sustain_s * fps)))

    accel = _sliding_slope(times, speeds, half)

    moving = speeds >= min_turn_speed
    turn = np.full(len(sub), np.nan)
    if moving.sum() >= 2 * half + 1:
        idx = np.nonzero(moving)[0]
        head = np.unwrap(np.array([sub[i].heading for i in idx], dtype=float))
        rates = np.degrees(_sliding_slope(times[idx], head, half))
        turn[idx] = rates

    pts = np.array([s.ground for s in sub], dtype=float)
    path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    curvature = 0.0 if path_len < 1e-6 else max(0.0, 1.0 - straight / path_len)
    lateral = 0.0
    if straight > 1e-6:
        direction = (pts[-1] - pts[0]) / straight
        normal = np.array([-direction[1], direction[0]])
        lateral = float(np.max(np.abs((pts - pts[0]) @ normal)))

    acc_f = accel[np.isfinite(accel)]
    turn_abs = np.abs(turn)
    turn_f = turn[np.isfinite(turn)]
    return {
        "mean_speed": float(speeds.mean()),
        "max_speed": float(speeds.max()),
        "speed_std": float(speeds.std()),
        "min_accel": _sustained(accel, n_sustain, "min"),
        "max_accel": _sustained(accel, n_sustain, "max"),
        "accel_std": float(acc_f.std()) if acc_f.size else 0.0,
        "max_turn_rate": _sustained(turn_abs, n_sustain, "max") if turn_f.size else 0.0,
        "turn_rate_std": float(turn_f.std()) if turn_f.size else 0.0,
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
        ``No module named '_loss'``. Không có cách vá phía đọc - phải train lại
        trên chính môi trường đang chạy, nên thông báo phải nói thẳng điều đó.
        """
        try:
            import joblib

            bundle = joblib.load(path)
            if bundle.get("feature_version") != FEATURE_VERSION:
                # Model cũ train trên đặc trưng kiểu cũ (cực trị thô ở đuôi
                # track) - đưa đặc trưng mới vào sẽ cho kết quả vô nghĩa.
                log.info(
                    "Behavior model %s train với đặc trưng phiên bản %s, cần %s - "
                    "bỏ qua tầng ML, chỉ dùng luật. Train lại: saferoad train-behavior",
                    path, bundle.get("feature_version"), FEATURE_VERSION,
                )
                self.model = None
                return
            self.model = bundle["model"]
            self.scaler = bundle.get("scaler")
            trained_with = bundle.get("sklearn_version")
            log.info("Đã nạp behavior classifier từ %s", path)
            if trained_with:
                import sklearn

                if sklearn.__version__ != trained_with:
                    log.warning(
                        "Behavior model train bằng scikit-learn %s, đang chạy %s "
                        "- nếu kết quả bất thường hãy chạy: saferoad train-behavior",
                        trained_with, sklearn.__version__,
                    )
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            hint = ""
            if "No module named" in str(exc) or "sklearn" in str(exc).lower():
                try:
                    import sklearn

                    hint = (
                        f" - lệch phiên bản scikit-learn (đang chạy "
                        f"{sklearn.__version__}); train lại bằng: saferoad train-behavior"
                    )
                except Exception:
                    hint = " - train lại bằng: saferoad train-behavior"
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
