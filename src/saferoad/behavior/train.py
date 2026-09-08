"""Huấn luyện bộ phân loại hành vi từ dữ liệu mô phỏng.

Nhãn được sinh **tự động** từ chính simulator: ta biết chính xác xe nào có pha
phanh gấp chủ động, xe nào đang rẽ cắt dòng, nên không cần gán nhãn tay. Đây là
lợi thế lớn của việc có một simulator — module ML có dữ liệu huấn luyện sạch
ngay từ đầu.

Mô hình: Gradient Boosting trên 10 đặc trưng động học (xem
:mod:`saferoad.behavior.classifier`). Chọn cây tăng cường thay vì mạng nơ-ron vì
tập dữ liệu nhỏ, đặc trưng đã có ý nghĩa vật lý rõ ràng, và mô hình chạy được
trên CPU của Jetson với độ trễ không đáng kể.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import BehaviorConfig, Config
from ..types import Track, TrackState, VehicleClass
from .classifier import FEATURE_NAMES, extract_features

log = logging.getLogger(__name__)


def _label_window(speeds: np.ndarray, times: np.ndarray, headings: np.ndarray) -> str:
    """Gán nhãn hành vi cho **một cửa sổ thời gian**, không phải cả track.

    Đây là điểm dễ sai nhất khi sinh nhãn tự động: nếu gán nhãn cho toàn bộ
    track rồi cắt cửa sổ trượt, thì một chiếc xe chỉ phanh gấp trong 1.4 giây sẽ
    khiến **mọi** cửa sổ của nó — kể cả 20 giây chạy hoàn toàn bình thường — đều
    mang nhãn "phanh gấp". Model học từ dữ liệu đó sẽ chỉ học được nhiễu.

    Nhãn phải mô tả đúng những gì xảy ra *trong* cửa sổ đang xét.
    """
    if len(speeds) < 3:
        return "binh_thuong"

    dt = np.diff(times)
    accel = np.diff(speeds) / np.maximum(dt, 1e-6)

    moving = speeds > 1.0
    turn_rate = np.zeros(1)
    if moving.sum() > 2:
        h = np.unwrap(headings[moving])
        tt = times[moving]
        if len(tt) > 1:
            turn_rate = np.degrees(np.diff(h) / np.maximum(np.diff(tt), 1e-6))

    if accel.size and accel.min() <= -3.5:
        return "phanh_gap"
    if accel.size and accel.max() >= 3.0:
        return "tang_toc_dot_ngot"
    if turn_rate.size and np.abs(turn_rate).max() >= 45.0:
        return "doi_huong_gap"
    return "binh_thuong"


def _tracks_from_simulation(vehicles: list) -> dict[int, tuple[Track, str]]:
    """Chuyển quỹ đạo mô phỏng thành ``Track`` kèm nhãn hành vi chuẩn."""
    out: dict[int, tuple[Track, str]] = {}

    for veh in vehicles:
        if veh.times.size < 10:
            continue
        track = Track(track_id=veh.vid, cls=veh.cls, confirmed=True)
        for i, t in enumerate(veh.times):
            pos = (float(veh.positions[i, 0]), float(veh.positions[i, 1]))
            vel = (float(veh.velocities[i, 0]), float(veh.velocities[i, 1]))
            track.history.append(
                TrackState(
                    frame_idx=i, t=float(t),
                    bbox=(0.0, 0.0, 1.0, 1.0), anchor=(0.0, 0.0),
                    ground=pos, velocity=vel, score=1.0,
                )
            )

        # --- Gán nhãn chuẩn từ chính quỹ đạo -------------------------- #
        speeds = np.linalg.norm(veh.velocities, axis=1)
        dt = np.diff(veh.times)
        accel = np.diff(speeds) / np.maximum(dt, 1e-6)

        headings = np.arctan2(veh.velocities[:, 1], veh.velocities[:, 0])
        moving = speeds > 0.5
        turn_rate = np.zeros(1)
        if moving.sum() > 3:
            h = np.unwrap(headings[moving])
            tt = veh.times[moving]
            turn_rate = np.degrees(np.diff(h) / np.maximum(np.diff(tt), 1e-6))

        label = "binh_thuong"
        if accel.size and accel.min() <= -3.5:
            label = "phanh_gap"
        elif accel.size and accel.max() >= 3.0:
            label = "tang_toc_dot_ngot"
        elif turn_rate.size and np.abs(turn_rate).max() >= 45.0:
            label = "doi_huong_gap"

        out[veh.vid] = (track, label)
    return out


def build_dataset(
    n_scenarios: int = 6, duration: float = 120.0, window: int = 15, seed0: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Sinh tập đặc trưng/nhãn từ nhiều kịch bản mô phỏng khác nhau.

    Mỗi track đóng góp **nhiều mẫu** — trích đặc trưng ở các cửa sổ trượt khác
    nhau dọc quỹ đạo — nên một kịch bản 120 s cho ra vài nghìn mẫu.
    """
    from ..simulation.scenario import build_scenario, simulate

    features: list[list[float]] = []
    labels: list[str] = []

    for k in range(n_scenarios):
        seed = seed0 + k
        vehicles = simulate(build_scenario(duration=duration, seed=seed), duration)
        tracks = _tracks_from_simulation(vehicles)
        log.info("Kịch bản seed=%d: %d track", seed, len(tracks))

        for track, _track_label in tracks.values():
            n = len(track.history)
            if n < window + 5:
                continue
            # Cửa sổ trượt: lấy mẫu mỗi 5 frame. Nhãn được tính LẠI cho từng
            # cửa sổ để mô tả đúng đoạn quỹ đạo đang xét.
            for end in range(window, n, 5):
                sub = Track(track_id=track.track_id, cls=track.cls,
                            history=track.history[:end], confirmed=True)
                feats = extract_features(sub, window)

                win = sub.history[-window:]
                speeds = np.array([s.speed for s in win], dtype=float)
                times = np.array([s.t for s in win], dtype=float)
                headings = np.array([s.heading for s in win], dtype=float)

                features.append([feats[name] for name in FEATURE_NAMES])
                labels.append(_label_window(speeds, times, headings))

    return np.asarray(features, dtype=float), np.asarray(labels)


def train_from_simulation(
    out_path: str = "models/behavior_clf.joblib",
    n_scenarios: int = 6,
    duration: float = 120.0,
) -> dict[str, Any]:
    """Huấn luyện và lưu bộ phân loại; trả về các chỉ số đánh giá."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(f"Sinh dữ liệu từ {n_scenarios} kịch bản mô phỏng...")
    X, y = build_dataset(n_scenarios=n_scenarios, duration=duration)
    if len(X) == 0:
        raise RuntimeError("Không sinh được mẫu huấn luyện nào")

    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    print(f"  {len(X)} mẫu, phân bố nhãn: {counts}")

    # Phân tầng theo nhãn để tập test có đủ mọi lớp.
    stratify = y if min(counts.values()) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    scaler = StandardScaler().fit(X_tr)
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, random_state=42,
        # Lớp "bình thường" áp đảo về số lượng — cân bằng lại để không bỏ sót
        # hành vi nguy hiểm (recall của lớp hiếm mới là thứ quan trọng).
        class_weight="balanced",
    )
    model.fit(scaler.transform(X_tr), y_tr)

    y_pred = model.predict(scaler.transform(X_te))
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    print("\n" + classification_report(y_te, y_pred, zero_division=0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURE_NAMES}, out
    )
    print(f"✓ Đã lưu model: {out}")

    return {
        "n_samples": int(len(X)),
        "label_counts": counts,
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class": {
            k: {kk: round(float(vv), 4) for kk, vv in v.items()}
            for k, v in report.items()
            if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
        },
        "model_path": str(out),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classes": sorted(set(y_te.tolist())),
    }
