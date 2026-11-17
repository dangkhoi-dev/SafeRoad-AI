"""Cấu hình pipeline - nạp từ YAML, có giá trị mặc định an toàn.

Toàn bộ ngưỡng kỹ thuật (TTC, PET, trọng số Risk Score...) đều nằm ở đây chứ
không rải rác trong code, để khi hiệu chỉnh theo hiện trường chỉ phải sửa 1 file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectionConfig:
    """Tham số cho khối detection."""

    weights: str = "models/yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"          # "cpu" | "cuda:0" | "mps"
    #: Độ chính xác số học khi suy luận: None = FP32 (mặc định, an toàn trên
    #: CPU), 16 = FP16, 8 = INT8. Ultralytics đã bỏ cờ ``half`` để chuyển sang
    #: ``quantize``; giữ giá trị ở dạng số để không phải đổi lại khi nâng cấp.
    quantize: int | None = None
    max_det: int = 300
    #: Bỏ qua bbox nhỏ hơn ngưỡng này (pixel²). Đối tượng nhỏ hơn ~150 px² nằm
    #: quá xa camera: sai số vị trí vài pixel đã làm vận tốc ước lượng sai hàng
    #: mét/giây, nên TTC tính từ đó chỉ sinh báo động giả. Lọc ở đây giúp cả
    #: tracking lẫn conflict ổn định hơn nhiều so với việc cố xử lý chúng.
    min_box_area: float = 150.0
    #: "yolo"   - chạy detector một lượt trên cả khung hình (nhanh nhất)
    #: "tiled"  - cắt khung hình thành lưới ô chồng lấn và chạy trên từng ô ở độ
    #:            phân giải gốc, rồi gộp lại. Chậm hơn nhiều lần nhưng bắt được
    #:            vật thể nhỏ ở xa mà lượt toàn khung bỏ sót - cần cho camera
    #:            hạ tầng đặt cao.
    #: "replay" - phát lại detection có sẵn (dùng khi đánh giá trên mô phỏng)
    backend: str = "yolo"
    #: Chỉ dùng khi backend = "tiled".
    tile_rows: int = 2
    tile_cols: int = 3
    #: Tỉ lệ chồng lấn giữa các ô kề nhau. Không có chồng lấn thì vật thể nằm
    #: đúng trên đường cắt sẽ bị xẻ đôi và không ô nào nhận ra nó.
    tile_overlap: float = 0.25
    #: Có chạy thêm một lượt trên toàn khung hình hay không. Ô nhỏ bỏ sót vật
    #: thể lớn hơn chính nó, nên lượt toàn khung bù lại phần đó.
    tile_full_frame: bool = True
    #: Ngưỡng IoU khi gộp kết quả của các ô - hai ô kề nhau nhìn thấy cùng một
    #: vật trong vùng chồng lấn sẽ cho hai hộp gần trùng.
    tile_merge_iou: float = 0.55
    #: Detector phát lại (chỉ dùng cho mô phỏng): độ lệch chuẩn nhiễu bbox (px),
    #: tỉ lệ bỏ sót và seed. Đặt ở cấu hình thay vì rải trong mã để lệnh ``run``
    #: và lệnh ``evaluate`` chắc chắn dùng cùng một chuỗi detection - trước đây
    #: hai lệnh dùng hai seed khác nhau nên dashboard hiện 214 sự kiện còn báo
    #: cáo đánh giá chấm 211.
    replay_noise_px: float = 1.5
    replay_miss_rate: float = 0.03
    replay_seed: int = 17


@dataclass
class TrackingConfig:
    """Tham số ByteTrack + Kalman."""

    track_high_thresh: float = 0.50   # ngưỡng "high score" cho vòng ghép 1
    track_low_thresh: float = 0.10    # ngưỡng "low score" cho vòng ghép 2
    new_track_thresh: float = 0.55    # ngưỡng khởi tạo track mới
    match_thresh: float = 0.80        # ngưỡng IoU-distance khi ghép
    track_buffer: int = 30            # số frame giữ track khi mất dấu
    min_hits: int = 3                 # số lần khớp trước khi track được confirm
    #: Hệ số làm mượt vận tốc (EMA). Càng nhỏ càng mượt nhưng càng trễ.
    velocity_alpha: float = 0.35
    #: Số mẫu tối thiểu trước khi tin vào vector vận tốc.
    min_states_for_velocity: int = 3
    #: Kẹp vận tốc về 0 cho đối tượng đứng yên: nếu quãng dịch chuyển ròng trong
    #: ``stationary_window_s`` giây chia cho thời gian đó nhỏ hơn
    #: ``stationary_speed`` (m/s) thì coi là đứng yên (giữ nguyên hướng thân xe
    #: lần cuối còn chạy). Ở xa camera, bbox của xe đỗ rung vài pixel = hơn một
    #: mét, đủ để ước lượng ra vận tốc vài m/s và sinh cảnh báo giả giữa hai xe
    #: đang đứng trong hàng chờ. 0 = tắt (mặc định, giữ nguyên benchmark mô phỏng).
    stationary_speed: float = 0.0
    stationary_window_s: float = 2.0


@dataclass
class HomographyConfig:
    """Quy đổi toạ độ ảnh sang mặt đất.

    ``image_points`` và ``world_points`` là 4 cặp điểm tương ứng. Khi không có
    calibration thực địa, đặt ``fallback_scale`` (mét trên mỗi pixel) để pipeline
    vẫn chạy được - kết quả TTC khi đó chỉ mang tính tương đối.
    """

    image_points: list[list[float]] = field(default_factory=list)
    world_points: list[list[float]] = field(default_factory=list)
    fallback_scale: float = 0.05      # 1 px ≈ 5 cm (giá trị an toàn cho cam ~12 m)
    #: Làm mượt quỹ đạo mặt đất bằng cửa sổ trung bình trượt.
    smooth_window: int = 5


@dataclass
class ConflictConfig:
    """Ngưỡng phát hiện xung đột - bám theo poster SafeRoad AI."""

    ttc_threshold: float = 3.0        # giây; TTC < 3.0s ⇒ ứng viên conflict
    pet_threshold: float = 1.5        # giây; PET < 1.5s ⇒ ứng viên conflict
    #: Chỉ xét cặp cách nhau dưới ngưỡng này (m) để giảm O(n²).
    max_pair_distance: float = 30.0
    #: Tốc độ tiếp cận tối thiểu (m/s). Đây là thứ phân biệt xung đột thật với
    #: dòng xe bám đuôi bình thường: hai xe nối đuôi cùng tốc độ có TTC rất nhỏ
    #: nhưng vận tốc tương đối ~0, và đó không phải tình huống nguy hiểm.
    min_rel_speed: float = 3.0
    #: Khoảng cách mặt-tới-mặt (m) mà cặp xe phải thực sự đạt tới thì mới ghi
    #: nhận là near-miss. Chỉ dựa vào TTC là chưa đủ - TTC là một *phép ngoại
    #: suy*, còn đây là kiểm chứng rằng hai xe đã thật sự đến sát nhau.
    proximity_gate: float = 3.0
    #: Tốc độ tối thiểu của ít nhất một đối tượng (m/s) - lọc xe đang dừng.
    min_abs_speed: float = 0.8
    #: Khoảng thời gian tối thiểu giữa hai lần ghi cùng một cặp (giây) -
    #: chống việc một near-miss bị đếm thành hàng chục sự kiện.
    dedup_window: float = 3.0
    #: Giới hạn thời gian mô phỏng tiến để tìm điểm giao (giây).
    horizon: float = 6.0
    #: --- PET đo từ quỹ đạo (xem saferoad.conflict.encroachment) ----------- #
    #: Dùng PET đo được để CHỐT sự kiện. Tắt đi thì quay về chỉ dùng PET ngoại
    #: suy - giữ lại để bảng ablation đo được đóng góp của bước này.
    use_measured_pet: bool = True
    #: Lấy lịch sử quỹ đạo từ bao nhiêu giây trước khi episode mở để đo PET.
    pet_lookback_s: float = 4.0
    #: Chờ tối đa bao lâu (giây) sau khi episode kết thúc để xe sau đi vào vùng
    #: xung đột. Quá thời gian này thì chốt với PET ngoại suy.
    pet_max_wait_s: float = 4.0
    #: Góc cắt tối thiểu (độ) để PET có nghĩa; nhỏ hơn là song song/cùng làn.
    pet_min_angle: float = 20.0
    #: Bỏ qua cặp hai người đi bộ - cùng quy ước với nhãn chuẩn (protocol v2):
    #: người đi bộ tự né nhau ở tốc độ thấp, mô hình hình tròn không tả được.
    ignore_pedestrian_pairs: bool = True
    #: Vùng phân tích - đa giác toạ độ pixel [[x, y], ...]. Chỉ xét đối tượng có
    #: điểm neo (giữa cạnh đáy bbox) nằm trong vùng. Dùng để giới hạn phân tích
    #: trong phần mặt đường mà homography còn đúng: ở xa camera, vài pixel sai
    #: số bbox đã thành vài mét và xe ở làn kề trông như chồng lên nhau. Rỗng =
    #: cả khung hình. Áp dụng như nhau cho hệ thống và cảnh báo tham chiếu.
    analysis_region_px: list = field(default_factory=list)
    #: Thời gian (giây) theo dõi thêm sau lúc gần nhất để đọc động học sau đó -
    #: phân biệt va chạm thật (xe dừng) với chồng thân xe do sai số đo.
    contact_settle_s: float = 2.0
    #: Gom các khung hình của cùng một tình huống thành một sự kiện. Chỉ tắt
    #: khi chạy ablation để đo đóng góp của bước gom.
    episode_grouping: bool = True
    #: Ngưỡng góc (độ) phân loại kiểu xung đột.
    angle_rear_end: float = 30.0      # < 30°  ⇒ cùng hướng
    angle_crossing_min: float = 60.0  # 60-120° ⇒ cắt ngang
    angle_crossing_max: float = 120.0
    angle_head_on: float = 150.0      # > 150° ⇒ đối đầu


@dataclass
class RiskConfig:
    """Trọng số Risk Score và tham số Risk Map.

    Công thức (theo poster)::

        RS = 100 * σ( w1/TTC + w2/PET + w3*v_r + w4*Type + w5*Time )

    Trọng số mặc định dưới đây được **hiệu chỉnh trên tập synthetic có ground
    truth**: chọn sao cho phân bố Risk Score trải đều trong [0, 100] thay vì dồn
    cục ở một đầu. Bộ trọng số chưa hiệu chỉnh đẩy trung vị lên 98/100 - khi mọi
    cảnh báo đều "rất nguy hiểm" thì thang điểm không còn phân biệt được gì và
    người vận hành sẽ bỏ qua tất cả.

    Với tập hiện tại, phân bố thu được là p10 ≈ 21, trung vị ≈ 55, p90 ≈ 89.
    Chuyển sang hiện trường khác nên chạy lại :meth:`RiskScorer.calibrate` để
    giữ thang điểm có ý nghĩa.
    """

    w_ttc: float = 0.52
    w_pet: float = 0.32
    w_rel_speed: float = 0.072
    w_type: float = 0.38
    w_time: float = 0.14
    bias: float = -2.69               # dịch sigmoid để trung vị Risk Score ≈ 55
    #: Kích thước ô lưới Risk Map (mét).
    grid_size: float = 2.0
    #: Bán kính lan toả nhiệt của một sự kiện (mét).
    kernel_radius: float = 6.0
    #: Hằng số suy giảm theo thời gian (giây) cho risk map "24h gần đây".
    decay_tau: float = 900.0
    #: Số điểm nóng hiển thị trên dashboard.
    top_k_hotspots: int = 5


@dataclass
class BehaviorConfig:
    """Phân loại hành vi nguy hiểm (rule-based + ML classifier)."""

    enabled: bool = True
    model_path: str = "models/behavior_clf.joblib"
    #: Ngưỡng gia tốc phanh gấp (m/s²), giá trị âm.
    harsh_brake: float = -3.5
    #: Ngưỡng gia tốc tăng tốc đột ngột (m/s²).
    harsh_accel: float = 3.0
    #: Ngưỡng tốc độ đổi hướng (độ/giây) coi là tạt đầu.
    sharp_turn_rate: float = 45.0
    #: Cửa sổ tính đặc trưng (số frame).
    window: int = 15


@dataclass
class PrivacyConfig:
    """Ẩn danh dữ liệu - tuân thủ Điều 5 Thể lệ."""

    enabled: bool = False
    blur_faces: bool = True
    blur_plates: bool = True
    #: Cường độ làm mờ (kernel Gaussian, số lẻ).
    blur_kernel: int = 31
    #: Tỉ lệ phần trên bbox người được coi là vùng mặt.
    face_ratio: float = 0.28
    #: Tỉ lệ phần dưới bbox xe được coi là vùng biển số.
    plate_band: tuple[float, float] = (0.55, 0.95)


@dataclass
class VideoConfig:
    """Nguồn video và tuỳ chọn xuất."""

    source: str = "data/samples/synthetic_intersection.mp4"
    start_frame: int = 0
    max_frames: int = 0               # 0 = xử lý hết
    stride: int = 1                   # bỏ bớt frame để tăng tốc
    resize_width: int = 0             # 0 = giữ nguyên
    fps_override: float = 0.0         # dùng khi metadata video sai
    write_overlay: bool = True
    overlay_path: str = "data/outputs/overlay.mp4"


@dataclass
class SiteConfig:
    """Mô tả hiện trường - hiển thị trên dashboard và báo cáo."""

    name: str = "Ngã tư Hàng Xanh, Q. Bình Thạnh, TP.HCM"
    short_name: str = "Hàng Xanh"
    camera_height_m: float = 12.0
    approaches: list[str] = field(
        default_factory=lambda: ["Hướng 1", "Hướng 2", "Hướng 3", "Hướng 4"]
    )
    lanes: int = 3


@dataclass
class Config:
    """Cấu hình gốc, gom toàn bộ các khối con."""

    site: SiteConfig = field(default_factory=SiteConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    db_path: str = "data/outputs/saferoad.db"
    output_dir: str = "data/outputs"

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Nạp cấu hình từ file YAML; khoá thiếu sẽ lấy giá trị mặc định."""
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg = cls.from_dict(raw)
        if overrides:
            cfg = cls.from_dict({**cfg.to_dict(), **overrides})
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        sections = {
            "site": SiteConfig,
            "video": VideoConfig,
            "detection": DetectionConfig,
            "tracking": TrackingConfig,
            "homography": HomographyConfig,
            "conflict": ConflictConfig,
            "risk": RiskConfig,
            "behavior": BehaviorConfig,
            "privacy": PrivacyConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sections.items():
            data = raw.get(key) or {}
            valid = {f for f in klass.__dataclass_fields__}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(
                    f"Cấu hình '{key}' có khoá không hợp lệ: {sorted(unknown)}. "
                    f"Khoá hợp lệ: {sorted(valid)}"
                )
            kwargs[key] = klass(**data)
        for scalar in ("db_path", "output_dir"):
            if scalar in raw:
                kwargs[scalar] = raw[scalar]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
