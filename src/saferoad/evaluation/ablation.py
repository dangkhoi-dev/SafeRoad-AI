"""Chạy baseline / ablation trên tập synthetic có ground truth.

Bảng ablation trả lời một câu hỏi cụ thể: **từng khối trong pipeline đóng góp
bao nhiêu?** Ta bật dần các khối và đo lại:

1. ``det+track`` — chỉ detection + tracking, TTC tính thẳng trên toạ độ **pixel**
   (không homography, không làm mượt). Đây là baseline ngây thơ.
2. ``+homography`` — thêm quy đổi sang mặt đất. Kỳ vọng: TTC MAE giảm mạnh, vì
   pixel không phải đơn vị vật lý.
3. ``+smoothing`` — thêm làm mượt quỹ đạo và ước lượng vận tốc bằng hồi quy.
   Kỳ vọng: giảm báo động giả do nhiễu vận tốc.
4. ``+pet`` — thêm tiêu chí PET bên cạnh TTC. Kỳ vọng: Recall tăng (bắt được
   tình huống "vừa lướt qua" mà TTC bỏ sót).
5. ``+proximity`` — thêm cổng khoảng cách. Kỳ vọng: Precision tăng mạnh.
6. ``full`` — bật thêm Risk Scoring và phân loại kiểu xung đột.

Ngoài ra ta quét mức nhiễu detector (``noise_px``) để đo độ bền của hệ thống
trước sai số của khối phía trước.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..detection.detector import ReplayDetector
from ..pipeline.runner import Pipeline
from ..types import Detection
from .metrics import (
    ConflictMetrics,
    build_gt_points,
    build_id_mapping,
    evaluate_conflicts,
    severity_breakdown,
)

log = logging.getLogger(__name__)


@dataclass
class AblationStage:
    """Một cấu hình trong bảng ablation."""

    key: str
    label: str
    description: str
    mutate: Any = None    # hàm nhận Config và sửa tại chỗ


def _no_homography(cfg: Config) -> None:
    """Tắt homography: dùng tỉ lệ phẳng, tức là gần như làm việc trên pixel."""
    cfg.homography.image_points = []
    cfg.homography.world_points = []
    cfg.homography.fallback_scale = 0.05
    cfg.homography.smooth_window = 1


def _no_smoothing(cfg: Config) -> None:
    cfg.homography.smooth_window = 1
    cfg.tracking.velocity_alpha = 1.0
    cfg.tracking.min_states_for_velocity = 2


def _no_pet(cfg: Config) -> None:
    """Vô hiệu tiêu chí PET bằng cách đặt ngưỡng về 0."""
    cfg.conflict.pet_threshold = 0.0


def _no_proximity(cfg: Config) -> None:
    """Bỏ cổng khoảng cách — chấp nhận mọi cặp có TTC/PET dưới ngưỡng."""
    cfg.conflict.proximity_gate = 1e9
    cfg.conflict.min_rel_speed = 0.3


STAGES: list[AblationStage] = [
    AblationStage(
        "det_track", "1. Detection + Tracking",
        "TTC tính trên pixel, không homography, không làm mượt, không PET/proximity",
        lambda c: (_no_homography(c), _no_smoothing(c), _no_pet(c), _no_proximity(c)),
    ),
    AblationStage(
        "homography", "2. + Homography",
        "Quy đổi toạ độ ảnh sang mặt đất (mét)",
        lambda c: (_no_smoothing(c), _no_pet(c), _no_proximity(c)),
    ),
    AblationStage(
        "smoothing", "3. + Trajectory smoothing",
        "Làm mượt quỹ đạo + ước lượng vận tốc bằng hồi quy tuyến tính",
        lambda c: (_no_pet(c), _no_proximity(c)),
    ),
    AblationStage(
        "pet", "4. + PET",
        "Thêm tiêu chí Post-Encroachment Time bên cạnh TTC",
        lambda c: _no_proximity(c),
    ),
    AblationStage(
        "proximity", "5. + Proximity gate",
        "Yêu cầu hai xe thực sự đến gần nhau, không chỉ dựa vào ngoại suy TTC",
        None,
    ),
    AblationStage(
        "full", "6. Full (+ Risk scoring)",
        "Bản đầy đủ: thêm Risk Score, phân loại xung đột và Explainable Risk",
        None,
    ),
]


@dataclass
class AblationResult:
    """Kết quả một dòng trong bảng ablation."""

    key: str
    label: str
    description: str
    metrics: dict[str, Any]
    n_events: int
    processing_fps: float
    latency_ms: dict[str, float] = field(default_factory=dict)


def run_ablation(
    base_cfg: Config,
    detections: dict[int, list[Detection]],
    ground_truth: list,
    vehicles: list,
    gt_points: dict,
    region: tuple | None = None,
    noise_px: float = 1.5,
    miss_rate: float = 0.03,
    stages: list[AblationStage] | None = None,
    progress: bool = True,
) -> list[AblationResult]:
    """Chạy toàn bộ bảng ablation trên cùng một video synthetic.

    ``detections`` là bbox chuẩn theo frame do simulator sinh; ``noise_px`` và
    ``miss_rate`` mô phỏng một detector không hoàn hảo để kết quả phản ánh điều
    kiện thực tế thay vì đầu vào lý tưởng.
    """
    stages = stages or STAGES
    results: list[AblationResult] = []

    for stage in stages:
        cfg = copy.deepcopy(base_cfg)
        cfg.video.write_overlay = False
        if stage.mutate is not None:
            stage.mutate(cfg)

        detector = ReplayDetector(detections, noise_px=noise_px, miss_rate=miss_rate, seed=17)
        pipe = Pipeline(cfg, detector=detector, write_db=False)
        result = pipe.run(progress_every=0)

        mapping, track_metrics = build_id_mapping(result.tracks, gt_points)
        conflict_metrics = evaluate_conflicts(
            result.events, ground_truth, mapping, region=region
        )

        row = conflict_metrics.to_dict()
        row["tracking"] = track_metrics.to_dict()

        results.append(
            AblationResult(
                key=stage.key, label=stage.label, description=stage.description,
                metrics=row, n_events=len(result.events),
                processing_fps=round(result.processing_fps, 2),
                latency_ms={k: round(v, 2) for k, v in result.latency.items()},
            )
        )
        if progress:
            print(
                f"  [{stage.key:11s}] P={row['precision']:.3f} R={row['recall']:.3f} "
                f"F1={row['f1']:.3f} TTC-MAE={row['ttc_mae']} n_events={len(result.events)}",
                flush=True,
            )

    return results


def run_noise_sweep(
    base_cfg: Config,
    detections: dict[int, list[Detection]],
    ground_truth: list,
    vehicles: list,
    gt_points: dict,
    region: tuple | None = None,
    noise_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 6.0),
    miss_rate: float = 0.03,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Quét mức nhiễu detector để đo độ bền của các khối phía sau.

    Ý nghĩa thực tiễn: khi thay YOLO11n bằng một model yếu hơn (hoặc camera rung,
    trời mưa), chất lượng bbox giảm — bảng này cho biết hệ thống chịu được tới
    mức nào trước khi các chỉ số tụt dưới ngưỡng cam kết.
    """
    rows: list[dict[str, Any]] = []
    for noise in noise_levels:
        cfg = copy.deepcopy(base_cfg)
        cfg.video.write_overlay = False
        detector = ReplayDetector(detections, noise_px=noise, miss_rate=miss_rate, seed=17)
        result = Pipeline(cfg, detector=detector, write_db=False).run(progress_every=0)

        mapping, track_metrics = build_id_mapping(result.tracks, gt_points)
        m = evaluate_conflicts(result.events, ground_truth, mapping, region=region)

        row = {"noise_px": noise, **m.to_dict(), "idf1": track_metrics.idf1,
               "n_events": len(result.events)}
        rows.append(row)
        if progress:
            print(
                f"  noise={noise:4.1f}px  P={m.precision:.3f} R={m.recall:.3f} "
                f"F1={m.f1:.3f} TTC-MAE={row['ttc_mae']} IDF1={track_metrics.idf1:.3f}",
                flush=True,
            )
    return rows


def save_report(
    ablation: list[AblationResult],
    noise_sweep: list[dict[str, Any]],
    severity: list[dict[str, Any]],
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Ghi toàn bộ kết quả đánh giá ra JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation": [
            {
                "key": r.key, "label": r.label, "description": r.description,
                "n_events": r.n_events, "processing_fps": r.processing_fps,
                "latency_ms": r.latency_ms, **r.metrics,
            }
            for r in ablation
        ],
        "noise_sweep": noise_sweep,
        "severity_breakdown": severity,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def to_markdown(ablation: list[AblationResult]) -> str:
    """Kết xuất bảng ablation dạng Markdown để dán vào báo cáo."""
    lines = [
        "| Cấu hình | Precision | Recall | F1 | TTC MAE (s) | Số sự kiện | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ablation:
        m = r.metrics
        mae = m["ttc_mae"]
        lines.append(
            f"| {r.label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{mae if mae is not None else '—'} | {r.n_events} | {r.processing_fps:.1f} |"
        )
    return "\n".join(lines)
