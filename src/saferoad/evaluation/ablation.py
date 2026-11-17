"""Ablation: đo đóng góp của từng khối **trên cùng một mẫu**.

Hai cách đọc, báo cáo cả hai
----------------------------
* **Cộng dồn** (bật dần từng khối, như bản cũ) - kể câu chuyện hệ thống được
  dựng lên thế nào, nhưng đóng góp của một khối phụ thuộc thứ tự bật.
* **Bỏ từng khối khỏi bản đầy đủ** (leave-one-out) - mỗi dòng chỉ khác bản đầy
  đủ đúng một khối, nên chênh lệch là đóng góp *biên* của khối đó khi mọi khối
  khác vẫn có mặt. Đây là cách hội đồng yêu cầu.

Mọi dòng dùng cùng bbox gốc, cùng seed nhiễu detector, cùng vùng đánh giá, cùng
hàm chấm (:func:`saferoad.evaluation.experiment.run_case`), và in kèm mẫu số.

Ngoài ra :func:`run_noise_sweep` quét nhiễu bbox, còn
:mod:`saferoad.evaluation.grid` chạy lưới mật độ × góc camera × điều kiện ánh
sáng (đại diện bằng chất lượng detector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from .experiment import SimCase, run_case, strip_private


@dataclass
class AblationStage:
    key: str
    label: str
    description: str
    mutate: Callable[[Config], Any] | None = None


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
    cfg.conflict.pet_threshold = 0.0


def _no_measured_pet(cfg: Config) -> None:
    cfg.conflict.use_measured_pet = False


def _no_proximity(cfg: Config) -> None:
    cfg.conflict.proximity_gate = 1e9


def _no_closing_speed(cfg: Config) -> None:
    cfg.conflict.min_rel_speed = 0.3


def _no_episode(cfg: Config) -> None:
    cfg.conflict.episode_grouping = False


def _chain(*fns):
    def run(cfg: Config) -> None:
        for fn in fns:
            fn(cfg)
    return run


#: Bật dần từng khối (dòng 1 là baseline ngây thơ).
CUMULATIVE: list[AblationStage] = [
    AblationStage("baseline", "0. Baseline: TTC < 3 s trên toạ độ ảnh",
                  "Không homography, không làm mượt, không PET, không cổng, mỗi khung hình một cảnh báo (chống trùng 3 s)",
                  _chain(_no_homography, _no_smoothing, _no_pet, _no_proximity,
                         _no_closing_speed, _no_episode)),
    AblationStage("homography", "1. + Homography (mét)", "Quy đổi toạ độ ảnh sang mặt đất",
                  _chain(_no_smoothing, _no_pet, _no_proximity, _no_closing_speed, _no_episode)),
    AblationStage("smoothing", "2. + Làm mượt quỹ đạo", "Làm mượt + vận tốc hồi quy",
                  _chain(_no_pet, _no_proximity, _no_closing_speed, _no_episode)),
    AblationStage("episode", "3. + Gom episode", "Một tình huống → một sự kiện, gán nhãn tại đỉnh",
                  _chain(_no_pet, _no_proximity, _no_closing_speed)),
    AblationStage("closing", "4. + Cổng tốc độ tiếp cận", "Loại bám đuôi cùng tốc độ",
                  _chain(_no_pet, _no_proximity)),
    AblationStage("pet", "5. + PET (ngoại suy + đo)", "Thêm tiêu chí PET",
                  _no_proximity),
    AblationStage("full", "6. + Cổng khoảng cách = bản đầy đủ", "Hai xe phải thực sự đến gần nhau",
                  None),
]

#: Bỏ đúng một khối khỏi bản đầy đủ.
LEAVE_ONE_OUT: list[AblationStage] = [
    AblationStage("full", "Bản đầy đủ", "Mọi khối", None),
    AblationStage("-homography", "Bỏ homography", "TTC/khoảng cách tính trên tỉ lệ phẳng", _no_homography),
    AblationStage("-smoothing", "Bỏ làm mượt quỹ đạo", "Vận tốc từ hiệu vị trí, không EMA", _no_smoothing),
    AblationStage("-episode", "Bỏ gom episode", "Mỗi khung hình thoả ngưỡng một cảnh báo", _no_episode),
    AblationStage("-closing", "Bỏ cổng tốc độ tiếp cận", "Chấp nhận cả cặp bám đuôi cùng tốc độ", _no_closing_speed),
    AblationStage("-pet", "Bỏ PET", "Chỉ dùng TTC", _no_pet),
    AblationStage("-measured_pet", "Bỏ PET đo (chỉ PET ngoại suy)", "Hành vi PET của bản cũ", _no_measured_pet),
    AblationStage("-proximity", "Bỏ cổng khoảng cách", "Không đòi hai xe thực sự đến gần", _no_proximity),
]


def _row(stage: AblationStage, rec: dict) -> dict[str, Any]:
    m = rec["metrics"]
    rows = rec["reconciliation"]["rows"]
    return {
        "key": stage.key,
        "label": stage.label,
        "description": stage.description,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "ttc_mae": m["ttc_mae"],
        "pet_final": (rec["pet_errors"].get("final") or {}).get("mae"),
        "n_emitted": rows["emitted"],
        "n_ignored_outside": rows["ignored_outside_eval_region"],
        "n_scored": rows["scored"],
        "near_miss_recall": rec["by_kind"]["near_miss"]["recall"],
        "collision_recall": rec["by_kind"]["collision"]["recall"],
        "false_alarms_per_hour": rec["reconciliation"]["per_hour"]["false_alarms_per_hour"],
        "idf1": rec["tracking"]["idf1"],
        "config_sha256": rec["config_sha256"],
        "checks_pass": rec["reconciliation"]["all_checks_pass"],
    }


def run_stages(cfg: Config, case: SimCase, stages: list[AblationStage],
               progress: bool = True) -> list[dict[str, Any]]:
    rows = []
    for st in stages:
        rec = run_case(cfg, case, mutate=st.mutate)
        row = _row(st, rec)
        rows.append(row)
        if progress:
            p, r = row["precision"], row["recall"]
            print(f"  [{st.key:14s}] P={p['value']} ({p['num']}/{p['den']})  "
                  f"R={r['value']} ({r['num']}/{r['den']})  F1={row['f1']}  "
                  f"phát ra={row['n_emitted']}", flush=True)
    return rows


def run_noise_sweep(cfg: Config, case: SimCase,
                    noise_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 6.0),
                    miss_rate: float | None = None, progress: bool = True) -> list[dict]:
    """Quét mức nhiễu bbox, mọi thứ khác giữ nguyên."""
    rows = []
    for noise in noise_levels:
        rec = run_case(cfg, case, noise_px=noise, miss_rate=miss_rate)
        m = rec["metrics"]
        rows.append({
            "noise_px": noise,
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "ttc_mae": m["ttc_mae"],
            "pet_final_mae": (rec["pet_errors"].get("final") or {}).get("mae"),
            "idf1": rec["tracking"]["idf1"],
            "id_switches": rec["tracking"]["id_switches"],
            "n_emitted": rec["reconciliation"]["rows"]["emitted"],
        })
        if progress:
            print(f"  nhiễu={noise:4.1f}px  P={m['precision']['value']} R={m['recall']['value']} "
                  f"F1={m['f1']} IDF1={rec['tracking']['idf1']}", flush=True)
    return rows


__all__ = ["CUMULATIVE", "LEAVE_ONE_OUT", "run_stages", "run_noise_sweep", "strip_private"]
