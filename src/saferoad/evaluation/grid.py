"""Lưới thực nghiệm: mật độ × góc camera × điều kiện ánh sáng (đại diện).

Mỗi ô của lưới chạy **cùng một kịch bản** (cùng seed mô phỏng) qua hai cấu
hình: bản đầy đủ và baseline, để so sánh trên cùng mẫu. Kịch bản mô phỏng chỉ
phụ thuộc mật độ và seed; góc camera chỉ đổi cách chiếu bbox; điều kiện ánh sáng
chỉ đổi chất lượng detector. Nhờ đó mỗi yếu tố được tách riêng.

Giới hạn phải nói rõ: "ánh sáng" ở đây là **hồ sơ nhiễu/bỏ sót của detector**
(xem :data:`saferoad.evaluation.experiment.LIGHT_PROFILES`), không phải ảnh ban
đêm thật. Ảnh hưởng thật của ánh sáng lên YOLO chỉ đo được trên video thật.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import Config
from .ablation import CUMULATIVE, LEAVE_ONE_OUT
from .experiment import LIGHT_PROFILES, build_case, run_case

DENSITIES: dict[str, float] = {"thua": 0.6, "vua": 1.1, "dong": 1.6}
CAMERAS: tuple[str, ...] = ("cao_18m", "thap_10m", "xien_45")


def run_grid(
    cfg: Config,
    duration: float = 180.0,
    seed: int = 7,
    densities: dict[str, float] | None = None,
    cameras: tuple[str, ...] | None = None,
    lights: dict[str, tuple[float, float]] | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    densities = densities or DENSITIES
    cameras = cameras or CAMERAS
    lights = lights or LIGHT_PROFILES
    baseline = next(s for s in CUMULATIVE if s.key == "baseline")
    no_meas = next(s for s in LEAVE_ONE_OUT if s.key == "-measured_pet")

    cells: list[dict[str, Any]] = []
    scen: dict[str, Any] = {}
    t_start = time.perf_counter()
    for dname, rate in densities.items():
        for cam in cameras:
            case = build_case(duration=duration, seed=seed, arrival_rate=rate, camera_name=cam)
            scen[f"{dname}/{cam}"] = {
                k: v for k, v in case.scenario_report.items() if k != "crashes"
            } | {"n_gt": len(case.ground_truth),
                 "n_gt_collision": sum(1 for g in case.ground_truth if g.kind == "collision")}
            for lname, (noise, miss) in lights.items():
                cell = {"density": dname, "arrival_rate": rate, "camera": cam,
                        "light": lname, "noise_px": noise, "miss_rate": miss,
                        "n_gt": len(case.ground_truth)}
                for tag, mut in (("full", None), ("baseline", baseline.mutate),
                                 ("old_pet", no_meas.mutate)):
                    rec = run_case(cfg, case, noise_px=noise, miss_rate=miss, mutate=mut)
                    m = rec["metrics"]
                    cell[tag] = {
                        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                        "ttc_mae": m["ttc_mae"],
                        "fa_per_hour": rec["reconciliation"]["per_hour"]["false_alarms_per_hour"],
                        "miss_per_hour": rec["reconciliation"]["per_hour"]["misses_per_hour"],
                        "idf1": rec["tracking"]["idf1"],
                        "n_emitted": rec["reconciliation"]["rows"]["emitted"],
                        "checks_pass": rec["reconciliation"]["all_checks_pass"],
                        "config_sha256": rec["config_sha256"],
                    }
                cells.append(cell)
                if progress:
                    f, b = cell["full"], cell["baseline"]
                    print(f"  {dname:5s} {cam:9s} {lname:10s} n_gt={cell['n_gt']:3d}  "
                          f"đầy đủ F1={f['f1']:.3f} (P={f['precision']['value']}, "
                          f"R={f['recall']['value']})  baseline F1={b['f1']:.3f}  "
                          f"[{time.perf_counter() - t_start:.0f}s]", flush=True)
    return {"cells": cells, "scenarios": scen,
            "params": {"duration": duration, "seed": seed, "densities": densities,
                       "cameras": list(cameras), "lights": lights}}
