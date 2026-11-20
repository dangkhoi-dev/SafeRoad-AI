"""Giao diện dòng lệnh của SafeRoad AI.

Các lệnh chính::

    saferoad simulate   # sinh video mô phỏng + nhãn chuẩn
    saferoad run        # chạy pipeline trên một video
    saferoad evaluate   # chấm điểm + bảng ablation trên tập synthetic
    saferoad serve      # bật dashboard
    saferoad anonymize  # ẩn danh một video (làm mờ mặt, biển số)
    saferoad train-behavior  # huấn luyện bộ phân loại hành vi
    saferoad prepare-real    # dựng video + nhãn từ dataset thật (MVTI)
    saferoad evaluate-real   # chấm mAP/IDF1 trên dữ liệu thật
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import yaml

from .config import Config
from .utils.bundle import load_bundle

LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-28s %(message)s"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
def cmd_simulate(args: argparse.Namespace) -> int:
    """Sinh video mô phỏng giao lộ kèm nhãn chuẩn."""
    from .simulation.render import render_video
    from .simulation.scenario import build_scenario, compute_ground_truth, simulate

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Sinh kịch bản ({args.duration:.0f}s, seed={args.seed})...")
    vehicles = build_scenario(
        duration=args.duration, seed=args.seed, arrival_rate=args.arrival_rate
    )
    print(f"      {len(vehicles)} phương tiện được lên lịch")

    print("[2/4] Chạy vi mô phỏng (IDM + đèn tín hiệu + phanh tránh + va chạm có vật lý)...")
    report: dict = {}
    vehicles = simulate(vehicles, args.duration, report=report)
    print(f"      {len(vehicles)} phương tiện có quỹ đạo")
    print("      Kiểm tra kịch bản: "
          f"lên lịch {report['n_scheduled']}, vào được {report['n_spawned']}, "
          f"bỏ ở điểm vào {report['n_dropped_at_entry']}, đi hết tuyến {report['n_completed_route']}, "
          f"va chạm {report['n_crashes']} vụ, vượt đèn đỏ {report['n_runs_red']}")

    print("[3/4] Tính nhãn chuẩn (oracle): near-miss và va chạm, PET đo từ mốc vào/ra...")
    from .simulation.camera import COVERAGE

    gt = compute_ground_truth(vehicles, region=COVERAGE)
    severe = sum(1 for g in gt if g.ttc < 1.5)
    n_coll = sum(1 for g in gt if g.kind == "collision")
    print(f"      {len(gt)} xung đột: {len(gt) - n_coll} near-miss + {n_coll} va chạm; "
          f"{severe} có TTC<1.5s")

    print("[4/4] Kết xuất video...")
    video_path = out_dir / "synthetic_intersection.mp4"
    detections, id_map, renderer = render_video(
        vehicles, args.duration, video_path, fps=args.fps,
        frame_w=args.width, frame_h=args.height, progress=True,
    )

    # Lưu toàn bộ dữ liệu chuẩn để bước evaluate dùng lại mà không phải mô phỏng lại.
    bundle_path = out_dir / "synthetic_groundtruth.pkl"
    with open(bundle_path, "wb") as fh:
        pickle.dump(
            {
                "vehicles": vehicles, "ground_truth": gt, "detections": detections,
                "id_map": id_map, "camera": renderer.camera, "fps": args.fps,
                "duration": args.duration, "seed": args.seed,
                "frame_size": (args.width, args.height),
                "arrival_rate": args.arrival_rate, "scenario_report": report,
                "pinhole": renderer.cam, "gt_version": 2,
            },
            fh,
        )

    summary = {
        "video": str(video_path),
        "ground_truth_bundle": str(bundle_path),
        "n_vehicles": len(vehicles),
        "n_ground_truth_conflicts": len(gt),
        "n_ground_truth_near_miss": len(gt) - n_coll,
        "n_ground_truth_collision": n_coll,
        "n_severe": severe,
        "seed": args.seed,
        "arrival_rate": args.arrival_rate,
        "scenario_check": {k: v for k, v in report.items() if k != "crashes"},
        "duration_s": args.duration,
        "fps": args.fps,
    }
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ Video: {video_path}\n✓ Nhãn chuẩn: {bundle_path}")
    return 0


# --------------------------------------------------------------------------- #
def _review_target(path) -> Path:
    """Không ghi đè danh sách ứng viên đã được người gán nhãn.

    Nhãn ``labels_<tập>_<người>.csv`` tham chiếu ``candidate_id`` của đúng danh
    sách cũ; chạy lại (ví dụ sau khi đổi luật) mà ghi đè thì nhãn lệch hàng. Khi đã
    có nhãn, danh sách mới được ghi vào thư mục con ``reruns/``.
    """
    path = Path(path)
    tag = path.stem[len("review_candidates_"):] if path.stem.startswith("review_candidates_") else ""
    if tag and path.exists() and any(path.parent.glob(f"labels_{tag}_*.csv")):
        new = path.parent / "reruns" / path.name
        new.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {path.name} đã có nhãn người rà soát - giữ nguyên, ghi danh sách mới vào {new}")
        return new
    return path


def cmd_run(args: argparse.Namespace) -> int:
    """Chạy pipeline end-to-end trên một video."""
    from .pipeline.runner import Pipeline

    cfg = Config.load(args.config) if args.config else Config()
    if args.source:
        cfg.video.source = args.source
    if args.max_frames:
        cfg.video.max_frames = args.max_frames
    if args.device:
        cfg.detection.device = args.device
    if args.weights:
        cfg.detection.weights = args.weights
    if args.no_overlay:
        cfg.video.write_overlay = False
    if args.anonymize:
        cfg.privacy.enabled = True
    if args.output:
        cfg.output_dir = args.output
        cfg.video.overlay_path = str(Path(args.output) / "overlay.mp4")
        cfg.db_path = str(Path(args.output) / "saferoad.db")

    # Dùng homography của camera mô phỏng khi chạy trên video synthetic.
    if args.ground_truth:
        bundle = load_bundle(args.ground_truth)
        cfg.homography.image_points = bundle["camera"].image_points
        cfg.homography.world_points = bundle["camera"].world_points
        if args.replay_detections:
            from .detection.detector import ReplayDetector

            dc = cfg.detection
            detector = ReplayDetector(
                bundle["detections"],
                noise_px=dc.replay_noise_px if args.noise_px is None else args.noise_px,
                miss_rate=dc.replay_miss_rate if args.miss_rate is None else args.miss_rate,
                seed=dc.replay_seed,
            )
            pipe = Pipeline(cfg, detector=detector)
        else:
            pipe = Pipeline(cfg)
    else:
        pipe = Pipeline(cfg)

    print(f"Đang xử lý: {cfg.video.source}")
    result = pipe.run()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Pipeline.export_json(result, out_dir / "results.json", cfg)
    if args.review_csv:
        from .evaluation.real_conflicts import export_system_candidates

        review_csv = _review_target(args.review_csv)
        n = export_system_candidates(review_csv, result.events, cfg.video.source,
                                     result.fps_source)
        print(f"  Danh sách rà soát ({n} cảnh báo): {review_csv}")

    summary = result.summary()
    print("\n" + "=" * 62)
    print("KẾT QUẢ")
    print("=" * 62)
    print(f"  Số frame đã xử lý     : {summary['n_frames']}")
    print(f"  Thời lượng            : {summary['duration_s']}s")
    print(f"  Tốc độ xử lý          : {summary['processing_fps']} FPS (trung vị)")
    print(f"  Độ trễ end-to-end     : {summary['total_latency_ms']} ms/frame")
    print(f"  Độ trễ p95            : {summary['frame_latency_p95_ms']} ms/frame")
    print(f"  Số đối tượng theo dõi : {summary['n_tracks']}")
    print(f"  Near-miss             : {summary['total_events']}")
    print(f"  Trong đó nghiêm trọng : {summary['severe_events']} (TTC<1.5s)")
    print(f"  Risk Score trung bình : {summary['avg_risk_score']}/100")
    if summary["by_type"]:
        print("  Phân loại xung đột    :")
        for k, v in sorted(summary["by_type"].items(), key=lambda kv: -kv[1]):
            print(f"      {k:14s} {v}")
    if result.overlay_path:
        print(f"\n  Video overlay : {result.overlay_path}")
    print(f"  Kết quả JSON  : {out_dir / 'results.json'}")
    print(f"  Database      : {cfg.db_path}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace) -> int:
    """Chấm điểm trên tập mô phỏng: kết quả chính + đối soát + ablation + quét nhiễu.

    Kết quả chính và ``results.json`` cho dashboard lấy từ **cùng một lần chạy**,
    nên số sự kiện trên dashboard và trong báo cáo luôn khớp nhau.
    """
    from .evaluation.ablation import CUMULATIVE, LEAVE_ONE_OUT, run_noise_sweep, run_stages
    from .evaluation.experiment import (
        case_from_bundle, file_hash, reproducibility, run_case, strip_private,
    )
    from .pipeline.runner import Pipeline

    bundle = load_bundle(args.ground_truth)
    if bundle.get("gt_version", 1) < 2:
        print("Bundle mô phỏng được tạo bằng phiên bản cũ (nhãn PET và va chạm theo định "
              "nghĩa cũ). Chạy lại: saferoad simulate --duration 300 --seed 42",
              file=sys.stderr)
        return 2
    case = case_from_bundle(bundle)
    cfg = Config.load(args.config)
    if args.noise_px is not None:
        cfg.detection.replay_noise_px = args.noise_px
    if args.miss_rate is not None:
        cfg.detection.replay_miss_rate = args.miss_rate

    gt = case.ground_truth
    n_coll = sum(1 for g in gt if g.kind == "collision")
    print(f"Nhãn chuẩn: {len(gt)} xung đột ({len(gt) - n_coll} near-miss, {n_coll} va chạm) "
          f"trên {len(case.vehicles)} phương tiện, {case.duration:.0f} s\n")

    print("── Kết quả chính (cấu hình đầy đủ) ─────────────────────────")
    main = run_case(cfg, case, with_behavior=True)
    m, rec = main["metrics"], main["reconciliation"]
    for k, v in rec["rows"].items():
        print(f"  {k:40s} {v}")
    for k, ok in rec["checks"].items():
        print(f"  [{'OK' if ok else 'LỆCH'}] {k}")
    print(f"  Precision = {m['precision']['num']}/{m['precision']['den']} = {m['precision']['value']}")
    print(f"  Recall    = {m['recall']['num']}/{m['recall']['den']} = {m['recall']['value']}")

    if not args.no_export:
        out_dir = Path(args.out).parent
        cfg_export = Config.load(args.config)
        Pipeline.export_json(main["_result"], out_dir / "results.json", cfg_export)
        print(f"  → results.json cho dashboard ghi từ CHÍNH lần chạy này: {out_dir / 'results.json'}")

    print("\n── Ablation cộng dồn (cùng mẫu) ─────────────────────────────")
    cumulative = run_stages(cfg, case, CUMULATIVE)
    print("\n── Ablation bỏ từng khối khỏi bản đầy đủ (cùng mẫu) ─────────")
    loo = run_stages(cfg, case, LEAVE_ONE_OUT)
    print("\n── Quét nhiễu detector ──────────────────────────────────────")
    sweep = run_noise_sweep(cfg, case)

    payload = {
        "version": 2,
        "setup": {
            "n_ground_truth": len(gt),
            "n_ground_truth_near_miss": len(gt) - n_coll,
            "n_ground_truth_collision": n_coll,
            "n_vehicles": len(case.vehicles),
            "duration_s": case.duration,
            "seed": bundle["seed"],
            "arrival_rate": bundle.get("arrival_rate"),
            "noise_px": cfg.detection.replay_noise_px,
            "miss_rate": cfg.detection.replay_miss_rate,
            "replay_seed": cfg.detection.replay_seed,
            "eval_region": list(case.region),
        },
        "scenario_check": {k: v for k, v in case.scenario_report.items() if k != "crashes"},
        "main": strip_private(main),
        "ablation": {"cumulative": cumulative, "leave_one_out": loo},
        "noise_sweep": sweep,
        "reproducibility": reproducibility(cfg, {
            "bundle": str(args.ground_truth),
            "bundle_sha256": file_hash(args.ground_truth),
        }),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n✓ Báo cáo đánh giá: {out}")
    print(f"\nTỔNG KẾT: Precision={m['precision']['value']}  Recall={m['recall']['value']}  "
          f"F1={m['f1']}  TTC-MAE={m['ttc_mae']}s  "
          f"PET đo MAE={(main['pet_errors'].get('measured') or {}).get('mae')}s  "
          f"IDF1={main['tracking']['idf1']}")
    return 0


def cmd_evaluate_grid(args: argparse.Namespace) -> int:
    """Lưới thực nghiệm mật độ × góc camera × điều kiện ánh sáng (đại diện)."""
    from .evaluation.experiment import reproducibility
    from .evaluation.grid import run_grid

    cfg = Config.load(args.config)
    print("Lưới: 3 mật độ × 3 camera × 3 hồ sơ detector, mỗi ô chạy bản đầy đủ, "
          "baseline và bản PET cũ trên cùng kịch bản.\n")
    grid = run_grid(cfg, duration=args.duration, seed=args.seed)
    payload = {"version": 2, **grid, "reproducibility": reproducibility(cfg)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n✓ {out}")
    return 0


# --------------------------------------------------------------------------- #
class _IgnoreClientDisconnect(logging.Filter):
    """Bỏ log khi phía trình duyệt tự đóng kết nối - không phải lỗi máy chủ."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, (ConnectionResetError, ConnectionAbortedError))


def cmd_serve(args: argparse.Namespace) -> int:
    """Bật dashboard web."""
    import uvicorn

    from .dashboard.server import create_app

    cfg = Config.load(args.config) if args.config else Config()
    if args.results:
        cfg.output_dir = str(Path(args.results).parent)
    app = create_app(cfg, results_path=args.results, video_path=args.video,
                     demo=args.demo, sample_video=args.sample_video)
    # Trình duyệt hay huỷ ngang yêu cầu (tua video, tải lại trang). Trên Windows,
    # vòng lặp Proactor ghi việc đó thành ERROR "WinError 10054" dù không có gì hỏng.
    logging.getLogger("asyncio").addFilter(_IgnoreClientDisconnect())
    print(f"\n  Dashboard: http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------- #
def cmd_anonymize(args: argparse.Namespace) -> int:
    """Ẩn danh video: làm mờ khuôn mặt và biển số."""
    from .detection.detector import build_detector
    from .privacy.anonymize import anonymize_video

    cfg = Config.load(args.config) if args.config else Config()
    cfg.privacy.enabled = True
    detector = build_detector(cfg.detection)
    stats = anonymize_video(args.source, args.out, detector, cfg.privacy)
    print(
        f"✓ Đã ẩn danh {stats['frames']} frame → {args.out}\n"
        f"  Vùng khuôn mặt làm mờ: {stats['faces_blurred']}\n"
        f"  Vùng biển số làm mờ  : {stats['plates_blurred']}"
    )
    return 0


# --------------------------------------------------------------------------- #
def cmd_train_behavior(args: argparse.Namespace) -> int:
    """Huấn luyện bộ phân loại hành vi từ dữ liệu mô phỏng."""
    from .behavior.train import train_from_simulation

    metrics = train_from_simulation(
        out_path=args.out, n_scenarios=args.scenarios, duration=args.duration
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0



# --------------------------------------------------------------------------- #
def cmd_prepare_real(args: argparse.Namespace) -> int:
    """Dựng video và nhãn chuẩn từ dataset thật (Multi-view Traffic Intersection)."""
    from .data.mvti import build_video, discover, load_coco

    root = Path(args.root)
    found = discover(root)
    if not found:
        print(
            f"Lỗi: không tìm thấy file annotation MVTI trong {root}.\n"
            f"Cần có 'infrastructure-mscoco.json' hoặc 'drone-mscoco.json'.",
            file=sys.stderr,
        )
        return 2

    view = args.view
    ann = found.get(view) or found.get("merged")
    if ann is None:
        print(f"Lỗi: không có annotation cho view '{view}'. Có: {list(found)}", file=sys.stderr)
        return 2

    print(f"[1/3] Nạp annotation {ann.name} (view={view})...")
    seq = load_coco(ann, root, view=view)
    info = seq.summary()
    for k, v in info.items():
        print(f"      {k}: {v}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Ghép {seq.n_frames} khung hình thành video @ {args.fps} FPS...")
    video = build_video(seq, out_dir / f"{view.lower()}.mp4", fps=args.fps,
                        max_frames=args.max_frames)

    print("[3/3] Lưu nhãn chuẩn...")
    bundle = out_dir / f"{view.lower()}_groundtruth.pkl"
    with open(bundle, "wb") as fh:
        pickle.dump(
            {
                "sequence": seq, "fps": args.fps, "view": view,
                "frame_size": (seq.width, seq.height),
            },
            fh,
        )
    (out_dir / f"{view.lower()}_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ Video: {video}\n✓ Nhãn chuẩn: {bundle}")
    return 0


# --------------------------------------------------------------------------- #
#: Số detection tối đa mỗi ảnh khi chấm mAP - quy ước của bộ chỉ số COCO.
MAX_DETS_PER_IMAGE = 100


def _environment() -> dict:
    """Phiên bản thư viện lúc chạy, ghi kèm báo cáo.

    Cùng mã nguồn và cùng seed, vài con số trung gian vẫn có thể lệch ở chữ số
    thứ ba giữa hai máy (khác phiên bản OpenCV/NumPy); ghi lại để truy được.
    """
    import platform

    import cv2
    import numpy

    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
    }


def _json_safe(obj):
    """Đổi NaN/inf thành None, đệ quy qua dict và list."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def cmd_evaluate_real(args: argparse.Namespace) -> int:
    """Detection → tracking → cảnh báo trên CÙNG một video thật có nhãn bbox + ID.

    Ba bảng trên cùng dữ liệu: mAP detection, IDF1/MOTA tracking, và cảnh báo
    xung đột của pipeline so với cảnh báo tham chiếu chạy trên quỹ đạo người gán
    nhãn. Ghi kèm danh sách ứng viên để người rà soát.
    """
    from .detection.detector import build_detector
    from .evaluation.detection import evaluate_detection
    from .evaluation.experiment import file_hash, reproducibility, strip_private
    from .evaluation.real import conflict_statistics, tracking_metrics_from_boxes
    from .evaluation.real_conflicts import (
        compare_to_reference, export_review_candidates, reference_events,
    )
    from .pipeline.runner import Pipeline
    from .types import Detection, VehicleClass

    bundle = load_bundle(args.ground_truth)
    seq = bundle["sequence"]
    fps = float(bundle.get("fps", 30))

    cfg = Config.load(args.config) if args.config else Config()
    cfg.video.source = args.source
    cfg.video.write_overlay = not args.no_overlay
    out = Path(args.out)
    tag = out.stem.replace("evaluation_", "")
    cfg.video.overlay_path = str(out.parent / f"overlay_{tag}.mp4")
    cfg.db_path = str(out.parent / f"saferoad_{tag}.db")
    if args.device:
        cfg.detection.device = args.device
    if args.weights:
        cfg.detection.weights = args.weights
    if args.max_frames:
        cfg.video.max_frames = args.max_frames
    if args.start_frame:
        cfg.video.start_frame = args.start_frame
    if getattr(args, "backend", None):
        cfg.detection.backend = args.backend
    if args.imgsz:
        cfg.detection.imgsz = args.imgsz

    op_conf = cfg.detection.conf
    map_conf = min(args.map_conf, op_conf)
    cfg.detection.conf = map_conf
    print(f"Chạy pipeline trên dữ liệu thật: {cfg.video.source}")
    print(f"  Ngưỡng chấm mAP: {map_conf}  ·  ngưỡng vận hành pipeline: {op_conf}")
    detector = build_detector(cfg.detection)

    # Lớp có trong dự đoán nhưng KHÔNG có trong bộ nhãn được gộp vào lớp cha
    # tương ứng trước khi chấm (vd. bộ nhãn đại lộ chỉ có "Car" gồm cả xe tải nhẹ).
    gt_classes = {d.cls for dets in seq.detections.values() for d in dets}
    remap = {}
    if VehicleClass.TRUCK not in gt_classes and VehicleClass.CAR in gt_classes:
        remap[VehicleClass.TRUCK] = VehicleClass.CAR
    captured: dict[int, list] = {}

    class _Recording:
        def __init__(self, inner):
            self.inner = inner

        @property
        def name(self) -> str:
            return self.inner.name

        def detect(self, frame, frame_idx=0):
            raw = self.inner.detect(frame, frame_idx)
            top = sorted(raw, key=lambda d: d.score, reverse=True)[:MAX_DETS_PER_IMAGE]
            captured[frame_idx] = [
                Detection(bbox=d.bbox, score=d.score, cls=remap.get(d.cls, d.cls)) for d in top
            ]
            return [d for d in raw if d.score >= op_conf]

    result = Pipeline(cfg, detector=_Recording(detector), write_db=not args.no_db).run()
    first = cfg.video.start_frame
    last = first + result.n_frames
    gt_dets = {f: d for f, d in seq.detections.items() if first <= f < last}

    # --- 1. Detection ---------------------------------------------------- #
    det_metrics = evaluate_detection(captured, gt_dets, operating_conf=op_conf)
    # Cùng dự đoán, bỏ qua hộp chuẩn nhỏ hơn min_box_area - vùng pipeline cố ý
    # không xử lý (quá xa camera). Báo cả hai để không che con số khắt khe hơn.
    det_metrics_big = evaluate_detection(captured, gt_dets, operating_conf=op_conf,
                                         min_area=cfg.detection.min_box_area)
    # Chỉ chấm trên đối tượng CHƯA xuất hiện trước khung --seen-before (tức là
    # chưa có trong phần video dùng để fine-tune): hộp của đối tượng đã gặp là
    # vùng bỏ qua. Trả lời câu hỏi "model học thuộc xe cũ hay nhận ra xe mới?".
    det_unseen = None
    if args.seen_before:
        seen_boxes = {
            (int(f), tuple(round(v, 1) for v in box))
            for tr in seq.tracks.values() if min(tr.frames) < args.seen_before
            for f, box in zip(tr.frames, tr.boxes)
        }
        n_unseen_tracks = sum(1 for tr in seq.tracks.values()
                              if min(tr.frames) >= args.seen_before
                              and any(first <= f < last for f in tr.frames))
        det_unseen = evaluate_detection(
            captured, gt_dets, operating_conf=op_conf, min_area=cfg.detection.min_box_area,
            ignore_gt=lambda f, d: (int(f), tuple(round(v, 1) for v in d.bbox)) in seen_boxes,
        ).to_dict()
        det_unseen["seen_before_frame"] = args.seen_before
        det_unseen["n_unseen_tracks"] = n_unseen_tracks
        det_unseen["n_test_tracks"] = sum(1 for tr in seq.tracks.values()
                                          if any(first <= f < last for f in tr.frames))

    # --- 2. Tracking ----------------------------------------------------- #
    mapping, track_metrics = tracking_metrics_from_boxes(
        result.tracks, seq, min_area=cfg.detection.min_box_area)

    # --- 3. Cảnh báo: hệ thống so với tham chiếu trên quỹ đạo người gán ---- #
    cfg_ref = Config.load(args.config) if args.config else Config()
    ref_events, ref_funnel, _ref_tracks = reference_events(cfg_ref, seq, fps)
    ref_events = [e for e in ref_events if first / fps <= e.t < last / fps]
    cmp_ = compare_to_reference(result.events, result.funnel, ref_events, mapping,
                                result.duration)
    review_path = _review_target(out.parent / f"review_candidates_{tag}.csv")
    n_review = export_review_candidates(
        review_path, result.events, ref_events, cmp_["_detail"], mapping,
        video=str(cfg.video.source), fps=fps)

    speed = None
    if cfg_ref.homography.image_points:
        from .geometry.calibration import speed_check

        speed = speed_check(seq, cfg_ref.homography, fps)
    calib = None
    if args.calibration_report and Path(args.calibration_report).exists():
        calib = json.loads(Path(args.calibration_report).read_text(encoding="utf-8"))
        calib = {k: calib[k] for k in ("source", "assumptions", "n_points", "fit",
                                       "leave_one_out", "check") if k in calib}

    payload = {
        "version": 2,
        "dataset": {**seq.summary(), "frames_evaluated": [first, last],
                    "stitch": bundle.get("stitch")},
        "thresholds": {"map_conf": map_conf, "pipeline_conf": op_conf,
                       "min_box_area": cfg.detection.min_box_area},
        "detector_backend": cfg.detection.backend,
        "weights": cfg.detection.weights,
        "class_remap_for_scoring": {k.value: v.value for k, v in remap.items()},
        "calibration": calib,
        "calibration_speed_check": speed,
        "detection": det_metrics.to_dict(),
        "detection_min_area": det_metrics_big.to_dict(),
        **({"detection_unseen_objects": det_unseen} if det_unseen else {}),
        "tracking": track_metrics.to_dict(),
        "alerts": {
            **strip_private(cmp_),
            "reference_funnel": ref_funnel,
            "system_statistics": conflict_statistics(result.events),
            "review_candidates_csv": str(review_path),
            "n_review_candidates": n_review,
        },
        "runtime": result.summary(),
        "reproducibility": reproducibility(cfg, {
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": file_hash(args.ground_truth),
            "weights_sha256": file_hash(cfg.detection.weights),
        }),
        "note": (
            "Bộ dữ liệu có nhãn bbox + ID nhưng KHÔNG có nhãn xung đột. 'alerts' so "
            "pipeline với cảnh báo tham chiếu (cùng thuật toán chạy trên quỹ đạo người "
            "gán nhãn) - đo sai số do detection + tracking, không phải độ đúng tuyệt "
            "đối. Độ đúng tuyệt đối cần nhãn người rà soát: xem review_candidates_*.csv "
            "và lệnh 'saferoad review-score'."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")

    d = det_metrics.to_dict()
    t = track_metrics.to_dict()
    m = cmp_["metrics_vs_reference"]
    print("\n" + "=" * 62)
    print("DETECTION → TRACKING → CẢNH BÁO (cùng một video)")
    print("=" * 62)
    print(f"  mAP@0.5 = {d['mAP50']:.4f} · mAP@0.5:0.95 = {d['mAP50_95']:.4f} · "
          f"P/R@{op_conf} = {d['precision50_op']:.3f}/{d['recall50_op']:.3f}")
    db = det_metrics_big.to_dict()
    print(f"  (bỏ qua {db['n_gt_ignored']} hộp chuẩn < {cfg.detection.min_box_area:g} px²) "
          f"mAP@0.5 = {db['mAP50']:.4f} · P/R@{op_conf} = "
          f"{db['precision50_op']:.3f}/{db['recall50_op']:.3f}")
    print(f"  IDF1 = {t['idf1']:.4f} · MOTA = {t['mota']:.4f} · ID switch = {t['id_switches']}")
    print(f"  Cảnh báo hệ thống: {len(result.events)} · tham chiếu: {len(ref_events)}")
    print(f"  So với tham chiếu: P = {m['precision']['num']}/{m['precision']['den']}, "
          f"R = {m['recall']['num']}/{m['recall']['den']}")
    ph = cmp_["reconciliation"]["per_hour"]
    print(f"  Báo nhầm/giờ = {ph['false_alarms_per_hour']} · bỏ sót/giờ = {ph['misses_per_hour']}")
    print(f"  Tốc độ: {result.processing_fps:.1f} FPS")
    print(f"\n✓ {out}\n✓ Danh sách rà soát ({n_review} ứng viên): {review_path}")
    return 0


def cmd_prepare_street(args: argparse.Namespace) -> int:
    """Nạp nhãn CVAT (đại lộ Kaggle), ghép quỹ đạo giữa các đoạn, lưu bundle."""
    from .data.cvat_video import load_cvat_video

    print(f"Nạp {args.annotations} ...")
    seq, stats = load_cvat_video(args.annotations, name=args.name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / f"{args.name}_groundtruth.pkl"
    with open(bundle, "wb") as fh:
        pickle.dump({"sequence": seq, "fps": args.fps, "view": args.name,
                     "frame_size": (seq.width, seq.height), "stitch": stats}, fh)
    info = {**seq.summary(), "stitch": stats}
    (out / f"{args.name}_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n✓ {bundle}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Tính homography từ điểm mốc + báo cáo sai số (fit / leave-one-out / kiểm tra)."""
    from .geometry.calibration import run_calibration

    rep = run_calibration(args.points, args.frame, args.out, args.image)
    print(json.dumps({k: rep[k] for k in ("n_points", "fit", "leave_one_out")}
                     | ({"check": rep["check"]} if "check" in rep else {}),
                     ensure_ascii=False, indent=2))
    if rep.get("warning"):
        print("CẢNH BÁO:", rep["warning"])
    print("\nDán khối sau vào mục 'homography' của file cấu hình:")
    print(yaml.safe_dump({"homography": rep["homography"]}, allow_unicode=True, sort_keys=False))
    print(f"✓ {args.out}" + (f"\n✓ {args.image}" if args.image else ""))
    return 0


def cmd_review_score(args: argparse.Namespace) -> int:
    """Chấm hệ thống theo nhãn người rà soát (kappa + P/R + báo nhầm/giờ)."""
    from .evaluation.real_conflicts import score_reviewed

    rep = score_reviewed(args.candidates, args.labels, args.duration, args.extra_missed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if rep.get("kappa_ok") is False:
        print("\nκ < 0,6: hai người hiểu protocol khác nhau - ngồi lại thống nhất rồi gán lại "
              "(protocol §8), KHÔNG lấy trung bình.")
    print(f"✓ {args.out}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saferoad",
        description="SafeRoad AI - phát hiện near-miss & bản đồ rủi ro giao thông",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Bật log chi tiết")
    sub = p.add_subparsers(dest="command", required=True)

    # simulate
    s = sub.add_parser("simulate", help="Sinh video mô phỏng + nhãn chuẩn")
    s.add_argument("--duration", type=float, default=120.0, help="Thời lượng (giây)")
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--arrival-rate", type=float, default=1.1, help="Số xe/giây")
    s.add_argument("--out", default="data/samples", help="Thư mục kết quả")
    s.set_defaults(func=cmd_simulate)

    # run
    r = sub.add_parser("run", help="Chạy pipeline trên một video")
    r.add_argument("--config", help="File cấu hình YAML")
    r.add_argument("--source", help="Đường dẫn video đầu vào")
    r.add_argument("--output", help="Thư mục kết quả")
    r.add_argument("--weights", help="Trọng số YOLO")
    r.add_argument("--device", help="cpu | cuda:0")
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--no-overlay", action="store_true", help="Không xuất video overlay")
    r.add_argument("--anonymize", action="store_true", help="Bật ẩn danh")
    r.add_argument("--review-csv", help="Ghi danh sách cảnh báo để người rà soát (CSV)")
    r.add_argument("--ground-truth", help="Bundle .pkl để lấy homography của camera mô phỏng")
    r.add_argument("--replay-detections", action="store_true",
                   help="Dùng bbox chuẩn thay vì chạy YOLO (chỉ cho video synthetic)")
    r.add_argument("--noise-px", type=float, default=None,
                   help="Ghi đè detection.replay_noise_px trong cấu hình")
    r.add_argument("--miss-rate", type=float, default=None,
                   help="Ghi đè detection.replay_miss_rate trong cấu hình")
    r.set_defaults(func=cmd_run)

    # evaluate
    e = sub.add_parser("evaluate", help="Chấm điểm + ablation trên tập synthetic")
    e.add_argument("--ground-truth", default="data/samples/synthetic_groundtruth.pkl")
    e.add_argument("--config", default="configs/synthetic.yaml", help="File cấu hình YAML")
    e.add_argument("--noise-px", type=float, default=None)
    e.add_argument("--miss-rate", type=float, default=None)
    e.add_argument("--no-export", action="store_true",
                   help="Không ghi đè results.json của dashboard")
    e.add_argument("--out", default="data/outputs/evaluation.json")
    e.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("evaluate-grid",
                       help="Lưới mật độ × góc camera × ánh sáng (đại diện), so baseline")
    g.add_argument("--config", default="configs/synthetic.yaml")
    g.add_argument("--duration", type=float, default=180.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/outputs/evaluation_grid.json")
    g.set_defaults(func=cmd_evaluate_grid)

    # serve
    v = sub.add_parser("serve", help="Bật dashboard web")
    v.add_argument("--config", help="File cấu hình YAML")
    v.add_argument("--results", default="data/outputs/results.json")
    v.add_argument("--video", default="data/outputs/overlay.mp4")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--demo", action="store_true",
                   help="Bật tải video lên và chạy pipeline thật trên máy chủ (bản demo)")
    v.add_argument("--sample-video", help="Video mẫu cho nút 'Chạy video mẫu'")
    v.set_defaults(func=cmd_serve)

    # anonymize
    a = sub.add_parser("anonymize", help="Ẩn danh video (mặt + biển số)")
    a.add_argument("--source", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--config")
    a.set_defaults(func=cmd_anonymize)

    # train-behavior
    t = sub.add_parser("train-behavior", help="Huấn luyện bộ phân loại hành vi")
    t.add_argument("--out", default="models/behavior_clf.joblib")
    t.add_argument("--scenarios", type=int, default=6)
    t.add_argument("--duration", type=float, default=120.0)
    t.set_defaults(func=cmd_train_behavior)

    # prepare-real
    pr = sub.add_parser("prepare-real", help="Dựng video + nhãn từ dataset thật (MVTI)")
    pr.add_argument("--root", default="data/raw/mvti",
                    help="Thư mục chứa Infrastructure/, Drone/ và các file *-mscoco.json")
    pr.add_argument("--view", default="Infrastructure", choices=["Infrastructure", "Drone"])
    pr.add_argument("--fps", type=int, default=30)
    pr.add_argument("--max-frames", type=int, default=0)
    pr.add_argument("--out", default="data/processed/mvti")
    pr.set_defaults(func=cmd_prepare_real)

    # evaluate-real
    er = sub.add_parser("evaluate-real", help="Chấm mAP + IDF1 trên dữ liệu thật")
    er.add_argument("--source", default="data/processed/mvti/infrastructure.mp4")
    er.add_argument("--ground-truth", default="data/processed/mvti/infrastructure_groundtruth.pkl")
    er.add_argument("--config")
    er.add_argument("--weights")
    er.add_argument("--device")
    er.add_argument("--max-frames", type=int, default=0)
    er.add_argument("--backend", choices=["yolo", "tiled"],
                    help="Ghi đè detection.backend trong config, để so sánh hai "
                         "detector với mọi tham số khác giữ nguyên")
    er.add_argument("--map-conf", type=float, default=0.001,
                    help="Ngưỡng tin cậy khi chấm mAP (chuẩn COCO là 0.001). "
                         "Pipeline vẫn chạy ở ngưỡng vận hành trong config.")
    er.add_argument("--no-overlay", action="store_true")
    er.add_argument("--imgsz", type=int, default=0, help="Ghi đè detection.imgsz")
    er.add_argument("--no-db", action="store_true", help="Không ghi SQLite (nhanh hơn khi chỉ chấm điểm)")
    er.add_argument("--seen-before", type=int, default=0,
                    help="Chấm thêm detection chỉ trên đối tượng xuất hiện lần đầu từ khung này "
                         "(đối tượng đã có trong phần huấn luyện thành vùng bỏ qua)")
    er.add_argument("--start-frame", type=int, default=0,
                    help="Chỉ chấm từ khung hình này (vd. phần test sau fine-tune)")
    er.add_argument("--calibration-report",
                    help="JSON do 'saferoad calibrate' ghi, để đưa sai số hiệu chuẩn vào báo cáo")
    er.add_argument("--out", default="data/outputs/evaluation_real.json")
    er.set_defaults(func=cmd_evaluate_real)

    ps = sub.add_parser("prepare-street",
                        help="Nạp nhãn CVAT của video đại lộ (Kaggle), ghép quỹ đạo")
    ps.add_argument("--annotations", default="data/raw/mvti/merged_annotations.json")
    ps.add_argument("--name", default="street")
    ps.add_argument("--fps", type=int, default=30)
    ps.add_argument("--out", default="data/processed/street")
    ps.set_defaults(func=cmd_prepare_street)

    ca = sub.add_parser("calibrate", help="Tính homography từ điểm mốc + báo cáo sai số")
    ca.add_argument("--points", required=True, help="YAML điểm mốc (xem geometry/calibration.py)")
    ca.add_argument("--frame", help="Ảnh hoặc video để vẽ lưới mét kiểm tra")
    ca.add_argument("--image", help="Ảnh lưới mét xuất ra")
    ca.add_argument("--out", default="data/outputs/calibration.json")
    ca.set_defaults(func=cmd_calibrate)

    rs = sub.add_parser("review-score", help="Chấm theo nhãn người rà soát (kappa, P/R, báo nhầm/giờ)")
    rs.add_argument("--candidates", required=True, help="review_candidates_*.csv")
    rs.add_argument("--labels", nargs="+", required=True,
                    help="CSV đã gán nhãn của từng người (cột candidate_id, label)")
    rs.add_argument("--duration", type=float, required=True, help="Thời lượng video đã rà soát (giây)")
    rs.add_argument("--extra-missed", type=int, default=0,
                    help="Số xung đột người xem thấy nhưng không có trong danh sách ứng viên")
    rs.add_argument("--out", default="data/outputs/review_score.json")
    rs.set_defaults(func=cmd_review_score)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nĐã huỷ.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
