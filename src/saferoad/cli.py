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
import pickle
import sys
from pathlib import Path

from .config import Config

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

    print("[2/4] Chạy vi mô phỏng (IDM + đèn tín hiệu + phanh tránh)...")
    vehicles = simulate(vehicles, args.duration)
    print(f"      {len(vehicles)} phương tiện có quỹ đạo")

    print("[3/4] Tính nhãn chuẩn near-miss (oracle)...")
    from .simulation.camera import COVERAGE

    gt = compute_ground_truth(vehicles, region=COVERAGE)
    severe = sum(1 for g in gt if g.ttc < 1.5)
    print(f"      {len(gt)} near-miss, trong đó {severe} nghiêm trọng (TTC<1.5s)")

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
            },
            fh,
        )

    summary = {
        "video": str(video_path),
        "ground_truth_bundle": str(bundle_path),
        "n_vehicles": len(vehicles),
        "n_ground_truth_conflicts": len(gt),
        "n_severe": severe,
        "duration_s": args.duration,
        "fps": args.fps,
    }
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ Video: {video_path}\n✓ Nhãn chuẩn: {bundle_path}")
    return 0


# --------------------------------------------------------------------------- #
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
        with open(args.ground_truth, "rb") as fh:
            bundle = pickle.load(fh)
        cfg.homography.image_points = bundle["camera"].image_points
        cfg.homography.world_points = bundle["camera"].world_points
        if args.replay_detections:
            from .detection.detector import ReplayDetector

            detector = ReplayDetector(
                bundle["detections"], noise_px=args.noise_px, miss_rate=args.miss_rate
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
    """Chấm điểm hệ thống trên tập synthetic có ground truth."""
    from .evaluation.ablation import run_ablation, run_noise_sweep, save_report, to_markdown
    from .evaluation.metrics import (
        build_gt_points,
        build_id_mapping,
        evaluate_conflicts,
        severity_breakdown,
    )
    from .detection.detector import ReplayDetector
    from .pipeline.runner import Pipeline
    from .simulation.camera import COVERAGE

    with open(args.ground_truth, "rb") as fh:
        bundle = pickle.load(fh)

    cfg = Config.load(args.config) if args.config else Config()
    cfg.video.source = args.source or str(
        Path(args.ground_truth).parent / "synthetic_intersection.mp4"
    )
    cfg.homography.image_points = bundle["camera"].image_points
    cfg.homography.world_points = bundle["camera"].world_points
    cfg.video.write_overlay = False

    gt = bundle["ground_truth"]
    vehicles = bundle["vehicles"]
    detections = bundle["detections"]
    gt_points = build_gt_points(
        detections, bundle["id_map"], vehicles, bundle["fps"],
        min_box_area=cfg.detection.min_box_area,
    )
    n_eval_points = sum(len(v) for v in gt_points.values())
    print(
        f"Nhãn chuẩn: {len(gt)} near-miss trên {len(vehicles)} phương tiện\n"
        f"Vùng đánh giá tracking: {n_eval_points} điểm-đối-tượng "
        f"(bbox ≥ {cfg.detection.min_box_area:.0f} px²)\n"
    )

    print("── Bảng ablation ───────────────────────────────────────────")
    ablation = run_ablation(
        cfg, detections, gt, vehicles, gt_points, region=COVERAGE,
        noise_px=args.noise_px, miss_rate=args.miss_rate,
    )

    print("\n── Quét mức nhiễu detector ─────────────────────────────────")
    sweep = run_noise_sweep(
        cfg, detections, gt, vehicles, gt_points, region=COVERAGE,
        miss_rate=args.miss_rate,
    )

    print("\n── Chi tiết theo mức nghiêm trọng (cấu hình đầy đủ) ────────")
    detector = ReplayDetector(detections, noise_px=args.noise_px,
                              miss_rate=args.miss_rate, seed=17)
    result = Pipeline(cfg, detector=detector, write_db=False).run(progress_every=0)
    mapping, track_metrics = build_id_mapping(result.tracks, gt_points)
    severity = severity_breakdown(result.events, gt, mapping, region=COVERAGE)
    for row in severity:
        if row.get("recall") is not None:
            print(f"  {row['band']:32s} n={row['n_gt']:3d}  Recall={row['recall']:.3f}")

    final = evaluate_conflicts(result.events, gt, mapping, region=COVERAGE)
    out = Path(args.out)
    save_report(
        ablation, sweep, severity, out,
        extra={
            "final": final.to_dict(),
            "tracking": track_metrics.to_dict(),
            "runtime": result.summary(),
            "setup": {
                "n_ground_truth": len(gt),
                "n_vehicles": len(vehicles),
                "noise_px": args.noise_px,
                "miss_rate": args.miss_rate,
                "duration_s": bundle["duration"],
                "seed": bundle["seed"],
            },
        },
    )

    print("\n" + to_markdown(ablation))
    print(f"\n✓ Báo cáo đánh giá: {out}")
    print(
        f"\nTỔNG KẾT: Precision={final.precision:.3f}  Recall={final.recall:.3f}  "
        f"F1={final.f1:.3f}  TTC-MAE={final.ttc_mae:.3f}s  IDF1={track_metrics.idf1:.3f}"
    )
    return 0


# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    """Bật dashboard web."""
    import uvicorn

    from .dashboard.server import create_app

    cfg = Config.load(args.config) if args.config else Config()
    if args.results:
        cfg.output_dir = str(Path(args.results).parent)
    app = create_app(cfg, results_path=args.results, video_path=args.video)
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
#: Số detection tối đa mỗi ảnh khi chấm mAP — quy ước của bộ chỉ số COCO.
MAX_DETS_PER_IMAGE = 100


def cmd_evaluate_real(args: argparse.Namespace) -> int:
    """Chấm điểm detection + tracking trên dữ liệu thật có nhãn."""
    from .detection.detector import build_detector
    from .evaluation.detection import evaluate_detection
    from .evaluation.real import conflict_statistics, tracking_metrics_from_boxes
    from .pipeline.runner import Pipeline

    with open(args.ground_truth, "rb") as fh:
        bundle = pickle.load(fh)
    seq = bundle["sequence"]

    cfg = Config.load(args.config) if args.config else Config()
    cfg.video.source = args.source
    cfg.video.write_overlay = not args.no_overlay
    cfg.video.overlay_path = str(Path(args.out).parent / "overlay_real.mp4")
    cfg.db_path = str(Path(args.out).parent / "saferoad_real.db")
    if args.device:
        cfg.detection.device = args.device
    if args.weights:
        cfg.detection.weights = args.weights
    if args.max_frames:
        cfg.video.max_frames = args.max_frames

    # mAP phải được tính trên detection ở ngưỡng tin cậy RẤT THẤP. Đây không
    # phải tiểu tiết: mAP là diện tích dưới đường precision-recall, và chấm nó
    # trên danh sách đã cắt ở ngưỡng vận hành (0,25) thì đường PR bị cụt ở giữa
    # chừng — mAP thu được thấp hơn giá trị thật, và thấp theo cách không so
    # sánh được với bất kỳ con số mAP nào khác trong tài liệu. Chuẩn COCO dùng
    # 0,001. Ngược lại, pipeline (tracking, xung đột) phải chạy ở đúng ngưỡng
    # vận hành, nếu không sẽ ngập detection rác.
    #
    # Chạy detector MỘT lần ở ngưỡng thấp rồi lọc lại cho pipeline — vừa đúng
    # cả hai mục đích, vừa không phải chạy YOLO hai lượt trên cùng video.
    op_conf = cfg.detection.conf
    map_conf = min(args.map_conf, op_conf)
    cfg.detection.conf = map_conf
    print(f"Chạy pipeline trên dữ liệu thật: {cfg.video.source}")
    print(f"  Ngưỡng chấm mAP: {map_conf}  ·  ngưỡng vận hành pipeline: {op_conf}")
    if cfg.detection.backend != "tiled":
        # Camera hạ tầng đặt cao khiến phương tiện chỉ chiếm vài chục pixel.
        # Suy luận một lượt trên khung hình thu nhỏ bỏ sót phần lớn trong số
        # đó; cắt ô và chạy ở độ phân giải gốc nâng recall lên khoảng gấp đôi.
        # Chạy nhầm cấu hình mặc định cho ra một con số mAP thấp không phản
        # ánh hệ thống, nên phải nói rõ ngay từ đầu chứ không để người dùng
        # phát hiện sau khi đã chờ xong.
        print(
            f"  CẢNH BÁO: đang dùng detector '{cfg.detection.backend}' "
            "(một lượt trên cả khung hình).\n"
            "           Với camera hạ tầng nên chạy kèm: "
            "--config configs/mvti.yaml (detector chia ô)."
        )
    detector = build_detector(cfg.detection)

    captured: dict[int, list] = {}

    class _Recording:
        def __init__(self, inner):
            self.inner = inner

        def detect(self, frame, frame_idx=0):
            raw = self.inner.detect(frame, frame_idx)
            # Chuẩn COCO chấm mAP với tối đa 100 detection mỗi ảnh. Giới hạn
            # này cũng cần cho bộ nhớ: ở ngưỡng 0,001 với detector chia ô, một
            # frame có thể trả về hàng nghìn hộp, nhân với vài nghìn frame là
            # hàng triệu đối tượng phải giữ trong RAM.
            captured[frame_idx] = sorted(
                raw, key=lambda d: d.score, reverse=True
            )[:MAX_DETS_PER_IMAGE]
            return [d for d in raw if d.score >= op_conf]

    result = Pipeline(cfg, detector=_Recording(detector), write_db=True).run()

    # --- Detection ---------------------------------------------------- #
    gt_dets = {
        f: d for f, d in seq.detections.items()
        if not args.max_frames or f < args.max_frames
    }
    det_metrics = evaluate_detection(captured, gt_dets)

    # --- Tracking ------------------------------------------------------ #
    _mapping, track_metrics = tracking_metrics_from_boxes(result.tracks, seq)

    # --- Xung đột (chỉ thống kê mô tả) --------------------------------- #
    conflicts = conflict_statistics(result.events)

    payload = {
        "dataset": seq.summary(),
        "thresholds": {"map_conf": map_conf, "pipeline_conf": op_conf},
        "detection": det_metrics.to_dict(),
        "tracking": track_metrics.to_dict(),
        "conflicts": conflicts,
        "runtime": result.summary(),
        "note": (
            "Dataset thật có nhãn bbox và ID đối tượng nhưng KHÔNG có nhãn xung đột. "
            "Vì vậy mục 'conflicts' chỉ là thống kê mô tả; Precision/Recall của "
            "near-miss được đo riêng trên tập mô phỏng có ground truth."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("KẾT QUẢ TRÊN DỮ LIỆU THẬT")
    print("=" * 62)
    d = det_metrics.to_dict()
    print(f"  Detection mAP@0.5      : {d['mAP50']:.4f}   (poster đặt mục tiêu ≥ 0.70)")
    print(f"  Detection mAP@0.5:0.95 : {d['mAP50_95']:.4f}")
    print(f"  Precision@0.5          : {d['precision50']:.4f}")
    print(f"  Recall@0.5             : {d['recall50']:.4f}")
    for k, v in d["per_class"].items():
        print(f"      {k:12s} AP50={v['AP50']:.3f}  n_gt={v['n_gt']}")
    t = track_metrics.to_dict()
    print(f"  Tracking IDF1          : {t['idf1']:.4f}   (poster đặt mục tiêu ≥ 0.70)")
    print(f"  Tracking MOTA          : {t['mota']:.4f}   (poster đặt mục tiêu ≥ 0.60)")
    print(f"  ID switch              : {t['id_switches']}")
    print(f"  Tốc độ xử lý           : {result.processing_fps:.1f} FPS")
    print(f"  Near-miss ghi nhận     : {conflicts.get('total', 0)}")
    print(f"\n✓ Báo cáo: {out}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="saferoad",
        description="SafeRoad AI — phát hiện near-miss & bản đồ rủi ro giao thông",
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
    r.add_argument("--ground-truth", help="Bundle .pkl để lấy homography của camera mô phỏng")
    r.add_argument("--replay-detections", action="store_true",
                   help="Dùng bbox chuẩn thay vì chạy YOLO (chỉ cho video synthetic)")
    r.add_argument("--noise-px", type=float, default=1.5)
    r.add_argument("--miss-rate", type=float, default=0.03)
    r.set_defaults(func=cmd_run)

    # evaluate
    e = sub.add_parser("evaluate", help="Chấm điểm + ablation trên tập synthetic")
    e.add_argument("--ground-truth", default="data/samples/synthetic_groundtruth.pkl")
    e.add_argument("--source", help="Video synthetic tương ứng")
    e.add_argument("--config", help="File cấu hình YAML")
    e.add_argument("--noise-px", type=float, default=1.5)
    e.add_argument("--miss-rate", type=float, default=0.03)
    e.add_argument("--out", default="data/outputs/evaluation.json")
    e.set_defaults(func=cmd_evaluate)

    # serve
    v = sub.add_parser("serve", help="Bật dashboard web")
    v.add_argument("--config", help="File cấu hình YAML")
    v.add_argument("--results", default="data/outputs/results.json")
    v.add_argument("--video", default="data/outputs/overlay.mp4")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
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
    er.add_argument("--map-conf", type=float, default=0.001,
                    help="Ngưỡng tin cậy khi chấm mAP (chuẩn COCO là 0.001). "
                         "Pipeline vẫn chạy ở ngưỡng vận hành trong config.")
    er.add_argument("--no-overlay", action="store_true")
    er.add_argument("--out", default="data/outputs/evaluation_real.json")
    er.set_defaults(func=cmd_evaluate_real)

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
