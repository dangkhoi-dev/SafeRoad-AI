"""Bản demo chạy thật trên Modal (https://modal.com) - thay cho Hugging Face Spaces.

Từ tháng 7/2026, Hugging Face chỉ cho chạy Space Docker/Gradio với tài khoản PRO.
Modal có 30 USD credit mỗi tháng và chỉ tính tiền lúc có người dùng (tự tắt khi
không ai truy cập), nên một bản demo cho hội đồng chấm gần như không tốn gì.

    pip install modal
    modal setup                              # đăng nhập một lần (mở trình duyệt)
    modal deploy scripts/deploy_modal.py     # in ra link https://...modal.run

Máy chủ trên Modal giống hệt bản Docker: FastAPI + dashboard, tab "Chạy thử video"
chạy YOLO11n → ByteTrack → quỹ đạo mặt đất → TTC/PET ngay khi người xem bấm.
Chỉ chép những gì máy chủ cần: mã nguồn, cấu hình, hai model, kết quả JSON/CSV,
ảnh/video minh hoạ và (nếu có) video mẫu data/samples/demo_sample.mp4.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    # PyTorch bản CPU trước, để pip không kéo bản CUDA vài GB.
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(ROOT / "pyproject.toml", f"{APP_DIR}/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", f"{APP_DIR}/README.md", copy=True)
    .add_local_dir(ROOT / "src", f"{APP_DIR}/src", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc", "*.egg-info"])
    .run_commands(f"pip install -e {APP_DIR}")
    .add_local_file(ROOT / "scripts" / "download_assets.py",
                    f"{APP_DIR}/scripts/download_assets.py", copy=True)
    .run_commands(f"cd {APP_DIR} && mkdir -p models && python scripts/download_assets.py")
    .env({
        "OMP_NUM_THREADS": "4",
        "SAFEROAD_MAX_SECONDS": "30",
        "SAFEROAD_SAMPLE_PRESET": "street",
        "SAFEROAD_JOBS_DIR": "/tmp/saferoad_jobs",
    })
    # Các bước dưới gắn file lúc container khởi động - sửa cấu hình/kết quả rồi
    # deploy lại không phải dựng lại image.
    .add_local_dir(ROOT / "configs", f"{APP_DIR}/configs")
    .add_local_file(ROOT / "models" / "behavior_clf.joblib", f"{APP_DIR}/models/behavior_clf.joblib")
    .add_local_file(ROOT / "models" / "saferoad_street_yolo11n.pt",
                    f"{APP_DIR}/models/saferoad_street_yolo11n.pt")
    .add_local_dir(ROOT / "data" / "outputs", f"{APP_DIR}/data/outputs",
                   ignore=~modal.FilePatternMatcher("*.json", "*.csv"))
    .add_local_dir(ROOT / "data" / "samples", f"{APP_DIR}/data/samples",
                   ignore=~modal.FilePatternMatcher("demo_sample.mp4", ".gitkeep"))
    .add_local_dir(ROOT / "docs" / "assets", f"{APP_DIR}/docs/assets")
)

app = modal.App("saferoad-ai", image=image)


@app.function(
    cpu=2.0,                 # 2 lõi vật lý
    memory=4096,
    max_containers=1,        # trạng thái các lượt chạy nằm trong bộ nhớ một container
    scaledown_window=900,    # giữ ấm 15 phút sau lượt truy cập cuối
    timeout=600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="saferoad-ai")
def web():
    import os

    os.chdir(APP_DIR)
    from saferoad.config import Config
    from saferoad.dashboard.server import create_app

    cfg = Config()
    cfg.output_dir = "data/outputs"
    sample = "data/samples/demo_sample.mp4"
    return create_app(
        cfg,
        results_path="data/outputs/results.json",
        video_path="docs/assets/demo_overlay.mp4",
        demo=True,
        sample_video=sample if os.path.exists(sample) else None,
    )
