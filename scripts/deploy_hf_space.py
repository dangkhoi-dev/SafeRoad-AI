"""Đẩy bản demo chạy thật lên Hugging Face Spaces (SDK Docker, CPU miễn phí).

    python scripts/deploy_hf_space.py --space <tên_tài_khoản_hf>/saferoad-ai

Lần đầu: tạo Space trên https://huggingface.co/new-space (SDK: Docker, Blank,
CPU basic, Public), tạo token quyền Write ở https://huggingface.co/settings/tokens.
Khi git hỏi mật khẩu thì dán token (hoặc đặt biến môi trường HF_TOKEN trước).

Script chỉ chép những gì máy chủ demo cần: mã nguồn, cấu hình, model hành vi,
kết quả JSON, video minh hoạ và (nếu có) video mẫu ``data/samples/demo_sample.mp4``.
Dữ liệu thô và trọng số YOLO không được đẩy lên - Docker tự tải YOLO khi dựng.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FRONT_MATTER = """---
title: SafeRoad AI
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Phát hiện near-miss từ video giao thông - chạy pipeline thật
---

# SafeRoad AI - bản demo chạy thật

Tải một đoạn video giao thông (≤ 30 giây được xử lý) ở tab **Chạy thử video**:
máy chủ chạy YOLO11n → ByteTrack → quỹ đạo mặt đất → TTC/PET → Risk Score và
hiển thị kết quả ngay trên dashboard. Mã nguồn: https://github.com/dangkhoi-dev/SafeRoad-AI
"""

INCLUDE = [
    "Dockerfile", ".dockerignore", "pyproject.toml", "src", "configs",
    "scripts/download_assets.py", "models/behavior_clf.joblib", "models/.gitkeep",
    "models/saferoad_street_yolo11n.pt",
    "docs/assets",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="vd. dangkhoi-dev/saferoad-ai")
    ap.add_argument("--workdir", default=str(ROOT / "build" / "hf_space"))
    args = ap.parse_args()

    work = Path(args.workdir)
    token = os.environ.get("HF_TOKEN", "")
    user = args.space.split("/")[0]
    url = (f"https://{user}:{token}@huggingface.co/spaces/{args.space}" if token
           else f"https://huggingface.co/spaces/{args.space}")
    if not (work / ".git").exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(work)], ROOT)

    # Xoá nội dung cũ (trừ .git) rồi chép bản mới.
    for p in work.iterdir():
        if p.name == ".git":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    out = work / "data" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for js in (ROOT / "data" / "outputs").glob("*.json"):
        shutil.copy2(js, out / js.name)
    for csv in (ROOT / "data" / "outputs").glob("review_candidates_*.csv"):
        shutil.copy2(csv, out / csv.name)
    samples = work / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (samples / ".gitkeep").write_text("", encoding="utf-8")
    sample = ROOT / "data" / "samples" / "demo_sample.mp4"
    if sample.exists():
        shutil.copy2(sample, samples / sample.name)
    (work / "README.md").write_text(FRONT_MATTER, encoding="utf-8")
    (work / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
                                         "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                         "*.pt filter=lfs diff=lfs merge=lfs -text\n",
                                         encoding="utf-8")

    run(["git", "lfs", "install"], work)
    run(["git", "add", "-A"], work)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=work,
                            capture_output=True, text=True).stdout
    if not status.strip():
        print("Không có thay đổi để đẩy.")
        return 0
    run(["git", "commit", "-m", "Cập nhật bản demo SafeRoad AI"], work)
    run(["git", "push"], work)
    print(f"\n✓ Đã đẩy. Space dựng lại trong vài phút: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
