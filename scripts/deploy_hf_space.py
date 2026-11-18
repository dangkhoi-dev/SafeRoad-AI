"""Đẩy bản demo chạy thật lên Hugging Face Spaces (SDK Docker, CPU miễn phí).

    python scripts/deploy_hf_space.py --space <tên_tài_khoản_hf>/saferoad-ai

Cần một token quyền Write (https://huggingface.co/settings/tokens) trong biến
môi trường HF_TOKEN. Nếu Space chưa có, script tự tạo (SDK Docker, CPU basic,
Public) rồi đẩy mã lên; Space tự dựng image trong vài phút.

Script chỉ chép những gì máy chủ demo cần: mã nguồn, cấu hình, model hành vi,
kết quả JSON, video minh hoạ và (nếu có) video mẫu ``data/samples/demo_sample.mp4``.
Dữ liệu thô và trọng số YOLO không được đẩy lên - Docker tự tải YOLO khi dựng.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
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
    # Không in token nằm trong URL.
    shown = [c.split("@")[-1] if c.startswith("https://") and "@" in c else c for c in cmd]
    print("  $", " ".join(shown))
    subprocess.run(cmd, cwd=cwd, check=True)


def _api(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"https://huggingface.co/api/{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def ensure_space(space: str, token: str) -> None:
    """Tạo Space (Docker, Public) nếu chưa có."""
    owner, name = space.split("/", 1)
    code, me = _api("GET", "whoami-v2", token)
    if code != 200:
        sys.exit(f"Token không hợp lệ (HTTP {code}). Tạo token quyền Write ở "
                 "https://huggingface.co/settings/tokens")
    body = {"type": "space", "name": name, "sdk": "docker", "private": False}
    if owner != me.get("name"):
        body["organization"] = owner
    code, res = _api("POST", "repos/create", token, body)
    if code in (200, 201):
        print(f"  ✓ Đã tạo Space {space}")
    elif code == 409:
        print(f"  ✓ Space {space} đã có")
    else:
        sys.exit(f"Không tạo được Space (HTTP {code}): {res.get('error', res)}")


def _git_identity() -> tuple[str, str]:
    def get(key: str) -> str:
        return subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return get("user.name") or "SafeRoad AI Team", get("user.email")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="vd. dangkhoi-dev/saferoad-ai")
    ap.add_argument("--workdir", default=str(ROOT / "build" / "hf_space"))
    args = ap.parse_args()

    work = Path(args.workdir)
    token = os.environ.get("HF_TOKEN", "")
    if "/" not in args.space:
        sys.exit("--space phải có dạng <tài_khoản>/<tên_space>")
    if token:
        ensure_space(args.space, token)
    else:
        print("  (chưa đặt HF_TOKEN - Space phải có sẵn, git sẽ hỏi mật khẩu: dán token)")
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
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.egg-info"))
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
    # Hugging Face từ chối file nhị phân không qua LFS, kể cả ảnh nhỏ.
    (work / ".gitattributes").write_text(
        "".join(f"*.{ext} filter=lfs diff=lfs merge=lfs -text\n"
                for ext in ("mp4", "joblib", "pt", "jpg", "jpeg", "png")),
        encoding="utf-8")
    name, email = _git_identity()
    run(["git", "config", "user.name", name], work)
    if email:
        run(["git", "config", "user.email", email], work)

    run(["git", "lfs", "install"], work)
    run(["git", "add", "-A"], work)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=work,
                            capture_output=True, text=True).stdout
    if not status.strip():
        print("Không có thay đổi để đẩy.")
        return 0
    run(["git", "commit", "-m", "Cập nhật bản demo SafeRoad AI"], work)
    run(["git", "push"], work)
    sub = args.space.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    print(f"\n✓ Đã đẩy. Space dựng lại trong 5-10 phút:\n"
          f"  Trang Space : https://huggingface.co/spaces/{args.space}\n"
          f"  Ứng dụng    : https://{sub}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
