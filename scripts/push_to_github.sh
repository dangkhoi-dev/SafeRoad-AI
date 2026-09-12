#!/usr/bin/env bash
# Đẩy repo lên GitHub. Chạy từ thư mục gốc của repo.
#
#   bash scripts/push_to_github.sh
#
set -euo pipefail

REMOTE_URL="https://github.com/dangkhoi-dev/SafeRoad-AI.git"

if [ ! -d .git ]; then
  echo "Lỗi: không thấy thư mục .git. Hãy chạy từ thư mục gốc SafeRoad-AI." >&2
  exit 1
fi

echo "Repo   : $(pwd)"
echo "Remote : $REMOTE_URL"
echo "Commit : $(git rev-list --count HEAD) commit trên nhánh $(git branch --show-current)"
echo

if git remote | grep -qx origin; then
  git remote set-url origin "$REMOTE_URL"
  echo "→ Đã cập nhật remote 'origin'"
else
  git remote add origin "$REMOTE_URL"
  echo "→ Đã thêm remote 'origin'"
fi

echo
echo "Đang đẩy lên GitHub…"
echo "(Nếu hỏi mật khẩu, hãy nhập Personal Access Token, KHÔNG phải mật khẩu tài khoản:"
echo " github.com → Settings → Developer settings → Personal access tokens → Fine-grained"
echo " → cấp quyền Contents: Read and write cho repo SafeRoad-AI)"
echo

git push -u origin main

echo
echo "✓ Xong. Kiểm tra tại: https://github.com/dangkhoi-dev/SafeRoad-AI"
