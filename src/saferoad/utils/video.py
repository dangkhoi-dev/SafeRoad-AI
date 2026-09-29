"""Mở VideoWriter với codec mà trình duyệt phát được.

OpenCV mặc định hay được gọi với fourcc ``mp4v`` (MPEG-4 Part 2) vì nó luôn mở
được. Nhưng Chrome, Edge và Firefox đều không giải mã được MPEG-4 Part 2, nên
video overlay ghi theo cách đó hiện ra ô đen trong tab Camera của dashboard.
Hàm này thử lần lượt các codec trình duyệt phát được (H.264) trước, rồi mới lùi
về ``mp4v`` kèm cảnh báo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def open_video_writer(path: str | Path, fps: float, size: tuple[int, int]
                      ) -> tuple[cv2.VideoWriter, str]:
    """Trả về (writer, fourcc đã dùng). Ném lỗi nếu không codec nào mở được."""
    path = str(path)
    attempts: list[tuple[str, int]] = []
    if path.lower().endswith(".mp4"):
        attempts.append(("avc1", cv2.CAP_FFMPEG))       # H.264 qua FFmpeg/openh264
        if os.name == "nt":
            attempts.append(("H264", cv2.CAP_MSMF))     # H.264 qua Media Foundation
    if path.lower().endswith(".webm"):
        attempts.append(("VP90", cv2.CAP_FFMPEG))
        attempts.append(("VP80", cv2.CAP_FFMPEG))
    attempts.append(("mp4v", cv2.CAP_FFMPEG))           # luôn mở được

    for fourcc, backend in attempts:
        try:
            w = cv2.VideoWriter(path, backend, cv2.VideoWriter_fourcc(*fourcc),
                                float(fps), size)
        except cv2.error:
            continue
        if w.isOpened():
            if fourcc == "mp4v":
                log.warning(
                    "Video %s ghi bằng codec mp4v - trình duyệt có thể không phát "
                    "được trong tab Camera. Mở bằng VLC, hoặc chuyển mã sang H.264.",
                    path,
                )
            else:
                log.info("Ghi video %s bằng codec %s", path, fourcc)
            return w, fourcc
        w.release()
    raise RuntimeError(f"Không mở được VideoWriter cho {path}")
