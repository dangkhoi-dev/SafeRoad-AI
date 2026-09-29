"""Đọc/ghi bundle nhãn chuẩn (pickle) sao cho mang được giữa Windows và Linux.

Bundle chứa đối tượng ``pathlib.Path``. Pickle lưu chúng dưới đúng lớp cụ thể
của hệ điều hành đã tạo ra chúng: tạo trên Windows thì là ``WindowsPath``, và
Linux từ chối dựng lại lớp đó (``cannot instantiate 'WindowsPath' on your
system``). Nghĩa là nhãn chuẩn chuẩn bị trên máy Windows của nhóm không mở
được trên Colab hay trên máy chạy CI - đúng hai nơi cần nó nhất.

Cách xử lý: khi đọc, mọi lớp đường dẫn cụ thể đều được dựng lại thành ``Path``
của hệ điều hành đang chạy từ các thành phần của nó.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_PATH_CLASSES = {"Path", "WindowsPath", "PosixPath", "PureWindowsPath", "PurePosixPath"}
_PATH_MODULES = {"pathlib", "pathlib._local"}   # Python 3.13 chuyển lớp sang _local


class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module in _PATH_MODULES and name in _PATH_CLASSES:
            return Path
        return super().find_class(module, name)


def load_bundle(path: str | Path) -> Any:
    """Đọc bundle, chấp nhận cả bundle tạo trên hệ điều hành khác."""
    with open(path, "rb") as fh:
        return _PortableUnpickler(fh).load()


def save_bundle(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
