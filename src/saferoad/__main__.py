"""Cho phép chạy bằng ``python -m saferoad``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
