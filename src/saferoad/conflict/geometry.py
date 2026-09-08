"""Hình học va chạm dùng mô hình đa hình tròn.

Mỗi phương tiện được phủ bằng 1-3 hình tròn đặt dọc trục thân xe (xem
:pyattr:`VehicleClass.circles`). So với một hình tròn ngoại tiếp duy nhất, cách
này bám sát hình chữ nhật thật nên không còn báo động giả giữa hai xe đi song
song ở hai làn cạnh nhau.

Các hàm ở đây là **nguyên hàm hình học thuần tuý** — không chứa quyết định nào
về "thế nào là một xung đột". Nhờ vậy cả thuật toán online lẫn bộ sinh nhãn
chuẩn (oracle) đều dùng chung được mà không làm hỏng tính độc lập của phép đánh
giá: hai bên khác nhau ở *cách quyết định* (tần số lấy mẫu, vận tốc ước lượng
hay giải tích, cực tiểu cục bộ hay toàn cục), chứ không phải ở phép đo khoảng
cách giữa hai khối hình.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import VehicleClass

#: Trả về khi hai đối tượng không hội tụ về phía nhau.
NO_COLLISION = float("inf")


def circle_centres(
    pos: tuple[float, float], heading: float, cls: VehicleClass
) -> list[tuple[float, float, float]]:
    """Toạ độ tuyệt đối các hình tròn của một xe: ``[(x, y, r), ...]``."""
    ct, st = math.cos(heading), math.sin(heading)
    return [
        (pos[0] + offset * ct, pos[1] + offset * st, r)
        for offset, r in cls.circles
    ]


def _pair_ttc(
    px: float, py: float, vx: float, vy: float, radius: float, horizon: float
) -> float:
    """TTC giữa hai hình tròn, từ vị trí và vận tốc **tương đối**.

    Giải phương trình bậc hai ``‖Δp + tΔv‖ = R`` và lấy nghiệm dương nhỏ nhất.
    """
    dist_sq = px * px + py * py
    if dist_sq <= radius * radius:
        return 0.0  # đã chồng lấn

    a = vx * vx + vy * vy
    if a < 1e-9:
        return NO_COLLISION
    b = 2.0 * (px * vx + py * vy)
    if b >= 0:
        return NO_COLLISION  # đang tách xa nhau

    c = dist_sq - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return NO_COLLISION  # lướt qua nhau, không chạm

    sqrt_disc = math.sqrt(disc)
    for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
        if 0.0 <= t <= horizon:
            return t
    return NO_COLLISION


def vehicle_ttc(
    pos_a: tuple[float, float], vel_a: tuple[float, float], cls_a: VehicleClass,
    pos_b: tuple[float, float], vel_b: tuple[float, float], cls_b: VehicleClass,
    horizon: float = 10.0,
) -> float:
    """TTC giữa hai phương tiện — nhỏ nhất trên mọi cặp hình tròn.

    Hướng của mỗi xe suy từ vector vận tốc. Xe gần như đứng yên không có hướng
    xác định nên lấy hướng mặc định 0 — sai số khi đó không quan trọng vì xe
    đứng yên không sinh xung đột do chuyển động của chính nó.
    """
    head_a = math.atan2(vel_a[1], vel_a[0]) if math.hypot(*vel_a) > 1e-6 else 0.0
    head_b = math.atan2(vel_b[1], vel_b[0]) if math.hypot(*vel_b) > 1e-6 else 0.0

    circles_a = circle_centres(pos_a, head_a, cls_a)
    circles_b = circle_centres(pos_b, head_b, cls_b)

    # Vận tốc tương đối chung cho mọi cặp hình tròn (bỏ qua chuyển động quay —
    # không đáng kể trong khoảng dự báo 1-3 giây).
    vx = vel_b[0] - vel_a[0]
    vy = vel_b[1] - vel_a[1]

    best = NO_COLLISION
    for ax, ay, ar in circles_a:
        for bx, by, br in circles_b:
            t = _pair_ttc(bx - ax, by - ay, vx, vy, ar + br, horizon)
            if t < best:
                best = t
                if best == 0.0:
                    return 0.0
    return best


def vehicle_gap(
    pos_a: tuple[float, float], head_a: float, cls_a: VehicleClass,
    pos_b: tuple[float, float], head_b: float, cls_b: VehicleClass,
) -> float:
    """Khoảng cách mặt-tới-mặt (m) giữa hai phương tiện; 0 nếu chồng lấn."""
    circles_a = circle_centres(pos_a, head_a, cls_a)
    circles_b = circle_centres(pos_b, head_b, cls_b)

    best = float("inf")
    for ax, ay, ar in circles_a:
        for bx, by, br in circles_b:
            d = math.hypot(bx - ax, by - ay) - ar - br
            if d < best:
                best = d
    return max(0.0, best)


# --------------------------------------------------------------------------- #
# Phiên bản vector hoá — dùng cho oracle, chạy trên cả chuỗi thời gian một lần
# --------------------------------------------------------------------------- #
def series_ttc_and_gap(
    pos_a: np.ndarray, vel_a: np.ndarray, cls_a: VehicleClass,
    pos_b: np.ndarray, vel_b: np.ndarray, cls_b: VehicleClass,
    horizon: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """TTC và khoảng cách mặt-tới-mặt cho cả một chuỗi thời gian.

    Tham số là các mảng ``(N, 2)`` vị trí và vận tốc đã đồng bộ theo thời gian.
    Trả về ``(ttc, gap)``, mỗi mảng có ``N`` phần tử; ``ttc`` bằng ``inf`` tại
    các thời điểm không hội tụ.
    """
    n = len(pos_a)
    speed_a = np.linalg.norm(vel_a, axis=1)
    speed_b = np.linalg.norm(vel_b, axis=1)
    head_a = np.where(speed_a > 1e-6, np.arctan2(vel_a[:, 1], vel_a[:, 0]), 0.0)
    head_b = np.where(speed_b > 1e-6, np.arctan2(vel_b[:, 1], vel_b[:, 0]), 0.0)

    dv = vel_b - vel_a
    aa = np.einsum("ij,ij->i", dv, dv)

    ttc = np.full(n, np.inf)
    gap = np.full(n, np.inf)

    for off_a, r_a in cls_a.circles:
        ca = pos_a + np.stack([off_a * np.cos(head_a), off_a * np.sin(head_a)], axis=1)
        for off_b, r_b in cls_b.circles:
            cb = pos_b + np.stack(
                [off_b * np.cos(head_b), off_b * np.sin(head_b)], axis=1
            )
            d = cb - ca
            dist = np.linalg.norm(d, axis=1)
            radius = r_a + r_b

            np.minimum(gap, dist - radius, out=gap)

            bb = 2.0 * np.einsum("ij,ij->i", d, dv)
            cc = dist**2 - radius**2
            disc = bb**2 - 4.0 * aa * cc

            with np.errstate(invalid="ignore", divide="ignore"):
                root = (-bb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * aa)

            valid = (aa > 1e-9) & (bb < 0) & (disc >= 0) & (root >= 0) & (root <= horizon)
            # Chồng lấn ngay tại thời điểm hiện tại ⇒ TTC = 0.
            overlap = dist <= radius
            candidate = np.where(overlap, 0.0, np.where(valid, root, np.inf))
            np.minimum(ttc, candidate, out=ttc)

    return ttc, np.maximum(gap, 0.0)
