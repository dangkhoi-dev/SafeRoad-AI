"""Mô hình camera pinhole cho cảnh mô phỏng.

Vì sao cần camera 3D thật thay vì chỉ một homography
----------------------------------------------------
Homography chỉ ánh xạ được **mặt phẳng mặt đất**. Nếu vẽ mỗi phương tiện bằng
hình chiếu bóng của nó xuống đất, bounding box thu được sẽ nhỏ hơn rất nhiều so
với thực tế — một người đi bộ chiếm vỏn vẹn 0.5 × 0.5 m mặt đất và cho ra bbox
~37 px², trong khi camera thật nhìn thấy một người **cao 1.7 m** với bbox lớn
gấp gần mười lần. Hệ quả: mọi người đi bộ đều nằm dưới ngưỡng phát hiện, và toàn
bộ nhóm xung đột với người đi bộ trở nên không thể phát hiện — không phải vì
thuật toán kém mà vì cảnh dựng sai.

Module này dựng một camera pinhole đầy đủ (nội tham số + ngoại tham số), nên:

* phương tiện được vẽ như khối hộp 3D, bbox phản ánh đúng chiều cao thật;
* homography mặt đất được **suy ra** từ chính ma trận camera, nên phép quy đổi
  mà pipeline dùng khớp chính xác với cảnh đã dựng — không có sai lệch ngầm nào
  giữa bộ sinh dữ liệu và bộ xử lý.

Quy ước toạ độ thế giới: X sang phải, Y hướng vào trong cảnh, Z hướng lên.
"""

from __future__ import annotations

import math

import numpy as np


class PinholeCamera:
    """Camera pinhole đặt trên cao, nhìn nghiêng xuống giao lộ.

    Tham số:
        width, height: kích thước ảnh (px).
        position: vị trí camera trong thế giới (X, Y, Z), mét.
        look_at: điểm camera hướng tới trên mặt đất (X, Y, Z).
        fov_deg: góc nhìn ngang (độ).
    """

    def __init__(
        self,
        width: int,
        height: int,
        position: tuple[float, float, float],
        look_at: tuple[float, float, float],
        fov_deg: float = 62.0,
    ):
        self.width = width
        self.height = height
        self.position = np.asarray(position, dtype=float)
        self.look_at = np.asarray(look_at, dtype=float)

        # --- Nội tham số ------------------------------------------------ #
        f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.K = np.array(
            [[f, 0.0, width / 2.0],
             [0.0, f, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )

        # --- Ngoại tham số (look-at) ------------------------------------ #
        # Trục quang của camera (OpenCV: +Z_cam hướng ra phía trước).
        forward = self.look_at - self.position
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        norm = np.linalg.norm(right)
        if norm < 1e-9:                      # camera nhìn thẳng đứng xuống
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm
        down = np.cross(forward, right)      # OpenCV: +Y_cam hướng xuống

        # Hàng của R là các trục camera biểu diễn trong hệ thế giới.
        self.R = np.stack([right, down, forward], axis=0)
        self.t = -self.R @ self.position

        # --- Homography mặt đất (Z = 0) --------------------------------- #
        # [u v 1]ᵀ ~ K [r1 r2 t] [X Y 1]ᵀ
        rt = np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H_img_from_world = self.K @ rt
        self.H_world_from_img = np.linalg.inv(self.H_img_from_world)

    # ------------------------------------------------------------------ #
    def project(self, points: np.ndarray) -> np.ndarray:
        """Chiếu các điểm 3D ``(N, 3)`` trong thế giới xuống ảnh ``(N, 2)``.

        Điểm nằm sau camera (z ≤ 0) được đẩy ra xa biên để không vẽ nhầm sang
        phía đối diện của khung hình.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        cam = (self.R @ pts.T).T + self.t
        z = cam[:, 2]
        safe_z = np.where(z > 1e-6, z, 1e-6)

        u = self.K[0, 0] * cam[:, 0] / safe_z + self.K[0, 2]
        v = self.K[1, 1] * cam[:, 1] / safe_z + self.K[1, 2]
        out = np.stack([u, v], axis=1)
        out[z <= 1e-6] = np.array([-1e5, -1e5])
        return out

    def project_ground(self, points_xy: np.ndarray) -> np.ndarray:
        """Chiếu các điểm ``(N, 2)`` trên mặt đất (Z = 0) xuống ảnh."""
        pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
        return self.project(np.column_stack([pts, np.zeros(len(pts))]))

    def image_to_ground(self, points_uv: np.ndarray) -> np.ndarray:
        """Chiếu ngược điểm ảnh ``(N, 2)`` về mặt đất ``(N, 2)``."""
        pts = np.atleast_2d(np.asarray(points_uv, dtype=float))
        homo = np.column_stack([pts, np.ones(len(pts))])
        world = (self.H_world_from_img @ homo.T).T
        return world[:, :2] / world[:, 2:3]

    # ------------------------------------------------------------------ #
    def ground_homography_points(
        self, world_points: list[tuple[float, float]]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Sinh cặp điểm ảnh/thế giới để nạp vào :class:`HomographyConfig`.

        Nhờ đó pipeline dùng đúng phép quy đổi mà camera đã dùng khi dựng cảnh:
        mọi sai số TTC đo được là sai số **thuật toán**, không phải sai lệch do
        hai bên hiểu hình học khác nhau.
        """
        world = np.asarray(world_points, dtype=float)
        image = self.project_ground(world)
        return image.tolist(), world.tolist()

    def cuboid_corners(
        self,
        centre: tuple[float, float],
        heading: float,
        length: float,
        width: float,
        height: float,
    ) -> np.ndarray:
        """8 đỉnh của khối hộp bao quanh phương tiện, trong toạ độ thế giới."""
        hl, hw = length / 2.0, width / 2.0
        ct, st = math.cos(heading), math.sin(heading)

        corners = []
        for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
            x = centre[0] + dx * ct - dy * st
            y = centre[1] + dx * st + dy * ct
            corners.append((x, y, 0.0))       # đáy
            corners.append((x, y, height))    # nóc
        return np.asarray(corners, dtype=float)


#: Vùng mặt đất mà camera mặc định bao phủ đủ tốt để phát hiện đáng tin cậy,
#: dạng ``(x_min, y_min, x_max, y_max)`` tính bằng mét. Xung đột xảy ra ngoài
#: vùng này không được đưa vào tập chấm điểm — không thể bắt hệ thống chịu
#: trách nhiệm cho thứ nằm ngoài tầm nhìn của cảm biến.
#:
#: Bề rộng bị giới hạn bởi **hàng gần camera nhất**: tầm nhìn là một hình nón,
#: nên ở y = 18 m khung hình chỉ phủ được |x| ≤ 15 m, trong khi ở y = 48 m nó
#: phủ tới |x| ≤ 29 m. Lấy hình chữ nhật lớn nhất nằm trọn trong hình nón đó.
#: Vùng này vẫn bao trọn mặt đường (rộng ±7.5 m) cùng biên dự phòng rộng rãi.
COVERAGE = (-14.5, 18.0, 14.5, 48.0)


def build_default_camera(
    width: int = 1280,
    height: int = 720,
    centre: tuple[float, float] = (0.0, 30.0),
    camera_height: float = 18.0,
    setback: float = 35.0,
    fov_deg: float = 56.0,
) -> PinholeCamera:
    """Camera giám sát điển hình, nhìn chếch xuống tâm giao lộ.

    Tham số mặc định được chọn để khung hình có bố cục giống camera giao thông
    thật: mép dưới khung ứng với khoảng 17 m trước tâm giao lộ, toàn bộ bề rộng
    ±19 m nằm trong khung, và mọi phương tiện trong vùng giao lộ — kể cả người
    đi bộ — đều đủ lớn để phát hiện.

    Đặt camera quá gần sẽ khiến vài chiếc xe ở tiền cảnh chiếm hết khung hình
    còn giao lộ bị đẩy lên sát đường chân trời; lùi xa và nâng cao khắc phục
    được điều đó.
    """
    position = (centre[0], centre[1] - setback, camera_height)
    return PinholeCamera(
        width=width, height=height,
        position=position, look_at=(centre[0], centre[1] + 8.0, 0.0),
        fov_deg=fov_deg,
    )
