"""PET đo được (measured PET) từ mốc vào/ra vùng xung đột của từng xe.

Hai cách tính PET trong hệ thống, và vì sao phải tách
-----------------------------------------------------
* **PET ngoại suy** (``pet_pred``, :func:`saferoad.conflict.indicators.post_encroachment_time`)
  - tính *trước khi* sự việc xảy ra, từ vị trí và vận tốc hiện tại: kéo dài hai
  tia vận tốc, tìm giao điểm, đoán thời điểm mỗi xe tới đó. Dùng được cho cảnh
  báo sớm, nhưng chỉ là phép đoán: nếu một bên phanh hoặc đổi hướng, con số thật
  sẽ khác.
* **PET đo** (``pet_measured``, module này) - tính *sau khi* cả hai xe đã đi qua,
  từ chính quỹ đạo đã ghi lại. Đây mới là định nghĩa gốc của Allen và cộng sự
  (1978): khoảng thời gian từ lúc xe thứ nhất **rời** vùng xung đột tới lúc xe
  thứ hai **đi vào** vùng đó.

Trước đây hệ thống chỉ có PET ngoại suy, còn nhãn chuẩn lại đo PET theo một cách
khác (khoảng cách thời gian giữa hai tâm xe tại điểm gần nhau nhất). Hai đại
lượng khác nhau được đem ra so với nhau, nên sai số PET báo cáo (3,09 s) lớn gấp
đôi chính ngưỡng 1,5 s. Module này cho cả thuật toán online lẫn bộ sinh nhãn
dùng **cùng một định nghĩa**, và lưu lại bốn mốc thời gian để người đọc tự kiểm.

Vùng xung đột được định nghĩa thế nào
-------------------------------------
Mỗi xe được phủ bằng các hình tròn dọc trục thân (cùng mô hình với TTC). Vùng
xung đột là phần giao giữa **dải quét** của xe A và dải quét của xe B trên mặt
đất. Một mẫu thời gian ``t_i`` của xe A nằm "trong vùng" khi thân xe A lúc đó
chạm dải quét của xe B ở *bất kỳ* thời điểm nào - tức là tồn tại ``s_j`` để thân
A tại ``t_i`` chồng lên thân B tại ``s_j``. Từ đó:

* ``t_a_in``/``t_a_out`` = mẫu đầu/cuối của A nằm trong vùng;
* ``t_b_in``/``t_b_out`` tương tự cho B;
* xe vào vùng trước là xe thứ nhất; ``PET = t_vào(xe sau) - t_ra(xe trước)``.

Ba trường hợp phải xử lý riêng
------------------------------
* **Quỹ đạo song song / cùng làn** - dải quét chồng lên nhau dọc cả đoạn đường,
  "vùng xung đột" dài hàng chục mét và PET mất ý nghĩa. Góc giữa hai hướng đi tại
  vùng nhỏ hơn ``min_angle`` (hoặc lớn hơn ``180 - min_angle``, tức đối đầu trên
  cùng một vệt) thì trả về trạng thái ``parallel`` - để TTC xử lý.
* **Xe đã qua giao điểm** - không còn là vấn đề: mốc thời gian lấy từ lịch sử
  quỹ đạo, không phụ thuộc vào việc giao điểm còn nằm phía trước hay không.
* **Chiếm vùng cùng lúc** - xe sau vào vùng khi xe trước chưa ra. Nếu hai thân
  xe thật sự chạm nhau ở cùng một thời điểm thì là ``contact``; nếu không thì là
  ``simultaneous`` với PET = 0 và ghi lại thời lượng chiếm chung.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..types import VehicleClass

#: Trạng thái của phép đo PET.
PET_MEASURED = "measured"          # đo được đủ 4 mốc
PET_PARALLEL = "parallel"          # quỹ đạo song song/cùng vệt - PET không xác định
PET_NO_ZONE = "no_zone"            # hai dải quét không giao nhau
PET_INCOMPLETE = "incomplete"      # một xe chưa ra khỏi vùng / mất dấu giữa vùng
PET_SIMULTANEOUS = "simultaneous"  # cùng chiếm vùng nhưng thân xe không chạm
PET_CONTACT = "contact"            # thân xe chồng lên nhau ở cùng thời điểm


@dataclass
class Encroachment:
    """Kết quả đo PET kèm đủ mốc thời gian để tự kiểm lại."""

    status: str
    pet: float | None = None
    first: str | None = None          # "a" | "b" - xe vào vùng trước
    t_a_in: float | None = None
    t_a_out: float | None = None
    t_b_in: float | None = None
    t_b_out: float | None = None
    crossing_angle: float | None = None
    zone_xy: tuple[float, float] | None = None
    #: Thời lượng hai xe cùng nằm trong vùng (giây), > 0 khi chiếm cùng lúc.
    overlap_s: float = 0.0

    @property
    def is_measured(self) -> bool:
        return self.status in (PET_MEASURED, PET_SIMULTANEOUS, PET_CONTACT)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        if self.zone_xy is not None:
            d["zone_xy"] = (round(self.zone_xy[0], 2), round(self.zone_xy[1], 2))
        return d


def _headings(vel: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Hướng đi (rad) từ vận tốc; mẫu gần như đứng yên lấy hướng hợp lệ gần nhất."""
    speed = np.linalg.norm(vel, axis=1)
    head = np.arctan2(vel[:, 1], vel[:, 0])
    valid = speed > 0.3
    if not valid.any():
        return np.full(len(vel), fallback)
    # Điền tiến rồi điền lùi các mẫu không có hướng.
    idx = np.where(valid, np.arange(len(vel)), -1)
    np.maximum.accumulate(idx, out=idx)
    first_valid = int(np.argmax(valid))
    idx[idx < 0] = first_valid
    return head[idx]


def _circles(pos: np.ndarray, head: np.ndarray, cls: VehicleClass) -> tuple[np.ndarray, np.ndarray]:
    """Tâm ``(N, k, 2)`` và bán kính ``(k,)`` các hình tròn phủ thân xe."""
    offs = np.array([o for o, _r in cls.circles], dtype=float)
    radii = np.array([r for _o, r in cls.circles], dtype=float)
    ct, st = np.cos(head), np.sin(head)
    cx = pos[:, 0:1] + offs[None, :] * ct[:, None]
    cy = pos[:, 1:2] + offs[None, :] * st[:, None]
    return np.stack([cx, cy], axis=2), radii


def footprint_overlap_matrix(
    pos_a: np.ndarray, head_a: np.ndarray, cls_a: VehicleClass,
    pos_b: np.ndarray, head_b: np.ndarray, cls_b: VehicleClass,
    chunk: int = 256,
) -> np.ndarray:
    """``M[i, j] = True`` khi thân A ở mẫu ``i`` chồng lên thân B ở mẫu ``j``."""
    ca, ra = _circles(pos_a, head_a, cls_a)
    cb, rb = _circles(pos_b, head_b, cls_b)
    rsum = ra[:, None] + rb[None, :]                       # (ka, kb)
    out = np.zeros((len(pos_a), len(pos_b)), dtype=bool)
    # Lọc thô bằng khoảng cách tâm xe để khỏi tính chi tiết các cặp xa nhau.
    reach = (max(abs(o) for o, _ in cls_a.circles) + ra.max()
             + max(abs(o) for o, _ in cls_b.circles) + rb.max())
    for s in range(0, len(pos_a), chunk):
        e = min(s + chunk, len(pos_a))
        d_centre = np.linalg.norm(pos_a[s:e, None, :] - pos_b[None, :, :], axis=2)
        near = d_centre <= reach
        if not near.any():
            continue
        ii, jj = np.nonzero(near)
        diff = ca[s:e][ii][:, :, None, :] - cb[jj][:, None, :, :]   # (P, ka, kb, 2)
        dist = np.linalg.norm(diff, axis=3)
        hit = (dist <= rsum[None, :, :]).any(axis=(1, 2))
        out[s + ii[hit], jj[hit]] = True
    return out


def _mean_direction(head: np.ndarray) -> float:
    return math.atan2(float(np.sin(head).mean()), float(np.cos(head).mean()))


def measure_pet(
    times_a: np.ndarray, pos_a: np.ndarray, vel_a: np.ndarray, cls_a: VehicleClass,
    times_b: np.ndarray, pos_b: np.ndarray, vel_b: np.ndarray, cls_b: VehicleClass,
    min_angle: float = 20.0,
    a_continues: bool = False,
    b_continues: bool = False,
) -> Encroachment:
    """Đo PET từ hai quỹ đạo đã ghi lại.

    Tham số ``*_continues`` cho biết track còn tiếp diễn sau mẫu cuối (đang chạy
    online). Khi đó nếu mẫu cuối vẫn nằm trong vùng thì mốc ra chưa biết, kết quả
    là ``incomplete`` thay vì một mốc ra sai.
    """
    times_a = np.asarray(times_a, dtype=float)
    times_b = np.asarray(times_b, dtype=float)
    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    vel_a = np.asarray(vel_a, dtype=float)
    vel_b = np.asarray(vel_b, dtype=float)
    if len(times_a) < 2 or len(times_b) < 2:
        return Encroachment(status=PET_INCOMPLETE)

    head_a = _headings(vel_a)
    head_b = _headings(vel_b)
    m = footprint_overlap_matrix(pos_a, head_a, cls_a, pos_b, head_b, cls_b)
    in_a = m.any(axis=1)
    in_b = m.any(axis=0)
    if not in_a.any():
        return Encroachment(status=PET_NO_ZONE)

    ia = np.nonzero(in_a)[0]
    ib = np.nonzero(in_b)[0]

    # Góc giữa hai hướng đi trong vùng - phân biệt cắt nhau với song song.
    da = _mean_direction(head_a[ia])
    db = _mean_direction(head_b[ib])
    angle = abs(math.degrees(math.atan2(math.sin(da - db), math.cos(da - db))))

    ii, jj = np.nonzero(m)
    zone = 0.5 * (pos_a[ii].mean(axis=0) + pos_b[jj].mean(axis=0))
    zone_xy = (float(zone[0]), float(zone[1]))

    # Người đi bộ không có "hướng thân" đủ dài để vùng quét bị kéo dọc, nên với
    # họ ngưỡng góc vẫn áp dụng như xe - đi dọc cạnh xe thì không có PET.
    if angle < min_angle or angle > 180.0 - min_angle:
        return Encroachment(status=PET_PARALLEL, crossing_angle=angle, zone_xy=zone_xy)

    # Mốc vào/ra lấy ở điểm giữa mẫu cuối-ngoài-vùng và mẫu đầu-trong-vùng: sai
    # số lấy mẫu không còn lệch về một phía (bản rời rạc thô luôn đo PET dài
    # hơn thật tới hai bước khung hình).
    t_a_in, t_a_out = _edge(times_a, ia[0], -1), _edge(times_a, ia[-1], +1)
    t_b_in, t_b_out = _edge(times_b, ib[0], -1), _edge(times_b, ib[-1], +1)

    # Mốc vào/ra chỉ tin được khi quỹ đạo bắt đầu TRƯỚC vùng và kết thúc SAU vùng.
    a_starts_inside = ia[0] == 0
    b_starts_inside = ib[0] == 0
    a_open = ia[-1] == len(times_a) - 1 and a_continues
    b_open = ib[-1] == len(times_b) - 1 and b_continues

    first = "a" if t_a_in <= t_b_in else "b"
    if first == "a":
        lead_out, follow_in = t_a_out, t_b_in
        lead_unknown_exit = a_open
        follow_unknown_entry = b_starts_inside
    else:
        lead_out, follow_in = t_b_out, t_a_in
        lead_unknown_exit = b_open
        follow_unknown_entry = a_starts_inside

    base = dict(
        first=first, t_a_in=t_a_in, t_a_out=t_a_out, t_b_in=t_b_in, t_b_out=t_b_out,
        crossing_angle=angle, zone_xy=zone_xy,
    )
    if lead_unknown_exit or follow_unknown_entry or (a_starts_inside and b_starts_inside):
        return Encroachment(status=PET_INCOMPLETE, **base)

    gap = follow_in - lead_out
    if gap >= 0:
        return Encroachment(status=PET_MEASURED, pet=gap, **base)

    # Hai xe cùng nằm trong vùng một lúc: kiểm tra có chạm nhau thật không, tức
    # là thân chồng lên nhau tại CÙNG một thời điểm.
    overlap_s = min(t_a_out, t_b_out) - max(t_a_in, t_b_in)
    contact = _contact_at_same_time(times_a, times_b, m)
    status = PET_CONTACT if contact else PET_SIMULTANEOUS
    return Encroachment(status=status, pet=0.0, overlap_s=max(0.0, overlap_s), **base)


def _edge(times: np.ndarray, i: int, direction: int) -> float:
    j = i + direction
    if 0 <= j < len(times):
        return 0.5 * (float(times[i]) + float(times[j]))
    return float(times[i])


def _contact_at_same_time(times_a: np.ndarray, times_b: np.ndarray, m: np.ndarray) -> bool:
    """Có cặp mẫu (i, j) trùng thời điểm (lệch dưới nửa bước) mà thân chồng nhau."""
    if len(times_b) < 2:
        return False
    step = float(np.median(np.diff(times_b))) if len(times_b) > 1 else 0.034
    jj = np.clip(np.searchsorted(times_b, times_a), 0, len(times_b) - 1)
    jj_prev = np.clip(jj - 1, 0, len(times_b) - 1)
    pick = np.where(
        np.abs(times_b[jj_prev] - times_a) < np.abs(times_b[jj] - times_a), jj_prev, jj
    )
    close = np.abs(times_b[pick] - times_a) <= 0.5 * step + 1e-9
    return bool(np.any(m[np.arange(len(times_a)), pick] & close))


# --------------------------------------------------------------------------- #
# Phân biệt near-miss, chồng thân xe và va chạm thật
# --------------------------------------------------------------------------- #
CONTACT_NONE = "near_miss"               # hai thân xe không chạm nhau
CONTACT_OVERLAP = "body_overlap"         # thân chồng nhau nhưng không có dấu hiệu va chạm
CONTACT_COLLISION = "collision_suspected"  # chồng thân + cả hai/một bên dừng gấp

CONTACT_VI = {
    CONTACT_NONE: "Suýt va chạm (không chạm)",
    CONTACT_OVERLAP: "Chồng thân xe - nhiều khả năng do sai số đo/che khuất",
    CONTACT_COLLISION: "Nghi va chạm thật - cần người xác nhận",
}


def classify_contact(
    min_gap_sync: float,
    speeds_before: tuple[float, float],
    speeds_after: tuple[float, float],
    stop_ratio: float = 0.35,
    min_speed_before: float = 1.5,
) -> str:
    """Phân loại kết cục của một tình huống theo khoảng cách đồng thời và động học.

    * Khoảng cách mặt-tới-mặt **cùng thời điểm** > 0 ⇒ ``near_miss``.
    * Bằng 0 (hai thân xe chồng nhau) mà ít nhất một bên đang chạy rồi **gần như
      dừng hẳn** ngay sau đó ⇒ ``collision_suspected``. Va chạm thật luôn để lại
      dấu vết động học: xe không thể đi xuyên qua nhau rồi giữ nguyên tốc độ.
    * Bằng 0 mà cả hai vẫn giữ tốc độ ⇒ ``body_overlap``: vật lý không cho phép
      điều đó, nên đây là sai số đo (chiếu sai điểm tiếp đất khi hai xe che
      nhau, bbox gộp) hoặc lỗi của trình mô phỏng. Không được đếm là va chạm.
    """
    if min_gap_sync > 0.0:
        return CONTACT_NONE
    stopped = False
    for before, after in zip(speeds_before, speeds_after):
        if before >= min_speed_before and after <= stop_ratio * before:
            stopped = True
    return CONTACT_COLLISION if stopped else CONTACT_OVERLAP
