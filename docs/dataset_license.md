# Bảng kê nguồn dữ liệu và điều kiện bản quyền

**Dự án:** SafeRoad AI · Cuộc thi Sáng tạo trẻ Quốc gia về AI 2026 (Bảng C)
**Tương ứng công việc:** mục 3.0 trong kế hoạch — *"Khảo sát public traffic dataset; lập danh mục kèm giấy phép"*

---

## 1. Tổng quan các nguồn dữ liệu đã sử dụng

| # | Nguồn | Loại | Dung lượng | Giấy phép | Vai trò trong dự án |
|---|---|---|---|---|---|
| 1 | **Multi-view Traffic Intersection Dataset (MVTI)** | 2.441 ảnh giao lộ 1024×640 + annotation COCO có `object_id` | ~101 MB | Theo điều khoản công bố trên Kaggle của tác giả — dùng cho nghiên cứu/học thuật | **Dữ liệu thật chính.** Đo mAP detection, IDF1/MOTA tracking, FPS thực tế |
| 2 | **UCSD Highway Traffic Database** | 254 clip .avi 320×240, 10 FPS, phân loại mật độ (light/medium/heavy) | ~32 MB | Học thuật, yêu cầu trích dẫn Chan & Vasconcelos (2005) | Kiểm thử độ bền detector ở độ phân giải thấp; tham chiếu phân loại mật độ |
| 3 | **Trình mô phỏng giao lộ SafeRoad** (tự viết) | Video 1280×720 sinh theo tham số + ground truth giải tích | sinh tại chỗ | **MIT** — mã nguồn của nhóm | **Bộ đo chính.** Nguồn *duy nhất* có nhãn near-miss chuẩn ⇒ đo Precision/Recall/TTC-MAE |
| 4 | **Video tự quay** (nhóm thực hiện) | Video giao lộ tại TP.HCM | theo thực tế | Do nhóm sở hữu; đã ẩn danh | Trình diễn trên bối cảnh giao thông Việt Nam |
| 5 | **Trọng số YOLO11n (COCO)** | Model pretrained | 5.6 MB | AGPL-3.0 (Ultralytics) | Detector nền; xem cảnh báo giấy phép ở §4 |

---

## 2. Chi tiết từng nguồn

### 2.1 Multi-view Traffic Intersection Dataset

* **Tác giả:** Andreas Møgelmose (Aalborg University)
* **Nguồn:** Kaggle — `andreasmoegelmose/multiview-traffic-intersection-dataset`
* **Nội dung:** một giao lộ có tín hiệu, quay đồng thời từ camera hạ tầng cố định
  và camera drone. Annotation dạng COCO gồm bounding box, segmentation và
  **`object_id` bền vững qua các khung hình**.
* **Vì sao chọn:** đây là điểm mấu chốt — `object_id` cho phép **đo tracking trên
  dữ liệu thật** (IDF1, MOTA, ID switch). Hầu hết dataset giao thông công khai chỉ
  có nhãn bbox rời rạc từng ảnh, không đo được chất lượng gán ID.
* **Thống kê thực tế đã nạp:** 2.441 ảnh · 14.488 bounding box · 19 quỹ đạo chuẩn ·
  độ dài quỹ đạo trung vị 646 khung hình.
* **Phân bố lớp:** `car` 7.675 · `lorry/truck/van` 3.948 · `bicycle` 1.503 · `bus` 1.362.
* **⚠ Giới hạn quan trọng:** **không có** lớp *người đi bộ* và *xe máy* — hai nhóm
  quan trọng bậc nhất với giao thông Việt Nam. Vì vậy hai lớp này chỉ được đánh
  giá trên tập mô phỏng, và điều đó được nêu rõ trong mọi bảng kết quả.

### 2.2 UCSD Highway Traffic Database

* **Tác giả:** Antoni B. Chan, Nuno Vasconcelos — Statistical Visual Computing Lab,
  University of California, San Diego
* **Nội dung:** video cao tốc I-5 (Seattle, WA) quay trong hai ngày 05–06/08/2004
  từ camera cố định, gán nhãn thủ công 3 mức mật độ. Video do Washington State
  Department of Transportation cung cấp.
* **Yêu cầu trích dẫn (bắt buộc theo README gốc):**
  > A. B. Chan and N. Vasconcelos, "Probabilistic Kernels for the Classification of
  > Auto-Regressive Visual Processes", *IEEE CVPR*, San Diego, 2005.
* **Lưu ý kỹ thuật từ README gốc:** khung hình đầu tiên của mỗi clip bị nhiễu tín
  hiệu — **phải bắt đầu xử lý từ khung thứ 2**.
* **Vai trò:** đây là dữ liệu **cao tốc một chiều**, không có giao cắt, nên **không
  dùng để đánh giá phát hiện xung đột**. Chỉ dùng kiểm thử độ bền của detector ở
  độ phân giải rất thấp (320×240).

### 2.3 Trình mô phỏng SafeRoad (tự phát triển)

* **Vì sao phải tự viết:** không một dataset giao thông công khai nào có **nhãn
  near-miss**. Chúng đều chỉ có bounding box. Mà Precision/Recall của phát hiện
  xung đột — cam kết trung tâm của đề tài — thì bắt buộc phải có nhãn đó.
* **Cách vận hành:** vi mô phỏng với đèn tín hiệu hai pha, mô hình car-following
  IDM, luật nhường đường của xe rẽ và phanh tránh khẩn cấp. Near-miss phát sinh
  từ đúng nguyên nhân ngoài đời (vượt đèn đỏ, rẽ cắt dòng, người đi bộ băng qua),
  **không có tình huống nào được dàn dựng thủ công**.
* **Ground truth:** tính từ quỹ đạo giải tích ở 60 Hz với vận tốc chính xác, lấy
  cực tiểu toàn cục — một quy trình **khác** với thuật toán online (30 Hz, vận tốc
  ước lượng từ bbox nhiễu, quyết định theo từng khung hình). Nhờ đó phép đo phản
  ánh sai số thật của hệ thống chứ không phải thuật toán tự chấm điểm cho chính nó.
* **Tái lập:** cùng `seed` cho ra kịch bản y hệt. Bộ dùng trong báo cáo:
  `seed=42`, 180 giây, 160 phương tiện, 74 near-miss chuẩn.

### 2.4 Video tự quay

* Thu tại giao lộ ở TP.HCM, camera đặt cao, giờ cao điểm.
* **Bắt buộc ẩn danh trước khi đưa vào hồ sơ** (Điều 5 Thể lệ): chạy
  `saferoad anonymize` để làm mờ khuôn mặt và biển số.
* Không quay khu vực riêng tư; chỉ quay không gian công cộng từ vị trí công cộng.

---

## 3. Tuân thủ quyền riêng tư

| Yêu cầu | Cách thực hiện |
|---|---|
| Làm mờ khuôn mặt | `saferoad/privacy/anonymize.py` — vùng 28% phía trên bbox người đi bộ, và 30% phía trên bbox xe máy/xe đạp (người ngồi trên xe) |
| Làm mờ biển số | Dải 55–95% chiều cao bbox phương tiện, thu hẹp 22% mỗi bên |
| Thời điểm áp dụng | **Ngay sau bước detection**, trước mọi thao tác ghi ra đĩa — mọi thứ lưu từ đó trở đi đều đã ẩn danh |
| Không lưu ảnh khuôn mặt | Database chỉ lưu ID số, lớp đối tượng, toạ độ và chỉ số — **không lưu ảnh cắt** |

**Ghi chú trung thực:** phương pháp ẩn danh hiện tại suy vùng nhạy cảm từ bbox của
detector chính, không dùng detector khuôn mặt chuyên dụng. Cách này thiên về **an
toàn** (làm mờ rộng hơn cần thiết) nhưng sẽ bỏ sót người đi bộ mà detector không
phát hiện được. Với dữ liệu công bố rộng rãi, nên chạy thêm một detector khuôn mặt
chuyên dụng.

---

## 4. Giấy phép phần mềm — điểm cần lưu ý

| Thành phần | Giấy phép | Ảnh hưởng |
|---|---|---|
| Mã nguồn SafeRoad AI | MIT | Tự do sử dụng |
| **Ultralytics YOLO11** | **AGPL-3.0** | ⚠ **Có tính lây lan.** Dùng cho nghiên cứu/dự thi thì hoàn toàn hợp lệ. Nếu **thương mại hoá**, phải mua giấy phép doanh nghiệp của Ultralytics **hoặc** thay bằng detector giấy phép dễ chịu hơn |
| PyTorch | BSD-3-Clause | Tự do |
| OpenCV | Apache-2.0 | Tự do |
| FastAPI, Uvicorn | MIT / BSD | Tự do |
| scikit-learn, NumPy, SciPy | BSD-3-Clause | Tự do |

Kiến trúc đã tách `BaseDetector` thành một giao diện riêng chính là để dự phòng
tình huống này: đổi sang detector khác chỉ cần cài đặt một lớp con, không đụng
tới phần còn lại của pipeline.

---

## 5. Ranh giới sử dụng — phần nào đo được bằng nguồn nào

Đây là bảng quan trọng nhất về mặt phương pháp. Trộn lẫn hai cột rồi báo cáo một
con số duy nhất sẽ là gian dối:

| Chỉ số | Đo trên | **Không** đo được trên |
|---|---|---|
| Detection mAP@0.5 | MVTI (dữ liệu thật, có nhãn bbox) | Mô phỏng (bbox là do chính ta vẽ ra) |
| Tracking IDF1 / MOTA | MVTI (có `object_id`) **và** mô phỏng | — |
| **Conflict Precision / Recall** | **Chỉ mô phỏng** (nguồn duy nhất có nhãn near-miss) | MVTI, UCSD, video tự quay |
| TTC MAE | Chỉ mô phỏng (nguồn duy nhất có TTC chuẩn) | Mọi dữ liệu thật |
| FPS / độ trễ | Mọi nguồn | — |

Trên dữ liệu thật, hệ thống chỉ báo cáo **thống kê mô tả** về near-miss (số lượng,
phân bố TTC, phân loại), **không** báo cáo Precision/Recall — vì không có gì để
đối chiếu.
