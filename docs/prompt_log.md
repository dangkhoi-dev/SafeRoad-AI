# Prompt Log — SafeRoad AI

**Dự án:** SafeRoad AI · Cuộc thi Sáng tạo trẻ Quốc gia về AI 2026 (Bảng C)
**Yêu cầu Thể lệ:** ghi đầy đủ System Prompt và toàn bộ hội thoại với công cụ AI

> **HƯỚNG DẪN CHO NHÓM:** file này là **khung mẫu**. Trước khi nộp, cần:
> 1. Xuất toàn bộ hội thoại với công cụ AI (Claude/ChatGPT/Copilot Chat…) ra PDF hoặc Markdown.
> 2. Tải lên Google Drive, **mở quyền "Bất kỳ ai có đường liên kết → Người xem"**.
> 3. Dán liên kết vào mục §1 bên dưới và kiểm tra lại bằng cửa sổ ẩn danh.
> 4. Điền §3 cho từng phiên làm việc.

---

## 1. Liên kết Prompt Log đầy đủ

| Nội dung | Liên kết | Trạng thái quyền |
|---|---|---|
| Toàn bộ hội thoại phát triển hệ thống | `<DÁN LIÊN KẾT GOOGLE DRIVE VÀO ĐÂY>` | ☐ Đã mở quyền xem |
| Ảnh chụp màn hình phiên làm việc | `<DÁN LIÊN KẾT>` | ☐ Đã mở quyền xem |
| Mã nguồn (GitHub) | https://github.com/dangkhoi-dev/SafeRoad-AI | ☐ Đã đặt public |

---

## 2. System Prompt đã sử dụng

Nhóm sử dụng công cụ AI ở chế độ trợ lý lập trình. System prompt (hoặc thiết lập
tương đương) như sau:

```
Bạn là kỹ sư AI hỗ trợ nhóm sinh viên xây dựng hệ thống thị giác máy tính
phát hiện xung đột giao thông (near-miss) từ camera giao lộ.

Yêu cầu:
- Viết Python sạch, có docstring giải thích LÝ DO chọn cách làm, không chỉ mô tả
  cách làm.
- Mọi ngưỡng kỹ thuật phải nằm trong file cấu hình, không rải rác trong mã.
- Khi một con số không đạt kỳ vọng, phải truy nguyên nhân gốc rồi mới sửa;
  không được chỉnh ngưỡng cho tới khi biểu đồ đẹp.
- Không bịa số liệu. Mọi chỉ số phải sinh ra từ mã chạy thật.
- Chú thích và tài liệu viết bằng tiếng Việt.
```

---

## 3. Nhật ký phiên làm việc

Điền mỗi phiên một mục.

### Phiên 1 — <NGÀY> — <NGƯỜI THỰC HIỆN>

**Mục tiêu:** <ví dụ: dựng khung detection + tracking>

**Prompt chính đã dùng:**
```
<dán prompt>
```

**Kết quả AI trả về:** <tóm tắt>

**Nhóm đã sửa/rà soát gì:** <ghi cụ thể — đây là phần thể hiện đóng góp của nhóm>

**Kết luận:** <đưa vào repo / bỏ / cần sửa tiếp>

---

### Phiên 2 — <NGÀY> — <NGƯỜI THỰC HIỆN>

**Mục tiêu:**
**Prompt chính:**
**Kết quả:**
**Nhóm sửa gì:**
**Kết luận:**

---

## 4. Các quyết định kỹ thuật quan trọng và lý do

Phần này ghi lại những chỗ nhóm **không** làm theo gợi ý đầu tiên, mà truy nguyên
nhân rồi chọn hướng khác. Đây là phần có giá trị nhất của nhật ký.

| # | Vấn đề gặp phải | Chẩn đoán nguyên nhân gốc | Quyết định |
|---|---|---|---|
| 1 | Mô phỏng sinh 352 near-miss trong 60 giây — vô lý | Xe đi xuyên qua nhau vì chưa có mô hình car-following | Thêm mô hình IDM |
| 2 | Toàn bộ mô phỏng tắc cứng, tốc độ trung bình 0.3 m/s | Cơ chế "ai cũng nhường ai" gây khoá chết; xe cắt ngang bị coi là "xe phía trước cùng làn" nên IDM hãm lại vô hạn | Thay bằng **đèn tín hiệu**; giới hạn car-following chỉ áp dụng cho xe cùng hướng (lệch < 45°) |
| 3 | 132 cảnh báo "đối đầu" giả trong 120 giây | Hình tròn ngoại tiếp biến ô tô rộng 1.8 m thành vật thể rộng 4.76 m ⇒ hai làn cạnh nhau luôn "va chạm" | Chuyển sang **mô hình đa hình tròn** ⇒ còn 13 |
| 4 | Tracking sinh 1.405 track cho 118 xe | Nhiễu 1.5 px trên bbox nhỏ (12×10 px) kéo IoU giữa 2 frame từ 0.93 xuống < 0.2 | Nới biên bbox trước khi tính IoU + lọc bbox quá nhỏ ⇒ còn 178 track, IDF1 = 0.983 |
| 5 | Recall chỉ 0.18 dù hệ thống phát hiện đủ số sự kiện | Sự kiện được phát **trước** thời điểm hai xe gần nhau nhất, lệch quá xa nhãn chuẩn | Chuyển sang **gom theo episode**, gán nhãn tại thời điểm TTC nhỏ nhất |
| 6 | 46% nhãn chuẩn có xe không bao giờ nằm trong vùng đánh giá; người đi bộ có bbox 37 px² | Cảnh dựng bằng hình chiếu bóng xuống đất, bỏ qua **chiều cao** vật thể | Dựng **camera pinhole 3D**, vẽ phương tiện dạng khối hộp ⇒ bbox người đi bộ 1.683 px² |
| 7 | Risk Score có trung vị 98/100 — không phân biệt được gì | Trọng số chưa hiệu chỉnh, sigmoid bão hoà | Hiệu chỉnh lại trên tập có nhãn ⇒ p10=21, trung vị=56, p90=89 |
| 8 | Bộ phân loại hành vi chỉ đạt macro-F1 0.50 | Nhãn gán cho **cả track** rồi cắt cửa sổ trượt ⇒ 20 giây chạy bình thường cũng mang nhãn "phanh gấp" | Gán nhãn theo **từng cửa sổ** ⇒ macro-F1 0.99 |
| 9 | YOLO11n COCO chỉ đạt recall 0.19 trên ảnh giao lộ thật, gán nhãn "train" cho một chiếc ô tô | **Lệch miền**: COCO chụp ngang tầm mắt, camera giao thông đặt cao nhìn chếch xuống | Thêm suy luận theo ô (recall 0.19→0.46) + chuẩn bị fine-tune trên MVTI |

---

## 5. Xác nhận

- ☐ Prompt Log đầy đủ đã tải lên Drive và **đã mở quyền xem**
- ☐ Đã kiểm tra liên kết bằng cửa sổ ẩn danh
- ☐ Bản kê khai công cụ (`docs/ke_khai_cong_cu.md`) đã điền đủ
- ☐ Repo GitHub đã public và có README hướng dẫn chạy
