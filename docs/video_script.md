# Kịch bản quay video — SafeRoad AI

**Tương ứng công việc:** mục 14.0 trong kế hoạch
**Yêu cầu Thể lệ:** 2 video, mỗi video **tối đa 5 phút**, **cả 3 thành viên phải xuất hiện** và trình bày phần mình phụ trách.

Chuẩn bị trước khi quay — chạy sẵn các lệnh này để có đủ hình ảnh:

```bash
python -m saferoad simulate --duration 180 --seed 42
python -m saferoad run --config configs/synthetic.yaml \
    --ground-truth data/samples/synthetic_groundtruth.pkl --replay-detections
python -m saferoad evaluate --out data/outputs/evaluation.json
python -m saferoad serve --results data/outputs/results.json
```

---

# VIDEO 1 — Thuyết trình (tối đa 5:00)

| Thời gian | Người trình bày | Nội dung | Hình ảnh trên màn hình |
|---|---|---|---|
| **0:00–0:25** | Đăng Khôi | Chào, giới thiệu nhóm và tên đề tài. Nêu ngay con số: mỗi năm Việt Nam có hàng nghìn người chết vì tai nạn giao thông, nhưng **ta chỉ đo được tai nạn đã xảy ra** | Slide bìa + ảnh giao lộ đông đúc |
| **0:25–1:00** | Huynh Hân | **Vấn đề.** Cách quản lý hiện tại là *phản ứng*: chờ tai nạn xảy ra rồi mới xử lý điểm đen. Nhưng trước mỗi tai nạn có **hàng trăm tình huống suýt va chạm** — dữ liệu đó đang bị bỏ phí hoàn toàn | Sơ đồ "kim tự tháp an toàn": 1 tai nạn ← ~300 near-miss |
| **1:00–1:30** | Huynh Hân | **Đối tượng dùng.** Sở GTVT, đơn vị quản lý hạ tầng: biết giao lộ nào nguy hiểm **trước khi** có người chết. Giá trị: chuyển từ phản ứng sang phòng ngừa | Ảnh dashboard, bảng Top điểm rủi ro |
| **1:30–2:30** | Đăng Khôi | **Phương pháp.** Giải thích TTC và PET bằng hình vẽ — không dùng công thức. "TTC là còn bao nhiêu giây nữa thì đụng nếu cả hai giữ nguyên tốc độ." Nêu pipeline 6 khối | Sơ đồ kiến trúc (`docs/assets/architecture.png`) |
| **2:30–3:15** | Mạnh Anh | **Cách chúng em đo.** Đây là điểm mạnh nhất — nhấn kỹ: *"Không dataset công khai nào có nhãn near-miss. Nên chúng em tự viết trình mô phỏng giao lộ có ground truth chính xác để đo, và dùng dataset thật để đo detection và tracking."* | Video mô phỏng cạnh video thật |
| **3:15–4:00** | Mạnh Anh | **Kết quả.** IDF1 = 0.98 · TTC MAE = 0.29 s · **Recall 0.82 ở dải nguy hiểm nhất (TTC < 1 s)** · 36 FPS. Nói thẳng: Precision 0.53 chưa đạt mục tiêu 0.75, và giải thích vì sao | Bảng ablation trên dashboard |
| **4:00–4:35** | Đăng Khôi | **Hạn chế & hướng phát triển.** Trung thực: chưa fine-tune trên dữ liệu Việt Nam; ranh giới "đông xe" và "xung đột" vốn mờ. Hướng đi: fine-tune YOLO, calibrate thực địa, triển khai Jetson | Slide 3 gạch đầu dòng |
| **4:35–5:00** | Cả 3 | Chốt lại giá trị đề tài, cảm ơn | Cả nhóm cùng khung hình |

**Lưu ý khi quay:**
- Mỗi người nói **liên tục ít nhất 40 giây** để giám khảo thấy rõ phần đóng góp.
- Không đọc slide. Nói bằng lời của mình.
- Nói rõ chỗ **chưa đạt** — giám khảo đánh giá cao sự trung thực hơn là con số đẹp.

---

# VIDEO 2 — Demo (tối đa 5:00)

| Thời gian | Người thực hiện | Thao tác | Điểm nhấn |
|---|---|---|---|
| **0:00–0:20** | Đăng Khôi | Mở terminal, chạy `python -m saferoad simulate --duration 60` | Cho thấy hệ thống **tự sinh dữ liệu kiểm chứng**, không phụ thuộc dataset ngoài |
| **0:20–1:10** | Đăng Khôi | Chạy `python -m saferoad run --config configs/synthetic.yaml` — để chạy thật, thấy log đếm near-miss tăng dần | Đọc to dòng cuối: FPS và độ trễ ms/frame |
| **1:10–2:00** | Mạnh Anh | Mở video overlay. Tua tới một near-miss. **Dừng hình** tại khoảnh khắc cảnh báo đỏ | Chỉ vào nhãn `TTC 0.8s HIGH` trên màn hình, giải thích khung màu đỏ nghĩa là gì |
| **2:00–2:40** | Mạnh Anh | `python -m saferoad serve` → mở trình duyệt → tab **Tổng quan** | 4 thẻ KPI, biểu đồ near-miss theo thời gian, biểu đồ tròn phân loại |
| **2:40–3:20** | Huynh Hân | Tab **Near-miss** → bấm vào một sự kiện rủi ro cao | **Đây là cảnh quan trọng nhất**: panel "Vì sao hệ thống cảnh báo" liệt kê lý do bằng tiếng Việt + thanh đóng góp của từng thành phần. Đọc to một lý do |
| **3:20–3:55** | Huynh Hân | Tab **Risk Map** | Bản đồ nhiệt, 5 điểm nóng đánh dấu ✕. Giải thích: "đây là những chỗ Sở GTVT nên ưu tiên xử lý" |
| **3:55–4:30** | Đăng Khôi | Tab **Đánh giá** | Bảng ablation: chỉ ra dòng "+ Proximity gate" làm Precision nhảy từ 0.10 lên 0.53 — **bằng chứng từng khối đều có đóng góp đo được** |
| **4:30–5:00** | Cả 3 | Tab **Hành vi**, rồi kết | Nhắc lại: toàn bộ chạy **offline**, không gửi dữ liệu đi đâu |

**Lưu ý kỹ thuật khi quay demo:**
- Quay màn hình ở **1920×1080**, phóng to trình duyệt lên **125%** cho dễ đọc.
- **Chạy thật, không dựng lại** — giám khảo phân biệt được.
- Nếu máy yếu, thêm `--max-frames 900` để rút ngắn thời gian chạy.
- Chuẩn bị sẵn `results.json` và `evaluation.json` phòng khi chạy trực tiếp bị lỗi.
- Bật phụ đề hoặc thuyết minh cho từng thao tác.

---

## Checklist trước khi nộp video

- ☐ Cả 3 thành viên xuất hiện trong **cả hai** video
- ☐ Mỗi video **≤ 5:00**
- ☐ Có tên đề tài và tên nhóm ở đầu video
- ☐ Âm thanh rõ, không ồn nền
- ☐ Video demo là **màn hình thật**, không phải slide mô phỏng
- ☐ Đã nêu rõ phần nào là mô phỏng, phần nào là dữ liệu thật
