# Bản kê khai công cụ AI, dataset, API và thư viện

**Dự án:** SafeRoad AI · Cuộc thi Sáng tạo trẻ Quốc gia về AI 2026 (Bảng C)
**Tương ứng công việc:** mục 15.0 trong kế hoạch · Điều khoản kê khai trung thực của Thể lệ

---

## 1. Nguyên tắc kê khai

Bản kê khai này phân định rõ **ba loại đóng góp** trong toàn bộ sản phẩm:

* **[TỰ LÀM]** — nhóm tự thiết kế và tự viết
* **[AI HỖ TRỢ]** — nhóm định hướng, công cụ AI sinh mã, nhóm rà soát và chịu trách nhiệm
* **[NGUỒN MỞ]** — kế thừa từ thư viện/dataset bên ngoài

---

## 2. Công cụ AI đã sử dụng

| Công cụ | Phiên bản/Model | Dùng vào việc gì | Prompt Log |
|---|---|---|---|
| Claude (Anthropic) | Claude Opus 4.6 | Sinh mã pipeline, dashboard, tài liệu; phân tích lỗi thuật toán | `docs/prompt_log.md` |
| GitHub Copilot | (nếu có sử dụng) | Gợi ý hoàn thiện dòng lệnh trong IDE | — |

**Mức độ can thiệp của AI:** phần lớn mã nguồn trong `src/saferoad/` được sinh với
sự hỗ trợ của công cụ AI dưới định hướng và yêu cầu cụ thể của nhóm. **Toàn bộ
quyết định thiết kế, việc chọn ngưỡng, phương pháp đánh giá và diễn giải kết quả
do nhóm thực hiện và chịu trách nhiệm.** Mọi con số trong báo cáo đều được sinh ra
từ mã trong repo và tái lập được bằng các lệnh nêu trong README.

---

## 3. Phân định chi tiết theo module

| Module | Nội dung | Phân loại | Ghi chú |
|---|---|---|---|
| `types.py` | Kiểu dữ liệu, mô hình đa hình tròn của phương tiện | **[TỰ LÀM]** + [AI HỖ TRỢ] | Ý tưởng đa hình tròn do nhóm chọn sau khi phát hiện lỗi cảnh báo giả của mô hình một hình tròn |
| `detection/` | Bọc YOLO, suy luận theo ô (SAHI) | [NGUỒN MỞ] + [AI HỖ TRỢ] | Model từ Ultralytics; lớp bọc và phần cắt ô do nhóm |
| `tracking/kalman.py` | Kalman filter cho bbox | **[TỰ LÀM]** + [AI HỖ TRỢ] | Cài đặt lại từ công thức, không copy thư viện |
| `tracking/bytetrack.py` | Ghép hai vòng theo ByteTrack | **[TỰ LÀM]** + [AI HỖ TRỢ] | **Thuật toán** theo bài báo ByteTrack (ECCV 2022); **mã nguồn tự cài đặt**, không dùng repo gốc |
| `geometry/` | Homography, làm mượt quỹ đạo, bù sai lệch điểm tiếp đất | **[TỰ LÀM]** + [AI HỖ TRỢ] | `cv2.getPerspectiveTransform` từ OpenCV; phần bù sai lệch là đóng góp riêng |
| `conflict/` | TTC, PET, phân loại, gom episode | **[TỰ LÀM]** + [AI HỖ TRỢ] | Công thức TTC/PET là kiến thức chuẩn ngành; cách gom episode và các cổng lọc là thiết kế của nhóm |
| `risk/` | Risk Score, Risk Map, Explainable Risk | **[TỰ LÀM]** + [AI HỖ TRỢ] | Dạng công thức theo poster đề tài; trọng số hiệu chỉnh bằng thực nghiệm |
| `behavior/` | Phân loại hành vi lai rule + ML | **[TỰ LÀM]** + [AI HỖ TRỢ] | Model từ scikit-learn; đặc trưng và nhãn do nhóm thiết kế |
| `privacy/` | Ẩn danh khuôn mặt, biển số | **[TỰ LÀM]** + [AI HỖ TRỢ] | — |
| `simulation/` | Vi mô phỏng + camera pinhole + oracle | **[TỰ LÀM]** + [AI HỖ TRỢ] | Mô hình IDM theo Treiber et al. (2000); toàn bộ phần còn lại tự viết |
| `evaluation/` | Ablation, mAP, IDF1/MOTA | **[TỰ LÀM]** + [AI HỖ TRỢ] | Định nghĩa chỉ số theo chuẩn COCO/MOT; mã tự cài đặt |
| `dashboard/` | FastAPI + giao diện web | **[TỰ LÀM]** + [AI HỖ TRỢ] | Biểu đồ vẽ tay bằng Canvas 2D, **không dùng thư viện biểu đồ ngoài** |
| Tài liệu | Báo cáo, protocol, README | **[TỰ LÀM]** + [AI HỖ TRỢ] | Số liệu sinh từ mã; diễn giải do nhóm |

---

## 4. Dataset

| Dataset | Nguồn | Giấy phép | Kê khai |
|---|---|---|---|
| Multi-view Traffic Intersection | Møgelmose, Kaggle | Học thuật | **[NGUỒN MỞ]** — dùng nguyên trạng, không sửa nhãn |
| UCSD Highway Traffic | Chan & Vasconcelos, UCSD | Học thuật, yêu cầu trích dẫn | **[NGUỒN MỞ]** — đã trích dẫn theo yêu cầu |
| Dữ liệu mô phỏng SafeRoad | Nhóm tự sinh | MIT | **[TỰ LÀM]** — sinh bằng mã trong repo |
| Video tự quay | Nhóm thu thập | Thuộc về nhóm | **[TỰ LÀM]** — đã ẩn danh |

Chi tiết đầy đủ: `docs/dataset_license.md`.

---

## 5. Thư viện và mô hình

| Thành phần | Phiên bản | Giấy phép | Vai trò |
|---|---|---|---|
| ultralytics (YOLO11n) | ≥ 8.3 | **AGPL-3.0** | Detector nền |
| PyTorch | ≥ 2.0 | BSD-3-Clause | Backend suy luận |
| OpenCV | ≥ 4.8 | Apache-2.0 | Xử lý ảnh, homography, I/O video |
| NumPy / SciPy | ≥ 1.24 / ≥ 1.10 | BSD-3-Clause | Tính toán số |
| scikit-learn | ≥ 1.3 | BSD-3-Clause | Bộ phân loại hành vi |
| FastAPI / Uvicorn | ≥ 0.110 / ≥ 0.27 | MIT / BSD | Backend dashboard |
| PyYAML | ≥ 6.0 | MIT | Đọc cấu hình |
| pytest | ≥ 7.4 | MIT | Kiểm thử |

**Không sử dụng API trả phí hoặc dịch vụ đám mây nào** trong pipeline. Toàn bộ hệ
thống chạy hoàn toàn cục bộ (offline).

---

## 6. Những gì hệ thống **không** làm

Kê khai rõ để tránh hiểu nhầm về năng lực sản phẩm:

* **Không** nhận dạng khuôn mặt hay danh tính cá nhân — ngược lại, chủ động làm mờ.
* **Không** đọc biển số — chủ động làm mờ.
* **Không** gửi dữ liệu ra ngoài; không có kết nối mạng nào trong pipeline xử lý.
* **Không** dự đoán tai nạn sẽ xảy ra; chỉ đo các chỉ số thay thế (TTC/PET) đã
  được kiểm chứng trong nghiên cứu an toàn giao thông.
* **Không** đưa ra quyết định điều khiển tự động; sản phẩm là công cụ **hỗ trợ
  phân tích** cho cán bộ quản lý hạ tầng.

---

## 7. Cam kết trung thực

Nhóm cam kết:

1. Mọi số liệu trong báo cáo được sinh ra từ mã nguồn trong repo và **tái lập
   được** bằng các lệnh ghi trong README.
2. **Không** chỉnh sửa kết quả đánh giá cho đẹp. Các chỉ số chưa đạt mục tiêu đề
   ra ban đầu được báo cáo **nguyên trạng**, kèm phân tích nguyên nhân (mục Đánh
   giá trong tài liệu chính).
3. Ranh giới giữa dữ liệu thật và dữ liệu mô phỏng được nêu rõ ở mọi bảng kết quả;
   **không** trộn hai loại để tạo ra một con số đẹp hơn.
4. Prompt Log được ghi đầy đủ trong `docs/prompt_log.md` kèm liên kết Google Drive
   đã mở quyền truy cập.
