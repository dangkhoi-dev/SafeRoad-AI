# Protocol định nghĩa Near-miss / Traffic Conflict - v2.0

> **Thay đổi so với v1.0** (Chặng 1, theo nhận xét hội đồng): PET tách hai cách tính và
> nhãn chuẩn dùng PET **đo** (§2.2); xung đột tách ba kết cục near-miss / chồng thân
> xe do sai số / va chạm (§3.1); thêm nhãn rà soát của người và cách chấm (§8). Mọi
> số liệu cũ trong bảng §4 được thay bằng số đo trong `data/outputs/evaluation.json`.

**Dự án:** SafeRoad AI · **Nhóm 2** · Cuộc thi Sáng tạo trẻ Quốc gia về AI 2026 (Bảng C)
**Tương ứng công việc:** mục 4.0 trong kế hoạch - *"Xây dựng protocol định nghĩa near-miss/conflict rõ ràng, nhất quán làm ground truth"*

---

## 1. Vì sao cần một protocol viết ra giấy

Cụm từ "suýt va chạm" nghe thì rõ, nhưng khi phải gán nhãn thật thì hai người
quan sát cùng một đoạn video rất dễ cho ra hai kết quả khác nhau. Ở giao lộ đông
xe máy như Việt Nam, khoảng cách 0.8 m giữa hai xe máy là chuyện **bình thường**,
trong khi cùng khoảng cách đó giữa một ô tô và một người đi bộ là **cực kỳ nguy
hiểm**.

Nếu không chốt định nghĩa trước, mọi con số Precision/Recall về sau đều vô nghĩa
vì chúng đo trên một mục tiêu di động. Tài liệu này chốt định nghĩa đó, và chính
nó được cài đặt trong `saferoad/simulation/scenario.py::compute_ground_truth`
(bộ sinh nhãn chuẩn), `saferoad/conflict/encroachment.py` (PET đo, kết cục) cùng
`saferoad/conflict/detector.py` (thuật toán online).

---

## 2. Hai chỉ số nền tảng

### 2.1 TTC - Time To Collision

Thời gian còn lại tới va chạm **nếu cả hai giữ nguyên vận tốc hiện tại**.

Mỗi phương tiện được phủ bằng 1-3 hình tròn dọc trục thân xe, bán kính bằng nửa
bề rộng xe (xem §5). Với một cặp hình tròn, đặt Δp = p_b − p_a, Δv = v_b − v_a
và R = r_a + r_b, va chạm xảy ra khi ‖Δp + tΔv‖ = R, dẫn tới phương trình bậc hai:

```
‖Δv‖² t² + 2(Δp · Δv) t + (‖Δp‖² − R²) = 0
```

TTC là **nghiệm dương nhỏ nhất trên mọi cặp hình tròn**. Không có nghiệm dương
nghĩa là hai đối tượng đang tách xa nhau hoặc sẽ lướt qua nhau.

### 2.2 PET - Post-Encroachment Time (hai cách tính, lưu riêng)

**Định nghĩa gốc** (Allen và cộng sự, 1978): khoảng thời gian từ lúc đối tượng thứ
nhất **rời khỏi vùng xung đột** tới lúc đối tượng thứ hai **đi vào** chính vùng đó.

**Vùng xung đột** là phần giao giữa dải quét thân xe A và dải quét thân xe B trên
mặt đất (thân xe phủ bằng các hình tròn như §5). Một mẫu thời gian của xe A nằm
trong vùng khi thân A lúc đó chạm dải quét của B ở bất kỳ thời điểm nào. Từ đó có
bốn mốc `a_in`, `a_out`, `b_in`, `b_out` (lấy ở điểm giữa hai mẫu kề biên).

| | PET ngoại suy (`pet_pred`) | PET đo (`pet_measured`) |
|---|---|---|
| Tính khi nào | Trước - từ vị trí, vận tốc tức thời | Sau - khi cả hai xe đã qua vùng |
| Công thức | `t_vào(sau) − t_ra(trước)` trên hai tia vận tốc, nửa chiều dài vùng = `L/2 + (W_kia/2)/sin θ` | `b_in − a_out` (A vào trước) từ quỹ đạo đã ghi |
| Dùng để | Mở episode, cảnh báo sớm | **Chốt sự kiện**, so với nhãn chuẩn |

**Trạng thái phép đo** (`pet_status`): `measured` · `parallel` (góc < 20° hoặc
> 160°: cùng làn/đối đầu cùng vệt - PET không xác định, để TTC xử lý) · `no_zone`
(hai dải quét không giao nhau) · `incomplete` (mất dấu giữa vùng) ·
`simultaneous` (cùng chiếm vùng nhưng không chạm, PET = 0) · `contact` (thân xe
chồng nhau cùng thời điểm).

**Xe đã qua giao điểm:** PET ngoại suy vẫn tính khi xe trước đã qua giao điểm
không quá 3 s (xe sau phải còn đang tiến tới). PET đo không phụ thuộc điều này vì
dùng lịch sử quỹ đạo.

---

## 3. Định nghĩa chính thức: khi nào là Near-miss

Một cặp đối tượng `(A, B)` được ghi nhận là **near-miss** khi thoả **đồng thời cả
bốn** điều kiện sau trong suốt quãng thời gian hai đối tượng cùng xuất hiện:

| # | Điều kiện | Ngưỡng | Vì sao |
|---|---|---|---|
| 1 | **Thực sự đến gần nhau** - khoảng cách mặt-tới-mặt nhỏ nhất | < 2.0 m (oracle) / 3.0 m (online) | TTC chỉ là *phép ngoại suy*. Nếu một bên kịp phanh và hai xe chưa bao giờ tới gần, đó là tình huống được xử lý **tốt**, không phải sự cố. |
| 2 | **Có nguy cơ va chạm** - cực tiểu TTC **hoặc** PET **đo** (§2.2; online chỉ dùng PET ngoại suy khi không đo được) | TTC < 3.0 s **hoặc** PET < 1.5 s | Ngưỡng theo thông lệ nghiên cứu conflict và theo cam kết trong poster đề tài. |
| 3 | **Có tốc độ tiếp cận thật** - vận tốc tương đối tại thời điểm căng nhất | > 2.5 m/s (oracle) / 3.0 m/s (online) | Đây là thứ phân biệt xung đột thật với **dòng xe bám đuôi bình thường**. Hai xe máy nối đuôi cách 0.9 m ở cùng tốc độ có TTC rất nhỏ theo mô hình hình học, nhưng đó là giao thông bình thường. |
| 4 | **Nằm trong vùng cảm biến** - vị trí xung đột trong tầm phủ của camera | vùng `COVERAGE` | Không thể quy trách nhiệm cho hệ thống về thứ nằm ngoài khung hình. |

Điều kiện 1 và 3 là phần **quan trọng nhất và cũng dễ bị bỏ quên nhất**. Chỉ dùng
TTC đơn thuần sẽ gán nhãn near-miss cho hàng trăm cặp xe đang lưu thông hoàn toàn
bình thường trong dòng đông đúc.

---

## 3.1 Ba kết cục - không đồng nhất xung đột với tai nạn

| Kết cục | Điều kiện | Ý nghĩa |
|---|---|---|
| `near_miss` | khoảng cách mặt-tới-mặt **cùng thời điểm** > 0 | suýt va chạm |
| `body_overlap` | khoảng cách = 0 nhưng cả hai xe giữ tốc độ sau đó | vật lý không cho phép ⇒ sai số đo / che khuất, **không** phải va chạm |
| `collision_suspected` | khoảng cách = 0 và ít nhất một xe gần như dừng hẳn trong 2,5 s | nghi va chạm thật - bắt buộc người xác nhận |

Nhãn chuẩn mô phỏng tách `near_miss` và `collision`; trình mô phỏng v2 có vật lý va
chạm (hai xe chạm nhau thì cùng dừng), không còn cho xe đi xuyên qua nhau.

## 4. Phân mức nghiêm trọng

| Mức | TTC nhỏ nhất | Diễn giải |
|---|---|---|
| **Rất nghiêm trọng** | < 1.0 s | Gần như không còn thời gian phản ứng |
| **Nghiêm trọng** | 1.0 - 1.5 s | Chỉ vừa đủ thời gian phản xạ |
| **Trung bình** | 1.5 - 2.5 s | Còn kịp phanh nếu chú ý |
| **Nhẹ** | 2.5 - 3.0 s | Đáng ghi nhận nhưng chưa nguy hiểm |

Recall theo từng dải (chỉ trên nhãn near-miss, va chạm báo riêng) nằm ở
`main.ttc_bands` trong `data/outputs/evaluation.json`.

---

## 5. Mô hình hình học của phương tiện

Mỗi đối tượng được phủ bằng **1-3 hình tròn** đặt dọc trục thân xe, bán kính bằng
nửa bề rộng xe:

| Lớp | Dài × Rộng × Cao (m) | Số hình tròn | Bán kính (m) |
|---|---|---|---|
| Người đi bộ | 0.5 × 0.5 × 1.70 | 1 | 0.25 |
| Xe đạp | 1.7 × 0.6 × 1.65 | 3 | 0.30 |
| Xe máy | 1.9 × 0.7 × 1.60 | 3 | 0.35 |
| Ô tô | 4.4 × 1.8 × 1.50 | 2 | 0.90 |
| Xe tải / Bus | 7.5 × 2.5 × 3.20 | 3 | 1.25 |

**Vì sao không dùng một hình tròn duy nhất.** Hình tròn ngoại tiếp một ô tô
4.4 × 1.8 m có bán kính ½·√(4.4² + 1.8²) ≈ 2.38 m - tức là mô hình hoá chiếc xe
như một vật thể **rộng 4.76 m**. Hậu quả đo được trong quá trình phát triển: hai
ô tô đi ngược chiều ở hai làn cách nhau 4 m bị gắn nhãn "đối đầu", sinh ra 132
cảnh báo giả trong 120 giây cho dòng xe hoàn toàn bình thường. Chuyển sang mô
hình đa hình tròn đưa con số đó xuống còn 13.

---

## 6. Phân loại kiểu xung đột

Dựa trên góc θ giữa hai vector vận tốc tại thời điểm TTC nhỏ nhất:

| Kiểu | Điều kiện | Hệ số nghiêm trọng |
|---|---|---|
| Người đi bộ | có người đi bộ tham gia (**ưu tiên cao nhất**) | 1.00 |
| Đối đầu | θ ≥ 150° | 0.95 |
| Cắt ngang | 60° ≤ θ ≤ 120° | 0.85 |
| Chuyển hướng | tốc độ đổi hướng > 45°/s | 0.70 |
| Chuyển làn | 30° < θ < 60° hoặc 120° < θ < 150° | 0.55 |
| Tạt đầu/đâm đuôi | θ < 30° | 0.45 |

Xung đột có người đi bộ **luôn** được gán nhãn riêng bất kể hình học, vì đây là
nhóm dễ tổn thương nhất và cần tách riêng trong mọi thống kê an toàn.

---

## 7. Gom sự kiện theo episode

Một tình huống suýt va chạm kéo dài 1-2 giây. Ở 30 FPS, nếu phát một sự kiện cho
mỗi khung hình thoả ngưỡng thì **một** tình huống sẽ biến thành 30-60 "sự kiện".

Quy tắc:

1. **Mở episode** khi cặp đối tượng lần đầu thoả điều kiện 2 + 3.
2. **Duy trì** episode, liên tục cập nhật TTC/PET/khoảng cách nhỏ nhất - kể cả ở
   những khung hình điều kiện tạm thời tắt (khoảnh khắc hai xe gần nhau **nhất**
   thường rơi vào lúc TTC đã hết ý nghĩa vì chúng đang lướt qua nhau).
3. **Đóng** episode sau 5 khung hình liên tiếp không còn xung đột.
4. **Phát một sự kiện duy nhất**, gán nhãn thời gian tại **thời điểm TTC nhỏ nhất**
   - khoảnh khắc căng thẳng nhất, không phải lúc bắt đầu hay lúc kết thúc.
5. **Gộp** hai episode của cùng một cặp nếu cách nhau dưới 8 giây.

---

## 8. Rà soát và gán nhãn của người (video thật)

1. Sinh danh sách ứng viên: `saferoad evaluate-real` (video có nhãn bbox) hoặc
   `saferoad run --review-csv ...` (video không nhãn).
2. Mở dashboard, tab **Rà soát nhãn**. Mỗi người gán **độc lập**, xem ở 0,25×
   quanh thời điểm, chọn một nhãn: `near_miss` · `collision` · `normal` (báo nhầm) ·
   `overlap_artifact` (hai xe chồng nhau trên ảnh do che khuất) · `unclear`.
3. Xem thêm toàn bộ video, đếm các xung đột **không có** trong danh sách (bỏ sót).
4. `saferoad review-score --candidates ... --labels a.csv b.csv --duration <giây>
   --extra-missed <số>`: tính Cohen's kappa (5 lớp và có/không xung đột).
5. Nếu κ < 0,6, ngồi lại thống nhất cách hiểu protocol rồi gán lại - **không** lấy
   trung bình hai kết quả bất đồng. Chỉ nhãn đồng thuận được dùng để chấm.

Ghi rõ số người gán nhãn, κ và thời lượng video trong báo cáo.

---

## 9. Giới hạn đã biết của protocol

Nêu ra để người đọc báo cáo đánh giá đúng phạm vi kết luận:

1. **Mô hình vận tốc không đổi.** TTC giả định cả hai giữ nguyên vận tốc. Khi có
   xe phanh gấp, giá trị TTC tức thời sẽ bi quan hơn thực tế. Đây là hạn chế
   chung của mọi phương pháp TTC, không riêng hệ thống này.

2. **Giả thiết mặt phẳng.** Homography giả định mọi đối tượng nằm trên **một** mặt
   phẳng. Với giao lộ dốc hoặc có cầu vượt, cần chia vùng và calibrate riêng.

3. **Ranh giới "dòng xe đông" và "xung đột" vốn mờ.** Ngay cả người gán nhãn có
   kinh nghiệm cũng bất đồng ở dải TTC 2-3 s. Đây là một lý do Precision không
   tiến sát 1.0, và cũng là phát hiện được ghi nhận trong tài liệu nghiên cứu về
   traffic conflict technique.

4. **Chưa mô hình hoá tương tác nhiều bên.** Hệ thống xét từng **cặp**; tình huống
   ba xe cùng liên quan sẽ được ghi thành nhiều cặp riêng lẻ.

---

## 10. Tài liệu tham khảo

1. Hayward, J.C. (1972). *Near-miss determination through use of a scale of danger.*
   Highway Research Record 384.
2. Allen, B.L., Shin, B.T., Cooper, P.J. (1978). *Analysis of traffic conflicts and
   collisions.* Transportation Research Record 667.
3. Zheng, L., Ismail, K., Meng, X. (2014). *Traffic conflict techniques for road
   safety analysis: open questions and some insights.* Canadian Journal of Civil
   Engineering, 41(7).
4. Treiber, M., Hennecke, A., Helbing, D. (2000). *Congested traffic states in
   empirical observations and microscopic simulations.* Physical Review E, 62(2).
   - mô hình IDM dùng trong trình mô phỏng.
5. Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every
   Detection Box.* ECCV 2022.

---

*Phiên bản 2.0 - Chặng 1. Ngưỡng không đổi so với v1.0; thay đổi nằm ở định nghĩa PET,
phân loại kết cục và quy trình rà soát. Toàn bộ đánh giá đã chạy lại.*
