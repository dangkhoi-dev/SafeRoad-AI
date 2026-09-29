# SafeRoad AI

**Hệ thống AI phát hiện near-miss, phân tích rủi ro và xây dựng bản đồ an toàn giao thông**

Cuộc thi Sáng tạo trẻ Quốc gia trong lĩnh vực Trí tuệ nhân tạo 2026 - Bảng C

[![CI](https://github.com/dangkhoi-dev/SafeRoad-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoi-dev/SafeRoad-AI/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

---

## Vấn đề

Quản lý an toàn giao thông ở Việt Nam đang chạy theo cơ chế **phản ứng**: một giao
lộ chỉ được coi là "điểm đen" sau khi đã đủ số vụ tai nạn được thống kê. Nghĩa là
dữ liệu để ra quyết định chỉ xuất hiện **sau khi đã có người thiệt mạng**.

Trong khi đó, trước mỗi vụ tai nạn có hàng trăm tình huống **suýt va chạm** mang
cùng cơ chế nguy hiểm nhưng kết thúc may mắn. Chúng diễn ra liên tục ngay trước
ống kính camera giao thông đã lắp sẵn - và không ai ghi nhận.

**SafeRoad AI biến luồng video giám sát sẵn có thành chỉ số rủi ro định lượng.**

---

## Kết quả đo được

Toàn bộ số liệu sinh ra từ mã trong repo này và chạy lại được bằng các lệnh ở mục
[Chạy thử](#chạy-thử). Tệp kết quả gốc nằm trong `data/outputs/`, mỗi tệp ghi kèm
phiên bản thư viện lúc chạy.

**Trên tập mô phỏng có đáp án** (300 giây, 315 phương tiện, 152 nhãn near-miss,
detector mô phỏng nhiễu 1,5 px và bỏ sót 3%):

| Chỉ số | Kết quả | Mục tiêu | |
|---|---:|---:|:--:|
| Tracking IDF1 | **0,983** | ≥ 0,70 | ✅ |
| Tracking MOTA | **0,966** | ≥ 0,60 | ✅ |
| TTC MAE | **0,286 s** | ≤ 0,30 s | ✅ |
| Recall khi TTC < 1 s (77 nhãn) | **0,792** | - | |
| Recall khi TTC < 1 s, chỉ nhãn chưa chạm (22 nhãn) | 0,773 | - | |
| Conflict Recall (gộp mọi mức) | 0,539 | ≥ 0,80 | ❌ |
| Conflict Precision | 0,471 | ≥ 0,75 | ❌ |

**Trên dữ liệu thật MVTI** (2.441 ảnh, YOLO11n trọng số COCO gốc, CPU 2 lõi, không GPU):

| Chỉ số | Một lượt toàn khung (mặc định) | Chia ô 2×3 (`--backend tiled`) |
|---|---:|---:|
| mAP@0,5 | 0,208 | 0,164 |
| mAP@0,5:0,95 | 0,139 | 0,104 |
| Precision / Recall ở ngưỡng 0,15 | 0,468 / 0,277 | 0,353 / 0,323 |
| IDF1 / MOTA | 0,235 / 0,061 | 0,262 / -0,022 |
| Tốc độ toàn hệ thống | 12,1 FPS | 1,4 FPS |

Các chỉ số chưa đạt được báo cáo nguyên trạng:

- Precision và Recall gộp chưa đạt mục tiêu. Với detector hoàn hảo (nhiễu 0 px) F1 = 0,583,
  nên khoảng 0,08 điểm F1 mất do nhiễu detector, phần còn lại là giới hạn của phương
  pháp. 127/214 cảnh báo có khoảng cách thật giữa hai xe dưới 1 m: phần lớn cảnh báo bị
  chấm là sai vẫn là hai xe đã rất gần nhau.
- Trình mô phỏng chưa có ràng buộc chống va chạm: 55/152 nhãn là lúc hai thân xe đã
  chồng lên nhau. `saferoad evaluate` in riêng Recall của nhóm này.
- Trên video thật, detector là nút thắt: camera đặt cao nhìn xuống khác hẳn ảnh COCO.
  Chia ô không giúp và chậm hơn khoảng 9 lần; hướng xử lý là fine-tune
  (`notebooks/01_finetune_yolo11n_colab.ipynb`).
- 12,1 FPS là tốc độ toàn hệ thống có YOLO trên CPU 2 lõi, chưa đạt 25 FPS. Phần sau
  phát hiện chỉ tốn 0,7 ms mỗi khung hình. FPS hiển thị trên dashboard của phiên mô
  phỏng dùng hộp phát hiện dựng sẵn nên không gồm YOLO (dashboard có ghi chú).

Phân tích đầy đủ: [`docs/BaoCao_SafeRoadAI_BangC.pdf`](docs/BaoCao_SafeRoadAI_BangC.pdf).

---

## Kiến trúc

![Kiến trúc hệ thống](docs/assets/architecture.png)

| Khối | Chức năng | Công nghệ |
|---|---|---|
| 1. Nguồn video | Thu nhận luồng hình ảnh | MP4 / RTSP / chuỗi ảnh |
| 2. AI Processing | Phát hiện · ẩn danh · theo vết · quỹ đạo mặt đất | YOLO11n · ByteTrack · Kalman · Homography |
| 3. Conflict Engine | TTC/PET · gom episode · phân loại xung đột | Mô hình đa hình tròn |
| 4. Risk Engine | Risk Score · Explainable Risk · bản đồ nhiệt | Sigmoid có trọng số |
| 5. Database | Lưu trữ bền vững, đọc/ghi song song | SQLite chế độ WAL |
| 6. Dashboard | 7 tab trực quan, chạy ngoại tuyến | FastAPI · HTML/Canvas thuần |

---

## Cài đặt

```bash
git clone https://github.com/dangkhoi-dev/SafeRoad-AI.git
cd SafeRoad-AI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,docs]"      # [dev,docs] = pytest + thư viện sinh báo cáo
python scripts/download_assets.py  # tải trọng số YOLO11n (~5,6 MB)
```

Yêu cầu: Python ≥ 3.10. Chạy được trên CPU; có GPU thì nhanh hơn.

---

## Chạy thử

Không cần tải dataset - hệ thống **tự sinh dữ liệu kiểm chứng**:

```bash
# 1. Sinh video mô phỏng + nhãn chuẩn (~2 phút)
python -m saferoad simulate --duration 300 --seed 42

# 2. Chạy pipeline end-to-end (~1 phút)
python -m saferoad run --config configs/synthetic.yaml \
    --ground-truth data/samples/synthetic_groundtruth.pkl --replay-detections

# 3. Chấm điểm + bảng ablation (~15 phút trên CPU)
python -m saferoad evaluate

# 4. Mở dashboard
python -m saferoad serve
#   → http://127.0.0.1:8000
```

Chạy trên video của bạn:

```bash
python -m saferoad run --source video_cua_ban.mp4 --anonymize
```

---

## Dùng với dataset thật

```bash
# Sắp xếp dữ liệu đã tải về vào cấu trúc chuẩn (xem trước trước khi làm thật)
python scripts/organize_dataset.py --source D:\SV7 --dry-run
python scripts/organize_dataset.py --source D:\SV7

# Dựng video + nhãn từ Multi-view Traffic Intersection Dataset
python -m saferoad prepare-real --root data/raw/mvti --view Infrastructure

# Chấm mAP detection + IDF1 tracking trên ảnh thật (~4 phút)
python -m saferoad evaluate-real --config configs/mvti.yaml

# So sánh với detector chia ô (~30 phút)
python -m saferoad evaluate-real --config configs/mvti.yaml --backend tiled --no-overlay \
    --out data/outputs/evaluation_real_tiled.json
```

---

## Bản trình diễn trực tuyến

```bash
node scripts/build_static_site.mjs    # dựng site/ từ kết quả trong data/outputs/
```

Thư mục `site/` là đúng giao diện dashboard, đọc dữ liệu tính sẵn thay cho API.
`vercel.json` đã cấu hình để Vercel dựng trang này khi import repo.

---

## Toàn bộ lệnh

| Lệnh | Chức năng |
|---|---|
| `saferoad simulate` | Sinh video mô phỏng + nhãn chuẩn near-miss |
| `saferoad run` | Chạy pipeline end-to-end trên một video |
| `saferoad evaluate` | Chấm Precision/Recall/TTC-MAE + bảng ablation |
| `saferoad serve` | Bật dashboard web |
| `saferoad anonymize` | Ẩn danh video (làm mờ mặt, biển số) |
| `saferoad train-behavior` | Huấn luyện bộ phân loại hành vi |
| `saferoad prepare-real` | Dựng video + nhãn từ dataset thật (MVTI) |
| `saferoad evaluate-real` | Chấm mAP + IDF1 trên dữ liệu thật |

---

## Bốn quyết định kỹ thuật đáng chú ý

### 1. Tự viết trình mô phỏng vì không dataset nào có nhãn near-miss

Mọi dataset giao thông công khai chỉ có bounding box. Mà Precision/Recall của
phát hiện xung đột lại là cam kết trung tâm của đề tài.

Trình mô phỏng chạy vi mô phỏng thật - đèn tín hiệu hai pha, car-following IDM,
gap acceptance cho xe rẽ, phanh tránh khẩn cấp. Near-miss **phát sinh** từ đúng
nguyên nhân ngoài đời (vượt đèn đỏ, rẽ cắt dòng, người đi bộ băng qua), không có
tình huống nào được dàn dựng thủ công.

Nhãn chuẩn được tính bằng quy trình **khác** thuật toán online: 60 Hz thay vì
30 Hz, vận tốc giải tích thay vì ước lượng từ bbox nhiễu, cực tiểu toàn cục thay
vì quyết định từng khung hình. Nhờ đó đây là nhãn độc lập, không phải thuật toán
tự chấm điểm cho chính nó.

### 2. Mô hình đa hình tròn thay cho hình tròn ngoại tiếp

Hình tròn ngoại tiếp một ô tô 4,4 × 1,8 m có bán kính 2,38 m - tức là mô hình hoá
chiếc xe như vật thể **rộng 4,76 m**. Hậu quả đo được: hai ô tô đi ngược chiều ở
hai làn cách nhau 4 m bị gắn nhãn "đối đầu", sinh ra **132 cảnh báo giả trong 120
giây** cho dòng xe hoàn toàn bình thường.

Phủ thân xe bằng 1-3 hình tròn nhỏ dọc trục (bán kính = nửa bề rộng xe) đưa con
số đó xuống còn **13**, mà vẫn giữ được công thức TTC dạng đóng.

### 3. Gom sự kiện theo episode, gán nhãn tại đỉnh nguy hiểm

Một tình huống suýt va chạm kéo dài 1-2 giây. Ở 30 FPS, phát sự kiện mỗi khung
hình sẽ biến **một** tình huống thành 30-60 "sự kiện".

Hệ thống theo dõi cả episode, đóng lại khi điều kiện chấm dứt, rồi phát **một**
sự kiện duy nhất gán nhãn tại **thời điểm TTC nhỏ nhất**. Riêng thay đổi này nâng
Recall từ 0,18 lên 0,42.

### 4. Cổng khoảng cách - thành phần có tác dụng lớn nhất

TTC chỉ là một **phép ngoại suy**. Nếu một bên kịp phanh và hai xe chưa bao giờ
tới gần nhau, đó là tình huống được xử lý **tốt**, không phải sự cố.

Thêm điều kiện "hai xe phải thực sự đến gần nhau" làm Precision tăng
**4,6 lần** (0,103 → 0,471), đổi lại Recall giảm
17 điểm phần trăm. F1 tăng từ 0,180 lên 0,503:

| Cấu hình | Precision | Recall | F1 |
|---|---:|---:|---:|
| Detection + Tracking | 0,035 | 0,020 | 0,025 |
| + Homography | 0,073 | 0,671 | 0,132 |
| + Làm mượt quỹ đạo | 0,154 | 0,678 | 0,251 |
| + PET | 0,103 | 0,711 | 0,180 |
| **+ Cổng khoảng cách** | **0,471** | **0,539** | **0,503** |

---

## Một phát hiện về YOLO trên camera giao thông

Đo trên 2.441 ảnh giao lộ thật theo chuẩn COCO, YOLO11n với trọng số COCO gốc chỉ đạt
**mAP@0,5 = 0,208**. Khi chạy ở kích thước ảnh 1280, có khung hình model gán nhãn
`airplane` cho 35 đối tượng.

Nguyên nhân chính là **lệch miền**: COCO chủ yếu là ảnh chụp ngang tầm mắt, còn camera
giao thông đặt cao nhìn chếch xuống.

Đã thử suy luận chia ô 2×3 (`--backend tiled`): Recall ở ngưỡng vận hành tăng
(0,277 → 0,323) nhưng Precision và mAP giảm, và chậm hơn khoảng 9 lần, nên
cấu hình mặc định giữ một lượt. Giải pháp căn cơ là fine-tune - xem
[`notebooks/01_finetune_yolo11n_colab.ipynb`](notebooks/01_finetune_yolo11n_colab.ipynb).

---

## Cấu trúc thư mục

```
SafeRoad-AI/
├── src/saferoad/
│   ├── types.py              # kiểu dữ liệu lõi, mô hình đa hình tròn
│   ├── config.py             # toàn bộ ngưỡng kỹ thuật ở một chỗ
│   ├── cli.py                # giao diện dòng lệnh
│   ├── detection/            # YOLO + suy luận theo ô
│   ├── tracking/             # ByteTrack + Kalman (tự cài đặt)
│   ├── geometry/             # homography, làm mượt quỹ đạo
│   ├── conflict/             # TTC, PET, gom episode
│   ├── risk/                 # Risk Score, Risk Map, Explainable Risk
│   ├── behavior/             # phân loại hành vi rule + ML
│   ├── privacy/              # ẩn danh mặt & biển số
│   ├── simulation/           # vi mô phỏng + camera pinhole + oracle
│   ├── evaluation/           # ablation, mAP, IDF1/MOTA
│   ├── data/                 # bộ nạp dataset thật
│   ├── storage/              # SQLite
│   ├── pipeline/             # điều phối + overlay
│   └── dashboard/            # FastAPI + giao diện web
├── configs/                  # default · synthetic · hangxanh · mvti
├── notebooks/                # fine-tune YOLO trên Colab
├── scripts/                  # sắp xếp dữ liệu, xuất YOLO, dựng báo cáo
├── tests/                    # 91 test
└── docs/                     # báo cáo kỹ thuật, protocol, giấy phép dữ liệu
```

---

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`docs/BaoCao_SafeRoadAI_BangC.pdf`](docs/BaoCao_SafeRoadAI_BangC.pdf) | Báo cáo kỹ thuật: phương pháp, kiến trúc, kết quả đánh giá, ablation |
| [`docs/protocol_near_miss_v1.md`](docs/protocol_near_miss_v1.md) | Định nghĩa chính thức near-miss dùng làm ground truth |
| [`docs/dataset_license.md`](docs/dataset_license.md) | Bảng kê nguồn dữ liệu và điều kiện bản quyền |

---

## Kiểm thử

```bash
pytest tests/ -q          # 91 test
```

Bộ test kiểm tra những chỗ **dễ sai nhất**, không chỉ chạy cho có:

- TTC/PET đúng trên các tình huống tính tay được;
- hai xe đi ngược chiều ở hai làn cạnh nhau **không** phải xung đột;
- một episode kéo dài sinh **đúng một** sự kiện;
- mô phỏng **không bị tắc nghẽn** (từng có lỗi khiến cả nút giao khoá chết);
- xe **không đi xuyên qua nhau**;
- một dự đoán **không** được khớp với hai nhãn chuẩn;
- mAP chấm đúng chuẩn COCO, MOTA âm không bị kẹp về 0.

Bộ test chạy tự động trên GitHub Actions ở mỗi lần đẩy mã (`.github/workflows/ci.yml`).

---

## Giấy phép

Mã nguồn: **MIT**.

⚠️ Ultralytics YOLO dùng **AGPL-3.0** - có tính lây lan. Dùng cho nghiên cứu và dự
thi thì hợp lệ; nếu thương mại hoá cần mua giấy phép doanh nghiệp hoặc thay
detector. Kiến trúc đã tách `BaseDetector` thành giao diện riêng để việc thay thế
chỉ cần cài đặt một lớp con.

Trích dẫn bắt buộc khi dùng UCSD Highway Traffic Database:

> A. B. Chan and N. Vasconcelos, "Probabilistic Kernels for the Classification of
> Auto-Regressive Visual Processes", *IEEE CVPR*, San Diego, 2005.

---

## Nhóm thực hiện

**Trần Phan Đăng Khôi** (KHMT, đội trưởng) · **Lại Thế Mạnh Anh** (Kỹ thuật Robot) · **Trần Huỳnh Hân** (Quản trị Kinh doanh)
