# Hướng dẫn triển khai — từng bước một

Tài liệu này dành riêng cho Khôi, để đưa dự án từ trạng thái hiện tại lên GitHub
và chạy được toàn bộ trên máy của bạn.

**Trạng thái hiện tại:** toàn bộ mã, tài liệu và 17 commit đã sẵn sàng trong
`D:\SV7\SafeRoad-AI`. Việc còn lại chỉ là cài môi trường, chạy thử và push.

---

## Phần 0 — Chuẩn bị (một lần, ~5 phút)

Mở **PowerShell** hoặc **Command Prompt**, rồi:

```powershell
cd D:\SV7\SafeRoad-AI

REM Kiểm tra Python (cần 3.10 trở lên)
python --version

REM Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate

REM Cài đặt (mất 3-5 phút vì phải tải PyTorch)
REM Dấu ngoặc kép bắt buộc trên PowerShell, [dev,docs] kéo thêm pytest + thư viện sinh báo cáo
pip install -e ".[dev,docs]"

REM Tải trọng số YOLO11n (~5.6 MB)
python scripts\download_assets.py
```

Nếu `python` không chạy được, thử `py -3` thay thế.

---

## Phần 1 — Chạy thử để chắc mọi thứ hoạt động (~6 phút)

```powershell
REM 1.1 Chạy bộ kiểm thử — phải thấy "78 passed"
pytest tests\ -q

REM 1.2 Sinh video mô phỏng + nhãn chuẩn (~2 phút)
python -m saferoad simulate --duration 300 --seed 42

REM 1.3 Chạy pipeline end-to-end (~1 phút)
python -m saferoad run --config configs\synthetic.yaml ^
    --ground-truth data\samples\synthetic_groundtruth.pkl --replay-detections

REM 1.4 Chấm điểm + bảng ablation (~3 phút)
python -m saferoad evaluate

REM 1.5 Mở dashboard
python -m saferoad serve
```

Rồi mở trình duyệt vào **http://127.0.0.1:8000** — bạn sẽ thấy dashboard 7 tab.
Nhấn `Ctrl+C` trong terminal để dừng.

**Nếu bước 1.2–1.4 chạy đúng, toàn bộ hệ thống đã hoạt động.**

---

## Phần 2 — Đẩy lên GitHub (~3 phút)

### 2.1 Tạo Personal Access Token

GitHub không còn cho dùng mật khẩu tài khoản để push. Bạn cần một token:

1. Vào https://github.com/settings/tokens?type=beta
2. Bấm **Generate new token**
3. Điền:
   - **Token name:** `saferoad-push`
   - **Expiration:** 30 days (hoặc tuỳ bạn)
   - **Repository access:** Only select repositories → chọn **SafeRoad-AI**
   - **Permissions** → Repository permissions → **Contents: Read and write**
4. Bấm **Generate token** và **copy chuỗi token** (chỉ hiện một lần)

### 2.2 Push

```powershell
cd D:\SV7\SafeRoad-AI
scripts\push_to_github.bat
```

Khi được hỏi:
- **Username:** `dangkhoi-dev`
- **Password:** dán **token** vừa copy (không phải mật khẩu tài khoản)

Hoặc làm thủ công:

```powershell
git remote add origin https://github.com/dangkhoi-dev/SafeRoad-AI.git
git push -u origin main
```

### 2.3 Nếu repo trên GitHub đã có sẵn file

Nếu bạn đã tạo repo kèm README hoặc .gitignore, push sẽ bị từ chối. Xử lý:

```powershell
git pull origin main --allow-unrelated-histories
REM Nếu có xung đột, giữ bản của mình:
git checkout --ours .
git add -A
git commit -m "merge: gộp với repo khởi tạo trên GitHub"
git push -u origin main
```

### 2.4 Kiểm tra

Mở https://github.com/dangkhoi-dev/SafeRoad-AI — phải thấy README hiển thị đầy đủ
kèm sơ đồ kiến trúc, và 18 commit trong lịch sử.

---

## Phần 3 — Chạy trên dataset thật (~20 phút)

### 3.1 Sắp xếp dữ liệu đã tải về

```powershell
REM Xem trước — KHÔNG đụng vào file nào
python scripts\organize_dataset.py --source D:\SV7 --dry-run

REM Nếu bảng xem trước trông đúng, chạy thật (di chuyển, gần như tức thời)
python scripts\organize_dataset.py --source D:\SV7
```

Kết quả sẽ là:

```
D:\SV7\SafeRoad-AI\data\raw\
├── mvti\              Infrastructure, Drone, *-mscoco.json
├── ucsd-highway\      video\, info.txt, *.mat
└── own-footage\       input-001.MOV, output.mp4
```

Nếu muốn giữ nguyên bản gốc thì thêm `--copy` (tốn thêm ~11 GB dung lượng).

### 3.2 Dựng video + nhãn từ dataset giao lộ thật

```powershell
python -m saferoad prepare-real --root data\raw\mvti --view Infrastructure
```

Bước này ghép 2.441 khung hình thành một video MP4 và trích 19 quỹ đạo chuẩn.
Mất khoảng 2-3 phút.

### 3.3 Chấm điểm trên dữ liệu thật

```powershell
python -m saferoad evaluate-real
```

Bạn sẽ nhận được mAP detection và IDF1 tracking trên **ảnh thật**.

⚠️ **Con số mAP sẽ thấp** (khoảng 0.09). Đây **không phải lỗi** — đó là phát hiện
thật của dự án: YOLO11n với trọng số COCO gốc không hợp với góc nhìn camera giao
thông. Xem Phần 4 để khắc phục.

Muốn thử ngay cách cải thiện tạm thời (cắt ô, chậm hơn ~6 lần nhưng recall gấp
2,4 lần):

```powershell
python -m saferoad evaluate-real --config configs\mvti.yaml ^
    --out data\outputs\evaluation_real_tiled.json
```

---

## Phần 4 — Fine-tune YOLO trên Colab (~1 giờ, cần GPU)

Đây là bước **nâng chất lượng lớn nhất** cho phần dữ liệu thật.

### 4.1 Xuất dataset sang định dạng YOLO

```powershell
python scripts\export_yolo_dataset.py --root data\raw\mvti --view Infrastructure --out data\yolo_mvti

REM Nén lại để tải lên Colab
python -c "import shutil; shutil.make_archive('yolo_mvti','zip','data/yolo_mvti')"
```

### 4.2 Tải lên Google Drive

Tải file `yolo_mvti.zip` vừa tạo lên Google Drive (thư mục gốc MyDrive).

### 4.3 Chạy notebook

1. Mở https://colab.research.google.com
2. **File → Upload notebook** → chọn `notebooks\01_finetune_yolo11n_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Chạy lần lượt từng ô. Bước train mất khoảng 40-60 phút.

Notebook đã xử lý sẵn hai điểm quan trọng:
- Đo **baseline trước khi train** để có số liệu so sánh cho báo cáo.
- Trộn thêm mẫu COCO có `person` và `motorcycle`, vì MVTI **không có** hai lớp
  này — nếu không, model fine-tune sẽ quên mất người đi bộ và xe máy.

### 4.4 Đưa trọng số mới về và chạy lại

```powershell
REM Chép file best.pt tải từ Colab vào models\
copy %USERPROFILE%\Downloads\best.pt models\saferoad_yolo11n.pt

python -m saferoad evaluate-real ^
    --weights models\saferoad_yolo11n.pt ^
    --out data\outputs\evaluation_real_finetuned.json
```

Bây giờ bạn có hai file: `evaluation_real.json` (trước) và
`evaluation_real_finetuned.json` (sau). Đó chính là bảng **"trước / sau
fine-tune"** để đưa vào báo cáo.

### 4.5 Cập nhật lại báo cáo

```powershell
node scripts\build_report.js
```

Script này đọc thẳng số liệu từ các file JSON, nên báo cáo tự cập nhật theo kết
quả mới. Không cần sửa tay con số nào.

---

## Phần 5 — Chạy trên video tự quay của nhóm

```powershell
REM Bước 1: ẩn danh (BẮT BUỘC theo Điều 5 Thể lệ)
python -m saferoad anonymize ^
    --source data\raw\own-footage\hangxanh.mp4 ^
    --out data\processed\hangxanh_anon.mp4

REM Bước 2: chạy phân tích
python -m saferoad run --source data\processed\hangxanh_anon.mp4 ^
    --output data\outputs\hangxanh
```

### Calibrate homography cho hiện trường thật

Đây là bước quyết định độ chính xác của TTC. Mở `configs\hangxanh.yaml` và sửa:

```yaml
homography:
  image_points:            # toạ độ pixel của 4 điểm mốc trên video
    - [320, 690]
    - [960, 690]
    - [820, 300]
    - [460, 300]
  world_points:            # toạ độ mét thật của đúng 4 điểm đó
    - [-9.0, 4.0]
    - [9.0, 4.0]
    - [9.0, 26.0]
    - [-9.0, 26.0]
```

Cách lấy 4 điểm mốc:
1. Chụp một khung hình từ video, mở bằng phần mềm xem ảnh có hiển thị toạ độ chuột.
2. Chọn 4 điểm dễ nhận (góc vạch kẻ, góc đảo giao thông, mép vạch dừng), ghi lại
   toạ độ pixel.
3. Dùng Google Maps (chế độ đo khoảng cách) để đo khoảng cách thật giữa các điểm
   đó, quy về hệ toạ độ mét với gốc tuỳ chọn.
4. Điền vào file cấu hình.

Sau đó chạy với `--config configs\hangxanh.yaml`. Sai số tái chiếu được in ra khi
chạy — nếu lớn hơn 0,5 m thì cần đo lại điểm mốc.

---

## Phần 6 — Chuẩn bị hồ sơ nộp

### Checklist

| Hạng mục | Trạng thái | Việc cần làm |
|---|---|---|
| Tài liệu PDF Bảng C | ✅ Có sẵn | `docs\BaoCao_SafeRoadAI_BangC.pdf` (13 trang) |
| Mã nguồn công khai | ⬜ Bạn làm | Đẩy lên GitHub (Phần 2), đặt repo **public** |
| Protocol near-miss | ✅ Có sẵn | `docs\protocol_near_miss_v1.md` |
| Bảng kê dataset + license | ✅ Có sẵn | `docs\dataset_license.md` |
| Bản kê khai công cụ AI | ⚠️ Cần điền | `docs\ke_khai_cong_cu.md` — điền tên công cụ nhóm dùng |
| Prompt Log | ⚠️ Cần điền | `docs\prompt_log.md` — xuất hội thoại, tải Drive, **mở quyền xem** |
| Video thuyết trình ≤5' | ⬜ Team quay | Kịch bản: `docs\video_script.md` |
| Video demo ≤5' | ⬜ Team quay | Kịch bản: `docs\video_script.md` |
| Sơ đồ kiến trúc | ✅ Có sẵn | `docs\assets\architecture.png` |
| Kết quả đánh giá | ✅ Có sẵn | `data\outputs\evaluation.json` |

### Ba việc bắt buộc phải tự làm

1. **Prompt Log** — mở `docs\prompt_log.md`, làm theo hướng dẫn trong file. Nhớ
   **mở quyền truy cập Drive** và kiểm tra bằng cửa sổ ẩn danh.
2. **Quay 2 video** — `docs\video_script.md` có kịch bản chi tiết theo từng phút,
   đã phân vai cho cả 3 thành viên. Điều Thể lệ yêu cầu: **cả 3 phải xuất hiện**
   và trình bày phần mình phụ trách.
3. **Đặt repo GitHub thành public** — Settings → General → Danger Zone → Change
   visibility.

---

## Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: saferoad` | Chưa kích hoạt venv hoặc chưa cài | `.venv\Scripts\activate` rồi `pip install -e ".[dev,docs]"` |
| `pytest ... is not recognized` | Cài thiếu nhóm `dev` | `pip install -e ".[dev,docs]"` |
| `Không nạp được behavior model (No module named '_loss')` | Model train bằng scikit-learn khác phiên bản máy đang chạy | `saferoad train-behavior` (2-3 phút, train lại tại chỗ) |
| Cột FPS trong bảng ablation thấp bất thường (< 10) | Máy ngủ giữa lúc đo (chỉ ảnh hưởng bản trước v1.0.1) | Cập nhật code mới nhất — FPS nay đo bằng trung vị độ trễ từng frame nên miễn nhiễm với việc máy ngủ |
| `Không tìm thấy trọng số models/yolo11n.pt` | Chưa tải model | `python scripts\download_assets.py` |
| `Không mở được video` | Sai đường dẫn | Kiểm tra `video.source` trong file cấu hình |
| Chạy rất chậm | Đang chạy CPU | Bình thường. Thêm `--max-frames 900` để rút ngắn, hoặc `--device cuda:0` nếu có GPU NVIDIA |
| `git push` báo 403 | Token sai hoặc thiếu quyền | Tạo lại token với quyền **Contents: Read and write** |
| `git push` báo "rejected" | Repo trên GitHub đã có commit | Xem mục 2.3 |
| Dashboard trắng trang | Chưa có `results.json` | Chạy `python -m saferoad run …` trước |
| mAP trên dữ liệu thật rất thấp | Lệch miền của YOLO COCO | Đây là kết quả đúng — xem Phần 4 để fine-tune |

---

## Tóm tắt: chuỗi lệnh ngắn nhất để có mọi thứ

```powershell
cd D:\SV7\SafeRoad-AI
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev,docs]"
python scripts\download_assets.py

pytest tests\ -q
python -m saferoad simulate --duration 300 --seed 42
python -m saferoad run --config configs\synthetic.yaml --ground-truth data\samples\synthetic_groundtruth.pkl --replay-detections
python -m saferoad evaluate
python -m saferoad serve

REM ở cửa sổ khác:
python scripts\organize_dataset.py --source D:\SV7 --dry-run
python scripts\organize_dataset.py --source D:\SV7
python -m saferoad prepare-real --root data\raw\mvti --view Infrastructure
python -m saferoad evaluate-real

scripts\push_to_github.bat
```
