# Hướng dẫn chạy và triển khai (Chặng 1)

Toàn bộ lệnh chạy trên Windows PowerShell tại `D:\SV7\SafeRoad-AI`.

## 1. Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,docs]"
python scripts\download_assets.py
pytest tests -q
```

## 2. Chạy lại số liệu

```powershell
python -m saferoad simulate --duration 300 --seed 42
python -m saferoad train-behavior
python -m saferoad evaluate
python -m saferoad evaluate-grid
python -m saferoad prepare-street --annotations data\raw\mvti\merged_annotations.json
python -m saferoad calibrate --points configs\calib_street.yaml --frame data\processed\street\street_labeled_0_300s.mp4 --image docs\assets\calib_street.jpg --out data\outputs\calibration_street.json
python -m saferoad evaluate-real --config configs\street.yaml --weights models\yolo11n.pt --no-db --no-overlay --source data\processed\street\street_labeled_0_300s.mp4 --ground-truth data\processed\street\street_groundtruth.pkl --calibration-report data\outputs\calibration_street.json --out data\outputs\evaluation_street.json
python scripts\street_reference_sensitivity.py
python scripts\export_street_yolo.py --video data\processed\street\street_labeled_0_300s.mp4 --ground-truth data\processed\street\street_groundtruth.pkl --out data\yolo_street
python scripts\train_street_yolo.py
python -m saferoad evaluate-real --config configs\street.yaml --weights models\yolo11n.pt --start-frame 6600 --no-db --no-overlay --source data\processed\street\street_labeled_0_300s.mp4 --ground-truth data\processed\street\street_groundtruth.pkl --calibration-report data\outputs\calibration_street.json --out data\outputs\evaluation_street_pretrained_test.json
python -m saferoad evaluate-real --config configs\street.yaml --weights models\saferoad_street_yolo11n.pt --start-frame 6600 --no-db --no-overlay --source data\processed\street\street_labeled_0_300s.mp4 --ground-truth data\processed\street\street_groundtruth.pkl --calibration-report data\outputs\calibration_street.json --out data\outputs\evaluation_street_finetuned_test.json
python -m saferoad run --config configs\street.yaml --no-overlay --source data\processed\street\street_heldout_300_741s.mp4 --output data\outputs\street_heldout --review-csv data\outputs\review_candidates_street_heldout.csv
python -m saferoad evaluate-real --config configs\mvti.yaml --no-db
python scripts\build_readme_tables.py
node scripts\build_improvement_report.js
```

`node scripts\build_improvement_report.js` cần LibreOffice để xuất PDF; không có thì mở
`.docx` bằng Word và Lưu thành PDF.

## 3. Dashboard tại máy

```powershell
python -m saferoad serve --demo --sample-video data\samples\demo_sample.mp4
```

Mở http://127.0.0.1:8000. Tab **Chạy thử video** chạy pipeline thật trên video tải lên;
tab **Rà soát nhãn** dùng để hai người gán nhãn độc lập.

## 4. Demo chạy thật trên Hugging Face Spaces

1. Tạo Space: https://huggingface.co/new-space → SDK **Docker** → **Blank** → **CPU basic** → Public.
2. Tạo token quyền **Write**: https://huggingface.co/settings/tokens
3. Cài Git LFS một lần: `git lfs install`
4. Đẩy:

```powershell
$env:HF_TOKEN = "hf_xxx"
python scripts\deploy_hf_space.py --space <tên_tài_khoản_hf>/saferoad-ai
```

Space tự dựng Docker trong 5-10 phút. Link demo: `https://huggingface.co/spaces/<tên_tài_khoản_hf>/saferoad-ai`.
Video mẫu `data\samples\demo_sample.mp4` là 20 s cắt từ đoạn held-out (giây 222-242 của
`street_heldout_300_741s.mp4`, chọn đoạn có xe đông để demo có cảnh báo); chỉ đẩy kèm khi
giấy phép của bộ dữ liệu cho phép. Tạo lại:

```powershell
ffmpeg -ss 222 -i data\processed\street\street_heldout_300_741s.mp4 -t 20 -c:v libx264 -crf 27 -preset veryfast -pix_fmt yuv420p -movflags +faststart -an data\samples\demo_sample.mp4
```

## 5. Trang tĩnh (Vercel, tuỳ chọn)

`node scripts\build_static_site.mjs` dựng thư mục `site\`; `vercel.json` đã cấu hình sẵn.

## 6. Đẩy mã lên GitHub

```powershell
git add -A
git add -f models\saferoad_street_yolo11n.pt
git commit -m "Chặng 1: PET đo, đối soát sự kiện, video thật, demo chạy thật"
git tag chang1-v2.0
git push origin main --tags
```
