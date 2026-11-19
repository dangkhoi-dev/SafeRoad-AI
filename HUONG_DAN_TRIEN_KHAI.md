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
python scripts\export_street_yolo.py --video data\processed\mvti\infrastructure.mp4 --ground-truth data\processed\mvti\infrastructure_groundtruth.pkl --out data\yolo_mvti --prefix mvti --train-end 1500 --test-start 1700 --train-step 5 --val-step 15
python scripts\train_street_yolo.py --data data\yolo_mvti\data.yaml --epochs 15 --name mvti --out models\saferoad_mvti_yolo11n.pt --results-csv data\outputs\train_mvti_yolo11n_results.csv
python -m saferoad evaluate-real --config configs\mvti.yaml --weights models\yolo11n.pt --no-db --no-overlay --start-frame 1700 --seen-before 1500 --out data\outputs\evaluation_mvti_pretrained_test.json
python -m saferoad evaluate-real --config configs\mvti.yaml --weights models\saferoad_mvti_yolo11n.pt --no-db --no-overlay --start-frame 1700 --seen-before 1500 --out data\outputs\evaluation_mvti_finetuned_test.json
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

## 4. Demo chạy thật trên Modal

Hugging Face chỉ cho chạy Space Docker với tài khoản PRO (từ 7/2026), nên bản demo
đặt trên Modal: 30 USD credit/tháng, tự tắt khi không ai truy cập.

```powershell
pip install modal
modal setup
modal deploy scripts\deploy_modal.py
```

Lần đầu dựng image mất khoảng 5 phút; link in ra có dạng
`https://<workspace>--saferoad-ai.modal.run`. Lần mở đầu tiên sau khi container đã tắt
chờ khoảng 30 giây. Có PRO thì vẫn dùng được `python scripts\deploy_hf_space.py`.

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
