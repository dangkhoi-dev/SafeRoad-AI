# data/raw — dữ liệu thô

Thư mục này **không được đưa lên git** (quá lớn). Cấu trúc mong đợi:

```
data/raw/
├── mvti/                    # Multi-view Traffic Intersection Dataset
│   ├── Infrastructure/      #   ảnh từ camera hạ tầng
│   ├── Drone/               #   ảnh từ drone
│   ├── infrastructure-mscoco.json
│   └── drone-mscoco.json
├── ucsd-highway/            # UCSD Highway Traffic Database
│   ├── video/               #   254 clip .avi
│   └── info.txt
└── own-footage/             # video tự quay của nhóm
```

Sắp xếp tự động từ thư mục tải về:

```bash
python scripts/organize_dataset.py --source <thư mục tải về> --dry-run
python scripts/organize_dataset.py --source <thư mục tải về>
```

Chi tiết nguồn và giấy phép: `docs/dataset_license.md`.
