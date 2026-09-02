# MobiVital — đồ án tốt nghiệp

Cải tiến mô hình dự báo dạng sóng nhịp thở cho MobiVital (radar UWB không tiếp
xúc): thay LSTM baseline bằng TCN nhân quả có RevIN.

## Chuẩn bị

```bash
pip install numpy

# Upstream MobiVital KHÔNG được sao chép vào repo này (họ không có LICENSE).
# Tải riêng lúc setup:
git clone https://github.com/nesl/mobivital-public.git external/mobivital
```

Bản upstream đang dùng: commit `4319731d2769d4134c92088dd846666e262f18e9`.

Đặt file CSV thô vào `data/raw/tripod/` (tải từ Zenodo, xem README upstream).

## Chạy lần lượt

| # | Lệnh | Việc |
|---|---|---|
| 1 | `python scripts/1_organize_raw.py` | Gom CSV chung → thư mục theo người (`data/raw/A/`, `data/raw/B/`, ...) |
| 2 | `python scripts/2_make_npz.py` | **Pipeline train/validate**: mỗi người → 1 file `.npz` |
| 3 | `python scripts/3_run_mobivital_prep.py` | **Pipeline test**: chạy `prep_breath_final.py` của MobiVital, nguyên bản |
| 4 | `python scripts/4_check_data.py` | Kiểm tra: số session, 4 fold, đối chiếu 2 pipeline |

Muốn đổi đường dẫn hay tuỳ chọn thì sửa thẳng các hằng số ở đầu mỗi script.

## Dữ liệu sau khi chạy

```
data/raw/
├── A/  (225 csv)   ...   L/  (119 csv)

data/processed/
├── by_user/                           ◄── PIPELINE TRAIN / VALIDATE
│   ├── A.npz   uwb (224, 1500, 120) complex64   gt (224, 1500) float32
│   ├── ...
│   └── L.npz
│
└── mobivital_original/                ◄── PIPELINE TEST
    ├── training_breath_tripod_data.npy    X (1289,1500,120) + y (1289,1500)
    └── testing_breath_tripod_data.npy     X  (537,1500,120) + y  (537,1500)
```

`gt` / `y_breath` ở cả hai pipeline đều đã chuẩn hoá về `[-1, 1]` bằng đúng công
thức `self_normalize` của MobiVital.

Session hợp lệ = file CSV đúng 1500 dòng; 48/1874 file bị loại vì thiếu dòng,
còn 1826 (pool `ABCDEFKL` 1289 + test `GHIJ` 537).

## Cấu trúc

```
scripts/                   chuẩn bị và kiểm tra dữ liệu (chạy ở máy)
notebooks/                 chạy trên COLAB: train, thí nghiệm
data/raw/                  CSV thô                       (không commit)
data/processed/by_user/            pipeline train/validate   (không commit → lên Drive)
data/processed/mobivital_original/ pipeline test             (không commit → lên Drive)
external/mobivital/        upstream clone                (không commit)
docs/PROTOCOL.md           LUẬT thí nghiệm: chia fold, cách chấm điểm, cách chọn
docs/RUNBOOK.md            chạy toàn bộ dự án từ đầu đến cuối
docs/PLAN.md               kế hoạch cấu trúc thư mục + luồng máy ↔ Colab
docs/superpowers/          spec thiết kế ban đầu
```

- **Chạy toàn bộ dự án từ đầu đến cuối**: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
- **Trước khi chạy bất kỳ thí nghiệm nào**, đọc [`docs/PROTOCOL.md`](docs/PROTOCOL.md).
- Luồng làm việc máy ↔ Colab, cấu trúc thư mục: [`docs/PLAN.md`](docs/PLAN.md).

## Nguyên tắc

- **Không trộn code upstream với code của đồ án.**
- **`GHIJ` là tập test**, chỉ chạm một lần ở bước cuối. Pool phát triển là
  `ABCDEFKL` (8 người).
