# Hướng dẫn chạy toàn bộ dự án, từ đầu đến cuối

Đọc kèm:
- `docs/PROTOCOL.md` — luật thí nghiệm (chia fold, cách chấm điểm, cách chọn)
- `docs/PLAN.md` — cấu trúc thư mục, luồng máy ↔ Colab

Ký hiệu: **[máy]** = chạy trên máy cá nhân (CPU đủ). **[Colab]** = chạy trên
Google Colab (cần GPU). ✅ = đã có. ⏳ = chưa viết.

---

## Phần 0 — Chuẩn bị một lần [máy]

```bash
# 0.1  Lấy code
git clone https://github.com/<ban>/THESIS_GRADUATE.git
cd THESIS_GRADUATE

# 0.2  Cài thư viện
pip install numpy scipy matplotlib tqdm torch

# 0.3  Lấy code upstream MobiVital (KHÔNG commit vào repo — họ không có LICENSE)
git clone https://github.com/nesl/mobivital-public.git external/mobivital
#     commit đang dùng: 4319731d2769d4134c92088dd846666e262f18e9

# 0.4  Tải dataset tripod (CSV thô) từ Zenodo, giải nén vào:
#      data/raw/tripod/*.csv        (1874 file, ~13 GB)
```

---

## Phần 1 — Dữ liệu cho PIPELINE TRAIN / VALIDATE [máy]

Đây là pipeline của mình: `.npz` theo từng người, để chia 4 fold.

```bash
# 1.1  Gom CSV chung vào thư mục theo người        ✅ scripts/1_organize_raw.py
python scripts/1_organize_raw.py
#      data/raw/tripod/*.csv  ->  data/raw/A/  data/raw/B/  ...  data/raw/L/

# 1.2  Mỗi thư mục người -> một file .npz            ✅ scripts/2_make_npz.py
python scripts/2_make_npz.py
#      data/raw/A/*.csv  ->  data/processed/by_user/A.npz   (keys: uwb, gt)
#      ... tới L.npz
#      gt đã chuẩn hoá về [-1, 1] bằng đúng công thức self_normalize của MobiVital
```

Kết quả: `data/processed/by_user/A.npz … L.npz` (12 file, ~2.5 GB).

---

## Phần 2 — Dữ liệu cho PIPELINE TEST [máy]

Đây là pipeline MobiVital: chạy `prep_breath_final.py` **nguyên bản, 0 dòng sửa**.

```bash
# 2.1  Dựng sân riêng + chạy script MobiVital        ✅ scripts/3_run_mobivital_prep.py
python scripts/3_run_mobivital_prep.py
```

Script này làm 3 việc, không đụng file của MobiVital:

1. Dựng thư mục tạm `_workdir/` bên trong thư mục kết quả, chứa
   `dataset/mobivital/tripod/*.csv` là **symlink** trỏ về `data/raw/*/*.csv`
   (0 byte thêm). Script MobiVital cần một thư mục phẳng vì nó gọi `os.listdir()`.
2. `cd` vào `_workdir/` rồi gọi `prep_breath_final.py`. Script MobiVital dùng
   đường dẫn tương đối (`./dataset/mobivital/tripod/`, `./data_final/`) nên chỉ
   cần đứng đúng chỗ là nó tự tìm ra.
3. Chuyển 2 file `.npy` nó sinh ra lên thư mục kết quả, xoá `_workdir/`.

Kết quả:

```
data/processed/mobivital_original/training_breath_tripod_data.npy   (8 người ABCDEFKL, gộp)
data/processed/mobivital_original/testing_breath_tripod_data.npy    (GHIJ, gộp)
```

Mỗi `.npy` chứa 2 mảng ghi nối tiếp: `X_uwb` (phức) rồi `y_breath` (đã
`self_normalize`). Đọc lại phải gọi `np.load` hai lần trên cùng file.

---

## Phần 2b — Kiểm tra dữ liệu [máy]

```bash
python scripts/4_check_data.py     # ✅
```

In ra shape/dtype/khoảng giá trị của cả hai pipeline, rồi đối chiếu xem có khớp
nhau không. Chạy xong nên thấy:

```
dev set  (ABCDEFKL) = 1289 session
test set (GHIJ)     =  537 session
tổng cộng           = 1826 session

                       by_user      mobivital
  dev set              1289         1289         KHỚP
  test set             537          537          KHỚP

  Sai lệch gt lớn nhất trên 1500 mẫu: 0.00e+00

KẾT LUẬN: dữ liệu OK, không phát hiện lỗi nào.
```

---

## Phần 3 — Đưa dữ liệu lên Google Drive [máy]

Upload một lần, Colab đọc thẳng, không xử lý lại.

```
Google Drive/
└── mobivital/
    └── processed/
        ├── by_user/                <- copy từ data/processed/by_user/
        │   ├── A.npz … L.npz
        └── mobivital_original/     <- copy từ data/processed/mobivital_original/
            ├── training_breath_tripod_data.npy
            └── testing_breath_tripod_data.npy
```

---

## Phần 4 — Đưa code lên GitHub [máy]

```bash
git add -A
git commit -m "..."
git push
```

`.gitignore` đã chặn `data/`, `external/mobivital/`, `runs/`, `*.npz`, `*.npy` —
chỉ code lên GitHub, dữ liệu ở Drive.

---

## Phần 5 — Chạy thí nghiệm trên Colab [Colab]

Mỗi notebook đầu phiên chạy phần setup (mount Drive + git clone + import).

```python
# ô setup, đầu MỌI notebook                          ⏳ notebooks/0_setup.ipynb
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<ban>/THESIS_GRADUATE.git
%cd /content/THESIS_GRADUATE
!git clone https://github.com/nesl/mobivital-public.git external/mobivital
!pip install -q numpy scipy matplotlib tqdm torch

DRIVE     = '/content/drive/MyDrive/mobivital'
DEV_DATA  = DRIVE + '/processed/by_user'             # A.npz … L.npz
TEST_DATA = DRIVE + '/processed/mobivital_original'  # 2 file .npy
RUN_DIR   = DRIVE + '/runs'                          # nơi lưu kết quả
```

### Thứ tự notebook

| notebook | pipeline | việc | ghi ra |
|---|---|---|---|
| ⏳ `1_explore.ipynb` | dev | xem dữ liệu, vẽ session, sinh hình cho slide | `docs/figures/*.png` |
| ⏳ `tn0_reproduce.ipynb` | **test** | TN0a (checkpoint MobiVital → eval GHIJ, so paper) + TN0b (train lại LSTM 3–4 seed → test GHIJ) | `runs/tn0/` |
| ⏳ `tn1_architecture.ipynb` | dev | TCN thường vs DS-TCN, 4 fold × seed | `runs/tn1/` |
| ⏳ `tn2_revin.ipynb` | dev | người thắng TN1, có/không RevIN | `runs/tn2/` |
| ⏳ `tn3_loss.ipynb` | dev | MSE vs MSE+Pearson (alpha) | `runs/tn3/` |
| ⏳ `tn4_threshold.ipynb` | dev | quét corr-threshold 0.70…0.95 | `runs/tn4/` |
| ⏳ `tn5_hparam.ipynb` | dev | Optuna: channels/kernel/blocks/dropout/lr/wd | `runs/tn5/` |
| ⏳ `tn6_confirm.ipynb` | dev | chạy lại 2 so sánh then chốt TN1–3 ở threshold + HP cuối | `runs/tn6/` |
| ⏳ `final_compare.ipynb` | **test** | LSTM + TCN (cấu hình đã khóa) train full pool → test GHIJ | `runs/final/` |

Mỗi notebook TN1–TN6 ghi lại: cấu hình đầy đủ, seed, commit git, `cv_score`,
`cv_std`, bảng điểm 8 người, `test_GHIJ` — vào file trong `runs/`, không chỉ để
trong output notebook (`docs/PROTOCOL.md` mục 7).

### Luật khi chạy (nhắc lại từ PROTOCOL)

- Mỗi thí nghiệm đổi **đúng một biến**, kế thừa người thắng của thí nghiệm trước.
- Chọn người thắng bằng `cv_score`. `test_GHIJ` chỉ để nhìn xu hướng.
- `test_GHIJ` lúc phát triển thấp hơn số công bố (model fold chỉ train 6 người) —
  bình thường.

---

## Phần 6 — Kết quả cuối [Colab]

`final_compare.ipynb`, chạy **một lần** sau khi TN1–TN6 đã khóa hết cấu hình:

```
cấu hình TCN đã chốt (từ TN1–TN6)
        |
   train lại trên ĐỦ 8 người ABCDEFKL      <- PIPELINE TEST (data từ TEST_DATA)
   train LSTM  trên ĐỦ 8 người ABCDEFKL      <- cùng pipeline
        |
   cả hai test GHIJ  ->  BẢNG SO SÁNH CUỐI
        |
   + hình scatter cv_score vs test_GHIJ (mọi cấu hình đã thử)
```

Báo cáo 3 số: MobiVital công bố (X) · mình dựng lại TN0b (Y ± Z) · TCN của mình
(W ± V).

---

## Tóm tắt một trang

```
[máy]  0  git clone repo + upstream, cài lib, tải CSV vào data/raw/tripod/
[máy]  1  python scripts/1_organize_raw.py      -> data/raw/A..L/
[máy]  1  python scripts/2_make_npz.py          -> data/processed/by_user/*.npz     (pipeline dev)
[máy]  2  python scripts/3_run_mobivital_prep.py    -> data/processed/mobivital_original/*.npy  (pipeline test)
[máy]  2b python scripts/4_check_data.py         -> in bảng kiểm tra, đối chiếu 2 pipeline
[máy]  3  upload data/processed/ lên Google Drive
[máy]  4  git push  (code lên GitHub, data ở Drive)
[Colab]5  0_setup -> 1_explore -> tn0 -> tn1 -> tn2 -> tn3 -> tn4 -> tn5 -> tn6
[Colab]6  final_compare  -> bảng so sánh + hình
```
