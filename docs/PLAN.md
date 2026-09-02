# Kế hoạch cấu trúc repo

## Luồng làm việc

```
        MÁY (CPU)                          COLAB (GPU)
  ┌───────────────────┐            ┌──────────────────────────┐
  │ scripts/1_...     │            │ notebooks/0_setup.ipynb  │
  │ scripts/2_...     │            │  - mount Google Drive    │
  │   ↓               │            │  - git clone repo        │
  │ data/processed/by_user/   │            │  - import mv_*           │
  │   *.npz  (2.5 GB) │            │   ↓                      │
  └────────┬──────────┘            │ notebooks/2_train...     │
           │ upload 1 lần          │ notebooks/3_exp...       │
           ▼                       │   ↓                      │
  Google Drive:                    │ Drive: runs/  (kết quả)  │
    mobivital/by_user/*.npz  ────┘                          │
                                   └──────────────────────────┘
```

- **Chuẩn bị dữ liệu**: chạy ở máy (chỉ đọc/cắt CSV, không cần GPU). Xong upload
  12 file `.npz` lên Google Drive một lần, không xử lý lại.
- **Train + thí nghiệm**: chạy trên Colab. Mỗi thực nghiệm ~3 tiếng.
- **Kết quả** (checkpoint, log, biểu đồ): lưu về Google Drive để không mất khi
  Colab ngắt.

## Cấu trúc thư mục

```
THESIS_GRADUATE/
├── README.md
├── .gitignore
│
├── mv_data.py            # đọc .npz, ghép fold train/val
├── mv_model.py           # TCN nhân quả + RevIN
├── mv_eval.py            # metric, official score (giống CELL 9 của notebook cũ)
│                         #   → cả scripts/ lẫn notebooks/ đều "import mv_data" v.v.
│                         #   → KHÔNG phải package, chỉ là file .py ở gốc repo
│
├── scripts/              # chạy ở MÁY, một lần, chuẩn bị dữ liệu
│   ├── 1_organize_raw.py     gom CSV chung → data/raw/A/ ... L/
│   └── 2_make_npz.py         data/raw/X/*.csv → data/processed/by_user/X.npz (uwb, gt)
│
├── notebooks/            # chạy trên COLAB
│   ├── 0_setup.ipynb        mount Drive + git clone + pip install + import mv_*
│   ├── 1_explore.ipynb      xem thử dữ liệu, vẽ vài session
│   ├── 2_train_one_fold.ipynb   train 1 fold để kiểm tra pipeline chạy được
│   └── 3_exp_architecture.ipynb thí nghiệm 1: TCN vs DS-TCN
│
├── data/                 # KHÔNG commit (xem .gitignore)
│   ├── raw/A/ ... L/         CSV thô
│   └── processed/A.npz ...   → upload lên Google Drive
│
├── external/mobivital/   # KHÔNG commit — clone riêng, commit 4319731
│
└── docs/
    ├── PLAN.md              (file này)
    └── superpowers/         spec thiết kế ban đầu
```

## Vì sao "vài file phẳng" mà không phải package

- `import mv_data` chạy được ở cả máy lẫn Colab, miễn là thư mục hiện tại là gốc
  repo. Không cần `__init__.py`, không cần `pip install -e .`.
- Mỗi file một việc → người mới đọc biết ngay tìm hàm ở đâu.
- Khi một file phình quá to mới tách nhỏ tiếp.

## Notebook trên Colab tái sử dụng code thế nào

`notebooks/0_setup.ipynb`, chạy đầu mỗi phiên:

```python
from google.colab import drive
drive.mount('/content/drive')

# 1. Code: clone repo từ GitHub (mỗi phiên clone lại cho mới)
!git clone https://github.com/<ban>/THESIS_GRADUATE.git
%cd /content/THESIS_GRADUATE
!pip install -q torch numpy

# 2. Dữ liệu: .npz đã để sẵn trên Drive, KHÔNG tạo lại
DATA_DIR = '/content/drive/MyDrive/mobivital/by_user'

# 3. Kết quả: lưu về Drive để sống sót khi runtime ngắt
RUN_DIR = '/content/drive/MyDrive/mobivital/runs'
```

Các notebook sau:

```python
%cd /content/THESIS_GRADUATE
import mv_data, mv_model, mv_eval

x, y = mv_data.load_user(DATA_DIR, "A")
```

## Notebook visualize (làm sau, ghi ra đây để không quên)

Sẽ có một thư mục riêng — `visualize/` — chứa notebook đi qua **từng bước xử lý
dữ liệu**, bắt đầu từ CSV thô:

```
visualize/
└── 1_data_pipeline.ipynb
```

Mỗi bước nhỏ trình bày **số và sóng song song nhau**: bên trái in giá trị thật
(vài dòng của mảng, shape, dtype), bên phải vẽ đồ thị tương ứng. Hội đồng nhìn
con số và hình dạng cùng lúc, tự đối chiếu được.

Các bước cần đi qua:

1. File CSV thô — `head()`, 1500 dòng x 254 cột
2. Bố cục cột — khoanh vùng `12:132` (thực), `132:252` (ảo), cột `-2` (nhịp thở)
3. Tách phần thực
4. Tách phần ảo
5. Ghép thành số phức, xem biên độ và pha
6. Ground truth — vẽ sóng 30 giây, đếm số nhịp thở, đối chiếu tần số
7. Kết quả `.npz` — shape cuối cùng
8. Thống kê toàn bộ — bảng session mỗi người, các file bị loại

Kèm widget chọn 1 file CSV bất kỳ để xem lại toàn bộ các bước trên cho file đó.

Hình sinh ra lưu vào `docs/figures/` để nhét thẳng vào slide.

## Các bước triển khai (làm lần lượt, dừng cho review từng bước)

1. **[xong]** `scripts/1_organize_raw.py`, `scripts/2_make_npz.py` — pipeline
   train/validate, đã chạy, `data/processed/by_user/*.npz` đã có.
2. **[xong]** `scripts/3_run_mobivital_prep.py` — pipeline test, chạy script gốc của
   MobiVital nguyên bản, ra 2 file `.npy` trong `data/processed/mobivital_original/`.
3. `visualize/1_data_pipeline.ipynb` — notebook đi qua từng bước xử lý (mục trên).
4. `mv_data.py` — `load_user`, ghép fold theo `AB / CE / DF / KL`.
5. `mv_model.py` — TCN + RevIN (port từ CELL 8).
6. `mv_eval.py` — official score (port từ CELL 9).
7. `notebooks/0_setup.ipynb` — bootstrap Colab.
8. `notebooks/tn0_reproduce.ipynb` — TN0a + TN0b.

## Protocol (đã chốt, chưa vào code)

- Pool phát triển `ABCDEFKL` (8 người). Test `GHIJ` (4 người), chạm 1 lần ở cuối.
- 4 fold, ghép cặp `AB / CE / DF / KL` (mỗi cặp = 1 người nhiều dữ liệu + 1 người
  ít, để 4 fold có lượng train xấp xỉ nhau).
- Điểm CV = trung bình điểm-mỗi-người qua 8 người pool. Báo cáo kèm độ lệch chuẩn.
- Thứ tự thí nghiệm: 1 kiến trúc, 2 RevIN, 3 loss, 4 corr-threshold, 5 hyperparameter.
```
