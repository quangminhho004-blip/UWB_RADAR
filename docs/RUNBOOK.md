# Chạy toàn bộ dự án

Máy cá nhân chỉ để **viết code và đẩy lên GitHub**. Mọi thứ chạy trên **Google Colab**,
dữ liệu và kết quả để trên **Google Drive**.

```
GitHub      code + notebook + docs                     vài MB
Drive       dữ liệu đã xử lý + checkpoint + kết quả    ~4.5 GB
Colab       clone code, clone MobiVital, chạy          xoá sạch sau mỗi phiên
```

---

## Phần 0 — Phiên Colab đầu tiên: chuẩn bị dữ liệu

Chạy **một lần duy nhất**, mất 30–45 phút. Tải dataset thẳng từ Zenodo xuống Colab,
không upload từ máy — mạng Colab nhanh hơn nhiều.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/quangminhho004-blip/UWB_RADAR.git /content/UWB_RADAR
%cd /content/UWB_RADAR
!git clone https://github.com/nesl/mobivital-public.git external/mobivital
!pip install -q einops

# Tải dataset (5.7 GB) rồi giải nén THẲNG vào thư mục MobiVital (13 GB)
# Zip đã có sẵn thư mục tripod/ nên giải nén vào .../mobivital/, không vào .../tripod/
!mkdir -p external/mobivital/dataset/mobivital
!wget -q --show-progress -O tripod.zip https://zenodo.org/records/15022885/files/tripod.zip
!unzip -q tripod.zip -d external/mobivital/dataset/mobivital/

# Dữ liệu cho pipeline gốc — script của MobiVital, nguyên bản
!python scripts/mobivital/setup_dataset.py
!cd external/mobivital && python dataset_preparation/prep_breath_final.py

# Dữ liệu của mình, đọc đúng bộ CSV đó
!python scripts/make_npz.py
!python scripts/check_data.py
!python scripts/make_windows.py

# Cất những gì cần giữ sang Drive
!mkdir -p /content/drive/MyDrive/mobivital
!tar -czf /content/drive/MyDrive/mobivital/windows.tar.gz -C data/processed windows
!tar -czf /content/drive/MyDrive/mobivital/by_user.tar.gz -C data/processed by_user
```

Bước 4 phải in ra `sai lệch gt lớn nhất = 0.00e+00`. Không phải thì dừng lại.

### Trên Drive sẽ có

```
MyDrive/mobivital/
├── windows.tar.gz     ~200 MB   cửa sổ cắt sẵn, để train
├── by_user.tar.gz     ~2 GB     uwb phức thô, để chấm điểm
└── runs/                        checkpoint và kết quả, sinh ra dần
```

---

## Phần 1 — Mọi phiên Colab sau: ô setup

Dán vào đầu mọi notebook. Mất khoảng 2 phút.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/quangminhho004-blip/UWB_RADAR.git /content/UWB_RADAR
%cd /content/UWB_RADAR
!git clone https://github.com/nesl/mobivital-public.git external/mobivital
!pip install -q einops

!mkdir -p data/processed
!tar -xzf /content/drive/MyDrive/mobivital/windows.tar.gz -C data/processed/
!tar -xzf /content/drive/MyDrive/mobivital/by_user.tar.gz -C data/processed/

# runs/ trỏ thẳng vào Drive: Colab ngắt phiên thì checkpoint vẫn còn
!mkdir -p /content/drive/MyDrive/mobivital/runs
!ln -s /content/drive/MyDrive/mobivital/runs runs

import sys
sys.path.insert(0, '/content/UWB_RADAR')
from src import mobivital_reference as mv
mv.info()
```

`src/mobivital_reference.py` tự nhận biết Colab qua sự tồn tại của `/content`, không phải
sửa gì.

**Điểm quan trọng:** `runs/` là lối tắt vào Drive. Colab hay ngắt phiên giữa chừng —
checkpoint ghi vào đó thì phiên sau chạy tiếp được.

### Kiểm tra GPU

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Phải ra `True`. Không thì **Runtime → Change runtime type → T4 GPU**.

---

## Phần 2 — Thứ tự notebook

| notebook | việc | pipeline |
|---|---|---|
| `TN0.ipynb` | dựng lại kết quả MobiVital bằng code gốc, rồi chứng minh pipeline mình cho ra đúng số đó | cả hai |
| `tn1_cv.ipynb` | TCN vs DS-TCN | dev, 4 fold |
| `tn2_revin.ipynb` | có / không RevIN | dev, 4 fold |
| `tn3_loss.ipynb` | MSE vs MSE + Pearson | dev, 4 fold |
| `tn4_threshold.ipynb` | quét ngưỡng lọc 0.70 … 0.95 | dev, 4 fold |
| `tn5_hparam.ipynb` | Optuna | dev, 4 fold |
| `tn6_confirm.ipynb` | kiểm lại kết luận TN1–TN3 ở cấu hình cuối | dev, 4 fold |
| `final_test.ipynb` | train đủ 8 người rồi test GHIJ | gốc |

Luật khi chạy, chi tiết ở `docs/PROTOCOL.md`:

- mỗi thí nghiệm đổi **đúng một biến**, kế thừa người thắng của thí nghiệm trước
- chọn người thắng bằng `cv_score` (4 fold trên `ABCDEFKL`)
- `test_GHIJ` chỉ để nhìn xu hướng, **không được dùng để chọn**

---

## Phần 3 — Lấy kết quả về máy

Tuỳ ý, chỉ khi cần. Mọi thứ đã nằm trên Drive.

```
MyDrive/mobivital/runs/
├── tn1/checkpoints/*.pth
├── tn1/scores.csv
├── tn2/...
└── final/...
```

---

## Phần 4 — Riêng TN0 và TN0.1 nên chạy ở máy

Hai thí nghiệm này **đối chiếu số với nhau**, mà bước chọn kênh là `argmax`: hai ứng viên
gần bằng điểm nhau thì GPU và CPU cho thứ hạng khác nhau, vì cộng số theo thứ tự khác.

Đã đo ở TN0: cùng checkpoint, bản chạy GPU và bản chạy CPU chọn khác kênh ở 251/537 buổi ghi.

Nên muốn so `trùng 537/537` thì phải **cùng loại thiết bị**. Chạy ở máy (CPU) là đơn giản
nhất. Nếu buộc phải chạy trên Colab thì ép CPU trước khi chấm.

Sau khi TN0.1 xanh thì mọi thí nghiệm sau chạy Colab GPU thoải mái.

---

## Tóm tắt một trang

```
[máy]   viết code -> git push

[Colab] phiên 1  tải Zenodo -> scripts 1..5 -> cất .tar.gz lên Drive     45 phút
[Colab] mọi phiên sau  ô setup 2 phút -> chạy notebook

[Colab] TN0 -> tn1 -> tn2 -> tn3 -> tn4 -> tn5 -> tn6 -> final_test
```
