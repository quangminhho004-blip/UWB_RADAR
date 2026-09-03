# runs/ — mỗi thực nghiệm một thư mục

Mọi thứ của một thực nghiệm nằm chung một chỗ: checkpoint, đường cong loss, bảng
lựa chọn kênh, điểm từng buổi ghi, metric.

```
runs/
├── summary.csv          bảng metric chung cả đồ án, mỗi lần chạy một dòng, 27 cột
│
├── tn0/                 <- notebooks/TN0.ipynb
│   ├── TN0a.txt  TN0b.txt  TN0c.txt      lựa chọn kênh, pipeline MobiVital
│   ├── ours_b.txt  ours_c.txt            lựa chọn kênh, pipeline đồ án
│   ├── scores_*.csv                      điểm từng buổi ghi (537 dòng)
│   ├── compare.csv                       bảng ĐẠT / KHÔNG ĐẠT
│   ├── summary.csv  README.txt           metric và ghi chú, do save_results.py sinh
│   └── ours_c/final.pth  curve.csv       trọng số và loss từng epoch
├── tn0.zip
│
├── tn1/                 <- scripts/run_cv.py --experiment tn1
│   ├── <cấu hình>_val_AB/final.pth  curve.csv
│   ├── <cấu hình>_val_CE/...
│   ├── scores_<cấu hình>_val_AB.csv
│   └── summary.csv  README.txt
├── tn1.zip
│
└── tn7/                 <- scripts/run_final_test.py --experiment tn7
```

`--experiment` là **bắt buộc** ở `run_cv.py` và `run_final_test.py`; nó quyết định
tên thư mục. Không có thùng dùng chung, không thực nghiệm nào ghi đè thực nghiệm
khác.

## Nén và mang đi

```bash
python scripts/save_results.py tn0     # -> runs/tn0.zip
unzip tn0.zip -d runs/                 # bung lại đúng chỗ cũ
```

Tệp nén **tự chứa đủ**: `summary.csv` bên trong đã lọc sẵn các dòng metric của
riêng thực nghiệm đó, tải về đọc được ngay mà không cần bảng chung.

## Cái gì lên GitHub, cái gì không

| | |
|---|---|
| lên GitHub | `.txt`, `.csv` — nhỏ, là bằng chứng cho hội đồng xem |
| không lên | `*.pth` (trọng số), `*.zip` — xem `.gitignore` |

Mã băm dữ liệu nằm riêng ở `data/checksums.txt` vì nó mô tả dữ liệu, không thuộc
thực nghiệm nào.
