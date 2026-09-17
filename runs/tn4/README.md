# TN4 — kiểm tra trên G H I J

Train đủ tám người ABCDEFKL rồi chấm một lần trên 537 phiên của G H I J. Bốn
cấu hình, mỗi cấu hình 3 seed. G H I J không dùng để chọn bất cứ thứ gì.

**macro** = trung bình theo người. **micro** = trung bình toàn bộ buổi ghi.

## Kết quả

| thư mục | | tham số | macro (3 seed) | micro (3 seed) | seed 0 / 1 / 2 |
|---|---|---:|---:|---:|---|
| `LSTM-352/` | mốc MobiVital, giữ nguyên | 1.502.713 | **0.810302 ± 0.015402** | 0.805309 ± 0.016308 | 0.8000 / 0.8029 / 0.8280 |
| `64-61__hybrid-a0.6/` | DS-TCN 64/RF61 · loss lai alpha 0,6 ★ **mô hình cuối** | **37.081** | **0.803590 ± 0.015350** | 0.798315 ± 0.016034 | 0.7869 / 0.8171 / 0.8068 |
| `64-121__pearson/` | DS-TCN 64/RF121 · Pearson thuần | 38.105 | **0.801739 ± 0.009968** | 0.796845 ± 0.010557 | 0.8058 / 0.7904 / 0.8090 |
| `64-121__mse/` | DS-TCN 64/RF121 · MSE thuần — đối chứng loss | 38.105 | **0.762191 ± 0.021433** | 0.755965 ± 0.022176 | 0.7814 / 0.7660 / 0.7391 |

## Hai điều đọc được

**Hàm loss có tác dụng rõ nhất.** Cùng kiến trúc 64/RF121, chỉ đổi loss: Pearson
thuần **0.801739** so với MSE thuần **0.762191** — chênh **+0,039548**, thắng cả
ba seed. Đây là phép so sạch nhất trong đồ án: cùng kiến trúc, cùng ba seed,
cùng dữ liệu, khác đúng một thứ.

**LSTM 352 đứng đầu, không phải DS-TCN.** Mô hình cuối thấp hơn mốc 0,0067 —
nhỏ hơn dao động seed của cả hai bên (0,0154), nên không xếp hạng được. Phát
biểu đúng là **ngang điểm với ít hơn 40,5 lần tham số**. Ngược với thứ hạng ở
TN1 trên tập phát triển; xem [docs/BANG_TCN.md](../../docs/BANG_TCN.md) mục 5.

## Cấu trúc thư mục

```
runs/tn4/
   <cấu hình>/
      seed0/  seed1/  seed2/
         final.pth       trọng số model
         curve.csv       loss 20 epoch
         scores.csv      Pearson 537 buổi ghi G H I J
         selection.txt   bảng lựa chọn kênh, 537 dòng
   summary.csv           12 dòng metric, cột run_id đầy đủ
   SOURCE_MANIFEST.json  SHA-256 mọi tệp của ba cấu hình DS-TCN
```

## Ghi chú về `LSTM-352/`

Ba lượt chạy của mốc này đặt `--experiment tn1_ghij` lúc chạy, nên log trong
`TN1_LSTM.ipynb` in ra `runs/tn1_ghij/`. Kết quả cùng giao thức với ba cấu hình
còn lại nên được xếp chung vào đây; cột `experiment` trong `summary.csv` đã đổi
thành `tn4`. `run_id` giữ nguyên, không đụng tới.

`SOURCE_MANIFEST.json` chỉ phủ ba thư mục DS-TCN — nó được lập lúc nhập kết quả
TN4, trước khi mốc LSTM chuyển vào.

## Dựng lại bảng

```bash
python3 scripts/compare_cv.py --experiment tn4 --final
```
