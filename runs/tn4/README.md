# TN4 — test cuối trên GHIJ

Train đủ ABCDEFKL, chấm một lần trên GHIJ. Bốn cấu hình, mỗi cấu hình 3 seed.

Điểm **macro** = trung bình theo người. **micro** = trung bình toàn bộ buổi ghi.

## Kết quả

| cấu hình | | macro (3 seed) | micro (3 seed) | seed 0 / 1 / 2 |
|---|---|---:|---:|---|
| `64-121__pearson/` | Ours-64/121 · Pearson thuần ★ ĐỀ XUẤT | **0.801739 ± 0.009968** | 0.796845 ± 0.010557 | 0.7904 / 0.8058 / 0.8090 |
| `64-61__hybrid-a0.6/` | Ours-64/61 · loss lai alpha 0,6  | **0.803590 ± 0.015350** | 0.798315 ± 0.016034 | 0.7869 / 0.8068 / 0.8171 |
| `64-121__mse/` | Ours-64/121 · MSE thuần nền để trừ | **0.762191 ± 0.021433** | 0.755965 ± 0.022176 | 0.7391 / 0.7660 / 0.7814 |
| `192-121__hybrid-a0.2/` | Ours-192/121 · alpha 0,2  | **0.800721 ± 0.010705** | 0.795857 ± 0.011006 | 0.7932 / 0.7960 / 0.8130 |

**Hiệu Pearson thuần − MSE thuần ở Ours-64/121 = +0,039548**, thắng cả 3 seed —
đóng góp của hàm loss đo trên tập kiểm tra độc lập.

## Cấu trúc thư mục

```
runs/tn4/
   <cấu hình>/
      seed0/  seed1/  seed2/
         final.pth       trọng số model
         curve.csv       loss 20 epoch
         scores.csv      Pearson 537 buổi ghi GHIJ
         selection.txt   bảng lựa chọn kênh, 537 dòng
   summary.csv           12 dòng metric gốc, cột run_id đầy đủ
   SOURCE_MANIFEST.json  SHA-256 mọi file + nguồn từ tệp nén Drive
```

Tên ngắn ↔ config_id đầy đủ: xem `SOURCE_MANIFEST.json` khoá `config_names`.
