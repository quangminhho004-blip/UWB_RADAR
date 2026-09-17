# TN4 — test cuối trên GHIJ

Train đủ ABCDEFKL, chấm một lần trên 537 phiên của G H I J. Ba cấu hình, mỗi
cấu hình 3 seed. G H I J không dùng để chọn bất cứ thứ gì.

Điểm **macro** = trung bình theo người. **micro** = trung bình toàn bộ buổi ghi.

## Kết quả

| cấu hình | | macro (3 seed) | micro (3 seed) | seed 0 / 1 / 2 |
|---|---|---:|---:|---|
| `64-61__hybrid-a0.6/` | DS-TCN 64/RF61 · loss lai alpha 0,6 ★ **mô hình cuối** | **0.803590 ± 0.015350** | 0.798315 ± 0.016034 | 0.7869 / 0.8068 / 0.8171 |
| `64-121__pearson/` | DS-TCN 64/RF121 · Pearson thuần | **0.801739 ± 0.009968** | 0.796845 ± 0.010557 | 0.7904 / 0.8058 / 0.8090 |
| `64-121__mse/` | DS-TCN 64/RF121 · MSE thuần — đối chứng loss | **0.762191 ± 0.021433** | 0.755965 ± 0.022176 | 0.7391 / 0.7660 / 0.7814 |

**Hiệu Pearson thuần − MSE thuần ở 64/RF121 = +0,039548**, thắng cả 3 seed. Đây
là phép so sạch nhất trong đồ án: cùng kiến trúc, cùng ba seed, cùng dữ liệu,
khác đúng một thứ.

## Mốc để so

Ba dòng trên chỉ đọc được khi đặt cạnh mốc ngoài họ TCN, ở
[`runs/tn1_ghij/`](../tn1_ghij/README.md):

| | tham số | macro |
|---|---:|---:|
| LSTM 352 *(kiến trúc MobiVital)* | 1.502.713 | **0.810302** ± 0.015402 |

**Trên tập kiểm tra độc lập, LSTM 352 đứng đầu, không phải DS-TCN.** DS-TCN thấp
hơn 0,0067 — nhỏ hơn dao động seed của cả hai bên (0,0154), nên không xếp hạng
được. Phát biểu đúng là **ngang điểm với ít hơn
40,5 lần tham số**. Xem [docs/BANG_TCN.md](../../docs/BANG_TCN.md) mục 5.

## Cấu trúc thư mục

```
runs/tn4/
   <cấu hình>/
      seed0/  seed1/  seed2/
         final.pth       trọng số model
         curve.csv       loss 20 epoch
         scores.csv      Pearson 537 buổi ghi GHIJ
         selection.txt   bảng lựa chọn kênh, 537 dòng
   summary.csv           9 dòng metric gốc, cột run_id đầy đủ
   SOURCE_MANIFEST.json  SHA-256 mọi file + nguồn từ tệp nén Drive
```

Tên ngắn ↔ config_id đầy đủ: xem `SOURCE_MANIFEST.json` khoá `config_names`.
