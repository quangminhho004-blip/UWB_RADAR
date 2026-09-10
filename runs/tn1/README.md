# runs/tn1 — kết quả TN1 (các cấu hình họ TCN trong nhánh submission)

`summary.csv` — dòng `fold=TONG` là cv_score của cả cấu hình (trung bình 4 fold).
Mỗi `<cấu hình>_<seed>_<fold>/` có `curve.csv` (loss 20 epoch); `scores_*.csv` là
Pearson từng buổi ghi. Bỏ `final.pth` cho nhẹ.

## Cấu hình có ở đây (tải từ tệp nén Drive)

| config_id | tên đọc | seed đủ |
|---|---|---|
| `ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9` | Ours-64/61 | 0, 1, 2 |
| `ds_tcn_c192_k3_n4_none_do0.2_dpel_mse_corr0.9` | Ours-192/61 | 0, 1, 2 |
| `tcn_c64_weight_mse_corr0.9` | TCN-64 WeightNorm | 0, 1, 2 |
| `tcn_c64_mse_corr0.9` | TCN-64 BatchNorm | **1, 2** (thiếu seed 0) |
| `ds_tcn_c64_mse_corr0.9` | DS-TCN-nền | **1** (thiếu seed 0, 2) |

## Ba dòng thiếu trong tệp nén Drive

Hai cấu hình nền (đã bị loại ở TN1) chưa gom đủ 3 seed vào một tệp nén. Giá trị
đầy đủ có trong `docs/BANG_TCN.md` và đầu ra
`notebooks/TN1_TCN_DSTCN_model_selection.ipynb`:

| cấu hình | seed 0 | seed 1 | seed 2 | cv_mean ± std |
|---|---:|---:|---:|---:|
| TCN-64 BatchNorm | 0,737742 | 0,746437 | 0,742833 | 0,742337 ± 0,004369 |
| DS-TCN-nền | 0,741375 | 0,742256 | 0,742697 | 0,742109 ± 0,000673 |

Muốn Drive đủ: mở lại `TN1_TCN_DSTCN_model_selection.ipynb`, chạy ô khôi phục +
ô cất kết quả.
