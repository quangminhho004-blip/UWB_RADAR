# runs/tn1 — kết quả TN1 (các cấu hình họ TCN trong nhánh submission)

`summary.csv` — dòng `fold=TONG` là cv_score của cả cấu hình (trung bình 4 fold).
Mỗi `<cấu hình>_<seed>_<fold>/` có `curve.csv` (loss 20 epoch); `scores_*.csv` là
Pearson từng buổi ghi. Bỏ `final.pth` cho nhẹ.

## Năm cấu hình, ĐỦ 3 seed

| config_id | tên đọc | cv_score (3 seed) |
|---|---|---:|
| `ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9` | Ours-64/61 ★ | 0,760878 ± 0,003095 |
| `ds_tcn_c192_k3_n4_none_do0.2_dpel_mse_corr0.9` | Ours-192/61 | 0,747955 ± 0,012189 |
| `tcn_c64_weight_mse_corr0.9` | TCN-64 WeightNorm | 0,746250 ± 0,002997 |
| `tcn_c64_mse_corr0.9` | TCN-64 BatchNorm | 0,742337 ± 0,004369 |
| `ds_tcn_c64_mse_corr0.9` | DS-TCN-nền | 0,742109 ± 0,000673 |

## Ghi chú

Ba dòng `_tong` — TCN-64 BatchNorm seed 0, DS-TCN-nền seed 0 và 2 — không có
trong tệp nén Drive (chỉ có dữ liệu từng fold). Đã **tính lại** từ
`scores_<cấu hình>_seed<N>_<fold>.csv` bằng đúng công thức của `run_cv.py`:
macro mỗi fold = trung bình theo người, cv_score = trung bình 4 fold. Ba giá trị
`0,737742 · 0,741375 · 0,742697` khớp đúng đầu ra
`notebooks/TN1_TCN_DSTCN_model_selection.ipynb`.
