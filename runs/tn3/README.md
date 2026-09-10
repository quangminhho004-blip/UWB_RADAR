# runs/tn3 — hàm loss lai (10 mức alpha · 4 fold · 1 seed)

Layout: `<cấu hình>/a<alpha>/seed0/<fold>/{curve.csv, scores.csv, final.pth}`.
`a0` = alpha 0,0 = Pearson thuần. `summary.csv` có 10 dòng TONG mỗi cấu hình.

| thư mục | notebook | config_id | đỉnh |
|---|---|---|---|
| `c64-rf61/` | `TN3_HybridLoss_DS_TCN_c64.ipynb` | ds_tcn_c64_k3_..._mse_pearson_a*_corr0.9 | alpha 0.6 → 0.780028 |
| `c64-rf121/` | `TN3_HybridLoss_DS_TCN_c64_rf121.ipynb` | ds_tcn_c64_k5_..._mse_pearson_a*_corr0.9 | alpha 0 → 0.780306 |
| `c192-rf121/` | `TN3_HybridLoss_DS_TCN_c192.ipynb` | ds_tcn_c192_k5_..._mse_pearson_a*_corr0.9 | alpha 0.2 → 0.776011 |

29/30 mức alpha hơn MSE thuần. Cả ba cấu hình mang xuống TN4.
