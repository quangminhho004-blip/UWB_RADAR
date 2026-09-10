# runs/tn2_rf — tầm nhìn DS-TCN (4 fold · 1 seed)

Layout: `<cấu hình>/seed0/<fold>/{curve.csv, scores.csv}`.
Nền tầm nhìn 61 là cấu hình TN1 đã chốt, không chạy lại ở đây — TN2 chỉ đổi kernel.

| thư mục | notebook | config_id | cv_score |
|---|---|---|---:|
| `c64-rf121/` | `TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb` | ds_tcn_c64_k5_... | 0.757855 |
| `c64-rf181/` | `TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb` | ds_tcn_c64_k7_... | 0.743657 |
| `c64-rf241/` | `TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb` | ds_tcn_c64_k9_... | 0.736970 |
| `c192-rf121/` | `TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb` | ds_tcn_c192_k5_... | 0.764428 |
| `c192-rf181/` | `TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb` | ds_tcn_c192_k7_... | 0.732562 |
| `c192-rf241/` | `TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb` | ds_tcn_c192_k9_... | 0.738416 |
| `c64-rf301/` `c64-rf361/` | `TN2_ReceptiveField_DS_TCN_c64.ipynb` | k11 / k13 — **vòng sàng lọc 1 fold val_KL**, không có dòng TONG | — |

Kết luận: tầm nhìn 121 tốt, 181 trở lên tệ rõ. Mức 121 (c64 hoặc c192) mang xuống TN3.
