# runs/tn2_rf — tầm nhìn DS-TCN (4 fold · 1 seed)

Layout: `<cấu hình>/seed0/<fold>/{curve.csv, scores.csv, final.pth}`.

Giữ nguyên DS-TCN 64, 4 khối, không chuẩn hoá, dropout 0,2 theo phần tử. Đổi
**đúng một thứ**: kernel. Tầm nhìn 61 là cấu hình TN1 đã chốt, không chạy lại ở
đây.

| thư mục | notebook | kernel | tầm nhìn | cv_score |
|---|---|---:|---:|---:|
| *(ở `runs/tn1/`)* | `TN1_DS_TCN_RF61_no_norm_do02_c64` | **3** | **61** | **0,760878** ± 0,003095 *(3 seed)* |
| `c64-rf121/` | `TN2_ReceptiveField_DS_TCN_c64_4fold` | 5 | 121 | 0,757855 |
| `c64-rf181/` | `TN2_ReceptiveField_DS_TCN_c64_4fold` | 7 | 181 | 0,743657 |
| `c64-rf241/` | `TN2_ReceptiveField_DS_TCN_c64_4fold` | 9 | 241 | 0,736970 |

Tầm nhìn tính bằng `1 + 2 × (kernel − 1) × Σ dilation`, với dilation 1-2-4-8.

**Kết luận: tầm nhìn càng rộng điểm càng thấp, đơn điệu.** Cửa sổ vào chỉ có 200
mẫu nên kernel 9 đã phủ dư (241 > 200) mà vẫn không giúp gì.

Chênh RF61 với RF121 là 0,0030 — xấp xỉ dao động seed của RF61, nên chưa tách
được hai cái này bằng CV. Vì vậy **cả hai** mang xuống TN3.

## Vòng sàng lọc một fold

`TN2_ReceptiveField_DS_TCN_c64.ipynb` còn chạy kernel 11 và 13 (tầm nhìn 301 và
361) nhưng **chỉ trên fold `val_KL`**, để xem đẩy tầm nhìn vượt hẳn cửa sổ có
đảo chiều xu hướng không. Không đảo: trên `val_KL` điểm giảm đều từ k5 tới k13.
Hai mức đó bị dừng ở vòng sàng lọc, không cho chạy tiếp bốn fold, và **không có
trong thư mục này**.

Một fold là sàng lọc, không phải kết luận: riêng trên `val_KL` thì k5 hơn k3,
ngược hẳn với thứ hạng bốn fold.

Kết quả gốc của k11/k13 nằm ở nhánh `submission` và trong `tn2_rf_ds_tcn_c64.zip`
trên Drive.
