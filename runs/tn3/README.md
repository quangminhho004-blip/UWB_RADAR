# runs/tn3 — hàm loss lai (10 mức alpha · 4 fold · 1 seed)

Layout: `<cấu hình>/a<alpha>/seed0/<fold>/{curve.csv, scores.csv, final.pth}`.
`a0` = alpha 0,0 = Pearson thuần. `summary.csv` có 10 dòng `TONG` mỗi cấu hình.

`loss = alpha × MSE + (1 − alpha) × (1 − Pearson)` — `alpha` là trọng số của
**MSE**, không phải ngưỡng chọn kênh.

| thư mục | notebook | tầm nhìn | đỉnh | mốc MSE thuần |
|---|---|---:|---|---|
| `c64-rf61/` | `TN3_HybridLoss_DS_TCN_c64` | 61 | alpha 0,6 → **0,780028** | 0,760878 *(TN1)* |
| `c64-rf121/` | `TN3_HybridLoss_DS_TCN_c64_rf121` | 121 | alpha 0 → **0,780306** | 0,757855 *(TN2)* |

**19 trong 20 mức alpha hơn MSE thuần.** Đỉnh hơn mốc 0,0192 (RF61) và 0,0225
(RF121). Ngoại lệ duy nhất: RF121 ở alpha 0,6 được 0,752385, thấp hơn mốc MSE
của chính nó (0,757855) 0,0055 — một seed, nên đọc là nhiễu chứ không phải một
vùng alpha xấu.

Nhưng bảng này **một seed**. Dải 0,776–0,780 của RF61 trải trong khoảng 0,004,
đúng cỡ dao động seed đo được ở TN1. Kết luận đọc được là *"đưa Pearson vào loss
thì tốt hơn MSE thuần"*, không phải *"0,6 là mức tối ưu"*.

Cả hai cấu hình mang xuống TN4, mỗi cái với alpha đỉnh của nó.
