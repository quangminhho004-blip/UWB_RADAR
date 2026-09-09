# Họ TCN đi qua bốn thực nghiệm — kiến trúc nào đi tiếp, kiến trúc nào dừng

Cột là bốn bậc thực nghiệm. Hàng là kiến trúc. Mũi tên `──►` nghĩa là kiến trúc
đó **được chọn, đi tiếp sang bậc sau**. Dấu `╳` nghĩa là **dừng ở đó**.

Điểm là **macro trên tập phát triển**, trừ cột TN4 là **macro trên GHIJ**.


## Sơ đồ

```
                              TN1              TN2              TN3            TN4
kiến trúc                  chọn kiến trúc   chọn tầm nhìn   chọn hàm loss   test GHIJ
                           4 fold · 3 seed  4 fold · 1 seed 4 fold · 1 seed 3 seed
═══════════════════════════════════════════════════════════════════════════════════════

TCN-64 BatchNorm             0,742337  ╳
  tcn_c64                    ± 0,004369
  TN1_TCN_DSTCN_model_selection

TCN-64 WeightNorm            0,746250  ╳
  tcn_c64_weight             ± 0,002997
  TN1_TCN_DSTCN_model_selection

ModernTCN-32                 0,740533  ╳
  modern_tcn_c32_n3          ± 0,004481
  TN1_ModernTCN

TCN-64 + RevIN                    ─         chưa chạy  ╳
  TN2_TCN_RevIN

───────────────────────────────────────────────────────────────────────────────────────

DS-TCN-nền                   0,742109  ──┬────────────────────────────────►  0,795782
  ds_tcn_c64                 ± 0,000673  │                                   ± 0,015413
  TN1_TCN_DSTCN_model_selection          │                                   TN1_final_evaluation
                                         │
DS-TCN-nền + RevIN                       └──►  0,729715  ╳
  ds_tcn_c64_revin                             ± 0,007271
  TN2_DS_TCN_RevIN

───────────────────────────────────────────────────────────────────────────────────────

Ours-64/61                   0,760877  ──►   nền   ──────►  0,780028  ──────►  0,803590
  ds_tcn_c64_k3_n4_none_...  ± 0,003095       của TN2        alpha 0,6         ± 0,015350
  TN1_DS_TCN_RF61_..._c64      ★ CHỌN                        TN3_..._c64       TN4_..._c64

Ours-192/61                  0,747955  ──►   nền   ╳
  ds_tcn_c192_k3_n4_none_... ± 0,012189       của TN2
  TN1_DS_TCN_RF61_..._c192

───────────────────────────────────────────────────────────────────────────────────────

Ours-64/121                       ─         0,757855  ──────►  0,780306  ──────►  0,801739
  ds_tcn_c64_k5_n4_none_...                                    alpha 0,0         ± 0,009968
  TN2_ReceptiveField_..._c64_4fold                             TN3_..._c64_rf121  TN4_..._c64_rf121
                                                                                   ★ ĐỀ XUẤT

Ours-64/181                       ─         0,743657  ╳
  ds_tcn_c64_k7_n4_none_...
  TN2_ReceptiveField_..._c64_4fold

Ours-64/241                       ─         0,736970  ╳
  ds_tcn_c64_k9_n4_none_...
  TN2_ReceptiveField_..._c64_4fold

───────────────────────────────────────────────────────────────────────────────────────

Ours-192/121                      ─         0,764428  ──────►  0,776011  ──────►  0,800721
  ds_tcn_c192_k5_n4_none_...                                   alpha 0,2         ± 0,010705
  TN2_ReceptiveField_..._c192_4fold                            TN3_..._c192       TN4_..._c192

Ours-192/181                      ─         0,732562  ╳
  ds_tcn_c192_k7_n4_none_...
  TN2_ReceptiveField_..._c192_4fold

Ours-192/241                      ─         0,736623  ╳
  ds_tcn_c192_k9_n4_none_...
  TN2_ReceptiveField_..._c192_4fold

───────────────────────────────────────────────────────────────────────────────────────

bản thử 192/121-BN                ─         0,756618  ╳       chạy ở --experiment tn_test,
  ds_tcn_c192_k5_n4_do0.2                                     không thuộc bậc nào
  TN_test_ds_tcn_192
```


## Bảng tra — cùng nội dung, dạng bảng

| kiến trúc | tham số | TN1 | TN2 | TN3 | TN4 | notebook |
|---|---:|---:|---:|---:|---:|---|
| TCN-64 BatchNorm | 151.513 | 0,742337 | ╳ | | | `TN1_TCN_DSTCN_model_selection` |
| TCN-64 WeightNorm | 150.745 | 0,746250 | ╳ | | | `TN1_TCN_DSTCN_model_selection` |
| ModernTCN-32 | 56.985 | 0,740533 | ╳ | | | `TN1_ModernTCN` |
| TCN-64 + RevIN | 151.513 | — | *chưa chạy* | | | `TN2_TCN_RevIN` |
| DS-TCN-nền | 56.281 | 0,742109 | → | | **0,795782** | `TN1_TCN_DSTCN_model_selection` · `TN1_final_evaluation` |
| DS-TCN-nền + RevIN | 56.281 | — | 0,729715 ╳ | | | `TN2_DS_TCN_RevIN` |
| **Ours-64/61** | **37.081** | **0,760877** ★ | nền | 0,780028 | **0,803590** | `TN1_DS_TCN_RF61_no_norm_do02_c64` → `TN3_HybridLoss_DS_TCN_c64` → `TN4_final_test_ds_tcn_c64` |
| Ours-192/61 | 307.801 | 0,747955 | nền ╳ | | | `TN1_DS_TCN_RF61_no_norm_do02_c192` |
| **Ours-64/121** | 38.105 | — | 0,757855 | **0,780306** | **0,801739** ★ | `TN2_ReceptiveField_DS_TCN_c64_4fold` → `TN3_HybridLoss_DS_TCN_c64_rf121` → `TN4_final_test_ds_tcn_c64_rf121` |
| Ours-64/181 | 39.129 | — | 0,743657 ╳ | | | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| Ours-64/241 | 40.153 | — | 0,736970 ╳ | | | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| **Ours-192/121** | 310.873 | — | **0,764428** | 0,776011 | 0,800721 | `TN2_ReceptiveField_DS_TCN_c192_4fold` → `TN3_HybridLoss_DS_TCN_c192` → `TN4_final_test_ds_tcn_c192` |
| Ours-192/181 | 313.945 | — | 0,732562 ╳ | | | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| Ours-192/241 | 317.017 | — | 0,736623 ╳ | | | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| bản thử 192/121-BN | 313.945 | — | 0,756618 ╳ | | | `TN_test_ds_tcn_192` |

`★` cấu hình đề xuất · `╳` dừng ở bậc đó · `→` đi thẳng tới TN4 làm mốc so


## Vì sao mỗi nhánh dừng

| kiến trúc | dừng vì |
|---|---|
| TCN-64 BatchNorm | gấp 2,7 lần tham số DS-TCN mà điểm ngang |
| TCN-64 WeightNorm | hơn DS-TCN 0,0041, nhỏ hơn dao động seed, không đủ để trả 2,7 lần tham số |
| ModernTCN-32 | thua Ours-64/61 0,0203, cùng ngân sách tham số |
| TCN-64 + RevIN | RevIN đã hại ở nhánh DS-TCN, không chạy tiếp nhánh này |
| DS-TCN-nền + RevIN | tệ hơn nền 0,0124, gấp 18 lần dao động seed |
| Ours-192/61 | 8 lần tham số của Ours-64/61 mà thấp hơn 0,0129 |
| Ours-64/181 · 64/241 | thua nền TN2 là 0,0172 và 0,0239 |
| Ours-192/181 · 192/241 | thua Ours-192/121 là 0,0319 và 0,0278 |
| bản thử 192/121-BN | chạy ngoài mạch, ở `--experiment tn_test` |


## Ba nhánh đi trọn bốn bậc

```
Ours-64/61     TN1 0,760877  ►  nền TN2  ►  TN3 0,780028  ►  TN4 0,803590 ± 0,015350
Ours-64/121    TN2 0,757855  ►  TN3 0,780306  ►  TN4 0,801739 ± 0,009968     ★ ĐỀ XUẤT
Ours-192/121   TN2 0,764428  ►  TN3 0,776011  ►  TN4 0,800721 ± 0,010705
```

Ba con số TN4 chênh nhau dưới 0,003, **nhỏ hơn dao động giữa các seed của cả ba**
— không xếp hạng được với nhau.

Chọn **Ours-64/121** làm cấu hình đề xuất không vì điểm cao nhất, mà vì nó là
cấu hình **duy nhất có nền MSE thuần chạy trên GHIJ cùng lượt**:

```
Ours-64/121  Pearson thuần   0,801739 ± 0,009968
Ours-64/121  MSE thuần       0,762191 ± 0,021433
                             +0,039548   thắng cả ba seed
```

Đó là bằng chứng cho đóng góp chính, đo **trên tập kiểm tra độc lập**.


## Không thuộc họ TCN

`LSTM-352`, `LSTM-67`, `BiLSTM-41`, `CNN-LSTM-58`, `GRU-77`, `MixLinear` — ở lại
TN1 làm mốc so, không đi tiếp. Xem `docs/BANG_DIEM.md`.
