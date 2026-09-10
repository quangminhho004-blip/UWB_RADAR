# Danh mục tệp nén kết quả — trên Google Drive `MyDrive/mobivital/`

Mỗi lần chạy notebook, `save_results.py` (hoặc `run_final_test.py`) nén thư mục
`runs/<thực nghiệm>/` rồi chép sang Drive. Đây là **bằng chứng gốc** — checkpoint,
điểm từng buổi ghi, đường cong loss, đều nằm trong đó.

Link Drive: điền vào ô ARTIFACT ↓ (bạn tự thêm, tôi không thấy Drive của bạn).


## Dữ liệu đã xử lý (không phải kết quả)

| tệp | dung lượng | chứa gì | ARTIFACT |
|---|---|---|---|
| `by_user.tar` | 2,6 GB | 12 file `A.npz…L.npz` — `uwb` thô số phức `(n,1500,120)`, `gt` `[-1,1]`, `files` | |
| `windows.tar.gz` | 106 MB | `windows/dev_cv/` 8 người + `windows/final_train/` — cửa sổ 200/25 đã lọc, đã chuẩn hoá, CHỈ để train | |


## TN0 — tái lập MobiVital

| tệp nén | notebook | chứa gì | ARTIFACT |
|---|---|---|---|
| `tn0.zip` | `TN0.ipynb` · `tn0_reproduce.ipynb` | `TN0a/b/c.txt` + `ours_b/c.txt` (bảng lựa chọn kênh), `scores_*.csv`, `compare.csv` (ĐẠT/KHÔNG ĐẠT), `ours_c/final.pth` + `curve.csv`, `summary.csv` | |


## TN1 — so kiến trúc (4 fold · 3 seed)

Mỗi zip chứa: `<cấu hình>_val_AB/final.pth` + `curve.csv` cho cả 4 fold × 3 seed,
`scores_<cấu hình>_<fold>.csv` (Pearson từng buổi ghi), `summary.csv` lọc riêng.

| tệp nén | notebook | kiến trúc | ARTIFACT |
|---|---|---|---|
| `tn1_ds_tcn_rf61_c64.zip` | `TN1_DS_TCN_RF61_no_norm_do02_c64` | **Ours-64/61** — 37.081 ts | |
| `tn1_ds_tcn_rf61_c192.zip` | `TN1_DS_TCN_RF61_no_norm_do02_c192` | Ours-192/61 — 307.801 ts | |
| `tn1_tcn_weightnorm.zip` | `TN1_TCN_DSTCN_model_selection` | DS-TCN-nền + TCN-64 BatchNorm + TCN-64 WeightNorm | |
| `tn1_modern_tcn.zip` | `TN1_ModernTCN` | ModernTCN-32 — 56.985 ts | |
| `tn1_bilstm_h41.zip` | `TN1_BiLSTM` | BiLSTM-41 | |
| `tn1_cnn_lstm_h58.zip` | `TN1_CNN_LSTM` | CNN-LSTM-58 | |
| `tn1_gru_h77.zip` | `TN1_GRU` | GRU-77 *(chưa chạy)* | |
| `tn1_mix_linear.zip` | `TN1_MixLinear` | MixLinear-63 | |
| `tn1_lstm*.zip` | `TN1_LSTM` · `TN1_LSTM_small` | LSTM-352, LSTM-67 (cả CV lẫn nhánh GHIJ) | |
| `tn1_ghij_lstm_h67.zip` | `TN1_LSTM_small` | LSTM-67 train đủ 8 người, test GHIJ | |
| `tn1_ghij_*.zip` | `TN1_final_evaluation` | LSTM-352 · DS-TCN-nền · TCN-64 — test GHIJ, tự sinh tên `<exp>_<run_id>.zip` | |
| `tn2_ds_tcn_revin.zip` | `TN1_DS_TCN_RevIN` | DS-TCN-nền + RevIN, 4 fold × 3 seed — nhánh phụ so kiến trúc; giải nén vào `runs/tn1/DS-TCN-nen+RevIN/`. Tên tệp mang `tn2` vì đó là experiment lúc chạy. | |
| `tn2_tcn_revin.zip` | `TN1_TCN_RevIN` | TCN-64 + RevIN *(chưa chạy)* | |


## TN2 — tầm nhìn (receptive field)

| tệp nén | notebook | chứa gì | ARTIFACT |
|---|---|---|---|
| `tn2_rf_c64_4fold.zip` | `TN2_ReceptiveField_DS_TCN_c64_4fold` | Ours-64/121, 64/181, 64/241 — kernel 5·7·9, đủ 4 fold, 1 seed | |
| `tn2_rf_c192_4fold.zip` | `TN2_ReceptiveField_DS_TCN_c192_4fold` | Ours-192/121, 192/181, 192/241 — đủ 4 fold, 1 seed | |
| `tn2_rf_ds_tcn_c64.zip` | `TN2_ReceptiveField_DS_TCN_c64` | vòng SÀNG LỌC 1 fold, kernel 5·7·9·11·13 | |
| `tn2_rf_ds_tcn_c192.zip` | `TN2_ReceptiveField_DS_TCN_c192` | vòng SÀNG LỌC 1 fold *(chưa có đầu ra)* | |


## TN3 — hàm loss lai (quét 10 mức alpha, 4 fold, 1 seed)

Mỗi zip tích luỹ: sau mỗi alpha `save_results.py` chạy lại nên zip mới nhất chứa
**mọi alpha đã xong tới thời điểm đó** — checkpoint từng fold, `scores_*.csv`,
`summary.csv` với đủ dòng `TONG` mỗi alpha.

| tệp nén | notebook | cấu hình | ARTIFACT |
|---|---|---|---|
| `tn3_ds_tcn_c64.zip` | `TN3_HybridLoss_DS_TCN_c64` | Ours-64/61 — 11 điểm alpha | |
| `tn3_ds_tcn_c64_k5.zip` | `TN3_HybridLoss_DS_TCN_c64_rf121` | Ours-64/121 — 11 điểm alpha | |
| `tn3_ds_tcn_c192.zip` | `TN3_HybridLoss_DS_TCN_c192` | Ours-192/121 — 11 điểm alpha | |


## TN4 — test cuối trên GHIJ (train đủ ABCDEFKL, 3 seed)

`run_final_test.py` tự nén sau **mỗi seed**, tên `tn4_<run_id>.zip`. Mỗi zip chứa:
`<run_id>/final.pth` + `curve.csv`, `<run_id>.txt` (bảng lựa chọn kênh 537 dòng),
`scores_<run_id>.csv` (Pearson từng buổi ghi), `summary.csv`.

| mẫu tên tệp | notebook | cấu hình | ARTIFACT |
|---|---|---|---|
| `tn4_ds_tcn_c64_k3_..._mse_pearson_a0.6_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64` | **Ours-64/61 alpha 0,6** | |
| `tn4_ds_tcn_c64_k5_..._mse_pearson_a0_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64_rf121` | **Ours-64/121 Pearson thuần** ★ đề xuất | |
| `tn4_ds_tcn_c64_k5_..._mse_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64_rf121` | Ours-64/121 MSE thuần (nền để trừ) | |
| `tn4_ds_tcn_c192_k5_..._mse_pearson_a0.2_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c192` | Ours-192/121 alpha 0,2 | |


## Khảo sát phụ

| tệp nén | notebook | chứa gì | ARTIFACT |
|---|---|---|---|
| `tn_mixlinear_c0.zip` · `c2.zip` · `c3.zip` | `TN_MixLinear_C0/C2/C3` | biến thể ghép nhánh MixLinear, 1 seed | |
| `tn_test_ds_tcn_c192.zip` | `TN_test_ds_tcn_192` | bản thử 192/121-BN (BatchNorm), 1 seed | |


## Trong MỖI zip luôn có

```
runs/<thực nghiệm>/
   <cấu hình>_<fold>/final.pth     trọng số model đã train
   <cấu hình>_<fold>/curve.csv     loss từng epoch
   scores_<cấu hình>_<fold>.csv    Pearson TỪNG buổi ghi (đọc lại tính micro/macro)
   <cấu hình>.txt                  bảng lựa chọn kênh (chỉ TN0 và TN4)
   summary.csv                     dòng metric của riêng thực nghiệm này
   README.txt                      sinh lúc nào · commit mã nào
```
