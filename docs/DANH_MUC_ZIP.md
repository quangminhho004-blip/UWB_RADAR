# Danh mục tệp nén kết quả — trên Google Drive `MyDrive/mobivital/`

Mỗi lần chạy notebook, `save_results.py` nén thư mục `runs/<thực nghiệm>/` rồi
chép sang Drive. Đây là **bằng chứng gốc** — checkpoint, điểm từng buổi ghi,
đường cong loss, đều nằm trong đó.

Tệp nén **tích luỹ**: sau mỗi seed script chạy lại, nên bản của seed cuối chứa
cả những seed trước.

Link Drive: điền vào ô ARTIFACT ↓.


## Dữ liệu đã xử lý (không phải kết quả)

| tệp | dung lượng | chứa gì | ARTIFACT |
|---|---|---|---|
| `by_user.tar` | 2,6 GB | 12 file `A.npz…L.npz` — `uwb` thô số phức `(n,1500,120)`, `gt` `[-1,1]`, `files` | |
| `windows.tar.gz` | 106 MB | `windows/dev_cv/` 8 người + `windows/final_train/` — cửa sổ 200/25 đã lọc, đã chuẩn hoá, CHỈ để train | |


## TN0 — tái lập MobiVital

| tệp nén | notebook | chứa gì | ARTIFACT |
|---|---|---|---|
| `tn0.zip` | `TN0` | `TN0a/b.txt` (bảng lựa chọn kênh), `scores_*.csv` | |

`runs/tn0/` trong repo đã có sẵn nội dung này, không cần tải lại.


## TN1 — bốn cấu hình (4 fold · 3 seed)

| tệp nén | notebook | cấu hình | trong git | ARTIFACT |
|---|---|---|---|---|
| `tn1_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed2.zip` | `TN1_DS_TCN_RF61_no_norm_do02_c64` | **DS-TCN 64 — 37.081 ts** | rồi | |
| `tn1_lstm_mse_corr0.9_seed2.zip` | `TN1_LSTM` | LSTM 352 — 1.502.713 ts | rồi | |
| `tn1_lstm_h67_mse_corr0.9_seed2.zip` | `TN1_LSTM_small` | LSTM 67 — 56.908 ts | rồi | |
| `tn1_cnn_lstm_h58.zip` | `TN1_CNN_LSTM` | CNN-LSTM 58 — 55.667 ts | rồi | |


## Mốc LSTM 352 trên G H I J (3 seed)

| tệp nén | notebook | cấu hình | trong git | ARTIFACT |
|---|---|---|---|---|
| `tn1_lstm.zip` *(thư mục `tn1_ghij/` bên trong)* | `TN1_LSTM` | LSTM 352 | rồi | |


## TN2 — tầm nhìn (4 fold · 1 seed)

| tệp nén | notebook | cấu hình | trong git | ARTIFACT |
|---|---|---|---|---|
| `tn2_rf_c64_4fold.zip` | `TN2_ReceptiveField_DS_TCN_c64_4fold` | kernel 5, 7, 9 | rồi | |
| `tn2_rf_ds_tcn_c64.zip` | `TN2_ReceptiveField_DS_TCN_c64` | vòng sàng lọc 1 fold, kernel 11 và 13 | **không** — ngoài phạm vi | |


## TN3 — hàm loss lai (10 mức alpha · 4 fold · 1 seed)

Mỗi tệp nén **tích luỹ**: sau mỗi alpha script chạy lại, nên bản mới nhất chứa
mọi alpha đã xong tới lúc đó.

| tệp nén | notebook | cấu hình | trong git | ARTIFACT |
|---|---|---|---|---|
| `tn3_ds_tcn_c64.zip` | `TN3_HybridLoss_DS_TCN_c64` | DS-TCN 64/RF61 — 10 mức alpha | rồi | |
| `tn3_ds_tcn_c64_k5.zip` | `TN3_HybridLoss_DS_TCN_c64_rf121` | DS-TCN 64/RF121 — 10 mức alpha | rồi | |


## TN4 — test trên G H I J (train đủ ABCDEFKL, 3 seed)

`run_final_test.py` tự nén sau **mỗi seed**, tên `tn4_<run_id>.zip`. Mỗi tệp
chứa `<run_id>/final.pth` + `curve.csv`, `<run_id>.txt` (bảng lựa chọn kênh 537
dòng), `scores_<run_id>.csv`, `summary.csv`.

| mẫu tên tệp | notebook | cấu hình | trong git | ARTIFACT |
|---|---|---|---|---|
| `tn4_ds_tcn_c64_k3_..._mse_pearson_a0.6_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64` | **DS-TCN 64/RF61 alpha 0,6 — mô hình cuối** | rồi | |
| `tn4_ds_tcn_c64_k5_..._mse_pearson_a0_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64_rf121` | DS-TCN 64/RF121 Pearson thuần | rồi | |
| `tn4_ds_tcn_c64_k5_..._mse_..._seed{0,1,2}.zip` | `TN4_final_test_ds_tcn_c64_rf121` | DS-TCN 64/RF121 MSE thuần — đối chứng | rồi | |


## Trong MỖI tệp nén luôn có

```
runs/<thực nghiệm>/
   <config_id>_<fold>/final.pth     trọng số model sau epoch cuối
   <config_id>_<fold>/curve.csv     loss từng epoch
   scores_<config_id>_<fold>.csv    Pearson TỪNG buổi ghi (đọc lại tính micro/macro)
   summary.csv                      dòng metric của riêng thực nghiệm này
   README.txt                       sinh lúc nào · commit mã nào
```

Trong repo, `runs/` xếp theo `<thực nghiệm>/<tên cấu hình>/seed<N>/<fold>/` —
cùng nội dung, chỉ khác cách lồng thư mục.
