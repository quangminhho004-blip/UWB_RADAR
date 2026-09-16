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

Mỗi tệp nén chứa `<config_id>_<fold>/{final.pth, curve.csv}` cho cả 4 fold ×
3 seed, `scores_<config_id>_<fold>.csv` (Pearson từng buổi ghi), và `summary.csv`
lọc riêng.

| tệp nén | notebook | cấu hình | trong git chưa | ARTIFACT |
|---|---|---|---|---|
| `tn1_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed2.zip` | `TN1_DS_TCN_RF61_no_norm_do02_c64` | **DS-TCN 64 — 37.081 ts** | **rồi**, `runs/tn1/DS-TCN-C64-RF61/` | |
| `tn1_lstm_mse_corr0.9_seed2.zip` | `TN1_LSTM` | LSTM 352 — 1.502.713 ts | chưa | |
| `tn1_lstm_h67_mse_corr0.9_seed2.zip` | `TN1_LSTM_small` | LSTM 67 — 56.908 ts | chưa | |
| `tn1_cnn_lstm_h58.zip` | `TN1_CNN_LSTM` | CNN-LSTM 58 — 55.667 ts | chưa | |

Ba dòng "chưa" lấy về bằng [`NAP_KET_QUA_TN1.ipynb`](../notebooks/NAP_KET_QUA_TN1.ipynb).


## Trong MỖI tệp nén luôn có

```
runs/<thực nghiệm>/
   <config_id>_<fold>/final.pth     trọng số model sau epoch cuối
   <config_id>_<fold>/curve.csv     loss từng epoch
   scores_<config_id>_<fold>.csv    Pearson TỪNG buổi ghi (đọc lại tính micro/macro)
   summary.csv                      dòng metric của riêng thực nghiệm này
   README.txt                       sinh lúc nào · commit mã nào
```

`NAP_KET_QUA_TN1.ipynb` xếp lại thành `runs/tn1/<tên>/seed<N>/<fold>/` — cùng
nội dung, chỉ khác cách lồng thư mục.
