# runs/tn1 — thực nghiệm 1, chọn kiến trúc (4 fold · 3 seed)

Layout: `<cấu hình>/seed<N>/<fold>/{curve.csv, scores.csv, final.pth}`.
`summary.csv` giữ cột `run_id` đầy đủ để truy nguồn; dòng `fold=TONG` là cv_score.
`final.pth` là trọng số sau epoch cuối mỗi fold (không kèm trạng thái Adam).

| thư mục | notebook sinh ra | config_id | CV macro |
|---|---|---|---:|
| `DS-TCN-C64-RF61/` | `TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb` | `ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed{0,1,2}` | **0,760878 ± 0,003095** |
| `LSTM-352/` | `TN1_LSTM.ipynb` | `lstm_mse_corr0.9_seed{0,1,2}` | 0,756998 ± 0,004141 |
| `LSTM-67/` | `TN1_LSTM_small.ipynb` | `lstm_h67_mse_corr0.9_seed{0,1,2}` | 0,753208 ± 0,001966 |
| `CNN-LSTM-58/` | `TN1_CNN_LSTM.ipynb` | `cnn_lstm_h58_c32_k5_mse_corr0.9_seed{0,1,2}` | 0,752666 ± 0,003749 |

`DS-TCN-C64-RF61` là cấu hình được chọn, mang xuống [TN2](../tn2_rf/README.md)
để chọn tầm nhìn, rồi [TN3](../tn3/README.md) để chọn hàm loss, rồi
[TN4](../tn4/README.md) để chấm trên G H I J.

Điểm G H I J của mốc LSTM 352 nằm cùng bảng TN4, ở [`runs/tn4/`](../tn4/README.md).
Bảng đầy đủ kèm cách đọc: [docs/BANG_TCN.md](../../docs/BANG_TCN.md).

## Ba thư mục baseline lấy ở đâu

`LSTM-352`, `LSTM-67` và `CNN-LSTM-58` train xong từ trước nhưng tệp kết quả chỉ
được nén lên Drive. Chúng được nạp về bằng
[`notebooks/NAP_KET_QUA_TN1.ipynb`](../../notebooks/NAP_KET_QUA_TN1.ipynb), xếp
đúng layout trên, và điểm tính lại từ `scores.csv` khớp bảng này.

Hai dòng `TONG` của seed2 (`LSTM-352` và `LSTM-67`) không có trong tệp nén:
`run_cv.py` ghi dòng đó sau khi xong cả bốn fold, nên ô lưu kết quả chạy trước
thì chưa kịp. Chúng được dựng lại từ `scores.csv` per-fold, đúng công thức
`run_cv.py` dùng — macro là trung bình bốn fold, `score_std` là độ lệch chuẩn
tổng thể. Công thức đã đối chiếu với bảy dòng `TONG` có sẵn, khớp từng chữ số.

## Dựng lại bảng

```bash
python3 scripts/compare_cv.py --experiment tn1
```

## Cột `device` để trống

`summary.csv` từng ghi tên đời GPU. Tên đó không đổi kết quả nào — cùng mã, cùng
seed, cùng dữ liệu thì điểm như nhau — nên đã xoá trắng, và `results.py` giờ chỉ
ghi `cuda` hay `cpu`.
