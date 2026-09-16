# runs/tn1 — thực nghiệm 1, chọn kiến trúc (4 fold · 3 seed)

Layout: `<cấu hình>/seed<N>/<fold>/{curve.csv, scores.csv, final.pth}`.
`summary.csv` giữ cột `run_id` đầy đủ để truy nguồn; dòng `fold=TONG` là cv_score.
`final.pth` là trọng số sau epoch cuối mỗi fold (không kèm trạng thái Adam).

| thư mục | notebook sinh ra | config_id | CV macro |
|---|---|---|---:|
| `DS-TCN-C64-RF61/` | `TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb` | `ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed{0,1,2}` | **0,760878 ± 0,003095** |
| `LSTM-352/` | `TN1_LSTM.ipynb` | `lstm_mse_corr0.9_seed{0,1,2}` | 0,756992 ± 0,004156 |
| `LSTM-67/` | `TN1_LSTM_small.ipynb` | `lstm_h67_mse_corr0.9_seed{0,1,2}` | 0,753208 ± 0,001967 |
| `CNN-LSTM-58/` | `TN1_CNN_LSTM.ipynb` | `cnn_lstm_h58_c32_k5_mse_corr0.9_seed{0,1,2}` | 0,752658 ± 0,003757 |

`DS-TCN-C64-RF61` là cấu hình được chọn. Bảng đầy đủ kèm cách đọc:
[docs/BANG_TCN.md](../../docs/BANG_TCN.md).

## Ba thư mục baseline lấy ở đâu

`LSTM-352`, `LSTM-67` và `CNN-LSTM-58` đã train xong từ trước, nhưng tệp kết quả
chỉ được nén lên Drive chứ chưa bao giờ vào git. Chạy
[`notebooks/NAP_KET_QUA_TN1.ipynb`](../../notebooks/NAP_KET_QUA_TN1.ipynb) để
lấy về và xếp đúng layout trên.

Notebook đó tự tính lại điểm macro từ `scores.csv` rồi so với bảng này; lệch quá
0,0001 là báo, không im.

## Dựng lại bảng

```bash
python3 scripts/compare_cv.py --experiment tn1
```

## Cột `device` để trống

`summary.csv` từng ghi tên đời GPU. Tên đó không đổi kết quả nào — cùng mã, cùng
seed, cùng dữ liệu thì điểm như nhau — nên đã xoá trắng, và `results.py` giờ chỉ
ghi `cuda` hay `cpu`.
