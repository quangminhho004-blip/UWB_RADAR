# runs/tn1_ghij — mốc LSTM trên G H I J

Layout: `<cấu hình>/seed<N>/{curve.csv, scores.csv, final.pth, selection.txt}`.

Hai kiến trúc mốc, train đủ tám người ABCDEFKL rồi chấm một lần trên 537 phiên
của G H I J — cùng giao thức với TN4, để hai bảng đặt cạnh nhau được.

| thư mục | notebook | config_id | macro | micro |
|---|---|---|---:|---:|
| `LSTM-352/` | `TN1_LSTM.ipynb` | `lstm_mse_corr0.9_seed{0,1,2}` | **0,810302** ± 0,015402 | 0,805309 |
| `LSTM-67/` | `TN1_LSTM_small.ipynb` | `lstm_h67_mse_corr0.9_seed{0,1,2}` | 0,801683 ± 0,002506 | 0,796531 |

`LSTM-352` là kiến trúc của MobiVital, giữ nguyên. `LSTM-67` thu nhỏ để cùng
ngân sách tham số với các cấu hình gọn.

## Vì sao hai dòng này quan trọng

Bảng TN4 toàn cấu hình DS-TCN. Không có mốc ngoài họ đó thì không đọc được.

Đặt cạnh nhau: **LSTM 352 (0,810302) đứng trên mô hình cuối của đồ án
(0,803590)** — ngược với thứ hạng ở TN1 trên tập phát triển. Chênh 0,0067 nhỏ
hơn dao động seed của DS-TCN (0,0154), nên không xếp hạng được, nhưng phải ghi
rõ chứ không chỉ trình bảng CV.

Xem [docs/BANG_TCN.md](../../docs/BANG_TCN.md) mục 5.

## Nguồn

Cả hai train xong từ trước, kết quả nằm trên Drive chứ không vào git. Nạp về
bằng [`notebooks/NAP_MOC_GHIJ.ipynb`](../../notebooks/NAP_MOC_GHIJ.ipynb), có
kiểm đủ 537 phiên mỗi lượt và so điểm với con số đã biết.

## Dựng lại bảng

```bash
python3 scripts/compare_cv.py --experiment tn1_ghij --final
```
