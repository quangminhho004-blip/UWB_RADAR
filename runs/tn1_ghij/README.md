# runs/tn1_ghij — mốc LSTM 352 trên G H I J

Layout: `LSTM-352/seed<N>/{curve.csv, scores.csv, final.pth, selection.txt}`.

Kiến trúc của MobiVital, giữ nguyên. Train đủ tám người ABCDEFKL rồi chấm một
lần trên 537 phiên của G H I J — **cùng giao thức với TN4**, để hai bảng đặt
cạnh nhau được.

| thư mục | notebook | config_id | macro | micro |
|---|---|---|---:|---:|
| `LSTM-352/` | `TN1_LSTM.ipynb` | `lstm_mse_corr0.9_seed{0,1,2}` | **0,810302** ± 0,015402 | 0,805309 |

## Vì sao chỉ có một dòng

Bảng TN4 toàn cấu hình DS-TCN. Không có mốc ngoài họ đó thì không đọc được số
nào. LSTM 352 là mốc duy nhất cần: nó là kiến trúc của bài báo mà đồ án cải
tiến.

Các mốc khác của TN1 — LSTM 67 và CNN-LSTM 58 — **không** mang đi chấm trên
G H I J. Chúng làm nhiệm vụ so ở cùng ngân sách tham số trên tập phát triển,
xong ở đó.

## Con số này nói gì

Đặt cạnh TN4: **LSTM 352 (0,810302) đứng trên mô hình cuối của đồ án
(0,803590)** — ngược với thứ hạng ở TN1 trên tập phát triển. Chênh 0,0067 nhỏ
hơn dao động seed của cả hai bên (0,0154), nên không xếp hạng được, nhưng phải
ghi rõ chứ không chỉ trình bảng CV.

Xem [docs/BANG_TCN.md](../../docs/BANG_TCN.md) mục 5.

## Nguồn

Train xong từ trước, kết quả nằm trên Drive trong `tn1_lstm.zip` chứ không vào
git. Nạp về bằng [`notebooks/NAP_MOC_GHIJ.ipynb`](../../notebooks/NAP_MOC_GHIJ.ipynb),
có kiểm đủ 537 phiên mỗi lượt và so điểm với con số đã biết.

## Dựng lại bảng

```bash
python3 scripts/compare_cv.py --experiment tn1_ghij --final
```
