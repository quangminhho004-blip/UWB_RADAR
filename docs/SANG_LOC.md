# Vòng sàng lọc: chạy một fold thay vì bốn

Từ TN2 trở đi, thử một cấu hình mới bằng **một** phép chia trước, cấu hình nào
lọt mới chạy đủ bốn fold.

```
python scripts/run_cv.py --experiment tn3 --model lstm --hidden 67 \
    --loss mse_pearson --alpha 0.7 --folds val_KL
```

`val_KL` là **train ABCDEF, chấm K và L** — 218.088 cửa sổ train. Không cần cắt
lại dữ liệu: `data/processed/windows/dev_cv/` vốn cắt riêng từng người "để ghép
fold tuỳ ý" (`scripts/make_windows.py` dòng 9).


## Vì sao chia hai vòng

Một cấu hình chạy đủ bốn fold tốn gấp bốn lần một fold. Khi số cấu hình cần thử
nhiều hơn số cấu hình đáng giữ, sàng trước rẻ hơn hẳn.

Đây là cách làm quen thuộc, không phải cắt xén: **sàng lọc** để loại nhanh
những cấu hình rõ ràng không đi tới đâu, rồi **xác nhận** bằng giao thức đầy đủ
cho những cấu hình còn lại. Kết luận trong luận văn chỉ lấy từ vòng xác nhận.


## HAI ĐIỀU BẮT BUỘC NHỚ

### 1. Điểm một fold KHÔNG so được với cv_score bốn fold

Đo trên tám cấu hình TN1 đã có đủ ba seed:

| cấu hình | 4 fold | chỉ val_KL | chênh |
|---|---:|---:|---:|
| LSTM-352 | 0,7570 | 0,8165 | +0,0595 |
| LSTM-67 | 0,7532 | 0,8229 | +0,0697 |
| CNN-LSTM-58 | 0,7527 | 0,7883 | +0,0357 |
| TCN-64 WeightNorm | 0,7462 | 0,7662 | +0,0200 |
| TCN-64 BatchNorm | 0,7423 | 0,7749 | +0,0325 |
| DS-TCN-64 | 0,7421 | 0,7624 | +0,0203 |
| ModernTCN-32 | 0,7405 | 0,7631 | +0,0226 |
| BiLSTM-41 | 0,7398 | 0,7987 | +0,0589 |
| **trung bình** | **0,7467** | **0,7866** | **+0,0399** |

`val_KL` là fold dễ nhất trong bốn fold. Điểm trên nó cao hơn trung bình
**+0,040**, và mức chênh không đều — từ +0,020 tới +0,070 tuỳ cấu hình.

Vì vậy **không đặt số sàng lọc vào cùng bảng với cv_score**, và không trừ chúng
cho nhau.

### 2. Thứ hạng có thể đảo

| xếp theo 4 fold | xếp theo riêng val_KL |
|---|---|
| LSTM-352 | LSTM-67 |
| LSTM-67 | LSTM-352 |
| CNN-LSTM-58 | **BiLSTM-41** |
| TCN-64 WeightNorm | CNN-LSTM-58 |
| TCN-64 BatchNorm | TCN-64 WeightNorm |
| DS-TCN-64 | ModernTCN-32 |
| ModernTCN-32 | TCN-64 BatchNorm |
| BiLSTM-41 | **DS-TCN-64** |

**BiLSTM-41 đứng chót trên bốn fold nhưng hạng ba nếu chỉ nhìn `val_KL`.**
DS-TCN-64 thì ngược lại, từ hạng sáu xuống chót.

Nên vòng sàng lọc chỉ dùng để **loại**, không dùng để **chọn**. Cấu hình nào ở
sát nhau thì đưa cả cụm sang vòng xác nhận, đừng lấy thứ hạng của một fold làm
căn cứ.

Dải điểm cũng rộng hơn hẳn — 0,0605 trên `val_KL` so với 0,0172 trên bốn fold —
nên chênh lệch trông to hơn thực tế.


## Script chặn nhầm lẫn ra sao

`run_cv.py` **không ghi dòng `TONG`** khi chạy thiếu fold. Dòng đó chứa
`cv_score` và là dòng `compare_cv.py` đọc để dựng bảng; ghi nó từ một fold là
đưa một con số cao giả vào bảng so sánh.

Bốn dòng của bốn fold vẫn ghi bình thường. Nên **chạy nốt các fold còn lại thì
dòng `TONG` tự có**, và fold đã chạy được bỏ qua không train lại:

```
python scripts/run_cv.py --experiment tn3 ... --folds val_KL   # vòng sàng lọc
python scripts/run_cv.py --experiment tn3 ...                  # chạy nốt 3 fold
```

Lần thứ hai không có `--folds` nên chạy đủ bốn; `val_KL` đã có kết quả nên bỏ
qua, chỉ train ba fold còn lại rồi ghi `TONG`.

`config_id` không phụ thuộc `--folds`, nên hai lần chạy trên cùng một tên cấu
hình. Tên fold sai thì script dừng ngay, không chạy rồi mới báo thiếu.


## Quy tắc đề nghị chốt trước khi chạy

1. Sàng bằng `--folds val_KL`, một seed.
2. So với **cùng cấu hình nền chạy cùng cách** — một fold với một fold, một
   seed với một seed. Không so với cv_score bốn fold ba seed.
3. Cấu hình nào không kém nền rõ rệt thì đưa sang vòng xác nhận: bỏ `--folds`,
   chạy đủ ba seed.
4. Cấu hình bị loại thì ghi là "không vượt nền trong vòng sàng lọc", **không**
   ghi là "đã chứng minh kém hơn".

Đây là quyết định về chi phí, không phải kiểm định thống kê. Một fold một seed
có thể loại nhầm một cấu hình tốt.
