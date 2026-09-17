# Bảng kết quả — TN1 đến TN4

Bốn thực nghiệm nối nhau. TN1 chọn kiến trúc, TN2 chọn tầm nhìn, TN3 chọn hàm
loss, TN4 chấm trên tập kiểm tra độc lập G H I J.

Số tham số dựng lại bằng `models.build_model` rồi đếm, không chép tay.


## Đọc bảng này thế nào

**macro** — trung bình điểm từng người, rồi trung bình các người. Số chính thức
của đồ án, theo [THESIS.md](THESIS.md) mục 2.

**micro** — trung bình toàn bộ phiên đo. Bài báo MobiVital công bố theo thước này.

**Chỉ so những dòng cùng một thực nghiệm.** Ghép hai thực nghiệm khác nhau là
đưa thêm một biến không kiểm soát được vào phép so. Cột **thực nghiệm** ghi rõ
dòng nào từ đâu ra.

**Chênh lệch nhỏ hơn dao động seed thì không xếp hạng được.** Dao động đo được
trong đồ án trải từ 0,0020 đến 0,0214.

**Số seed không giống nhau giữa các bảng.** TN1 và TN4 chạy 3 seed; TN2 và TN3
chạy 1 seed cho mỗi cấu hình. Bảng 1 seed không có cột dao động, nên chênh lệch
nhỏ ở đó đọc yếu hơn.


## 1. TN1 — chọn kiến trúc. 4 fold trên ABCDEFKL, 3 seed

| cấu hình | họ | tham số | tầm nhìn | CV macro | seed_std |
|---|---|---:|---:|---:|---:|
| **DS-TCN 64, k3 n4** | tích chập | **37.081** | 61 | **0,760878** | 0,003095 |
| LSTM 352 | hồi quy | 1.502.713 | cả 200 mẫu | 0,756998 | 0,004141 |
| LSTM 67 | hồi quy | 56.908 | cả 200 mẫu | 0,753208 | 0,001966 |
| CNN-LSTM 58 | lai | 55.667 | cả 200 mẫu | 0,752666 | 0,003749 |

`LSTM 352` là kiến trúc của MobiVital, giữ nguyên, làm mốc.

**Cùng ngân sách tham số:** DS-TCN hơn LSTM 67 **0,0077** và hơn CNN-LSTM 58
**0,0082**, với ít hơn 1,5 lần tham số. Cả hai chênh lệch lớn hơn dao động seed
của mọi cấu hình trong bảng.

**So với mốc MobiVital:** hơn **0,0039** với ít hơn **40,5 lần** tham số. Con số
0,0039 nhỏ hơn dao động seed của LSTM 352 (0,0041), nên phát biểu an toàn là
**ngang điểm**, không phải "vượt".


## 2. TN2 — tầm nhìn. 4 fold, 1 seed

Giữ nguyên DS-TCN 64, 4 khối, không chuẩn hoá, dropout 0,2; chỉ đổi kernel.
Tầm nhìn tính bằng `1 + 2 × (kernel − 1) × Σ dilation`.

| kernel | tầm nhìn | tham số | CV macro | thực nghiệm |
|---:|---:|---:|---:|---|
| **3** | **61** | **37.081** | **0,760878** ± 0,003095 | TN1, 3 seed |
| 5 | 121 | 38.105 | 0,757855 | TN2, 1 seed |
| 7 | 181 | 39.129 | 0,743657 | TN2, 1 seed |
| 9 | 241 | 40.153 | 0,736970 | TN2, 1 seed |

Tầm nhìn **càng rộng càng tệ**, đơn điệu. Cửa sổ vào chỉ có 200 mẫu, nên k9 đã
phủ dư (241 > 200) mà vẫn không giúp gì.

RF61 giữ ngôi đầu, và RF121 được mang tiếp sang TN3 để xem hàm loss có đổi thứ
hạng không. Chênh RF61 với RF121 là 0,0030 — xấp xỉ dao động seed của RF61, nên
**chưa tách được hai cái này bằng CV**.


## 3. TN3 — hàm loss lai. 4 fold, 1 seed, quét 10 mức alpha

`loss = alpha × MSE + (1 − alpha) × (1 − Pearson)`. `alpha = 0` là Pearson
thuần, `alpha = 1` là MSE thuần.

| alpha | DS-TCN 64/RF61 | DS-TCN 64/RF121 |
|---:|---:|---:|
| 0 | 0,776667 | **0,780306** |
| 0,1 | 0,769390 | 0,763183 |
| 0,2 | 0,775264 | 0,771996 |
| 0,3 | 0,771931 | 0,776213 |
| 0,4 | 0,779266 | 0,771848 |
| 0,5 | 0,779419 | 0,763457 |
| 0,6 | **0,780028** | 0,752386 |
| 0,7 | 0,771665 | 0,761892 |
| 0,8 | 0,776611 | 0,760013 |
| 0,9 | 0,764941 | 0,758236 |
| **1 (MSE thuần)** | 0,760878 *(TN1, 3 seed)* | 0,757855 *(TN2, 1 seed)* |

**Đây là kết quả mạnh nhất của đồ án.** **19 trong 20** mức alpha hơn MSE thuần.
Mức tốt nhất hơn MSE thuần **0,0192** (RF61) và **0,0225** (RF121) — lớn hơn dao
động seed nhiều lần.

Ngoại lệ duy nhất là RF121 ở alpha 0,6: **0,752386**, thấp hơn mốc MSE của chính
nó 0,0055. Một seed nên đọc là nhiễu, nhưng phải ghi ra chứ không làm tròn thành
"mọi mức đều hơn".

Nhưng **đừng đọc từng mức alpha như một xếp hạng**: dải 0,776–0,780 của RF61
trải trong khoảng 0,004, đúng cỡ dao động seed, mà bảng này chỉ có 1 seed. Kết
luận đọc được là **"đưa Pearson vào loss thì tốt hơn MSE thuần"**, không phải
"0,6 là mức tối ưu".

Giữ `RF61 + alpha 0,6` và `RF121 + alpha 0` mang xuống TN4.


## 4. TN4 — kiểm tra độc lập trên G H I J. Train đủ ABCDEFKL, 3 seed

537 buổi ghi của bốn người chưa từng dùng để chọn bất cứ thứ gì.

| cấu hình | tham số | macro | std | thực nghiệm |
|---|---:|---:|---:|---|
| **LSTM 352** *(mốc MobiVital)* | 1.502.713 | **0,810302** | 0,015402 | TN1 GHIJ |
| **DS-TCN 64/RF61, alpha 0,6** | **37.081** | **0,803590** | 0,015350 | TN4 |
| DS-TCN 64/RF121, Pearson thuần | 38.105 | 0,801739 | 0,009968 | TN4 |
| DS-TCN 64/RF121, MSE thuần *(đối chứng)* | 38.105 | 0,762191 | 0,021433 | TN4 |


## 5. Ba điều đọc được từ TN4

### Hàm loss là thứ có tác dụng rõ nhất

| cùng kiến trúc 64/RF121, chỉ đổi loss | macro GHIJ |
|---|---:|
| Pearson thuần | **0,801739** ± 0,009968 |
| MSE thuần | 0,762191 ± 0,021433 |

Chênh **0,0395**, gấp gần bốn lần dao động seed của bên cao hơn. Đây là phép
so **sạch** — cùng kiến trúc, cùng ba seed, cùng dữ liệu, khác đúng một thứ.
Và nó khớp chiều với TN3 trên tập phát triển.

### Trên tập độc lập, DS-TCN NGANG chứ không hơn các mốc LSTM

| | tham số | macro GHIJ | chênh |
|---|---:|---:|---:|
| LSTM 352 | 1.502.713 | **0,810302** ± 0,015402 | — |
| DS-TCN 64/RF61 α0,6 | **37.081** | 0,803590 ± 0,015350 | **−0,0067** |

DS-TCN **thấp hơn** mốc MobiVital 0,0067 — nhỏ hơn dao động seed của cả hai bên
(0,0154 và 0,0154), nên không xếp hạng được.

Phát biểu đúng: **DS-TCN 37 nghìn tham số cho kết quả ngang LSTM 1,5 triệu
tham số trên tập kiểm tra độc lập** — ít hơn 40,5 lần. Không nói "tốt hơn".

### Thứ hạng ở TN1 KHÔNG giữ nguyên sang TN4

| | CV (ABCDEFKL) | GHIJ |
|---|---:|---:|
| DS-TCN 64 | **0,760878** | 0,803590 |
| LSTM 352 | 0,756998 | **0,810302** |

Trên tập dùng để chọn, DS-TCN đứng đầu. Trên tập không dùng để chọn, LSTM 352
đứng đầu. Đây là dấu hiệu cấu hình được chọn hợp với tám người ABCDEFKL hơn là
hợp với bài toán nói chung — **phải ghi rõ khi báo cáo**, đừng chỉ trình bảng CV.

Khoảng cách đảo chiều chỉ 0,0039 → −0,0067, cả hai đều trong dao động seed, nên
cách đọc chặt nhất là: **hai kiến trúc này không phân biệt được bằng dữ liệu
hiện có; cái đáng nói là chênh lệch số tham số.**


## 6. Giới hạn phải ghi khi báo cáo

**TN2 và TN3 chỉ có 1 seed.** Chênh lệch dưới ~0,004 ở hai bảng đó không kết
luận được. Riêng khoảng cách "có Pearson so với MSE thuần" thì đủ lớn.

**DS-TCN 64/RF61 khác các mốc ở nhiều thứ cùng lúc** — họ kiến trúc, số khối,
chuẩn hoá, dropout, tầm nhìn. Không quy kết quả cho riêng phép tích chập
depthwise.

**Tầm nhìn 61 trên cửa sổ 200 mẫu** nghĩa là model chỉ thấy 1,2 giây gần nhất,
trong khi một nhịp thở khoảng 4 giây. Nó vẫn cho điểm cao nhất TN1 và TN2. Đây
là quan sát đo được, chưa có giải thích được kiểm chứng.

**G H I J đã được chấm ở TN0** (phần tái lập MobiVital). Không mô tả nó là "chỉ
mở đúng một lần".

**Cột micro của các bảng CV còn trống.** `run_cv.py` ghi `score_micro` vào dòng
từng fold nhưng không ghi vào dòng `TONG`, mà bảng chỉ đọc dòng `TONG`. Chữa
được mà không phải chạy lại: micro bốn fold bằng trung bình micro từng fold có
trọng số theo số phiên đo. Bảng GHIJ thì có đủ cả hai thước.


## Dựng lại các bảng trên

```bash
python3 scripts/compare_cv.py --experiment tn1
python3 scripts/compare_cv.py --experiment tn2_rf
python3 scripts/compare_cv.py --experiment tn3
python3 scripts/compare_cv.py --experiment tn4 --final
python3 scripts/compare_cv.py --experiment tn1_ghij --final
```

Muốn tính lại từ gốc thì đọc thẳng `scores.csv` — điểm Pearson của từng phiên
đo nằm ở đó.
