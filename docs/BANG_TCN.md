# Họ TCN — mọi cấu hình đã chạy

Chỉ `tcn` và `ds_tcn`. Các kiến trúc khác xem `docs/BANG_DIEM.md`.

Số tham số và tầm nhìn dựng lại bằng `models.build_model` rồi đếm, không chép tay.


## Đọc bảng này thế nào

**macro** — trung bình điểm từng người, rồi trung bình các người. Số chính thức
của đồ án, theo `docs/PROTOCOL.md` mục 4.

**micro** — trung bình toàn bộ phiên đo. Bài báo gốc công bố theo thước này.

**Chỉ so những dòng chạy chung một lượt.** Ghép hai lượt chạy khác nhau là đưa
thêm một biến không kiểm soát được vào phép so. Cột "lượt" ghi rõ.

**Chênh lệch nhỏ hơn `seed_std` thì không xếp hạng được.** Dao động đo được
trong đồ án trải từ 0,0007 đến 0,0154.


## 1. Tập phát triển — 4 fold trên ABCDEFKL

| cấu hình | tham số | tầm nhìn | macro | seed | lượt |
|---|---:|---:|---:|:---:|---|
| DS-TCN-192 k5n4 | 310.873 | 121 | **0,764427** | 1 | tn2_rf |
| DS-TCN-64 k3n4 | **37.081** | 61 | **0,760877** ± 0,003095 | **3** | tn1 |
| DS-TCN-64 k5n4 | 38.105 | 121 | 0,757855 | 1 | tn2_rf |
| DS-TCN-192 k5n4 BatchNorm | 313.945 | 121 | 0,756618 | 1 | tn1 |
| DS-TCN-192 k3n4 | 307.801 | 61 | 0,747955 ± 0,012189 | **3** | tn1 |
| TCN-64 WeightNorm | 150.745 | 253 | 0,746250 ± 0,002997 | **3** | tn1 |
| DS-TCN-64 k7n4 | 39.129 | 181 | 0,743657 | 1 | tn2_rf |
| TCN-64 BatchNorm | 151.513 | 253 | 0,742337 ± 0,004369 | **3** | tn1 |
| DS-TCN-64 bản đầu | 56.281 | 253 | 0,742109 ± 0,000673 | **3** | tn1 |
| DS-TCN-64 k9n4 | 40.153 | 241 | 0,736969 | 1 | tn2_rf |
| DS-TCN-192 k9n4 | 317.017 | 241 | 0,736623 | 1 | tn2_rf |
| DS-TCN-192 k7n4 | 313.945 | 181 | 0,732561 | 1 | tn2_rf |
| DS-TCN-64 + RevIN | 56.281 | 253 | 0,729715 ± 0,007271 | **3** | tn1 |

**Cột micro của bảng này còn trống.** `run_cv.py` ghi `score_micro` vào dòng
từng fold nhưng **quên ghi vào dòng `TONG`**, mà bảng chỉ đọc dòng `TONG`. Chữa
được mà không phải chạy lại: micro bốn fold bằng trung bình micro từng fold có
trọng số theo số phiên đo, cả hai số đều đã nằm trong dòng từng fold. Bốn fold
chia hết tám người nên mỗi phiên đo đếm đúng một lần.


## 2. Tập kiểm tra độc lập — GHIJ, train đủ ABCDEFKL

| cấu hình | tham số | tầm nhìn | micro | macro | seed |
|---|---:|---:|---:|---:|:---:|
| **DS-TCN-64 k3n4, loss lai 0,6** | **37.081** | 61 | **0,798314** ± 0,016034 | **0,803590** ± 0,015350 | 3 |
| DS-TCN-192 k5n4, loss lai 0,2 | 310.873 | 121 | 0,795856 ± 0,011006 | 0,800721 ± 0,010705 | 3 |
| DS-TCN-64 bản đầu, MSE | 56.281 | 253 | 0,791232 ± 0,015774 | 0,795782 ± 0,015413 | 3 |

Mốc đối chiếu ngoài họ TCN, cùng pipeline và cùng ba seed: LSTM-352 *(kiến trúc
MobiVital, 1.502.713 tham số)* micro **0,805308** ± 0,016308, macro **0,810302**
± 0,015403.

Chênh giữa hai thước ổn định ở mọi cấu hình: macro cao hơn micro khoảng 0,005,
và **thứ hạng không đổi**.


## 3. Ba phép so đọc được

### Depthwise so với tích chập thường — cùng lượt `tn1`, cùng tầm nhìn 253

| | tham số | macro |
|---|---:|---:|
| TCN-64 WeightNorm | 150.745 | 0,746250 ± 0,002997 |
| TCN-64 BatchNorm | 151.513 | 0,742337 ± 0,004369 |
| DS-TCN-64 | **56.281** | 0,742109 ± 0,000673 |

DS-TCN dùng **ít hơn 2,7 lần tham số** mà điểm ngang TCN BatchNorm (chênh
0,0002). So với WeightNorm thì thấp hơn 0,0041, xấp xỉ dao động seed của chính
nó — chưa xếp hạng được.

### RevIN — cùng lượt, cùng kiến trúc, khác đúng một lớp

| | macro |
|---|---:|
| DS-TCN-64 | 0,742109 ± 0,000673 |
| DS-TCN-64 + RevIN | 0,729715 ± 0,007271 |

RevIN **làm tệ đi 0,0124**, gấp hơn 18 lần dao động seed của bản không RevIN.

Cách đọc đề xuất: RevIN chia cho độ lệch chuẩn, tức **xoá biên độ**. Mà biên độ
ở bài toán này là manh mối thật — đo được tương quan 0,53 giữa biên độ cửa sổ và
chất lượng ứng viên, và chọn kênh chỉ bằng biên độ không dùng model đã đạt
0,6243 so với 0,2251 khi chọn ngẫu nhiên.

### Tầm nhìn — cùng lượt `tn2_rf`, giữ nguyên 4 khối, chỉ đổi kernel

| tầm nhìn | c64 | c192 |
|---:|---:|---:|
| 121 | 0,757855 | **0,764427** |
| 181 | 0,743657 | 0,732561 |
| 241 | 0,736969 | 0,736623 |

Tầm nhìn 121 hơn 181 là **0,0142** (c64) và **0,0319** (c192), đều lớn hơn dao
động seed. Mức 61 chạy ở lượt `tn1` nên để riêng, không đặt chung bảng này.


## 4. Ba chỗ phải ghi khi báo cáo

**Dòng một seed không so được với dòng ba seed.** `DS-TCN-192 k5n4` đứng đầu
bảng phát triển nhưng mới một seed; `DS-TCN-64 k3n4` xếp sau nhưng có đủ ba.

**Cùng cấu hình cùng seed chạy hai lần vẫn lệch.** Đo được ở `DS-TCN-192 k3n4`:
seed 0 chạy hai lần cho 0,762714 và 0,757493, lệch 0,0052. Nên `seed_std` chưa
phải toàn bộ nguồn dao động.

**Điểm trên tập phát triển luôn lạc quan hơn tập kiểm tra**, vì tập phát triển
đã được dùng để chọn cấu hình. So hai cột ở hai mục trên là so hai loại số khác
nhau.
