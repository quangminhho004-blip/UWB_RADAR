# Họ TCN — điểm từng seed

Chỉ `tcn` và `ds_tcn`. Mỗi dòng là **một seed**, không phải một cấu hình — để
thấy dao động thật chứ không chỉ thấy trung bình.

Số tham số và tầm nhìn dựng lại bằng `models.build_model` rồi đếm, không chép tay.


## Thước đo — đọc kỹ trước khi so

| | nghĩa là gì | dùng ở đâu |
|---|---|---|
| **macro** | tính điểm từng người trước, rồi trung bình các người | **số chính thức của đồ án**, `docs/THESIS.md` mục 2 |
| **micro** | trung bình toàn bộ phiên đo, gộp chung | thước bài báo gốc công bố |

Macro cao hơn micro khoảng 0,005 ở mọi cấu hình đo được, và **thứ hạng không
đổi** giữa hai thước.

**Bảng 1 chỉ có macro.** `run_cv.py` ghi `score_micro` vào dòng từng fold nhưng
không ghi vào dòng tổng, mà bảng đọc dòng tổng. Không phải mất dữ liệu — tính
lại được từ dòng từng fold.

**Bảng 2 có cả hai**, vì `run_final_test.py` ghi đủ.


## Quy tắc so sánh

**Chỉ so những dòng cùng cột thực nghiệm.** Cột đó ghi rõ dòng nào chạy chung
một lượt. `TN2` có hai vòng riêng — RevIN và tầm nhìn — chạy ở hai thư mục kết
quả khác nhau, nên hai vòng đó cũng không đặt chung với nhau.

**Chênh lệch nhỏ hơn `seed_std` thì không xếp hạng được.** Dao động đo được
trong đồ án trải từ 0,0007 đến 0,0154.


## 1. Tập phát triển — 4 fold trên ABCDEFKL, thước **macro**

| cấu hình | tham số | tầm nhìn | seed | **macro** | trung bình | seed_std | thực nghiệm | notebook |
|---|---:|---:|:---:|---:|---:|---:|---|---|
| DS-TCN-192 k5n4 | 310.873 | 121 | 0 | 0,764427 | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| DS-TCN-64 k3n4 | 37.081 | 61 | 0 | 0,758244 | 0,760877 | 0,003095 | TN1 | `TN1_DS_TCN_RF61_no_norm_do02_c64` |
| ↳ |  |  | 1 | 0,760101 |  |  |  |  |
| ↳ |  |  | 2 | 0,764286 |  |  |  |  |
| DS-TCN-64 k5n4 | 38.105 | 121 | 0 | 0,757855 | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| DS-TCN-192 k5n4 BatchNorm | 313.945 | 121 | 0 | 0,756618 | — | — | thử riêng | `TN_test_ds_tcn_192` |
| DS-TCN-192 k3n4 | 307.801 | 61 | 0 | 0,757493 | 0,747955 | 0,012189 | TN1 | `TN1_DS_TCN_RF61_no_norm_do02_c192` |
| ↳ |  |  | 1 | 0,752150 |  |  |  |  |
| ↳ |  |  | 2 | 0,734222 |  |  |  |  |
| TCN-64 WeightNorm | 150.745 | 253 | 0 | 0,749663 | 0,746250 | 0,002997 | TN1 | `TN1_TCN_DSTCN_model_selection` |
| ↳ |  |  | 1 | 0,745041 |  |  |  |  |
| ↳ |  |  | 2 | 0,744047 |  |  |  |  |
| DS-TCN-64 k7n4 | 39.129 | 181 | 0 | 0,743657 | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| TCN-64 BatchNorm | 151.513 | 253 | 0 | 0,737742 | 0,742337 | 0,004369 | TN1 | `TN1_TCN_DSTCN_model_selection` |
| ↳ |  |  | 1 | 0,746437 |  |  |  |  |
| ↳ |  |  | 2 | 0,742833 |  |  |  |  |
| DS-TCN-64 bản đầu | 56.281 | 253 | 0 | 0,741375 | 0,742109 | 0,000673 | TN1 | `TN1_TCN_DSTCN_model_selection` |
| ↳ |  |  | 1 | 0,742256 |  |  |  |  |
| ↳ |  |  | 2 | 0,742697 |  |  |  |  |
| DS-TCN-64 k9n4 | 40.153 | 241 | 0 | 0,736969 | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| DS-TCN-192 k9n4 | 317.017 | 241 | 0 | 0,736623 * | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| DS-TCN-192 k7n4 | 313.945 | 181 | 0 | 0,732561 | — | — | TN2 tầm nhìn | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| DS-TCN-64 + RevIN | 56.281 | 253 | 0 | 0,723775 | 0,729715 | 0,007271 | TN2 RevIN | `TN2_DS_TCN_RevIN` |
| ↳ |  |  | 1 | 0,737823 |  |  |  |  |
| ↳ |  |  | 2 | 0,727547 |  |  |  |  |

`*` Dòng `DS-TCN-192 k9n4` lấy từ **đầu ra notebook**, không có trong tệp nén
trên Drive — ô nén có lẽ không chạy sau cấu hình cuối. Mọi dòng khác đọc từ
`summary.csv` trong tệp nén.


## 2. Tập kiểm tra độc lập — GHIJ, train đủ ABCDEFKL

| cấu hình | tham số | tầm nhìn | seed | **micro** | **macro** | TB micro | TB macro | thực nghiệm | notebook |
|---|---:|---:|:---:|---:|---:|---:|---:|---|---|
| DS-TCN-64 k3n4, loss lai 0,6 | 37.081 | 61 | 0 | 0,780911 | 0,786905 | 0,798314 ± 0,016034 | 0,803590 ± 0,015350 | TN4 | `TN4_final_test_ds_tcn_c64` |
| ↳ |  |  | 1 | 0,812486 | 0,817114 |  |  |  |  |
| ↳ |  |  | 2 | 0,801546 | 0,806750 |  |  |  |  |
| DS-TCN-192 k5n4, loss lai 0,2 | 310.873 | 121 | 0 | 0,790855 | 0,796040 | 0,795856 ± 0,011006 | 0,800721 ± 0,010705 | TN4 | `TN4_final_test_ds_tcn_c192` |
| ↳ |  |  | 1 | 0,788239 | 0,793153 |  |  |  |  |
| ↳ |  |  | 2 | 0,808474 | 0,812969 |  |  |  |  |
| DS-TCN-64 bản đầu, MSE | 56.281 | 253 | 0 | 0,776340 | 0,781091 | 0,791232 ± 0,015774 | 0,795782 ± 0,015413 | TN1 | `TN1_final_evaluation` |
| ↳ |  |  | 1 | 0,789594 | 0,794429 |  |  |  |  |
| ↳ |  |  | 2 | 0,807761 | 0,811827 |  |  |  |  |

Mốc đối chiếu ngoài họ TCN, cùng pipeline và cùng ba seed — **LSTM-352**, kiến
trúc MobiVital, 1.502.713 tham số:

| seed | micro | macro |
|:---:|---:|---:|
| 0 | 0,794211 | 0,799989 |
| 1 | 0,797682 | 0,802909 |
| 2 | 0,824032 | 0,828007 |
| **trung bình** | **0,805308** ± 0,016308 | **0,810302** ± 0,015403 |


## 3. Ba phép so đọc được

### Depthwise so với tích chập thường — cùng **TN1**, cùng tầm nhìn 253

| | tham số | macro |
|---|---:|---:|
| TCN-64 WeightNorm | 150.745 | 0,746250 ± 0,002997 |
| TCN-64 BatchNorm | 151.513 | 0,742337 ± 0,004369 |
| DS-TCN-64 | **56.281** | 0,742109 ± 0,000673 |

DS-TCN dùng **ít hơn 2,7 lần tham số** mà điểm ngang TCN BatchNorm (chênh
0,0002). So với WeightNorm thì thấp hơn 0,0041, xấp xỉ dao động seed của chính
nó — chưa xếp hạng được.

### RevIN — cùng **TN2** vòng RevIN, cùng kiến trúc, khác đúng một lớp

| | macro |
|---|---:|
| DS-TCN-64 | 0,742109 ± 0,000673 |
| DS-TCN-64 + RevIN | 0,729715 ± 0,007271 |

RevIN **làm tệ đi 0,0124**, gấp hơn 18 lần dao động seed của bản không RevIN.

Cách đọc đề xuất: RevIN chia cho độ lệch chuẩn, tức **xoá biên độ**. Mà biên độ
ở bài toán này là manh mối thật — đo được tương quan 0,53 giữa biên độ cửa sổ và
chất lượng ứng viên, và chọn kênh chỉ bằng biên độ không dùng model đã đạt
0,6243 so với 0,2251 khi chọn ngẫu nhiên.

### Tầm nhìn — cùng **TN2** vòng tầm nhìn, giữ nguyên 4 khối, chỉ đổi kernel

| tầm nhìn | c64 | c192 | |
|---:|---:|---:|---|
| 61 | **0,760877** | 0,747955 | nền, kế thừa từ TN1, 3 seed |
| 121 | 0,757855 | **0,764427** | 1 seed |
| 181 | 0,743657 | 0,732561 | 1 seed |
| 241 | 0,736969 | 0,736623 | 1 seed |

TN2 giữ nguyên cấu hình TN1 đã chốt, **chỉ đổi kernel**, nên nền không chạy lại.

Mức 181 và 241 thua nền rõ, chênh lớn hơn dao động seed. Mức 61 và 121 không
phân biệt được: ở c64 nền nhỉnh hơn 0,0030, ở c192 mức 121 nhỉnh hơn 0,0165.


## 4. Hai chỗ phải ghi khi báo cáo

**Dòng một seed không so được với dòng ba seed.** `DS-TCN-192 k5n4` đứng đầu
bảng 1 nhưng mới một seed; `DS-TCN-64 k3n4` xếp sau nhưng có đủ ba.

**Cùng cấu hình cùng seed chạy hai lần vẫn lệch.** Đo được ở `DS-TCN-192 k3n4`:
seed 0 chạy hai lần cho 0,762714 và 0,757493, lệch 0,0052. Nên `seed_std` chưa
phải toàn bộ nguồn dao động — nó chỉ đo dao động do khởi tạo, không đo dao động
do phiên chạy và thiết bị.
