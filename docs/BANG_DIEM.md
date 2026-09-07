# Bảng điểm — mọi kết quả đã có

Cập nhật mỗi khi có kết quả mới. Điểm là **Pearson macro theo người**: trung
bình từng người, rồi trung bình các người.

Ba cột phải đọc kèm nhau, đừng chỉ nhìn điểm:

    seed    3 seed mới tính được seed_std. 1 seed thì chưa kết luận được gì.
    fold    4 fold là giao thức đầy đủ. 1 fold là vòng sàng lọc, điểm cao hơn
            trung bình +0,040 và có thể đảo thứ hạng — xem docs/SANG_LOC.md
    máy     Colab và macOS chênh nhau 0,014, đo được ở TN0. KHÔNG trừ thẳng
            hai loại cho nhau.


## MỐC

| | điểm |
|---|---:|
| MobiVital công bố trong bài báo | **0,819** |
| Trần trên chọn kênh, tính trên dev ABCDEFKL | **0,9120** |
| Trần trên trước bộ lọc `invert_detector` | 0,9133 |
| Chọn kênh chỉ bằng biên độ, không dùng model | 0,6243 |
| Chọn kênh ngẫu nhiên | 0,2251 |

Trần trên là điểm đạt được nếu **luôn chọn đúng kênh tốt nhất**. Nó nhìn nhịp
thở thật để chọn, nên không phải cấu hình chạy được — chỉ là giới hạn trên.


## TN1 — so kiến trúc, 4 fold, 3 seed

Trừ hai dòng cuối, mọi cấu hình chạy trên Colab với cùng giao thức: 20 epoch,
Adam lr 1e-4, batch 64, MSE, `corr` 0,9.

| cấu hình | tham số | cv_score | seed_std | seed | GHIJ | máy |
|---|---:|---:|---:|:---:|---:|---|
| **DS-TCN-64 k3n4 no_norm do0.2** | **37.081** | **0,760878** | 0,003095 | 3 | chưa | Colab |
| LSTM-352 | 1.502.713 | 0,756992 | 0,004156 | 3 | 0,810302 ± 0,015399 | Colab |
| LSTM-67 | 56.908 | 0,753208 | 0,001967 | 3 | 0,801683 ± 0,002507 | Colab |
| CNN-LSTM-58 | 55.667 | 0,752658 | 0,003757 | 3 | chưa | Colab |
| TCN-64 WeightNorm | 150.745 | 0,746250 | 0,003006 | 3 | chưa | Colab |
| TCN-64 BatchNorm | 151.513 | 0,742333 | 0,004358 | 3 | chưa | Colab |
| DS-TCN-64 | 56.281 | 0,742117 | 0,000677 | 3 | 0,795783 ± 0,015413 | Colab |
| ModernTCN-32 | 56.985 | 0,740533 | 0,004481 | 3 | chưa | Colab |
| BiLSTM-41 | 57.507 | 0,739817 | 0,004630 | 3 | chưa | Colab |
| MixLinear-63 | 63 | 0,672429 | 0,006570 | 3 | 0,687685 | cv Colab, **GHIJ macOS** |
| GRU-77 | 56.466 | **chưa chạy** | — | — | chưa | — |

Điểm từng seed:

| cấu hình | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| DS-TCN-64 k3n4 no_norm do0.2 | 0,7582 | 0,7601 | 0,7643 |
| LSTM-352 | 0,7607 | 0,7525 | 0,7578 |
| LSTM-67 | 0,7536 | 0,7511 | 0,7550 |
| CNN-LSTM-58 | 0,7497 | 0,7569 | 0,7513 |
| TCN-64 WeightNorm | 0,7497 | 0,7450 | 0,7440 |
| TCN-64 BatchNorm | 0,7378 | 0,7464 | 0,7428 |
| DS-TCN-64 | 0,7414 | 0,7423 | 0,7427 |
| ModernTCN-32 | 0,7438 | 0,7354 | 0,7424 |
| BiLSTM-41 | 0,7423 | 0,7345 | 0,7427 |
| MixLinear-63 | 0,6760 | 0,6764 | 0,6648 |


## TN1 — DS-TCN 192 kênh, tầm nhìn 61

Cùng kiến trúc với dòng đầu bảng trên, chỉ khác số kênh.

| cấu hình | tham số | cv_score | seed | fold | máy |
|---|---:|---:|:---:|:---:|---|
| DS-TCN-192 k3n4 no_norm do0.2 | 307.801 | 0,762714 | **1** | 4 | Colab |

Điểm từng fold seed 0: `val_AB` 0,8013 · `val_CE` 0,7809 · `val_DF` 0,6267 ·
`val_KL` 0,8419

**Seed 1 và 2 chưa chạy.** Seed 0 đã nén sang Drive, chạy tiếp thì được bỏ qua.


## TN2 — RevIN

| cấu hình | tham số | cv_score | seed_std | seed | máy |
|---|---:|---:|---:|:---:|---|
| DS-TCN-64 | 56.281 | 0,742117 | 0,000677 | 3 | Colab |
| DS-TCN-64 **+ RevIN** | 56.281 | 0,729717 | 0,007278 | 3 | Colab |
| | | **−0,012400** | | | |

RevIN **làm tệ đi** 0,0124, gấp hơn 18 lần `seed_std` của bản không RevIN. Nó
cũng làm dao động giữa seed tăng 11 lần.

TCN-64 + RevIN: **chưa chạy lại** sau khi sửa công thức `std` cho khớp
Kim et al. Bản chạy dở trước đó đã bỏ.


## Bản chạy thử trên macOS — KHÔNG dùng cho luận văn

Chênh lệch máy đo được ở TN0: cùng mã, cùng dữ liệu, MacBook 0,798748 còn Colab
0,812839, lệch **0,014**. Không đặt cạnh cột GHIJ của Colab.

| cấu hình | tham số | GHIJ | seed | thiết bị |
|---|---:|---:|:---:|---|
| DS-TCN-192 k5n4 do0.2 | 313.945 | 0,785587 | 1 | MPS |
| MixLinear-63 | 63 | 0,687685 | 1 | CPU |
| LSTM-158 | 306.703 | **hỏng giữa chừng** | — | MPS |

Mức tăng từ dev sang GHIJ, tách theo máy:

    Colab   +0,0485 .. +0,0537   (3 cấu hình)
    macOS   +0,0153 .. +0,0290   (2 cấu hình)


## Cấu hình DS-TCN do nhóm tối ưu

Chạy bằng mã và giao thức khác, **không đặt chung bảng TN1**.

| cấu hình | tham số | val_KL | GHIJ | seed |
|---|---:|---:|---:|:---:|
| RF121 (kernel 5, 4 khối) | 310.873 | 0,8587 | 0,808637 ± 0,012651 | 3 |
| RF61 (kernel 3, 4 khối) | 307.801 | 0,8546 | 0,794983 ± 0,011648 | 3 |

Điểm `val_KL` của hai dòng này **được dùng để chọn chính chúng**, nên lạc quan.
Điểm GHIJ thì sạch — GHIJ chưa bao giờ dùng để chọn cấu hình.

Khảo sát loss trên cùng bộ mã đó, `val_KL`:

| loss | val_KL |
|---|---:|
| MSE 0,7 + Pearson 0,3 | **0,8657** ± 0,0050 |
| MSE 0,5 + Pearson 0,5 | 0,8603 |
| Pearson thuần | 0,8586 |
| **MSE thuần** | **0,8337** ± 0,0028 |

Đổi loss được **+0,0320** mà không đụng kiến trúc — lớn hơn toàn bộ khoảng cách
giữa chín kiến trúc TN1 (0,0172).


## Đang chờ chạy

| việc | thời gian | ghi chú |
|---|---|---|
| DS-TCN-192 seed 1, 2 | ~4 giờ | seed 0 đã có |
| GRU-77, 3 seed | ~2 giờ | notebook đã có |
| MixLinear C0 / C2 / C3 | ~45 phút mỗi cái | chạy song song được |
| TN2 tầm nhìn, 5 cấu hình | ~2,5 giờ | 1 fold mỗi cấu hình |
| GHIJ cho 5 cấu hình còn thiếu | ~25 phút mỗi cấu hình mỗi seed | |
| TN3 đổi loss | — | còn lỗi `config_id` không ghi `alpha` |


## Nguồn số

Mọi số `cv_score` và `seed_std` tính lại từ điểm từng fold trong đầu ra
notebook, không chép tay. Số GHIJ lấy từ `docs/CAU_HINH_TN1.txt` và đầu ra
`run_final_test.py`. Trần trên tính bằng `scripts/analyze_oracle_dev.py`.
