# Bảng điểm — mọi kết quả đã có

Cập nhật mỗi khi có kết quả mới. Điểm là **Pearson macro theo người**: trung
bình từng người, rồi trung bình các người.

Ba cột phải đọc kèm nhau, đừng chỉ nhìn điểm:

    seed    3 seed mới tính được seed_std. 1 seed thì chưa kết luận được gì.
    fold    4 fold là giao thức đầy đủ. 1 fold là vòng sàng lọc, điểm cao hơn
            trung bình +0,040 và có thể đảo thứ hạng — xem docs/SANG_LOC.md
    máy     Colab và macOS chênh nhau 0,014, đo được ở TN0. KHÔNG trừ thẳng
            hai loại cho nhau.


Cây nhánh DS-TCN, xem thực nghiệm nào kế thừa cấu hình nào và nhánh nào
cụt: [`docs/NHANH_DS_TCN.md`](NHANH_DS_TCN.md).

Riêng họ TCN — mọi cấu hình `tcn` và `ds_tcn` gom một chỗ, kèm cả micro lẫn
macro: [`docs/BANG_TCN.md`](BANG_TCN.md).


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

| cấu hình | tham số | cv_score | seed_std | seed | fold |
|---|---:|---:|---:|:---:|:---:|
| DS-TCN-64 k3n4 no_norm do0.2 | **37.081** | **0,760877** | 0,003095 | 3 | 4 |
| DS-TCN-192 k3n4 no_norm do0.2 | 307.801 | **0,747955** | 0,012189 | 3 | 4 |

Điểm từng seed:

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| c64 | 0,758244 | 0,760101 | 0,764286 |
| c192 | 0,757493 | 0,752150 | 0,734222 |

**Tám lần tham số, điểm thấp hơn 0,0129.** Chênh lệch này lớn hơn `seed_std` của
bản c64 (0,003095) nhưng xấp xỉ `seed_std` của bản c192 (0,012189), nên đọc là
**c192 không hơn c64**, chưa đọc được là c192 kém hơn.

`seed_std` của c192 lớn gấp bốn lần c64 — seed 2 tụt hẳn xuống 0,734222. Model
to hơn không những không giúp mà còn kém ổn định hơn.

**Một quan sát về nguồn dao động.** Seed 0 của c192 **chạy hai lần ra hai số:
0,762714 và 0,757493**, lệch 0,0052. Lần chạy lại do ô khôi phục kết quả im lặng
không làm gì, nên `run_cv.py` không thấy kết quả cũ và train lại. Nghĩa là
`seed_std` **chưa phải toàn bộ nguồn dao động** — cùng seed, khác phiên và khác
thiết bị vẫn lệch. Số trong bảng lấy từ tệp nén mới nhất trên Drive.


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


## TN2 — tầm nhìn DS-TCN

Chia hai bảng theo nguyên tắc **chỉ đặt cạnh nhau những cấu hình chạy chung một
lượt**. Ghép hai lượt chạy khác nhau vào một bảng là đưa thêm một biến không
kiểm soát được vào phép so.

### Chạy chung — thực nghiệm `tn2_rf`, 4 fold, 1 seed

Giữ nguyên 4 khối, chỉ đổi kernel, nên tham số gần như đứng yên trong khi tầm
nhìn gấp đôi. Điểm đổi thì gần như chắc do tầm nhìn.

| tầm nhìn | kernel | c64 *(38–40k)* | c192 *(311–317k)* |
|---:|---:|---:|---:|
| 121 | 5 | 0,757855 | **0,764428** |
| 181 | 7 | **0,743657** | 0,732562 |
| 241 | 9 | **0,736970** | 0,736623 |

Tương quan tầm nhìn với điểm: **−0,979** (c64) và **−0,802** (c192). Tầm nhìn
dài hơn cho điểm thấp hơn, ở cả hai bề rộng kênh cách nhau tám lần.

Ba điểm mỗi cột thì tương quan chưa nói được nhiều. Thứ đọc được chắc hơn là
**mức 121 hơn hẳn hai mức kia**: hơn 181 là 0,0142 (c64) và 0,0319 (c192), đều
lớn hơn dao động giữa các seed đo được trong đồ án (0,0007–0,0154).

### Chạy riêng ở TN1 — chỉ để tham chiếu, không đặt chung bảng trên

| tầm nhìn | kernel | tham số | cv_score | seed |
|---:|---:|---:|---:|:---:|
| 61 | 3 | 37.081 | **0,760878** ± 0,003095 | 3 |
| 61 | 3 | 307.801 | **0,747955** ± 0,012189 | 3 |

Hai dòng này chạy ở thực nghiệm `tn1`, khác lượt với ba mức trên. Đặt chung
bảng thì không biết chênh lệch đến từ tầm nhìn hay từ lần chạy.

Riêng dòng `c192` còn có chuyện seed 0 chạy hai lần ra hai số khác nhau — xem
mục **TN1 — DS-TCN 192 kênh, tầm nhìn 61** ở trên.

### Vòng sàng lọc trước đó, c64, 1 fold `val_KL`

| kernel | tầm nhìn | điểm `val_KL` | train_mse | train_pearson |
|---:|---:|---:|---:|---:|
| 5 | 121 | **0,8458** | 0,02127 | 0,5851 |
| 7 | 181 | 0,8176 | 0,02059 | 0,5921 |
| 9 | 241 | 0,8041 | 0,01939 | 0,6041 |
| 11 | 301 | 0,7940 | 0,01861 | 0,6130 |
| 13 | 361 | 0,7759 | 0,01840 | 0,6128 |

Điểm tụt 0,070 trong khi `train_mse` **giảm 13%** và `train_pearson` **tăng** —
model dự báo giỏi hơn nhưng chọn kênh dở hơn. Cơ chế có thể giải thích: tiêu chí
chọn kênh là "ứng viên nào tự dự báo được chính nó tốt nhất", nên model tầm nhìn
ngắn chỉ đoán giỏi sóng thật sự tuần hoàn, còn model tầm nhìn dài đoán giỏi mọi
sóng trơn, kể cả kênh nhiễu có cấu trúc. Đây là **cách đọc đề xuất, chưa chứng
minh**.

Điểm một fold cao hơn `cv_score` bốn fold trung bình +0,040 và có thể đảo thứ
hạng, nên bảng này chỉ dùng để **loại**, không dùng để **chọn**.


## TN3 — hàm loss lai, 4 fold, 1 seed

`alpha` là trọng số của MSE: `loss = alpha·MSE + (1 − alpha)·(1 − Pearson)`. Nên
`alpha = 1` là MSE thuần, `alpha = 0` là Pearson thuần.

Ba cấu hình, mỗi cấu hình đủ mười một điểm.

| alpha | c64 k3 RF61 | c64 k5 RF121 | c192 k5 RF121 |
|---:|---:|---:|---:|
| 0,0 | 0,776667 | **0,780306** | 0,772640 |
| 0,1 | 0,769390 | 0,763183 | 0,771848 |
| 0,2 | 0,775264 | 0,771996 | **0,776011** |
| 0,3 | 0,771931 | 0,776213 | 0,774584 |
| 0,4 | 0,779266 | 0,771848 | 0,768337 |
| 0,5 | 0,779419 | 0,763457 | 0,769950 |
| 0,6 | **0,780028** | *0,752386* | 0,765333 |
| 0,7 | 0,771665 | 0,761892 | 0,769263 |
| 0,8 | 0,776611 | 0,760013 | 0,770282 |
| 0,9 | 0,764941 | 0,758236 | 0,771893 |
| **1,0 — MSE thuần** | **0,760878** *(3 seed)* | **0,757855** | **0,764428** |

### Điều lặp lại được

**29/30 mức alpha hơn MSE thuần của chính cấu hình đó.** Ba cấu hình khác nhau
cả số kênh lẫn tầm nhìn. Mức duy nhất thua là `c64 k5 RF121 alpha 0,6`, thấp hơn
0,0055.

### Điều KHÔNG lặp lại được

**Thứ hạng giữa các alpha không chuyển được giữa các cấu hình.**

| | đỉnh | tốp 3 |
|---|---|---|
| c64 k3 RF61 | alpha **0,6** | 0,6 · 0,5 · 0,4 |
| c64 k5 RF121 | alpha **0,0** | 0,0 · 0,3 · 0,2 |
| c192 k5 RF121 | alpha **0,2** | 0,2 · 0,3 · 0,0 |

Ba đỉnh nằm ở ba chỗ khác nhau. Alpha 0,6 là đỉnh của cấu hình thứ nhất nhưng là
**mức duy nhất trong cả đồ án thua MSE thuần** ở cấu hình thứ hai — mà hai cấu
hình đó chỉ khác nhau **tầm nhìn**.

Biên độ dao động bên trong mỗi cột là 0,0151 · 0,0279 · 0,0107, cùng cỡ hoặc lớn
hơn cận trên của `seed_std` đo được ở TN1 (0,0007–0,0154).

**Hiệu ứng "có lai thì hơn" nằm trên nhiễu. Hiệu ứng "alpha nào tốt hơn alpha
nào" chìm trong nhiễu.**

### Phát biểu đúng

Nói được: **thêm thành phần Pearson vào loss hơn MSE thuần, ở cả ba cấu hình,
29 trên 30 mức.** Không nói được alpha nào tối ưu.

Nếu phải chọn một giá trị, **alpha = 0,2 hoặc 0,3** là lựa chọn có cơ sở nhất —
hai mức duy nhất nằm trong nửa trên của **cả ba** cột.

### Chú thích về `cv_std` (~0,07)

Đó là **độ lệch giữa bốn fold**, không phải giữa seed. `val_DF` thấp hơn ba fold
kia ở **mọi** mức alpha và ở **cả ba** cấu hình. Nó dịch cả cột xuống như nhau
nên không làm hỏng phép so.


## Khảo sát ghép nhánh MixLinear — 4 fold, 1 seed

Nhánh phụ là **đề xuất của đồ án**, không thuộc MixLinear nguyên bản.

| | không nhánh phụ | có nhánh phụ |
|---|---|---|
| **không MixLinear** | — | **C0** 0,654737 *(929)* |
| **có MixLinear** | **B0** 0,676037 *(63)* | **C2** 0,661746 *(992)* |

**C3** = **0,709744** *(992 tham số, thêm GELU giữa hai lớp nhánh phụ)*

| phép so | chênh | sạch không |
|---|---:|---|
| **C3 − C2** — phi tuyến giúp gì | **+0,0480** | **sạch**: cùng 992 tham số, cùng trọng số khởi tạo, khác đúng một `GELU` |
| C2 − B0 — thêm nhánh tuyến tính | −0,0143 | lệch 929 tham số |
| C2 − C0 — thêm MixLinear | +0,0070 | lệch 63 tham số |

Hai điều đọc được. **Thêm nhánh tuyến tính làm tệ đi**: 992 tham số thua 63 tham
số, nên không phải cứ thêm tham số là tốt. Và **toàn bộ lợi ích đến từ hàm phi
tuyến**, không từ số tham số — C2 và C3 cùng kích thước, cùng khởi tạo.

Chú thích khi báo cáo: ở C2 bias lớp đầu (4 tham số) bị hấp thụ vào bias lớp sau
vì hai `Linear` không phi tuyến hợp lại thành một phép affine, nên C2 chỉ có
**988** tham số tác dụng độc lập. Chênh 0,4%, không hỏng phép so.

Cả nhóm này ở thang điểm thấp hơn hẳn TN1 (0,65–0,71 so với 0,74–0,76) vì model
quá nhỏ. Không so trực tiếp với bảng TN1.


## TN4 — test cuối trên GHIJ, 3 seed

Train đủ ABCDEFKL, chấm một lần trên GHIJ. Đây là **số công bố**.

| cấu hình | tham số | GHIJ | seed_std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|---:|---:|
| **DS-TCN-64 RF61, alpha 0,6** | **37.081** | **0,803591** | 0,015350 | 0,786906 | 0,817115 | 0,806751 |
| DS-TCN-192 RF121, alpha 0,2 | 310.873 | 0,800721 | 0,010705 | 0,796041 | 0,793154 | 0,812969 |

Đặt cạnh các mốc **cùng pipeline, cùng Colab, cùng 3 seed**:

| | tham số | GHIJ | so với DS-TCN-64 |
|---|---:|---:|---:|
| LSTM-352 *(kiến trúc MobiVital)* | 1.502.713 | 0,810302 ± 0,015399 | −0,0067 |
| **DS-TCN-64 RF61 alpha 0,6** | **37.081** | **0,803591 ± 0,015350** | — |
| LSTM-67 | 56.908 | 0,801683 ± 0,002507 | +0,0019 |
| DS-TCN-64 bản đầu | 56.281 | 0,795783 ± 0,015413 | +0,0078 |

**Cả ba phép so đều nhỏ hơn `seed_std` của hai bên.** Không phân biệt được cấu
hình nào hơn. Phát biểu đúng: DS-TCN-64 đạt điểm **ngang** kiến trúc MobiVital
trên tập test, với **ít hơn 40,5 lần tham số**.

### Mức tăng dev sang GHIJ không giữ nguyên như dự tính

| | dev | GHIJ | mức tăng |
|---|---:|---:|---:|
| ba cấu hình train bằng MSE | — | — | **+0,0485 .. +0,0537** |
| DS-TCN-64 alpha 0,6 | 0,780028 | 0,803591 | **+0,0236** |
| DS-TCN-192 alpha 0,2 | 0,776011 | 0,800721 | **+0,0247** |

Hai cấu hình loss lai chỉ tăng bằng **một nửa**. Lợi ích +0,019 mà TN3 đo được
trên dev **không chuyển hết sang tập test**.

**Chưa giải thích được nguyên nhân**, vì ba cấu hình MSE kia khác cả kiến trúc
lẫn hàm loss — hai biến đổi cùng lúc. Muốn tách thì phải chạy **đúng kiến trúc
DS-TCN-64 RF61 với MSE thuần trên GHIJ**, hiện còn thiếu (dev đã có: 0,760878).

### Người H và I kéo điểm xuống ở mọi cấu hình

| | G | H | I | J |
|---|---:|---:|---:|---:|
| DS-TCN-64 alpha 0,6, ba seed | 0,910–0,918 | **0,635–0,686** | **0,696–0,760** | 0,902–0,912 |
| DS-TCN-192 alpha 0,2, ba seed | 0,911–0,917 | **0,636–0,697** | **0,717–0,745** | 0,883–0,899 |

MobiVital cũng vậy — TN0a cho H 0,6907 và I 0,7658. Nên đây là **đặc tính của
hai người đó**, không phải điểm yếu riêng của cấu hình nào.


## Đang chờ chạy

| việc | thời gian | ghi chú |
|---|---|---|
| **TN3 quét alpha cho DS-TCN-64 RF121** | **~10 giờ** | **10 mức, 1 giờ mỗi mức, chạy rời được** |
| **TN4 GHIJ cho DS-TCN-64 RF121** | **~2 giờ** | **3 seed loss lai + 3 seed MSE thuần, tách đóng góp hàm loss trên tập test** |
| GHIJ cho DS-TCN-64 RF61 MSE thuần | ~1 giờ | 3 seed |
| DS-TCN-192 seed 1, 2 | ~4 giờ | seed 0 đã có |
| GRU-77, 3 seed | ~2 giờ | notebook đã có |
| MixLinear C0 / C2 / C3 seed 1, 2 | ~45 phút mỗi cái | seed 0 đã có |
| TN2 tầm nhìn, seed 1, 2 | ~1,8 giờ mỗi seed mỗi bề rộng | seed 0 đã có |
| GHIJ cho các cấu hình còn thiếu | ~25 phút mỗi cấu hình mỗi seed | |

## Nguồn số

Mọi số `cv_score` và `seed_std` tính lại từ điểm từng fold trong đầu ra
notebook, không chép tay. Số GHIJ lấy từ `docs/CAU_HINH_TN1.txt` và đầu ra
`run_final_test.py`. Trần trên tính bằng `scripts/analyze_oracle_dev.py`.
