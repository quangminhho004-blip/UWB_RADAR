# Khoảng trống nghiên cứu — dùng cho slide

Mỗi mục ghi kèm **số đo** và **câu chú thích giới hạn**. Đừng tách con số ra
khỏi chú thích của nó khi đưa lên slide.


## 1. Phát biểu

> **MobiVital tối ưu *tiêu chí* chọn kênh, nhưng chưa khảo sát *mô hình* thực
> thi tiêu chí đó — dù chính đặc tính của mô hình quyết định ứng viên nào
> thắng.**


## 2. Vì sao đó là khoảng trống thật

Pipeline MobiVital:

    240 ứng viên (120 bin × 2 phép biến đổi)
            |
       lọc invert_detector < 0,8            -> còn khoảng 126
            |
       LSTM chấm từng ứng viên bằng "nó tự dự báo chính nó tốt đến đâu"
       (200 mẫu vào -> 25 mẫu ra, cộng Pearson qua 52 cửa sổ)
            |
       argmax

Bảng 4 của bài báo (arXiv 2503.11064) so **MobiVital / SNR / CFAR / Variance /
Oracle** — toàn bộ là *tiêu chí* chọn kênh. Bộ dự báo thì cố định, không có
khảo sát nào:

| thành phần | MobiVital đặt | bài báo có so không |
|---|---|---|
| kiến trúc | LSTM 352, 2 lớp, **1.502.713 tham số** | không |
| hàm loss | **MSE** | không |
| tầm nhìn | phủ toàn bộ cửa sổ 200 mẫu | không |

Chỗ này quan trọng vì tiêu chí là *"ứng viên nào tự dự báo được chính nó tốt
nhất"*. Nên **thiên kiến quy nạp của bộ dự báo chính là thước đo**. Một mô hình
dự báo được mọi sóng trơn sẽ chấm cao cả kênh nhiễu có cấu trúc. Đây không phải
suy đoán — TN2 đo được.


## 3. Ba câu hỏi con, ba thực nghiệm

### 3.1 Kiến trúc nào? — TN1

So chín kiến trúc ở **cùng ngân sách khoảng 56 nghìn tham số**, 4 fold, 3 seed.

| | tham số | GHIJ |
|---|---:|---:|
| LSTM-352 *(kiến trúc MobiVital)* | 1.502.713 | 0,810302 ± 0,015399 |
| **DS-TCN-64 tầm nhìn 61, loss lai** | **37.081** | **0,803591 ± 0,015350** |
| LSTM-67 | 56.908 | 0,801683 ± 0,002507 |
| DS-TCN-64 bản đầu | 56.281 | 0,795783 ± 0,015413 |

**Chú thích bắt buộc:** cả ba chênh lệch đều nhỏ hơn `seed_std` của hai bên.
Phát biểu đúng là **"đạt điểm ngang, với ít hơn 40,5 lần tham số"**, không phải
"tốt hơn".

### 3.2 Tầm nhìn bao nhiêu? — TN2

Bai et al. (arXiv 1803.01271, mục A.1) khuyên chọn `k` và `d` sao cho tầm nhìn
phủ đủ ngữ cảnh. Đo được **ngược lại**:

| tầm nhìn | c64 | c192 |
|---:|---:|---:|
| 61 | 0,760878 | 0,762714 |
| 121 | 0,757855 | 0,764428 |
| 181 | 0,743657 | 0,732562 |
| 241 | 0,736970 | 0,736623 |

Tương quan tầm nhìn với điểm: **−0,973** (c64) và **−0,845** (c192). Lặp lại ở
hai bề rộng kênh cách nhau ba lần.

Và đây là chỗ đáng chú ý nhất. Khi tầm nhìn dài ra:

    train_mse       0,02127 -> 0,01840    giảm 13%, dự báo CHÍNH XÁC hơn
    train_pearson   0,5851  -> 0,6128     tăng, bắt hình dạng TỐT hơn
    điểm chọn kênh  0,8458  -> 0,7759     TỤT 0,070

**Mô hình dự báo giỏi hơn thì chọn kênh dở hơn.**

**Chú thích bắt buộc:** cơ chế đề xuất — tầm nhìn ngắn chỉ đoán giỏi sóng thật
sự tuần hoàn nên nó lọc giúp — là **giả thuyết, chưa chứng minh**. Và kết luận
chỉ đúng trong dải đã thử: 61 và 121 cùng một nhóm, không xếp hạng được với
nhau (chênh 0,0030, nhỏ hơn dao động seed).

### 3.3 Hàm loss nào? — TN3

Cả pipeline **huấn luyện bằng MSE** nhưng **chấm bằng Pearson**. Pearson bất
biến với thang đo, MSE thì không:

    dự báo A: đúng hình dạng, biên độ gấp 3    MSE 2,0000   Pearson +1,0000
    dự báo B: phẳng lì, đoán bừa số 0          MSE 0,5000   Pearson  0,0000

MSE chọn B, Pearson chọn A. Hàm loss lai khép chỗ lệch đó:

    loss = alpha × MSE + (1 − alpha) × (1 − Pearson)

Quét mười mức alpha trên hai cấu hình, mỗi mức đủ 4 fold:

**20/20 mức alpha đều hơn MSE thuần**, từ +0,0009 tới +0,0192.

**Chú thích bắt buộc:** thứ hạng *giữa các alpha* thì không chuyển được giữa hai
cấu hình — tốp 3 của c64 (0,6 · 0,5 · 0,4) và của c192 (0,2 · 0,3 · 0,0) không
giao nhau mức nào, tương quan hình dạng hai đường cong −0,44. Nói được "có lai
thì hơn MSE thuần", **không** nói được alpha nào tối ưu.

Và mức tăng này **chưa kiểm được trên tập test**: TN4 chưa có nền MSE cùng kiến
trúc để trừ. Đây là giới hạn phải nêu, không được lờ đi.


## 4. Hai đóng góp phụ

**Trần trên trên tập dev = 0,9120.** Bài báo chỉ công bố oracle 0,943 trên tập
test. Đo trên dev cho biết mọi cải tiến chỉ ăn được trong khoảng đó, và cấu
hình tốt nhất hiện còn cách 0,132.

**Một lỗi trong mã MobiVital.** `mobivital_gen.py` dòng 139 ghi cứng
`invert_bit = 0`. `invert_detector` đáng lẽ loại sóng lộn ngược nhưng vài chục
buổi ghi vẫn lọt, mà cờ sửa dấu luôn để 0. So từng buổi ghi giữa hai lần chạy:
khoảng 30 buổi bị chọn phải kênh có tín hiệu đảo dấu, Pearson tụt tới −1,87
tức từ +0,93 xuống −0,94. Sửa được là khoảng **0,02 điểm**.


## 5. Câu chữ — dùng được và không dùng được

| đừng viết | viết thế này |
|---|---|
| "Cải thiện MobiVital" | "Đạt điểm ngang với ít hơn 40,5 lần tham số" |
| "Hàm loss lai cải thiện kết quả" | "Hơn MSE thuần ở 20/20 mức alpha **trên tập dev**" |
| "Tầm nhìn ngắn tốt hơn" | "Tầm nhìn 61–121 là một nhóm, tách rõ khỏi 181–241" |
| "alpha 0,6 là tối ưu" | "alpha 0,6 là mức tốt nhất **đo được** ở cấu hình này" |
| "192 kênh không bằng 64 kênh" | "Không đo được khác biệt, trong khi chi phí chênh 8,3 lần" |

Nguyên tắc chung: **chênh lệch nhỏ hơn `seed_std` thì không được phát biểu
thành thứ hạng.** `seed_std` đo được trong đồ án trải 0,0007–0,0154.


## 6. Cái gì vào phần chính, cái gì xuống phụ lục

Phần chính chỉ nên có thứ trả lời một trong ba câu hỏi ở mục 3.

**Phần chính — không được bỏ:**

- TN1 bảng chín kiến trúc, và bảng GHIJ **có đủ dòng LSTM-352**
- TN2 cả hai bề rộng kênh — bỏ một cái là mất luôn lập luận "lặp lại được"
- TN3 cả hai cấu hình, **kèm việc thứ hạng alpha không chuyển được**
- TN4 toàn bộ bảng, kể cả khi có dòng cao hơn dòng của mình

**Xuống phụ lục — vì trả lời câu hỏi khác, không phải vì bất lợi:**

- Khảo sát ghép nhánh MixLinear (B0 · C0 · C2 · C3). Đây là câu hỏi về mô hình
  cực nhỏ (63–992 tham số), không phải về bộ dự báo cho MobiVital. Thang điểm
  0,65–0,71 cũng không so được với bảng TN1.
- Vòng sàng lọc TN2 một fold, gồm kernel 11 và 13. Vòng 4 fold đã thay thế nó.
- Chi tiết tái lập TN0a/TN0b/TN0c — thuộc chương phương pháp, không phải kết quả.
- Các bản chạy trên macOS. Đã tách sẵn, chênh lệch máy 0,014.

**Lý do phải giữ dòng bất lợi:** hội đồng đọc `docs/BANG_DIEM.md` hoặc hỏi
"LSTM-352 được bao nhiêu" là ra ngay. Bảng thiếu dòng đó **yếu hơn** bảng có nó,
vì lập luận chính của đồ án không phải "thắng" mà là "ngang điểm, nhỏ hơn 40,5
lần" — mà muốn nói "ngang" thì bắt buộc phải trưng con số để so.
