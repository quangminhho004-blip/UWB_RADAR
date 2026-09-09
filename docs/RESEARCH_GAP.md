# Khoảng trống nghiên cứu — bản viết cho slide

Mỗi mục kèm số đo và câu chú thích giới hạn. Khi lên slide, **đừng tách con số
ra khỏi chú thích của nó**.


## 1. MobiVital làm gì

Radar UWB chia không gian trước mặt thành 120 lớp theo khoảng cách. Người nằm ở
lớp nào thì không biết trước. Mỗi lớp lại rút ra được 2 chuỗi tín hiệu. Tổng
cộng **240 chuỗi ứng viên**, chỉ một cái chứa nhịp thở.

Bài toán: **chọn đúng chuỗi đó, mà không được nhìn đáp án.**

Cách MobiVital làm: huấn luyện một model dự báo, rồi cho nó chấm từng ứng viên
bằng câu hỏi *"chuỗi này có tự dự báo được chính nó không"*. Chuỗi tuần hoàn thì
dự báo được, nhiễu thì không. Chọn chuỗi nào dự báo được tốt nhất.


## 2. Chỗ bài báo không kiểm

Bài báo so **tiêu chí chọn** của họ với ba tiêu chí khác — Bảng 4, gồm SNR,
CFAR, Variance. Nhưng cái **model** thực hiện tiêu chí đó thì họ chỉ mô tả, một
câu, rồi thôi:

> *"Our autoregressive reconstruction model is a lightweight 2-layer LSTM model
> followed by a linear layer. […] The Mean Squared Error (MSE) between the
> predicted segment and the future segment is used as the loss function."*

Ba lựa chọn nằm gọn trong câu đó, và **không cái nào được so với phương án khác**:

| | MobiVital chọn | bài báo có so không |
|---|---|---|
| kiến trúc | LSTM 2 lớp, **1.502.713 tham số** | không |
| hàm mục tiêu | **MSE** | không |
| tầm nhìn | phủ hết 200 mẫu đầu vào | không |

Bài báo có một ablation, nhưng là cho **bộ phát hiện đảo pha**, không phải cho
model.

**Vì sao đây là khoảng trống thật, không phải bới lông tìm vết.** Tiêu chí ở đây
là *"chuỗi nào tự dự báo được chính nó"*. Nên **model chính là cái thước**. Đổi
model là đổi thước. Một model quá mạnh sẽ dự báo được cả nhiễu trơn, và chọn
nhầm — đây không phải suy đoán, nhóm đo được ở mục 3.2.


## 3. Ba câu hỏi nhóm đặt ra

### 3.1 Kiến trúc nào hợp? — TN1

So chín kiến trúc, tất cả ở **cùng ngân sách khoảng 56 nghìn tham số**, cùng
giao thức, 3 seed.

| | tham số | điểm trên tập kiểm tra độc lập |
|---|---:|---:|
| LSTM-352 *(kiến trúc MobiVital)* | 1.502.713 | 0,810302 ± 0,015399 |
| **DS-TCN-64, hàm loss lai** | **37.081** | **0,803591 ± 0,015350** |
| LSTM-67 | 56.908 | 0,801683 ± 0,002507 |
| DS-TCN-64 bản đầu | 56.281 | 0,795783 ± 0,015413 |

**Chú thích bắt buộc.** Chênh lệch giữa các dòng nhỏ hơn độ lệch giữa các seed
của chính chúng. Nên câu đúng là **"đạt điểm ngang, với ít hơn 40,5 lần tham
số"** — không phải "tốt hơn".

### 3.2 Tầm nhìn bao nhiêu là đủ? — TN2

Sách vở về TCN nói tầm nhìn phải phủ hết ngữ cảnh. Đo ra **ngược lại**:

| tầm nhìn | 64 kênh | 192 kênh |
|---:|---:|---:|
| 121 | 0,757855 | 0,764428 |
| 181 | 0,743657 | 0,732562 |
| 241 | 0,736970 | 0,736623 |

Tầm nhìn dài ra thì điểm tụt, và tụt đều ở cả hai bề rộng kênh. Ba mức này chạy
chung một lượt nên so được với nhau; mức 61 chạy ở lượt khác nên để riêng.

Chỗ đáng nói nhất: cùng lúc đó, **model dự báo giỏi lên**.

```
sai số dự báo    0,02127 -> 0,01840     giảm 13%
điểm chọn kênh   0,8458  -> 0,7759      tụt 0,070
```

**Model dự báo càng giỏi thì chọn kênh càng dở.** Nghe ngược, nhưng hợp với mục
2: thước đo mạnh quá thì đo cái gì cũng "được", mất khả năng phân biệt.

**Chú thích bắt buộc.** Cách giải thích trên là **giả thuyết nhóm đề xuất, chưa
chứng minh**. Và kết luận chỉ đúng trong dải đã thử: 61 và 121 nằm cùng một
nhóm, chênh nhau 0,0030, không xếp hạng được với nhau.

### 3.3 Hàm mục tiêu nào đúng? — TN3

Cả hệ thống **huấn luyện bằng sai số bình phương** nhưng **chấm bằng tương quan
Pearson**. Hai thước đo này không cùng ý:

```
dự báo A: đúng hình dạng, biên độ gấp 3    sai số 2,0000   Pearson +1,0000
dự báo B: phẳng lì, đoán bừa số 0          sai số 0,5000   Pearson  0,0000
```

Sai số chọn B. Pearson chọn A. Model được thưởng khi **đoán an toàn**, trong khi
lúc chấm lại cần nó **bắt đúng hình dạng**.

Nhóm trộn hai thứ lại, quét 10 mức trên 2 cấu hình, mỗi mức chạy đủ 4 lần chia
dữ liệu. Kết quả: **20/20 mức đều hơn sai số bình phương thuần**, từ +0,0009 tới
+0,0192.

**Chú thích bắt buộc.** Chỉ nói được *"trộn thì hơn"*. **Không** nói được tỉ lệ
nào tốt nhất — ba mức tốt nhất của hai cấu hình không trùng nhau mức nào. Và
mức tăng này đo trên tập phát triển; trên tập kiểm tra độc lập chưa có nền để
trừ ra, nên chưa kiểm được.


## 4. Hai thứ nhỏ nhưng nên có slide

**Giới hạn trên = 0,9120.** Điểm đạt được nếu luôn chọn đúng ứng viên tốt nhất.
Bài báo chỉ công bố con số này trên tập kiểm tra; nhóm đo thêm trên tập phát
triển. Nó cho biết mọi cải tiến chỉ ăn được trong khoảng đó.

**Một lỗi trong mã của tác giả.** Cờ sửa dấu sóng bị ghi cứng bằng 0, không bao
giờ bật. Bộ phát hiện đảo pha đáng lẽ loại sóng lộn ngược, nhưng vài chục phiên
đo vẫn lọt. So từng phiên đo giữa hai lần chạy: có phiên tương quan tụt tới
−1,87, tức từ +0,93 xuống −0,94 — sóng bị lật hẳn. Sửa được là khoảng 0,02 điểm.


## 5. Nói thế nào cho đúng

| đừng nói | nói thế này |
|---|---|
| "Cải thiện MobiVital" | "Đạt điểm ngang, với ít hơn 40,5 lần tham số" |
| "Hàm loss lai cải thiện kết quả" | "Hơn sai số bình phương ở 20/20 mức, **trên tập phát triển**" |
| "Tầm nhìn ngắn tốt hơn" | "Tầm nhìn 61–121 là một nhóm, tách rõ khỏi 181–241" |
| "Tỉ lệ trộn 0,6 là tối ưu" | "0,6 là mức tốt nhất **đo được** ở cấu hình này" |
| "192 kênh thua 64 kênh" | "Không đo được khác biệt, trong khi chi phí chênh 8,3 lần" |
| "Nhóm bổ sung bước tính nhịp/phút mà bài báo không có" | Bài báo **có làm**, mục 5.3.5, sai số 0,68 nhịp/phút. Nhưng **không phát hành mã** — nhóm tái lập theo mô tả |

Nguyên tắc chung: **chênh lệch nhỏ hơn độ lệch giữa các seed thì không được
phát biểu thành thứ hạng.** Độ lệch đo được trong đồ án trải từ 0,0007 đến
0,0154.


## 6. Cái gì vào phần chính, cái gì xuống phụ lục

Tiêu chí: **có trả lời một trong ba câu hỏi ở mục 3 hay không** — không phải có
lợi hay bất lợi.

**Phần chính, không được bỏ**

- Bảng chín kiến trúc, và bảng điểm cuối **có đủ dòng LSTM-352**
- TN2 cả hai bề rộng kênh — bỏ một cái là mất lập luận "lặp lại được"
- TN3 cả hai cấu hình, **kèm việc thứ hạng không chuyển được giữa chúng**
- Bảng điểm cuối đầy đủ, kể cả khi có dòng cao hơn dòng của mình

**Xuống phụ lục, vì trả lời câu hỏi khác**

- Khảo sát MixLinear và các biến thể ghép nhánh. Đây là câu hỏi về model cực nhỏ
  63–992 tham số, không phải về model chọn kênh. Thang điểm cũng khác hẳn.
- Vòng sàng lọc một lần chia dữ liệu. Vòng đầy đủ đã thay thế nó.
- Chi tiết tái lập bài báo gốc — thuộc chương phương pháp.
- Các bản chạy trên máy cá nhân. Chênh lệch máy 0,014, đã tách riêng sẵn.

**Vì sao phải giữ dòng bất lợi.** Lập luận chính của đồ án không phải "thắng" mà
là **"ngang điểm, nhỏ hơn 40,5 lần"**. Muốn nói "ngang" thì bắt buộc phải trưng
con số để so. Bỏ dòng LSTM-352 đi thì còn lại một con số lơ lửng, không chứng
minh được gì — mà hội đồng mở bảng điểm ra là thấy ngay.
