# Thiết kế ba cấu hình MixLinear để Claude review

Ngày: 07/09/2026. Trạng thái: **đã review, đã triển khai, chưa có kết quả**.

## Kết quả review và những gì đã đổi so với bản đề xuất

Ba thay đổi so với tài liệu này, đã thống nhất:

1. **Bỏ C1 (tăng `lpf` lên 10)** khỏi vòng sàng lọc. Ba suất chạy dành cho C0,
   C2, C3 để tập trung vào hai câu hỏi chính.

2. **Thêm C0**: nhánh phụ đứng một mình, không có MixLinear, 929 tham số. Lý
   do: C2 trừ B0 đổi cùng lúc hai thứ — thêm đường dự báo trực tiếp VÀ thêm 929
   tham số vào một model 63 tham số. C0 cho đọc được theo chiều còn lại.

   C0 xử lý trung bình y hệt: `linear(x - mu) + mu`. Không có bước này thì đối
   chứng khác cả tiền xử lý.

3. **Đính chính về "cùng 992 tham số"**: ở C2, bias lớp đầu bị hấp thụ vào bias
   lớp sau, vì hai `Linear` không phi tuyến hợp lại thành đúng một phép affine.
   C2 có **988** tham số tác dụng độc lập so với 992 của C3. Đã kiểm bằng số.
   Chênh 0,4%, không hỏng phép so, nhưng phải chú thích khi báo cáo.

Ba lưu ý khi đọc kết quả:

- C0 gần C2 **không** chứng minh "MixLinear vô dụng" — chỉ nói được là chưa
  thấy lợi ích rõ trong điều kiện đã thử, nhất là với một seed.
- C0 (929) vẫn khác C2 (992) 63 tham số, nên chưa tách hoàn toàn kiến trúc khỏi
  số tham số.
- **B0 có `seed_std` 0,0066**, ba seed trải 0,6648–0,6764. Chênh lệch dưới
  khoảng 0,007 trên một seed là ngẫu nhiên, không đọc được gì.

Triển khai: `src/models.py` lớp `LowRankLinear` và `MixLinearPlus`; model key
`low_rank_linear`, `mix_linear_linear`, `mix_linear_mlp`; cờ
`--correction_hidden`; notebook `notebooks/TN_MixLinear_ablation.ipynb`; nhóm
kết quả `tn_mixlinear_ablation`.

## 1. Quyết định cần review

Giữ MixLinear 63 tham số đã chạy làm baseline. Thử ba cấu hình mới:

1. MixLinear giữ thêm hệ số FFT, không thêm nhánh.
2. MixLinear 63 + nhánh tuyến tính hai lớp, chiều giữa bằng 4.
3. MixLinear 63 + nhánh MLP cùng kích thước, thêm GELU giữa hai lớp.

Mục đích là phân biệt tác dụng của giữ thêm thông tin phổ, thêm đường dự báo trực tiếp và thêm phi tuyến. Không chạy đồng thời cả ba cấu hình chia đoạn từng đề xuất trước đó. Không cố tăng chiều giữa nhánh FFT lên 60 chỉ để đạt khoảng 1.000 tham số.

Hai nhánh bổ sung là **biến thể đề xuất của đồ án**, không phải kiến trúc MixLinear nguyên bản. Chưa có bằng chứng chúng cải thiện điểm chọn kênh UWB.

## 2. Bảng cấu hình chốt để review

| Thuộc tính | B0: baseline đã chạy | C1: thêm hệ số FFT | C2: thêm Linear | C3: thêm MLP |
|---|---:|---:|---:|---:|
| Input / output | 200 / 25 | 200 / 25 | 200 / 25 | 200 / 25 |
| Số tín hiệu đầu vào | 1 | 1 | 1 | 1 |
| `period_len` | 10 | 10 | 10 | 10 |
| Số đoạn lịch sử | 20 | 20 | 20 | 20 |
| `lpf` | 5 | **10** | 5 | 5 |
| `mix_hidden` của FFT | 2 | 2 | 2 | 2 |
| `mix_alpha` | 0,5 | 0,5 | 0,5 | 0,5 |
| Nhánh bổ sung | Không | Không | 200 → 4 → 25 | 200 → 4 → GELU → 25 |
| Bias của hai lớp bổ sung | — | — | Có | Có |
| Dropout / norm bổ sung | Không | Không | Không | Không |
| RevIN | Tắt | Tắt | Tắt | Tắt |
| Tổng tham số thực | **63** | **83** | **992** | **992** |

`mix_hidden=2` thuộc nhánh FFT gốc; chiều ẩn `4` thuộc nhánh bổ sung. Hai tham số này độc lập. `mix_alpha` trộn hai nhánh gốc, không phải alpha của hybrid loss và không điều khiển nhánh bổ sung.

## 3. Phần MixLinear được giữ lại

Theo bản cài đặt trong `src/models.py`, đối chiếu với [mã tác giả](https://github.com/aitianma/MixLinear/blob/main/models/MixLinear.py): trừ trung bình cửa sổ, Conv1D có nối tắt, chia đoạn, xử lý song song miền thời gian và miền tần số, trộn dự báo rồi cộng lại trung bình.

Thông số cụ thể của nền B0 trong project:

- Conv1D: 1 → 1 kênh, kernel 11, stride 1, padding 5, không bias; cộng với tín hiệu trước Conv. Kernel được học, không bảo đảm luôn là bộ lọc làm mượt.
- Nhánh thời gian: đệm 20 đoạn thành 25 ô, xếp lưới 5 × 5; hai Linear(5, 2), không bias, tác động lần lượt lên hai chiều lưới. Thu được 40 mẫu, lấy 25 mẫu đầu.
- Nhánh tần số: FFT trên trục 20 đoạn; lấy 5 hệ số đầu; Linear phức 5 → 2 → 3, không bias, không kích hoạt; IFFT và lấy phần thực. Thu được 30 mẫu, lấy 25 mẫu đầu.
- Đầu ra: `0.5 * time_prediction + 0.5 * frequency_prediction + input_mean`.

Đây là cấu hình áp dụng cho input 200/output 25 của đồ án, không phải khẳng định tác giả dùng cùng cấu hình cho mọi dataset. Không thêm ReLU/GELU vào nhánh FFT gốc.

## 4. C1 — MixLinear với lpf = 10

Giữ B0, chỉ đổi `lpf: 5 → 10`. Nhánh tần số thành Linear phức **10 → 2 → 3**. Conv, nhánh thời gian, cách chia đoạn, tỷ trọng trộn đều giữ nguyên.

Giả thuyết: lấy thêm hệ số FFT có thể giữ lại thông tin hữu ích mà B0 bỏ đi. Đây không phải bảo đảm tốt hơn; thêm hệ số cũng có thể thêm nhiễu. Giữ nguyên chiều giữa 2 nên phép ánh xạ phổ vẫn có hạng tối đa 2.

Đếm tham số:

```text
Conv                             11 số thực
Hai Linear miền thời gian        2 × (5 × 2) = 20 số thực
Hai Linear miền tần số           (10 × 2 + 2 × 3) = 26 số phức
Tổng                             11 + 20 + 2 × 26 = 83 số thực
```

`lpf` đếm hệ số trong FFT của chuỗi đã chia đoạn, không phải số Hz hoặc số bin radar.

## 5. C2 và C3 — nhánh bổ sung dùng chung đặc tả

### 5.1. Đầu vào và cách ghép

Chốt nhánh phụ nhận **toàn bộ 200 mẫu đã trừ trung bình cửa sổ**, trước Conv/chia đoạn; không chia độ lệch chuẩn. Cả C2 và C3 dùng cùng lựa chọn này.

```python
# Mã mô tả thiết kế, chưa phải class đã có trong project.
# x: (batch, 200), đúng tensor đầu vào hiện tại của pipeline.
mu = x.mean(dim=1, keepdim=True)       # (batch, 1)
z = x - mu                           # (batch, 200)
base_prediction = base_mixlinear(x)  # (batch, 25), đã cộng lại mu bên trong
correction = correction_branch(z)   # (batch, 25)
prediction = base_prediction + correction
```

Không cộng `mu` lần thứ hai. Không thêm hệ số trộn có thể học, gate, lớp norm, dropout hoặc hàm kích hoạt sau phép cộng cuối. Nhánh phụ nhận cùng dữ liệu lịch sử mà nhánh gốc được phép nhìn.

### 5.2. C2 — MixLinear + Linear

```python
correction_branch = nn.Sequential(
    nn.Linear(200, 4, bias=True),
    nn.Linear(4, 25, bias=True),
)
```

Hai lớp không có kích hoạt nên hợp lại thành một ánh xạ affine; phần ma trận có hạng tối đa 4. Đây là nhánh tuyến tính phân tích thành hai ma trận để tiết kiệm tham số, **không phải** lớp Linear(200, 25) đầy đủ. Bias làm phép ánh xạ là affine theo nghĩa toán học.

Giả thuyết: thêm đường dự báo trực tiếp có thể bù hạn chế của cách chia đoạn/chia sẻ trọng số trong B0, dù vẫn không có phi tuyến.

### 5.3. C3 — MixLinear + MLP

```python
correction_branch = nn.Sequential(
    nn.Linear(200, 4, bias=True),
    nn.GELU(approximate="none"),
    nn.Linear(4, 25, bias=True),
)
```

Khác C2 đúng ở GELU. Giả thuyết: phép biến đổi phi tuyến có thể tạo hiệu chỉnh mà nhánh affine cùng kích thước không biểu diễn được. Chiều 4 là lựa chọn thử nghiệm nhỏ theo ngân sách khoảng 1.000 tham số, không phải giá trị tối ưu được chứng minh hay lấy nguyên từ một bài báo.

### 5.4. Số tham số và huấn luyện

```text
Linear(200, 4), có bias          200 × 4 + 4 = 804
Linear(4, 25), có bias           4 × 25 + 25 = 125
Nhánh bổ sung                   929
MixLinear nền                   63
Tổng mỗi biến thể               992 tham số thực
```

GELU không có tham số học được. C2/C3 bằng số tham số nhưng chưa chắc bằng latency.

- Train **toàn bộ model từ đầu**, đồng thời cập nhật nền MixLinear và nhánh phụ; không đóng băng nền, không nạp checkpoint B0 để fine-tune.
- Một loss MSE trên `prediction` cuối. Không thêm loss riêng cho nhánh phụ, không tạo nhãn residual riêng.
- Dùng khởi tạo mặc định của các Linear PyTorch cho cả hai biến thể; không zero-init riêng một nhánh.
- Tạo module theo cùng thứ tự: MixLinear nền → Linear đầu → kích hoạt không tham số → Linear cuối. Với cùng seed, kiểm tra C2/C3 có cùng trọng số khởi tạo tương ứng và cùng thứ tự minibatch. GELU/Identity không tiêu thụ RNG.
- Khi báo cáo, “hiệu chỉnh” mô tả cách cộng đầu ra. Vì train chung, không khẳng định nhánh phụ chỉ học phần sai của một MixLinear đã cố định.

## 6. Pipeline và điều kiện huấn luyện

| Mục | Giá trị chung |
|---|---|
| Dữ liệu phát triển | ABCDEFKL |
| Fold validation | AB, CE, DF, KL; mỗi fold train 6 người còn lại |
| Input / target | 200 mẫu lịch sử → 25 mẫu tiếp theo theo windows hiện có |
| Ngưỡng lọc train `corr` | 0,9 |
| Loss | MSE |
| Optimizer | Adam, learning rate 0,0001, weight decay 0 |
| Batch size / epochs | 64 / 20 |
| Checkpoint để chấm | Sau epoch cuối, cùng cách TN1 hiện tại |
| Vòng sàng lọc | Seed 0, đủ 4 fold cho mỗi cấu hình mới |
| Vòng xác nhận | Bổ sung seed 1 và 2 theo quy tắc mục 7 |

Giữ nguyên cách tạo windows và nguồn dữ liệu train hiện có, kể cả các cửa sổ từ GT nếu pipeline đang đưa chúng vào. Không đổi target radar thành GT của đai đo trong thiết kế này.

Model tiếp tục dự báo chuỗi để phục vụ **chọn ứng viên (bin, method)** bằng scoring hiện tại. Không ghép dự báo thành sóng hô hấp mới, không đưa GT của session validation/test vào input. Điểm chính vẫn là chất lượng sóng radar được chọn so với GT, macro theo người; không thay bằng MSE dự báo.

## 7. Quy tắc sàng lọc và đọc kết quả

Đề xuất khóa quy tắc này trước khi train:

1. Chạy C1/C2/C3 seed 0: tổng **12 lần train fold mới**. So với **B0 seed 0**, không so số một seed với mean ba seed của B0. Chỉ tái sử dụng B0 nếu cấu hình, dữ liệu và scoring tương thích.
2. Nếu C1 có `cv_score` seed 0 không thấp hơn B0 seed 0, chạy thêm seed 1, 2 cho C1.
3. Nếu ít nhất một trong C2/C3 đạt điều kiện trên, chạy thêm seed 1, 2 cho **cả C2 và C3**, để đối chứng tác dụng GELU trên cùng tập seed.
4. Nếu không cấu hình nào đạt, dừng nhánh thử này theo ngân sách. Ghi là không vượt baseline trong vòng sàng lọc; không kết luận đã chứng minh chúng luôn kém.

Vòng xác nhận tốn thêm 8 lần train fold cho mỗi cấu hình được giữ. Tối đa 36 lần train fold mới nếu cả ba được xác nhận. Quy tắc trên là quyết định tiết kiệm chi phí, **không phải kiểm định thống kê**; một seed có thể loại nhầm cấu hình tốt.

Các so sánh cần báo cáo:

- **C1 − B0:** tác dụng tăng `lpf` trong cấu hình này.
- **C2 − B0:** tác dụng thêm nhánh affine, kèm tăng số tham số.
- **C3 − C2:** tác dụng thay Identity bằng GELU trong cặp cấu hình cùng kích thước và điều kiện train.
- **C3 − B0:** hiệu quả tổng thể của biến thể; riêng so sánh này không tách được tác dụng phi tuyến khỏi việc tăng tham số.

Lưu `cv_score` từng seed, điểm từng fold/từng người, số tham số thực, thời gian train và các `scores_*.csv` chứa bin/method. Báo mean và sample std (`ddof=1`) qua các seed hoàn chỉnh; `seed_std` khác độ lệch chuẩn giữa fold. Không dùng dải seed chồng nhau để khẳng định hai model tương đương.

Không dùng điểm GHIJ để chọn cấu hình hoặc quyết định chạy tiếp. GHIJ đã được xem trong lịch sử project nên không mô tả là test chưa từng được nhìn. Tài liệu này không yêu cầu chạy thêm test trong vòng sàng lọc.

## 8. Tên cấu hình và phạm vi triển khai dự kiến

Đề xuất dùng nhóm kết quả `tn_mixlinear_ablation` để tránh tranh chấp số TN. Đây là tên nhóm chạy; vị trí chương/thực nghiệm trong luận văn có thể chốt sau. `docs/PROTOCOL.md` hiện còn mô tả TN4 là ngưỡng corr, nên không tự gán nhóm mới thành TN4.

- B0/C1 dùng model key hiện có `mix_linear` và ID hiện có, phân biệt bằng `lpf`.
- C2 đề xuất model key `mix_linear_linear`; C3 đề xuất `mix_linear_mlp`. Hai key này **chưa được triển khai**.
- ID của nhánh bổ sung phải ghi chiều phụ 4, chẳng hạn hậu tố `_h4`, cùng `period_len`, `lpf`, seed và fold. Không đổi ID baseline cũ.
- Nếu triển khai CLI, chiều nhánh phụ dùng tên riêng như `--correction_hidden`, không dùng `--mix_hidden`. Ghi đầy đủ config và revision vào kết quả theo cơ chế đang có.
- Chỉ cần wrapper nhỏ quanh `MixLinear` và factory/CLI tương ứng. Không viết lại phần FFT, không refactor framework train/scoring.

Không có lệnh CLI cho C2/C3 trong tài liệu để tránh người đọc tưởng tính năng đã có. Claude cần review thiết kế trước khi dựng code/notebook.

Kiểm tra tối thiểu khi triển khai: input `(B,200)` ra `(B,25)`; output/loss/gradient hữu hạn ở forward-backward; gradient đến cả nền và nhánh phụ; số tham số 83/992/992; khi ép correction bằng 0 thì đầu ra khớp nền; C2/C3 cùng trọng số ban đầu theo seed; ID không ghi đè kết quả cũ.

## 9. Cơ sở tham khảo và giới hạn phát biểu

- **MixLinear:** đối chiếu [mã nguồn chính thức](https://github.com/aitianma/MixLinear/blob/main/models/MixLinear.py) cho hai nhánh gốc. Nhánh phụ trong tài liệu này là đề xuất bổ sung, không gán cho tác giả MixLinear.
- **LSTNet, SIGIR 2018:** kết hợp thành phần neural với autoregression tuyến tính; tác giả nêu vai trò AR trong xử lý vấn đề nhạy với thang đo của phần neural. Đây là tiền lệ cho kết hợp thành phần tuyến tính và phi tuyến, không phải bằng chứng cho nhánh MLP chiều 4 trên UWB. [Bài báo](https://arxiv.org/abs/1703.07015).
- **N-BEATS, ICLR 2020:** dùng các khối fully connected với kết nối residual để dự báo chuỗi. Đây là tham khảo về thiết kế dự báo qua các khối hiệu chỉnh; không phải bản cài đặt MixLinear + MLP và không xác nhận hiệu quả của cấu hình này. [Bài báo](https://arxiv.org/abs/1905.10437).

Hai tham khảo trên cung cấp cơ sở thiết kế chung. Lựa chọn GELU, chiều 4, đầu vào đã trừ mean và cách ghép cụ thể là quyết định thiết kế cần kiểm chứng, không phải công thức được hai bài báo chứng minh. Không gọi đây là kiến trúc mới chưa từng có nếu chưa khảo sát tính mới riêng.

## 10. Những điểm nhờ Claude review

1. Nhánh phụ nhận `x - mean(x)` và cộng sau đầu ra MixLinear có nhất quán với pipeline không?
2. Cặp Linear/MLP cùng 992 tham số, khác GELU, có đủ rõ để làm đối chứng không?
3. Quy tắc sàng lọc seed 0 và xác nhận cặp C2/C3 có phù hợp ngân sách không?
4. Có chỗ nào vô tình đổi preprocessing, cách train hoặc tiêu chí chọn kênh so với B0 không?
5. Tên model/ID đề xuất cần nối vào những script nào để train, đọc kết quả và nạp checkpoint đồng nhất?

Số tham số 63, 83 và 992 đã được kiểm tra bằng lớp MixLinear hiện tại và phép đếm Linear trong môi trường project. Đây là xác minh kích thước, không phải xác minh chất lượng dự báo.
