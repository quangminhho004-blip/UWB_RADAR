# Báo cáo thực nghiệm TCN/DS-TCN và cơ sở chọn cấu hình cuối

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar

**Phạm vi:** TCN và DS-TCN; tập trung giải thích lựa chọn cấu hình và quá trình khảo sát.

**Cấu hình cuối được sử dụng:** **DS-TCN-64/RF121, Pearson loss thuần**.

**Đối chiếu:** mã, notebook và bảng kết quả trong repo ngày 09/09/2026.

## 1. Cách trình bày lựa chọn cấu hình

Không cần chứng minh từng con số của cấu hình là tối ưu toàn cục. Cần giải thích được:

1. **Nó làm gì trong mô hình?**
2. **Vì sao lựa chọn đó phù hợp để đưa vào khảo sát?**
3. **Nó được kế thừa hay được chọn từ kết quả nào?**
4. **Thực nghiệm đã kiểm chứng riêng nó, hay mới kiểm chứng cả tổ hợp?**

Có ba loại căn cứ:

| Loại căn cứ | Ví dụ trong đồ án |
|---|---|
| Kế thừa nguyên lý từ tài liệu | Causal/dilated convolution, residual block, depthwise + pointwise |
| Thiết lập cố định để giới hạn phạm vi khảo sát | 4 block, dropout 0,2, không norm; Adam 1e-4, batch 64, 20 epoch |
| Chọn/giữ dựa trên kết quả | 64 kênh làm nhánh chính; giữ RF61/RF121; chọn alpha riêng cho từng cấu hình |

**Cấu hình cuối không cần được mô tả là thắng mọi bảng.** Nó là cấu hình được sử dụng trong hệ thống của nhóm, có kết quả validation và test được báo cáo đầy đủ. Các cấu hình khác vẫn giữ vai trò đối chứng.

## 2. Vì sao chọn hướng DS-TCN?

Mô hình trong pipeline nhận lịch sử của một ứng viên radar và dự báo đoạn tiếp theo. Khả năng tự dự báo được dùng để chọn ứng viên; sóng đầu ra cuối là sóng radar của ứng viên được chọn.

TCN cho phép kiểm soát ngữ cảnh qua kernel, dilation và số block. DS-TCN giữ cấu trúc xử lý chuỗi này nhưng tách convolution thành depthwise và pointwise, tạo hướng giảm số trọng số ở cùng số kênh.

Cơ sở học thuật:

- **Bai et al., 2018:** causal convolution, dilation và residual block cho xử lý chuỗi. [Bài TCN, mục 3](https://arxiv.org/html/1803.01271v2#S3)
- **Howard et al., 2017:** nguyên lý tách depthwise và pointwise. Đồ án áp dụng cho 1D, không sử dụng nguyên mạng MobileNet cho ảnh. [Bài MobileNets](https://arxiv.org/html/1704.04861v1)

Với C kênh vào/ra, kernel k, bỏ qua bias:

```text
Conv1D thường             : k × C² trọng số
Depthwise + pointwise 1D  : k × C + C² trọng số
```

Đây là phép đếm từ cấu trúc lớp. Ít tham số không tự đồng nghĩa tăng tốc tương ứng trên thiết bị; hiệu năng vận hành cần được đo riêng.

**Tên phù hợp trong thesis:** “Cấu hình DS-TCN được điều chỉnh cho nhiệm vụ dự báo phục vụ chọn tín hiệu hô hấp”. Không cần gọi đây là một họ kiến trúc hoàn toàn mới.

## 3. Hai cấu hình nền và vai trò của bản 192 kênh

### 3.1. Nguồn gốc cấu hình

Hai notebook [64/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb) và [192/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c192.ipynb) ghi rằng cấu trúc 4 block, không norm, dropout theo phần tử 0,2 được dựng lại từ cấu hình nhóm đã khảo sát trước.

Trong TN1 hiện tại, cấu trúc đó được đánh giá lại với **MSE, Adam 1e-4, batch 64, 20 epoch**, cùng giao thức với các ứng viên khác. Vì vậy, việc kế thừa cấu trúc không đồng nghĩa tái lập nguyên toàn bộ chế độ train cũ.

Không gán cho Bai hoặc Howard việc đề xuất chính xác tổ hợp 64/192 kênh, 4 block, không norm, dropout 0,2 cho MobiVital. Các bài báo là cơ sở của thành phần; các giá trị cụ thể thuộc thiết kế của nhóm.

### 3.2. Có nên bỏ DS-TCN-192 khỏi báo cáo?

**Giữ như một đối chứng dung lượng, trình bày gọn; không đặt nó ngang vai trò cấu hình cuối.**

Bản 192 trả lời câu hỏi: **khi giữ cấu trúc và tăng số kênh, chất lượng có cải thiện đủ để cần mạng lớn hơn không?** 192 = 3 × 64 kênh, nhưng tham số tăng hơn 8 lần do phần pointwise phụ thuộc C².

Số **192** đến từ cấu hình đã có trong khảo sát của nhóm, không phải một giá trị được lý thuyết hô hấp quy định. Vai trò đối chứng dung lượng là cách dùng kết quả đã có để đánh giá lựa chọn 64 kênh; không viết rằng đã tìm kiếm đầy đủ mọi số kênh.

| Cấu hình nền | Tham số | CV macro, 3 seed × 4 fold |
|---|---:|---:|
| **64/RF61** | **37.081** | **0,7609 ± 0,0031** |
| 192/RF61 | 307.801 | 0,7480 ± 0,0122 |

Ở RF61, bản lớn không có điểm trung bình cao hơn. Tuy nhiên, nhóm vẫn khảo sát RF trên bản lớn để xem tăng ngữ cảnh có làm thay đổi kết quả ở mức dung lượng này không. Ở TN2, 192/RF121 đạt 0,7644 trên seed 0, nên được giữ để thử loss.

**Kết luận bảo vệ được:** bản 192 là phép đối chiếu cần thiết đã có, nhưng các kết quả không cho thấy phải sử dụng 192 kênh để có cấu hình cuối hữu ích. Không kết luận 64 luôn tốt hơn 192 trong mọi điều kiện.

Trong chương chính, chỉ cần một dòng trong bảng TN1/TN2/TN4 và đoạn giải thích trên. Chi tiết từng seed hoặc đường cong của bản 192 có thể để phụ lục.

## 4. Cấu hình cuối: mỗi tham số dùng để làm gì và vì sao chọn?

### 4.1. Bảng giải thích chính

| Thành phần | Giá trị cuối | Vai trò và lý do sử dụng | Căn cứ hiện có |
|---|---|---|---|
| Đầu vào / đầu ra | 200 → 25 mẫu | Giữ giao diện dự báo của pipeline; thay model mà không thay cách tạo mẫu/chấm ứng viên. | Kế thừa pipeline, không quét độ dài trong nhánh này. |
| Phép tích chập | Depthwise + pointwise | Lọc theo thời gian trong từng kênh rồi trộn kênh; giảm trọng số so với conv thường cùng kích thước. | Cơ sở kiến trúc và số tham số đếm được. |
| Số kênh | **64** | Giữ mô hình nhỏ; bản nền có điểm CV tốt, không cần mặc định tăng lên 192. | Đối chiếu 64/192 ở TN1 và khảo sát sau đó; không phải tìm kiếm toàn bộ số kênh. |
| Số block | **4** | Giữ độ sâu cố định, tạo nền gọn để khảo sát kernel/RF. | Kế thừa cấu trúc sơ bộ; chưa có ablation chỉ thay số block ở cấu hình cuối. |
| DS convolution mỗi block | **2** | Tạo nhánh biến đổi gồm hai tầng trước khi cộng residual. | Kế thừa cách tổ chức block của TCN, chuyển conv sang DS. |
| Dilation | **1, 2, 4, 8** | Mở rộng khoảng lịch sử mà không cần tăng kernel quá lớn ở từng tầng. | Lịch dilation tăng gấp đôi, giữ nguyên trong TN2. |
| Kernel | **5** | Với 4 block hiện tại tạo RF121; thuộc nhóm được giữ sau TN2. | Khảo sát k3/k5/k7/k9; không nói k5 thắng k3 khi cùng MSE. |
| RF lý thuyết | **121 mẫu** | Cho mô hình sử dụng khoảng lịch sử rộng hơn RF61; tiếp tục đánh giá bằng loss phù hợp. | TN2 giữ ứng viên; TN3 khảo sát loss riêng. |
| Norm trong block | **Không** | Giữ cấu trúc đơn giản và không sử dụng thống kê BatchNorm trong block. | Lựa chọn cố định của cấu hình được kế thừa; chưa đo được lợi ích riêng của bỏ norm. |
| Dropout | **nn.Dropout(0.2)** | Regularization theo phần tử trong lúc train. | Kế thừa cấu hình; không có bằng chứng 0,2 là mức tối ưu độc lập. |
| ReLU | Sau mỗi DS convolution | Tạo phi tuyến giữa các tầng. | Thiết lập giữ cố định; không khảo sát hàm kích hoạt. |
| Residual | Cộng đầu vào block với nhánh biến đổi | Giữ đường truyền trực tiếp qua block. | Kế thừa nguyên lý residual; code cộng trực tiếp, không thêm ReLU sau phép cộng. |
| Đầu dự báo | Đặc trưng cuối → Linear(64,25) | Dùng đặc trưng tại thời điểm mới nhất để dự báo trực tiếp 25 mẫu. | Thiết lập chung của các cấu hình TCN trong repo. |
| RevIN | **Tắt** | Giữ nhánh RF/loss theo cấu hình nền; không bổ sung một phép chuẩn hóa nữa. | Khảo sát RevIN riêng trên DS-TCN nền không cải thiện; chưa kiểm riêng trên cấu hình cuối. |
| Loss | **Pearson thuần, α = 0** | Tối ưu mức khớp hình dạng của dự báo với tương lai radar. | CV cao nhất quan sát được trong dải alpha đã thử cho 64/RF121; có MSE đối chứng ở TN4. |
| Tổng tham số | **38.105** | Kích thước cấu hình cuối. | Từ cấu trúc được cài đặt; không phải mục tiêu ấn định trước. |

Mã đối chiếu: [TCNBlock và TCN trong models.py](../src/models.py).

### 4.2. Không norm: lý do thế nào là đủ?

“Không norm” nghĩa là không có BatchNorm/WeightNorm trong các block này; **không có nghĩa bỏ chuẩn hóa tín hiệu của pipeline**.

Về kỹ thuật, BatchNorm dùng thống kê trong batch khi train và thông thường dùng running statistics khi eval. Bỏ BatchNorm loại bỏ cơ chế này khỏi block, làm cấu trúc đơn giản hơn. Đây là một lý do thiết kế để khảo sát cấu hình không norm. [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm1d.html)

Điểm CV xác nhận **tổ hợp không norm đang dùng có thể hoạt động tốt**. Nó chưa xác nhận cơ chế BatchNorm là nguyên nhân làm cấu hình khác có điểm thấp hơn.

Câu dùng trong thesis:

> Nhóm giữ cấu hình không sử dụng lớp chuẩn hóa bên trong block từ khảo sát sơ bộ và đánh giá lại cùng các ứng viên TCN/DS-TCN khác. Lựa chọn này giữ block đơn giản; kết quả được diễn giải ở cấp cấu hình hoàn chỉnh, không quy mức tăng điểm riêng cho việc bỏ chuẩn hóa.

Không dùng lý do “tín hiệu đã chuẩn hóa nên BatchNorm vô ích”, vì chuẩn hóa tín hiệu và đặc trưng trung gian có vai trò khác nhau.

### 4.3. Dropout 0,2: vì sao đúng con số này?

Dropout có cơ sở regularization: làm mô hình ít phụ thuộc vào một số kích hoạt trong quá trình học. [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)

**Con số 0,2 là một thiết lập kế thừa và được cố định**, không phải kết quả tối ưu hóa độc lập của các notebook TN1–TN4 hiện tại. Giữ nó giúp các phép so RF/loss không đồng thời thay mức dropout.

Ở train, nn.Dropout loại ngẫu nhiên các phần tử với xác suất 0,2. Nó khác nn.Dropout1d, vốn loại cả kênh. Khi eval, dropout không tiếp tục loại ngẫu nhiên kích hoạt. [PyTorch Dropout](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Dropout.html), [Dropout1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Dropout1d.html)

Câu dùng trong thesis:

> Dropout theo phần tử với xác suất 0,2 được giữ từ cấu hình sơ bộ nhằm regularization khi train. Nhóm cố định lựa chọn này trong các khảo sát receptive field và loss. Thực nghiệm hiện tại chưa đánh giá riêng mức dropout tối ưu hoặc ưu thế của dropout theo phần tử so với theo kênh.

Đó là giải thích hợp lệ về phạm vi nghiên cứu. Không cần gán cho một bài báo việc chứng minh “0,2 tốt nhất cho hô hấp”.

### 4.4. Bốn block, kernel 5 và RF121 liên hệ thế nào?

```text
4 block
Mỗi block có 2 DS convolution
Dilation theo block: 1, 2, 4, 8

RF = 1 + 2 × (kernel − 1) × (1 + 2 + 4 + 8)

kernel 3 → RF61
kernel 5 → RF121
kernel 7 → RF181
kernel 9 → RF241
```

Bốn block là độ sâu cố định; kernel 5 là lựa chọn được đưa đi tiếp trong quá trình khảo sát. Không trình bày hai con số này như hai nghiệm tối ưu độc lập.

Với norm tắt và đầu dự báo dùng đặc trưng cuối, RF121 khiến đầu ra phụ thuộc tối đa 121 mẫu cuối của cửa sổ. Ở 50 Hz, từ mẫu đầu đến mẫu cuối trong RF là 120/50 = **2,4 giây**. Đây là ngữ cảnh dự báo, không phải khẳng định một nhịp thở dài 2,4 giây.

### 4.5. Kiểm lại 38.105 tham số

Với C = 64, k = 5, bias bật:

| Thành phần | Số tham số |
|---|---:|
| Conv1d đầu vào, 1 → 64, kernel 1 | 128 |
| Một DS convolution: depthwise + pointwise | (64 × 5 + 64) + (64 × 64 + 64) = 4.544 |
| 4 block × 2 DS convolution | 8 × 4.544 = 36.352 |
| Linear(64,25) | 64 × 25 + 25 = 1.625 |
| **Tổng** | **38.105** |

ReLU, dropout, phép cộng residual và Identity không thêm tham số học được.

## 5. Siêu tham số huấn luyện — giải thích theo giao thức

| Thiết lập | Giá trị | Lý do dùng trong nhánh thực nghiệm |
|---|---|---|
| Optimizer | Adam | Giữ chế độ train nền của pipeline, không đổi optimizer khi so cấu hình/RF/loss. |
| Learning rate | 1e-4 | Giá trị kế thừa; chưa có quét LR riêng để gọi là tối ưu. |
| Weight decay | 0 | Giữ mặc định Adam trong runner; không đưa thêm biến regularization vào các phép so. |
| Batch | 64 | Giữ cùng thiết lập loader giữa các ứng viên; không phải tối ưu batch đã chứng minh. |
| Epoch | 20 | Cố định số vòng học cho các cấu hình trong giao thức. Cùng epoch không đồng nghĩa cùng chi phí tính toán. |
| Checkpoint | Epoch cuối | Tránh khác nhau ở tiêu chí chọn epoch giữa các ứng viên trong runner hiện tại. |
| Corr lọc train | > 0,9 | Kế thừa cách chọn chuỗi train chất lượng của pipeline; không quét ngưỡng trong nhánh này. |
| Stride tạo mẫu | 25 | Giữ cách lấy cặp history/future của pipeline. |
| TN1 | 3 seed × 4 fold | Quan sát điểm và dao động của các cấu hình nền. |
| TN2 RF mới / TN3 | 1 seed × 4 fold mỗi cấu hình/alpha | Khảo sát trong ngân sách hiện có; không tự tạo seed_std từ một seed. |
| TN4 | 3 seed; train ABCDEFKL, test GHIJ | Đánh giá các tổ hợp cuối qua nhiều lần huấn luyện. |

Nguồn: [mobivital_reference.py](../src/mobivital_reference.py), [training.py](../src/training.py), [run_cv.py](../scripts/run_cv.py), [run_final_test.py](../scripts/run_final_test.py).

## 6. TN1 — bằng chứng để giữ cấu hình nền

Bảng chỉ gồm TCN và DS-TCN. Điểm **CV Pearson macro theo người**, 3 seed × 4 fold; làm tròn 4 chữ số.

| Cấu hình | Block / RF | Norm | Dropout | Tham số | CV macro ± seed_std |
|---|---|---|---|---:|---:|
| TCN-64 | 6 / 253 | BatchNorm | 0 | 151.513 | 0,7423 ± 0,0044 |
| TCN-64 | 6 / 253 | WeightNorm | 0 | 150.745 | 0,7463 ± 0,0030 |
| DS-TCN nền, C64 | 6 / 253 | BatchNorm | 0 | 56.281 | 0,7421 ± 0,0007 |
| **DS-TCN-64/RF61** | **4 / 61** | **Không** | **0,2 phần tử** | **37.081** | **0,7609 ± 0,0031** |
| DS-TCN-192/RF61 | 4 / 61 | Không | 0,2 phần tử | 307.801 | 0,7480 ± 0,0122 |

**Lý do chọn nền 64/RF61:** trong nhóm này, cấu hình vừa có điểm trung bình cao nhất vừa có số tham số thấp nhất. Đây là bằng chứng giữ cấu hình, không phải bằng chứng bỏ norm hoặc thêm dropout riêng lẻ tạo ra toàn bộ mức tăng.

**Lý do giữ bản 192:** đối chiếu dung lượng trong khảo sát tiếp theo. Nó không được gọi là thắng TN1.

Nguồn: [TN1_TCN_DSTCN_model_selection](../notebooks/TN1_TCN_DSTCN_model_selection.ipynb), hai notebook nền và [BANG_TCN_TUNG_SEED.md](BANG_TCN_TUNG_SEED.md).

### Hai phép thử bổ sung thuộc họ TCN

| Phép thử | Quan sát | Kết luận trong phạm vi dữ liệu |
|---|---|---|
| DS-TCN nền RF253 + RevIN, 3 seed × 4 fold | 0,7297 ± 0,0073, thấp hơn mốc 0,7421 | Không giữ nhánh RevIN này; không suy rộng thành mọi norm đều có hại. |
| DS-TCN-192/RF121, BatchNorm, dropout theo kênh, seed 0 × 4 fold | Khoảng 0,7566; bản không norm + dropout phần tử khoảng 0,7644 | Hai tổ hợp khác cả norm và loại dropout; chưa tách riêng ảnh hưởng norm. |

Notebook: [TN2_DS_TCN_RevIN](../notebooks/TN2_DS_TCN_RevIN.ipynb), [TN_test_ds_tcn_192](../notebooks/TN_test_ds_tcn_192.ipynb).

## 7. TN2 — khảo sát RF và giữ RF121

So **cùng seed 0, cùng 4 fold, cùng MSE**; RF61 lấy đúng seed 0 của TN1:

| Kernel / RF | C64 | C192 |
|---|---:|---:|
| 3 / RF61 | 0,758244 | 0,757493 |
| 5 / RF121 | 0,757855 | 0,764428 |
| 7 / RF181 | 0,743657 | 0,732562 |
| 9 / RF241 | 0,736970 | 0,736623 |

**64/RF121 không thắng 64/RF61 trong bảng MSE.** Nó có điểm rất gần RF61 trong lượt khảo sát, trong khi RF181/RF241 thấp hơn, nên nhóm giữ cả RF61 và RF121 để khảo sát loss.

Ở C192, RF121 có điểm cao nhất trong bốn mức đã khảo sát, nên được mang sang TN3 như đối chứng dung lượng.

TN2 đổi kernel khi giữ số kênh và độ sâu. Do đó RF và số trọng số depthwise cùng thay đổi: k3 → k5 ở C64 làm tham số tăng từ 37.081 lên 38.105. Nên gọi chính xác là **khảo sát kernel/RF**, không tuyên bố RF được cô lập khỏi mọi thay đổi cấu trúc.

Nguồn: [TN2 C64, 4 fold](../notebooks/TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb), [TN2 C192, 4 fold](../notebooks/TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb). Kết quả một fold trong notebook sàng lọc không được nhập vào bảng này.

## 8. TN3 — chọn loss cho từng cấu hình/RF

Mỗi cấu hình thử α = 0,0 đến 0,9, seed 0 × 4 fold. MSE nền lấy từ cùng seed ở TN1/TN2.

| Cấu hình | MSE nền | Alpha có CV cao nhất quan sát được | CV với alpha đó |
|---|---:|---:|---:|
| 64/RF61 | 0,758244 | 0,6 | 0,780028 |
| **64/RF121** | **0,757855** | **0,0 — Pearson thuần** | **0,780306** |
| 192/RF121 | 0,764428 | 0,2 | 0,776011 |

Lý do thử Pearson loss là tiêu chí tự chấm dự báo của bộ chọn ứng viên cũng dùng Pearson. Đây là giả thuyết về mức phù hợp giữa mục tiêu train và cách chấm; kết quả được kiểm tra bằng chất lượng sóng được chọn.

**Alpha = 0 của 64/RF121 được chọn theo kết quả TN3 của chính cấu hình đó.** Không lấy alpha tốt nhất của RF61 áp sang RF121.

Hai RF ở C64 đều đã chạy cùng dải alpha, nên có dữ liệu so cùng loss trên validation. Các giá trị được chọn là tốt nhất quan sát trong phạm vi đã thử, không phải tối ưu phổ quát.

Nguồn: [TN3 64/RF61](../notebooks/TN3_HybridLoss_DS_TCN_c64.ipynb), [TN3 64/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c64_rf121.ipynb), [TN3 192/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c192.ipynb).

## 9. TN4 — báo cáo cấu hình cuối và các đối chứng

Train đủ ABCDEFKL, test GHIJ, mỗi tổ hợp 3 seed:

| Vai trò | Cấu hình | Loss | Tham số | Test macro ± seed_std |
|---|---|---|---:|---:|
| **Cấu hình cuối được sử dụng** | **64/RF121** | **Pearson thuần** | **38.105** | **0,8017 ± 0,0100** |
| Đối chứng loss | 64/RF121 | MSE | 38.105 | 0,7622 ± 0,0214 |
| Đối chứng tổ hợp RF/loss khác | 64/RF61 | Hybrid α = 0,6 | 37.081 | 0,8036 ± 0,0154 |
| Đối chứng dung lượng lớn | 192/RF121 | Hybrid α = 0,2 | 310.873 | 0,8007 ± 0,0107 |
| Mốc DS-TCN ban đầu | C64/RF253 | MSE | 56.281 | 0,7958 ± 0,0154 |

RF61 có điểm test trung bình nhỉnh hơn cấu hình cuối; báo cáo giữ nguyên quan sát này. Không cần đổi nhãn kết quả để cấu hình cuối trở thành “thắng tuyệt đối”.

Với 64/RF121, đổi MSE sang Pearson loss tăng trung bình **0,039548** và tăng ở cả ba seed đã chạy:

| Seed | MSE | Pearson | Chênh lệch |
|---|---:|---:|---:|
| 0 | 0,781448 | 0,805832 | +0,024384 |
| 1 | 0,766025 | 0,790376 | +0,024351 |
| 2 | 0,739099 | 0,809009 | +0,069910 |

Đây là bằng chứng về loss trên **cùng kiến trúc/RF**. Các hàng RF61 hybrid và RF121 Pearson so **tổ hợp**, không tách ảnh hưởng RF.

Nguồn: [TN4 64/RF121](../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb), [TN4 64/RF61](../notebooks/TN4_final_test_ds_tcn_c64.ipynb), [TN4 C192](../notebooks/TN4_final_test_ds_tcn_c192.ipynb), [mốc DS-TCN](../notebooks/TN1_final_evaluation.ipynb).

## 10. Cách giải thích lựa chọn cuối trước hội đồng

### “Vì sao dùng 64/RF121?”

> Nhóm sử dụng DS-TCN 64 kênh, RF121 làm cấu hình cuối. Bản 64 kênh cung cấp cấu hình gọn để nghiên cứu tiếp. RF121 được giữ sau khảo sát RF và được đánh giá với nhiều hàm loss; Pearson loss cho điểm CV cao nhất quan sát được trên cấu hình này. Ở đánh giá cuối, nhóm báo cáo cả Pearson và MSE đối chứng, cùng các cấu hình RF61 và 192 kênh.

Đoạn này giải thích cơ sở và mức hỗ trợ thực nghiệm của cấu hình được sử dụng. Nó không giả định cấu hình là nghiệm duy nhất hoặc có điểm test cao nhất.

### “Sao RF61 cao hơn mà không sử dụng RF61?”

> Nhóm ghi nhận RF61 với hybrid loss có điểm test trung bình cao hơn khoảng 0,0019. Hai hàng khác cả RF và loss, nên kết quả phản ánh hai tổ hợp cuối. Cấu hình được sử dụng là 64/RF121, và nhóm công bố đầy đủ kết quả đối chứng thay vì khẳng định nó vượt mọi ứng viên.

Không viện dẫn “nhỏ hơn std nên bằng nhau” để biến quan sát thành kiểm định tương đương.

### “Bỏ norm và dropout 0,2 có thật sự tốt hơn không?”

> Nhóm chưa tách riêng đóng góp của hai lựa chọn này. Chúng là thiết lập cố định trong cấu hình sơ bộ được đưa vào đánh giá lại. Kết quả hỗ trợ hiệu quả của cả tổ hợp; các khảo sát chính sau đó tập trung vào kernel/RF và loss.

### “Vậy số 192 để làm gì?”

> Để đối chiếu với mức dung lượng lớn hơn của cùng họ mạng. Bản này được khảo sát RF và loss riêng, giúp đánh giá có cần tăng đáng kể số tham số hay không. Nó là đối chứng, còn hệ thống sử dụng bản 64 kênh.

### “Vì sao đúng learning rate, epoch, batch này?”

> Nhóm giữ chế độ train nền thống nhất để hạn chế số biến thay đổi giữa các thực nghiệm. Luận văn không khẳng định các siêu tham số đó là tối ưu cho từng mô hình.

## 11. Đoạn tổng hợp có thể đưa vào luận văn

> Nhóm xây dựng hướng thực nghiệm trên họ TCN và DS-TCN. Cấu trúc DS-TCN bốn block, không có lớp chuẩn hóa trong block và dropout theo phần tử 0,2 được kế thừa từ khảo sát sơ bộ, sau đó đánh giá lại cùng các cấu hình khác trong giao thức hiện tại. Hai mức số kênh 64 và 192 được giữ để khảo sát dung lượng. Cấu hình 64/RF61 cho điểm CV tốt với số tham số nhỏ, tạo cơ sở cho nhánh nghiên cứu chính.
>
> Khi cố định số kênh và độ sâu, nhóm khảo sát kernel để thay đổi receptive field. Các cấu hình 64/RF61, 64/RF121 và 192/RF121 được giữ để đánh giá các hàm mất mát. Nhóm sử dụng 64/RF121 với Pearson loss làm cấu hình cuối, đồng thời báo cáo các tổ hợp còn lại và MSE đối chứng trên cùng kiến trúc. Kết quả được diễn giải theo phạm vi khảo sát: lựa chọn cấu hình hoàn chỉnh, ảnh hưởng của kernel/RF và ảnh hưởng của loss; không quy mức cải thiện riêng cho việc bỏ norm hay dropout 0,2.

## 12. Ghi chú nguồn và phạm vi kết luận

- Điểm chính là Pearson **macro theo người**, không phải phần trăm chính xác hoặc sai số nhịp thở/phút.
- TN2/TN3 mới một seed mỗi thiết lập; TN4 ba seed không thay thế phép kiểm soát từng biến ở các bước trước.
- Không suy ra ưu thế thống kê từ việc so khoảng cách trung bình với seed_std.
- Những lý do kỹ thuật trong tài liệu giải thích tính hợp lý của thiết kế; không chứng minh mọi lý do đều đã được ghi trước lần khảo sát lịch sử.
- Tài liệu lịch sử ghi nhận GHIJ từng được xem trong project. Khi viết protocol, mô tả đúng lịch sử sử dụng test; không tuyên bố chưa từng nhìn nếu không đúng.
- PROTOCOL.md và NHANH_DS_TCN.md còn một số kế hoạch/trạng thái cũ. Các đường đi ở đây đối chiếu notebook hiện có; không suy từ tên notebook rằng một lượt đã chạy.
- Các số được làm tròn ở bảng tổng hợp. Khi xuất bảng chính thức, dùng một bộ summary/scores nhất quán; một vài bảng trong repo khác nhau ở chữ số thập phân thứ sáu.

**Nguồn nội bộ:** [models.py](../src/models.py), [training.py](../src/training.py), [losses.py](../src/losses.py), [BANG_NHANH_TCN.md](BANG_NHANH_TCN.md), [BANG_TCN_TUNG_SEED.md](BANG_TCN_TUNG_SEED.md), [SO_DO_NHANH.md](SO_DO_NHANH.md).

**Nguồn học thuật:** [Bai et al., 2018](https://arxiv.org/abs/1803.01271); [Howard et al., 2017](https://arxiv.org/abs/1704.04861); [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html).
