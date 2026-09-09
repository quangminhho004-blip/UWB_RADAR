# Tóm tắt thesis — thực nghiệm TCN và DS-TCN

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar

**Nhánh nộp:** `submission`

**Cấu hình cuối sử dụng:** **DS-TCN 64/RF121, Pearson loss thuần, 38.105 tham số**.

## 1. Mục tiêu và bài toán

Nghiên cứu mô hình dự báo dùng để chọn tín hiệu hô hấp từ các ứng viên radar UWB theo pipeline MobiVital. Model nhận 200 mẫu lịch sử và dự báo 25 mẫu tiếp theo của chính ứng viên. Dự báo được đối chiếu với tương lai radar đã quan sát để chấm khả năng tự dự báo và chọn ứng viên.

Sóng được đánh giá cuối là **sóng radar của ứng viên được chọn**, không phải ghép các forecast thành sóng đai. GT đai dùng để chọn dữ liệu train và đánh giá kết quả; GT không tham gia lựa chọn ứng viên lúc inference.

Câu hỏi chính: **cấu hình TCN/DS-TCN nào, với RF và loss nào, phù hợp cho nhiệm vụ chọn tín hiệu?**

## 2. Giao thức và cách đọc điểm

- Dev: **ABCDEFKL**; bốn fold validation **AB, CE, DF, KL**, mỗi fold train trên sáu người còn lại.
- Test cuối: **GHIJ**; model cuối train đủ tám người dev.
- TN1: **3 seed × 4 fold**.
- TN2 RF mới và TN3: **seed 0 × 4 fold** cho mỗi cấu hình/alpha.
- TN4: **3 seed 0, 1, 2**, không chạy CV bốn fold.
- Điểm chính: **Pearson macro theo người**; tính trung bình theo phiên trong mỗi người, rồi trung bình các người.
- Nền train: Adam, LR 1e-4, weight decay 0, batch 64, 20 epoch, checkpoint epoch cuối; MSE cho TN1/TN2.

Các phép sàng lọc một fold được ghi riêng. Không lấy điểm KL thay cho CV macro. Độ lệch chuẩn giữa seed không phải kiểm định thống kê về tương đương.

Tập test được tách theo người. Tài liệu lịch sử có ghi nhận GHIJ từng được xem trong project; không mô tả nó là chưa từng được nhìn trong toàn bộ quá trình. Khảo sát/chọn RF và loss ở các bảng chính dùng validation.

## 3. Thứ tự notebook chính

Các đường dẫn dưới mở notebook trong repo. Khi chạy Colab, ô setup tải mã của **nhánh submission**.

| Bước | Notebook | Nhiệm vụ |
|---|---|---|
| Chuẩn bị | [DATA_PREPARE](../notebooks/DATA_PREPARE.ipynb) | Dựng và lưu dữ liệu xử lý để dùng chung. |
| **TN0** | [TN0](../notebooks/TN0.ipynb) | Tái lập điểm và đối chiếu bộ chọn kênh với MobiVital; giữ LSTM ở đây vì là mốc kiểm chứng. |
| **TN1** | [TCN và DS-TCN nền](../notebooks/TN1_TCN_DSTCN_model_selection.ipynb) | Khảo sát các cấu hình TCN/DS-TCN ban đầu. |
| **TN1** | [DS-TCN 64/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb) | Nền gọn, 3 seed × 4 fold. |
| **TN1** | [DS-TCN 192/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c192.ipynb) | Đối chứng dung lượng lớn, 3 seed × 4 fold. |
| **TN2** | [RF nhóm 64](../notebooks/TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb) | Khảo sát RF121/181/241, so với RF61 nền. |
| **TN2** | [RF nhóm 192](../notebooks/TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb) | Khảo sát RF ở mức dung lượng lớn hơn. |
| **TN3** | [Loss 64/RF61](../notebooks/TN3_HybridLoss_DS_TCN_c64.ipynb) | Quét alpha, giữ α = 0,6. |
| **TN3** | [Loss 64/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c64_rf121.ipynb) | Quét alpha, giữ α = 0. |
| **TN3** | [Loss 192/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c192.ipynb) | Quét alpha, giữ α = 0,2. |
| **TN4** | [64/RF61](../notebooks/TN4_final_test_ds_tcn_c64.ipynb) | Test hybrid α = 0,6, 3 seed. |
| **TN4** | [64/RF121 — cấu hình cuối](../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb) | Test Pearson thuần và MSE đối chứng, mỗi loss 3 seed. |
| **TN4** | [192/RF121](../notebooks/TN4_final_test_ds_tcn_c192.ipynb) | Test hybrid α = 0,2, 3 seed. |

## 4. Quá trình lựa chọn

### TN0 — kiểm tra pipeline

Output TN0a: hai pipeline cùng 0,819481 từ tệp lựa chọn của tác giả. TN0b: cùng checkpoint, chọn trùng **537/537 kênh** trong phép đối chiếu. Đây là kiểm tra tính nhất quán của evaluator trước khi khảo sát mô hình khác; không phải điểm CV của TN1.

### TN1 — giữ cấu hình gọn và đối chứng dung lượng

| Cấu hình | Tham số | CV macro, 3 seed |
|---|---:|---:|
| TCN-64 BatchNorm, RF253 | 151.513 | 0,7423 ± 0,0044 |
| TCN-64 WeightNorm, RF253 | 150.745 | 0,7463 ± 0,0030 |
| DS-TCN nền C64, RF253 | 56.281 | 0,7421 ± 0,0007 |
| **DS-TCN 64/RF61** | **37.081** | **0,7609 ± 0,0031** |
| DS-TCN 192/RF61 | 307.801 | 0,7480 ± 0,0122 |

64/RF61 có điểm trung bình cao nhất và ít tham số nhất trong bảng nên được giữ làm nền gọn. Bản 192 được giữ để khảo sát ảnh hưởng của tăng dung lượng, không gọi là thắng TN1.

### TN2 — giữ nhiều RF để nghiên cứu tiếp

So cùng **seed 0, bốn fold, MSE**:

| RF | Kernel | C64 | C192 |
|---|---:|---:|---:|
| 61 | 3 | 0,758244 | 0,757493 |
| 121 | 5 | 0,757855 | 0,764428 |
| 181 | 7 | 0,743657 | 0,732562 |
| 241 | 9 | 0,736970 | 0,736623 |

Giữ **64/RF61, 64/RF121, 192/RF121**. RF121 không thắng RF61 ở C64 với MSE, nhưng có điểm gần nhau trong lượt khảo sát; nhóm tiếp tục thử loss trên cả hai. Đổi kernel cũng làm thay số trọng số depthwise, nên đây là khảo sát **kernel/RF khi giữ độ sâu và số kênh**.

### TN3 — chọn loss riêng cho từng cấu hình

`Loss = α × MSE + (1 − α) × (1 − Pearson)`.

| Cấu hình | Alpha được giữ | CV macro, seed 0 × 4 fold |
|---|---:|---:|
| 64/RF61 | 0,6 | 0,780028 |
| **64/RF121** | **0 — Pearson thuần** | **0,780306** |
| 192/RF121 | 0,2 | 0,776011 |

Đây là các alpha có điểm cao nhất quan sát được trong dải đã thử, không phải giá trị tối ưu phổ quát.

### TN4 — cấu hình sử dụng và các đối chứng

| Vai trò | Cấu hình/loss | Test macro, 3 seed |
|---|---|---:|
| **Cấu hình cuối sử dụng** | **64/RF121, Pearson thuần** | **0,8017 ± 0,0100** |
| Đối chứng loss | 64/RF121, MSE | 0,7622 ± 0,0214 |
| Đối chứng tổ hợp RF/loss | 64/RF61, hybrid α = 0,6 | 0,8036 ± 0,0154 |
| Đối chứng dung lượng | 192/RF121, hybrid α = 0,2 | 0,8007 ± 0,0107 |

64/RF121 với Pearson tăng trung bình **0,039548** so với MSE trên cùng cấu hình và tăng ở cả ba seed đã chạy. RF61 hybrid có điểm test trung bình nhỉnh hơn; báo cáo giữ nguyên kết quả đó, không gọi cấu hình cuối là thắng mọi bảng.

## 5. Vì sao cấu hình cuối có các tham số này?

| Thành phần | Thiết lập | Lý do thiết kế và mức bằng chứng |
|---|---|---|
| DS convolution | Depthwise theo thời gian + pointwise trộn kênh | Giảm trọng số so với conv thường cùng kích thước. |
| Số kênh | 64 | Nhánh gọn có điểm CV tốt; bản 192 cung cấp đối chứng dung lượng. |
| Block / dilation | 4 block; 1, 2, 4, 8 | Giữ độ sâu cố định để khảo sát RF qua kernel. Không khẳng định 4 block tối ưu độc lập. |
| Kernel / RF | 5 / 121 | Được giữ sau TN2 và khảo sát loss ở TN3. |
| Norm | Không có trong block | Giữ cấu trúc đơn giản, không dùng thống kê BatchNorm trong block. Đây là lý do để khảo sát, chưa chứng minh bỏ norm tự làm tăng điểm. |
| Dropout | Theo phần tử, p = 0,2 | Regularization khi train. Cửa sổ 200 mẫu trượt 25 mẫu có mức chồng lấn lớn; số cửa sổ không phải số quan sát độc lập. Điều này tạo động cơ regularization, không chứng minh riêng p = 0,2 tối ưu. |
| Loss | Pearson thuần | Mức α = 0 có CV cao nhất quan sát được ở 64/RF121; có đối chứng cùng kiến trúc tại TN4. |
| Input/output | 200 → 25 | Giữ giao diện và cách chấm của pipeline. |
| LR / batch / epoch | 1e-4 / 64 / 20 | Thiết lập chung cho các phép so; không tối ưu riêng từng mô hình. |

Phải phân biệt **lý do đưa một thiết lập vào khảo sát** và **bằng chứng cho hiệu quả của cả cấu hình**. TN1 đổi nhiều thành phần giữa một số ứng viên; không quy toàn bộ chênh lệch cho dropout hoặc norm.

Bảng giải thích đầy đủ, số tham số và câu trả lời hội đồng: [BAO_CAO_QUA_TRINH_THUC_NGHIEM.md](BAO_CAO_QUA_TRINH_THUC_NGHIEM.md).

## 6. Notebook bổ sung được giữ

| Notebook | Vai trò / trạng thái |
|---|---|
| [Mốc DS-TCN trên GHIJ](../notebooks/TN1_final_evaluation.ipynb) | Kết quả test của DS-TCN nền; giữ lệnh/output DS-TCN. |
| [RF C64, một fold](../notebooks/TN2_ReceptiveField_DS_TCN_c64.ipynb) | Sàng lọc sơ bộ, tách khỏi kết quả CV bốn fold. |
| [RF C192, một fold](../notebooks/TN2_ReceptiveField_DS_TCN_c192.ipynb) | Sàng lọc sơ bộ. |
| [DS-TCN + RevIN](../notebooks/TN2_DS_TCN_RevIN.ipynb) | Nhánh phụ đã khảo sát, không được giữ cho cấu hình cuối. |
| [TCN + RevIN](../notebooks/TN2_TCN_RevIN.ipynb) | Notebook chuẩn bị; chưa có kết quả hợp lệ theo bảng hiện tại, không tính là phép thử đã hoàn tất. |
| [DS-TCN-192 thử BatchNorm](../notebooks/TN_test_ds_tcn_192.ipynb) | Phép thử bổ sung; khác cả norm và loại dropout với nhánh chính. |

Các notebook ngoài phạm vi không nằm trong danh mục bản nộp. Danh mục máy đọc được: [SUBMISSION_NOTEBOOKS.json](SUBMISSION_NOTEBOOKS.json). Mã hỗ trợ nhiều model vẫn được giữ để không phá runner và phần tái lập TN0; đó không phải danh mục thực nghiệm của bản nộp.

**Lưu ý tái lập:** output train/test cũ được giữ làm bằng chứng các lần chạy đã ghi. Ô clone đã được đổi sang nhánh submission cho lần chạy mới; output setup cũ có thể hiển thị commit cũ. Không coi việc chỉnh notebook trình bày là một lượt train mới.

## 7. Tài liệu đọc tiếp

- [Sơ đồ nhánh TN1–TN4](SO_DO_NHANH.md).
- [Báo cáo chi tiết và lý do từng tham số](BAO_CAO_QUA_TRINH_THUC_NGHIEM.md).
- [Bảng TCN từng seed](BANG_TCN_TUNG_SEED.md).
- [Pipeline train/inference](PIPELINE_2.md).
- [TN0 — ghi chú tái lập](../notebooks/TN0.md).

Các bảng/tài liệu lịch sử trong repo có thể còn mô hình ngoài phạm vi hoặc trạng thái cũ. Dùng danh mục và quá trình trong tài liệu này để xác định nội dung bản nộp.

Cơ sở kiến trúc: [Bai et al., 2018](https://arxiv.org/abs/1803.01271), [Howard et al., 2017](https://arxiv.org/abs/1704.04861). Cơ sở regularization: [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html). Các bài này không quy định chính xác tổ hợp tham số của đồ án.
