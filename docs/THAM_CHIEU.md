# Tham chiếu cho từng tham số kiến trúc

Mỗi con số trong `src/models.py` đều phải chỉ được nguồn. Bảng này ghi rõ mục
nào, hình nào, phương trình nào — để trích thẳng vào luận văn.

## Hai bài báo

| ký hiệu | bài |
|---|---|
| **Bai 2018** | Bai, S., Kolter, J. Z., Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271 |
| **Howard 2017** | Howard, A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861 |


## Từng thành phần kiến trúc

| thành phần | giá trị dùng | nguồn | nguyên văn |
|---|---|---|---|
| tích chập nhân quả, đệm bên trái | có, đệm `(k−1)·d` | **Bai 2018, mục 3.2** | *"TCN = 1D FCN + causal convolutions"*; *"zero padding of length (kernel size − 1) is added"* |
| công thức tích chập giãn | `F(s) = Σᵢ₌₀ᵏ⁻¹ f(i)·x_{s−d·i}` | **Bai 2018, mục 3.3, phương trình (2)** | |
| độ giãn tăng theo luỹ thừa | `d = 2ⁱ` ở tầng `i` | **Bai 2018, mục 3.3** | *"we increased d exponentially with the depth of the network (i.e., d = O(2ⁱ) at level i of the network)"* |
| tầm nhìn một tầng | `(k−1)·d` | **Bai 2018, mục 3.3** | *"the effective history of one such layer is (k − 1)d"* |
| nối tắt | `o = Act(x + F(x))` | **Bai 2018, mục 3.4, phương trình (3)** | |
| **hai tầng conv mỗi khối** | 2 | **Bai 2018, mục 3.4 + Hình 1(b)** | *"Within a residual block, the TCN has two layers of dilated causal convolution and non-linearity, for which we used the rectified linear unit (ReLU)"* |
| hàm kích hoạt | ReLU | **Bai 2018, mục 3.4** | như trên |
| dropout theo kênh | `Dropout1d` sau mỗi conv | **Bai 2018, mục 3.4** | *"a spatial dropout was added after each dilated convolution for regularization: at each training step, a whole channel is zeroed out"* |
| nhánh 1×1 trên đường tắt | **không dùng** | **Bai 2018, mục 3.4** | *"we use an additional 1x1 convolution ... when residual input and output have different dimensions"* — ở đây mọi khối giữ nguyên số kênh nên không cần |
| depthwise + pointwise | `Conv1d(groups=C)` rồi `Conv1d(kernel=1)` | **Howard 2017, mục 3.1, phương trình (3) và (5)** | *"factorize a standard convolution into a depthwise convolution and a 1×1 convolution called a pointwise convolution"* |
| tỉ lệ giảm chi phí | `1/N + 1/D_K²` | **Howard 2017, mục 3.1** | *"MobileNet uses 3×3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions"* |
| **BatchNorm** | dùng, thay WeightNorm | **Howard 2017, mục 3.1** | *"MobileNets use both batchnorm and ReLU nonlinearities for both layers"* — **lệch Bai 2018 mục 3.4** (WeightNorm), xem phần dưới |


## Từng siêu tham số

| tham số | giá trị | nguồn và lý do |
|---|---|---|
| `kernel_size` | **3** | **Howard 2017, mục 3.1** dùng 3×3. **Bai 2018, Bảng 2** dùng `k=3` cho 4 trên 10 bài toán. Kernel nhỏ, tầm nhìn xa nhờ giãn chứ không nhờ kernel to. |
| `n_blocks` | **6** | **Ràng buộc bắt buộc**, không phải lựa chọn. **Bai 2018, mục A.1**: *"The most important factor for picking parameters is to make sure that the TCN has a sufficiently large receptive field by choosing k and d that can cover the amount of context needed for the task."* Xem tính toán bên dưới. |
| `dropout` | **0.0** | **Bai 2018, Bảng 2** dùng `0.0` cho *The Adding Problem* — bài hồi quy giá trị liên tục, gần dự báo dạng sóng nhất. Và `nn.LSTM` của MobiVital cũng `dropout=0.0`; TN1 so **kiến trúc** nên phải giữ regularization giống nhau, không thêm biến thứ hai. |
| `channels` | **64** (nhẹ) hoặc **200 / 352** (ngang tham số) | **CỐ Ý LỆCH** **Bai 2018, mục A.1**: *"the number of hidden units was chosen so that the model size is approximately at the same level as the recurrent models with which we are comparing."* Thu nhỏ model chính là mục tiêu của đồ án, nên nhóm "nhẹ" đi ngược nguyên tắc này một cách có chủ đích. Nhóm "ngang tham số" thì tuân thủ đúng, để tách bạch hai câu hỏi. |


## Tính tầm nhìn — vì sao đúng 6 khối

Khối hai tầng conv nên tầm nhìn gấp đôi so với khối một tầng:

```
tầm nhìn = (k − 1) × 2 × Σ 2^i  + 1        i = 0 .. n−1
```

| k | n | tầm nhìn | đủ 200 mẫu |
|---|---|---|---|
| 3 | 4 | 61 | không |
| 3 | 5 | 125 | không |
| **3** | **6** | **253** | **đủ** |
| 3 | 7 | 509 | đủ, thừa |

Cửa sổ vào 200 mẫu (MobiVital cố định). Nhịp thở khoảng 0.25 Hz, lấy mẫu 50 Hz
nên một nhịp cũng đúng 200 mẫu. **6 khối là số nhỏ nhất phủ hết.**

Đo thực nghiệm xác nhận: đổi mẫu ở **vị trí 0** trong 200 mẫu vẫn làm đầu ra
thay đổi — tầm nhìn phủ toàn bộ cửa sổ.


## Ba chỗ lệch bài báo, và lý do

| lệch | Bai 2018 | đồ án dùng | lý do |
|---|---|---|---|
| chuẩn hoá | WeightNorm (mục 3.4) | BatchNorm | Nhánh `ds_tcn` theo **Howard 2017 mục 3.1** vốn dùng BatchNorm. Dùng chung một loại cho cả hai nhánh thì TN1 mới cô lập đúng một biến là phép tích chập. |
| số kênh ẩn | chọn ≈ model đem so (mục A.1) | 64 ở nhóm nhẹ | Thu nhỏ model là mục tiêu đồ án. Nhóm "ngang tham số" (200 / 352) chạy song song để vẫn trả lời được câu hỏi theo đúng nguyên tắc bài báo. |
| nhánh 1×1 | thêm khi lệch số kênh (mục 3.4) | không có | Mọi khối giữ nguyên số kênh nên không bao giờ lệch. |


## Số tham số đo được

`kernel=3`, `n_blocks=6`, hai tầng conv mỗi khối:

| model | kênh | tham số | so LSTM |
|---|---|---|---|
| LSTM (MobiVital) | hidden 352 | 1.502.713 | — |
| TCN | 64 | 151.513 | −90% |
| DS-TCN | 64 | 56.281 | −96% |
| TCN | 200 | 1.452.625 | −3% |
| DS-TCN | 352 | 1.525.945 | +2% |


## Truyền tham số vào lệnh chạy

```bash
python scripts/run_cv.py --experiment tn1 --model ds_tcn \
    --channels 64 --kernel_size 3 --n_blocks 6 --dropout 0.0
```

Cả `run_cv.py` và `run_final_test.py` đều nhận bốn cờ này. Bỏ trống thì dùng
mặc định trong bảng trên. Model `lstm` bỏ qua chúng.
