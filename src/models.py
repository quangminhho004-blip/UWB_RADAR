"""Các model dự báo. Tất cả cùng một giao diện.

    from src import models
    model = models.build_model("ds_tcn", revin=True)
    pred = model(torch.randn(64, 200))     # -> (64, 25)

GIAO DIỆN BẮT BUỘC

    vào   (batch, 200)   200 mẫu quá khứ
    ra    (batch,  25)   25 mẫu tiếp theo

Đúng như LSTMMultiStep của MobiVital. Sai shape là hỏng cả chuỗi: bộ chọn kênh
gọi model 52 lần mỗi ứng viên, rồi so từng cửa sổ 25 mẫu.

CHÍN MODEL, CHIA LÀM BA HỌ

    hồi quy       đọc lần lượt từng mẫu, mang trạng thái đi theo
      lstm        LSTMMultiStep của MobiVital, làm mốc so sánh
      gru         như lstm nhưng tế bào ít cổng hơn
      bilstm      đọc cả hai chiều, phải lấy đặc trưng từ h_n
      cnn_lstm    nén 200 xuống 50 bằng tích chập rồi mới đưa vào lstm

    tích chập     nhìn cả cửa sổ một lúc, không mang trạng thái
      tcn         nhân quả, giãn dần — Bai et al. 2018
      ds_tcn      như trên, tách depthwise + pointwise, ít tham số hơn nhiều
      modern_tcn  chia đoạn trước rồi dùng kernel lớn — Luo & Wang 2024

    tuyến tính    gần như không có phi tuyến nào
      mix_linear  ghép nhánh thời gian với nhánh tần số, 63 tham số

SỐ THAM SỐ ĐO ĐƯỢC

Tám cấu hình dưới cùng một ngân sách khoảng 56-57k, để so kiến trúc chứ không
so kích cỡ. Hai dòng đầu và mix_linear nằm ngoài dải đó, có chủ ý.

    cấu hình                       tham số
    lstm-352                     1.502.713    baseline MobiVital
    tcn-64                         151.513
    tcn-64 weightnorm              150.745
    bilstm-41                       57.507
    modern_tcn-32                   56.985
    lstm-67                         56.908    mốc của dải cùng ngân sách
    gru-77                          56.466
    ds_tcn-64                       56.281
    cnn_lstm-58                     55.667
    mix_linear                          63    ít hơn 900 lần

Tự in lại các số này: xem cuối docs/CAU_HINH_TN1.txt

RevIN là lớp bọc, dùng được với tcn, ds_tcn và modern_tcn.

TÀI LIỆU THAM CHIẾU

    Bai, Kolter & Koltun (2018), arXiv:1803.01271 -- TCN
    Howard et al. (2017), arXiv:1704.04861        -- depthwise separable
    Sainath, Vinyals, Senior, Sak (2015), ICASSP  -- CLDNN, tiền lệ cnn_lstm
    Kim, Kim, Tae, Park, Choi, Choo (2022), ICLR  -- RevIN
    Luo & Wang (2024), ICLR                       -- ModernTCN
    github.com/aitianma/MixLinear                 -- MixLinear

Trích dẫn từng tham số: xem docs/THAM_CHIEU.md
"""

import math

import torch
import torch.nn as nn

# torch >= 2.1 chuyển weight_norm sang parametrizations; bản cũ vẫn dùng được
# nhưng có cảnh báo sắp bỏ.
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from src import mobivital_reference as mv


class RevIN(nn.Module):
    """Chuẩn hoá theo từng mẫu, rồi trả lại thang đo cũ ở đầu ra.

    Kim, Kim, Tae, Park, Choi, Choo (2022), ICLR — "Reversible Instance
    Normalization for Accurate Time-Series Forecasting against Distribution
    Shift".

    Mỗi cửa sổ 200 mẫu có mức nền và biên độ riêng: người thở sâu hay nông,
    ngồi gần hay xa radar. Model phải học vừa hình dạng vừa mấy thứ đó.

    RevIN gỡ phần đó ra: trừ trung bình, chia độ lệch chuẩn, cho model chỉ lo
    hình dạng. Xong thì nhân lại và cộng lại vào đầu ra.

    MỘT CHỖ LỆCH BÀI GỐC, CÓ LÝ DO

    Bài gốc có thêm hai tham số học được gamma và beta, áp sau khi chuẩn hoá.
    Ở đây bỏ chúng đi, vì TN2 so CÙNG một kiến trúc có và không có RevIN —
    thêm tham số học được là hai cấu hình khác số tham số, không còn cô lập
    đúng một biến. Bỏ chúng thì bật hay tắt RevIN cho ra đúng cùng số tham số.

    TRẠNG THÁI GIỮA HAI LƯỢT GỌI

    `normalize` cất mean và std vào chính đối tượng để `denormalize` dùng lại.
    Trong một lượt forward thì normalize luôn chạy trước, nên không sao. Nhưng
    gọi `denormalize` riêng lẻ là lấy nhầm giá trị của lượt trước.
    """

    def normalize(self, x):
        # Bám đúng mã tác giả: phương sai KHÔNG hiệu chỉnh Bessel, và epsilon
        # cộng TRONG căn chứ không ngoài.
        #     tác giả   sqrt(var(unbiased=False) + eps)
        #     dễ viết   std() + eps          <- lệch hai chỗ
        # Với cửa sổ gần phẳng hai công thức lệch nhau nhiều; trên dữ liệu này
        # cửa sổ phẳng nhất đo được có std 0,0092 nên lệch tối đa 5%, còn
        # 99,85% cửa sổ lệch dưới 2%. Lệch nhỏ, nhưng sửa thì miễn phí.
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean


def apply_weight_norm(module):
    """Bọc WeightNorm lên mọi lớp tích chập bên trong.

    WeightNorm không phải một lớp đặt thêm vào luồng dữ liệu như BatchNorm. Nó
    viết lại chính trọng số của lớp tích chập:

        w = g * v / ||v||

    `v` là hướng, `g` là một số học được quyết định độ dài. Model học hai thứ
    đó tách rời nhau. Vì vậy khi bật WeightNorm thì KHÔNG đặt thêm lớp chuẩn
    hoá nào nữa — xem `_one_layer`.

    Nhánh tách depthwise có hai lớp tích chập nên phải bọc cả hai.
    """
    if isinstance(module, nn.Sequential):
        return nn.Sequential(*[apply_weight_norm(m) for m in module])
    return weight_norm(module)


class TCNBlock(nn.Module):
    """Một khối tích chập nhân quả, theo Bai et al. 2018 (arXiv:1803.01271).

    Bám đúng Hình 1(b) và mục 3.4 của bài báo: mỗi khối có HAI tầng tích chập
    nhân quả giãn, mỗi tầng kèm phi tuyến, rồi cộng nhánh tắt.

        "Within a residual block, the TCN has two layers of dilated causal
         convolution and non-linearity, for which we used the rectified linear
         unit (ReLU)."                                    -- Bai et al., muc 3.4

    "Nhân quả" (mục 3.2): mẫu thứ t chỉ được nhìn các mẫu <= t. Làm bằng cách
    đệm thêm bên TRÁI đúng (kernel_size - 1) * dilation rồi cắt phần thừa bên
    phải — chính là cách bài báo mô tả.

    "Giãn" (mục 3.3, phương trình 2): bỏ cách quãng khi lấy mẫu. Tầm nhìn của
    một tầng là (k - 1) * d.

    HAI CHỖ LỆCH BÀI BÁO, CÓ LÝ DO

    1. Chuẩn hoá dùng BatchNorm thay vì WeightNorm (Bai mục 3.4). Lý do: nhánh
       ds_tcn theo MobileNets (Howard et al. 2017, arXiv:1704.04861) mục 3.1 —
       "MobileNets use both batchnorm and ReLU nonlinearities for both layers".
       Dùng chung một loại chuẩn hoá cho cả hai nhánh thì TN1 mới cô lập đúng
       một biến là phép tích chập.

    2. Không có nhánh 1x1 trên đường tắt. Bài báo thêm nó khi số kênh vào và ra
       khác nhau (mục 3.4, Hình 1b); ở đây mọi khối giữ nguyên số kênh nên
       không cần.

    Dropout dùng Dropout1d — xoá cả một kênh, đúng "spatial dropout" bài báo
    nói ở mục 3.4.
    """

    def __init__(self, channels, kernel_size, dilation, dropout, separable,
                 norm="batch"):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation

        # Hai tầng giống hệt nhau, cùng độ giãn — Bai et al. Hình 1(b).
        self.layer_one = self._one_layer(channels, kernel_size, dilation,
                                         dropout, separable, norm)
        self.layer_two = self._one_layer(channels, kernel_size, dilation,
                                         dropout, separable, norm)

    def _one_layer(self, channels, kernel_size, dilation, dropout, separable,
                   norm):
        """Một tầng: tích chập giãn -> chuẩn hoá -> ReLU -> dropout."""
        if separable:
            # Depthwise: mỗi kênh một bộ lọc riêng, không trộn kênh.
            # Pointwise: kernel 1, chỉ trộn kênh.
            # Howard et al. 2017 mục 3.1, phương trình (3) và (5). Chi phí giảm
            # còn 1/N + 1/D_K^2 lần so với tích chập thường.
            conv = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=dilation, groups=channels),
                nn.Conv1d(channels, channels, 1))
        else:
            conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

        if norm == "weight":
            conv = apply_weight_norm(conv)
            norm_layer = nn.Identity()      # WeightNorm nằm trong chính conv
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d(channels)
        else:
            raise ValueError("norm phải là 'batch' hoặc 'weight', nhận " + str(norm))

        return nn.ModuleDict({
            "conv": conv,
            "norm": norm_layer,
            "act": nn.ReLU(),
            "drop": nn.Dropout1d(dropout),
        })

    def _run_layer(self, layer, x):
        x = nn.functional.pad(x, (self.left_pad, 0))   # đệm bên trái
        x = layer["conv"](x)
        x = layer["norm"](x)
        x = layer["act"](x)
        return layer["drop"](x)

    def forward(self, x):
        residual = x
        x = self._run_layer(self.layer_one, x)
        x = self._run_layer(self.layer_two, x)
        return x + residual                             # Bai mục 3.4, pt. (3)


class TCN(nn.Module):
    """Chồng nhiều khối TCN, giãn gấp đôi mỗi khối.

    RÀNG BUỘC CHÍNH: TẦM NHÌN PHẢI PHỦ HẾT CỬA SỔ VÀO

        "The most important factor for picking parameters is to make sure that
         the TCN has a sufficiently large receptive field by choosing k and d
         that can cover the amount of context needed for the task."
                                                   -- Bai et al., muc A.1

    Cửa sổ vào 200 mẫu. Nhịp thở khoảng 0.25 Hz, lấy mẫu 50 Hz, nên một nhịp
    cũng đúng 200 mẫu. Tầm nhìn với khối hai tầng:

        (k - 1) * 2 * sum(2^i, i = 0..n-1) + 1

        k=3, n=6  ->  253  >= 200   ĐỦ
        k=3, n=5  ->  125  <  200   THIẾU

    Nên mặc định n_blocks = 6, kernel_size = 3.

    dropout mặc định 0.0, hai lý do:
      - Bai et al. Bảng 2 dùng dropout 0.0 cho Adding Problem, bài hồi quy liên
        tục gần với dự báo dạng sóng nhất
      - LSTM của MobiVital cũng dropout = 0.0; TN1 so KIẾN TRÚC nên phải giữ
        regularization giống nhau, không thêm biến thứ hai

    channels mặc định 64 là CỐ Ý LỆCH bài báo. Bai mục A.1 chọn số kênh sao cho
    model to xấp xỉ model hồi quy đem so; ở đây thu nhỏ model chính là mục tiêu
    của đồ án. Đây là giới hạn của TN1, TN5 sẽ quét lại.

    norm mặc định "batch" cũng LỆCH bài báo — Bai mục 3.4 dùng WeightNorm. Lý do
    chọn BatchNorm là để nhánh ds_tcn (Howard 2017 mục 3.1, vốn dùng BatchNorm)
    và nhánh tcn chuẩn hoá giống nhau, nhờ đó so hai nhánh chỉ đổi đúng một biến
    là phép tích chập.

    Nhưng lập luận đó chỉ đòi hai nhánh GIỐNG NHAU, không đòi phải là BatchNorm;
    chọn WeightNorm cho cả hai cũng thoả. Nên bản đúng chuẩn Bai chạy được bằng
    norm="weight", và kết luận về tcn chỉ nên phát biểu kèm tên kiểu chuẩn hoá
    đã dùng.
    """

    def __init__(self, channels=64, kernel_size=3, n_blocks=6,
                 dropout=0.0, separable=False, revin=False, norm="batch"):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNBlock(channels, kernel_size, 2 ** i,
                                   dropout, separable, norm))
        self.blocks = nn.Sequential(*blocks)

        self.output_linear = nn.Linear(channels, mv.FUTURE_LENGTH)
        self.revin = RevIN() if revin else None

    def forward(self, x):
        if self.revin is not None:
            x = self.revin.normalize(x)

        x = x.unsqueeze(1)          # (batch, 200) -> (batch, 1, 200)
        x = self.input_conv(x)
        x = self.blocks(x)
        x = x[:, :, -1]             # chỉ lấy mẫu cuối cùng
        y = self.output_linear(x)   # -> (batch, 25)

        if self.revin is not None:
            y = self.revin.denormalize(y)
        return y


class CNNLSTM(nn.Module):
    """Tích chập rút đặc trưng cục bộ rồi đưa vào LSTM. Dự báo đa bước.

    KHÔNG PHẢI ConvLSTM

    `ConvLSTM` (Shi et al. 2015, NeurIPS) là kiến trúc khác: đưa tích chập vào
    BÊN TRONG ô LSTM, thay phép nhân ma trận bằng tích chập. Ở đây tích chập
    đứng TRƯỚC, rút đặc trưng rồi mới đưa vào LSTM thường. Tên đúng là CNN-LSTM.

    Ý tưởng ghép tích chập với hồi quy theo CLDNN (Sainath, Vinyals, Senior,
    Sak 2015, ICASSP) — "Convolutional, Long Short-Term Memory, Fully Connected
    Deep Neural Networks". Khác CLDNN ở chỗ bỏ khối DNN phía sau, thay bằng một
    tầng tuyến tính, vì bài toán chỉ cần xuất 25 mẫu.

    LUỒNG DỮ LIỆU

        (batch, 200)              200 mẫu quá khứ
          -> (batch, 1, 200)
          -> Conv1d(1, 32, k=5, s=2, p=2) -> BatchNorm -> ReLU   -> (b, 32, 100)
          -> Conv1d(32, 32, k=5, s=2, p=2) -> BatchNorm -> ReLU  -> (b, 32,  50)
          -> đổi trục                                            -> (b, 50, 32)
          -> LSTM(input_size=32, hidden, 2 tầng, MỘT chiều)
          -> output[:, -1, :]                                    -> (b, hidden)
          -> Linear(hidden, 25)                                  -> (b, 25)

    VÌ SAO output[:, -1, :] Ở ĐÂY LÀ ĐÚNG

    LSTM này MỘT chiều, nên bước cuối đã đọc hết chuỗi. Chỗ phải tránh
    `output[:, -1, :]` là LSTM HAI chiều — xem lớp BiLSTM bên dưới.

    ĐƯỢC GÌ SO VỚI LSTM THUẦN

    Mỗi bước LSTM nhận 32 số mô tả một ĐOẠN sóng, thay vì một mẫu đơn lẻ. Và
    chuỗi ngắn đi bốn lần, từ 200 bước tuần tự xuống 50 — đây chính là nút thắt
    tốc độ của LSTM, vì 200 bước không song song hoá được.

    GIỚI HẠN PHẢI GHI KHI BÁO CÁO

    Kiến trúc này đổi ĐỒNG THỜI hai thứ: cách trích đặc trưng, và độ dài chuỗi
    đưa vào LSTM. Nếu nó thắng thì chưa biết nhờ cái nào. Đối chứng rẻ để tách
    hai nguyên nhân: thay hai tầng tích chập bằng AvgPool 200 xuống 50 rồi đưa
    vào LSTM — nếu bản AvgPool cũng thắng thì công là của việc rút ngắn chuỗi,
    không phải của đặc trưng tích chập.

    THAM SỐ MẶC ĐỊNH

        conv_channels 32, conv_kernel 5, hai tầng stride 2, hidden 58
        -> 55.667 tham số, xấp xỉ DS-TCN-64 (56.281) và LSTM-67 (56.908)

    Chọn 32 kênh để phần tích chập nhẹ, dồn ngân sách cho LSTM — thứ TN1 chứng
    minh là hợp bài toán. kernel 5 phủ 0,1 giây ở tần số lấy mẫu 50 Hz, cỡ một
    đoạn dốc của sóng thở.
    """

    def __init__(self, hidden_size=58, conv_channels=32, conv_kernel=5,
                 num_layers=2, future_len=25):
        super().__init__()
        pad = conv_kernel // 2          # giữ độ dài chia đôi đúng khi stride 2

        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, conv_kernel, stride=2, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel,
                      stride=2, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, future_len)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                   # (b, 200) -> (b, 1, 200)

        x = self.conv(x)                         # -> (b, 32, 50)
        x = x.transpose(1, 2)                    # -> (b, 50, 32)

        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])          # một chiều nên bước cuối là đủ


class BiLSTM(nn.Module):
    """LSTM hai chiều, dự báo đa bước. Cùng giao diện với LSTMMultiStep.

    Nhận 200 mẫu quá khứ, xuất thẳng 25 mẫu tiếp theo trong MỘT lần —
    không đoán từng mẫu rồi nạp ngược vào. Trong luận văn gọi là
    "BiLSTM dự báo đa bước", bảng kết quả viết ngắn là BiLSTM-41.

    VÌ SAO ĐỌC HAI CHIỀU LÀ HỢP LỆ

    Model nhận 200 mẫu quá khứ và đoán 25 mẫu tiếp theo. Cả 200 mẫu vào đều CÓ
    SẴN lúc dự báo, nên đọc chúng theo chiều nào cũng được. "Tương lai" cần đoán
    là mẫu 201-225, model không hề thấy. Không có rò rỉ.

    Bản một chiều của MobiVital chỉ dùng trạng thái ở bước cuối, nên thông tin từ
    mẫu thứ 1 phải sống sót qua 200 bước cổng mới tới được đầu ra. Đọc thêm chiều
    ngược thì đầu ra thấy được cả hai đầu cửa sổ.

    CHỖ RẤT DỄ SAI: KHÔNG DÙNG output[:, -1, :]

    Với LSTM hai chiều, `output` có dạng (batch, 200, 2*hidden):

        output[:, t, :hidden]    chiều xuôi tại bước t
        output[:, t, hidden:]    chiều ngược tại bước t

    Lấy `output[:, -1, :]` thì nửa đầu là chiều xuôi đã đọc hết 200 mẫu — đúng.
    Nhưng nửa sau là chiều ngược MỚI ĐỌC ĐÚNG MỘT MẪU, vì với chiều ngược thì
    bước cuối chính là bước đầu tiên nó xử lý. Nửa thông tin gần như trống.

    Cách đúng là lấy từ `h_n`, dạng (num_layers*2, batch, hidden):

        h[-2]   tầng cuối, chiều xuôi, đã đọc hết
        h[-1]   tầng cuối, chiều ngược, đã đọc hết

    Ghép hai cái đó mới ra đặc trưng đầy đủ hai chiều.
    """

    def __init__(self, hidden_size, num_layers, future_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, future_len)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)                  # (batch, 200) -> (batch, 200, 1)

        _, (h, _) = self.lstm(x)
        features = torch.cat([h[-2], h[-1]], dim=1)   # (batch, 2*hidden)
        return self.linear(features)


class ModernTCNBlock(nn.Module):
    """Một khối ModernTCN, bám theo lớp Block trong mã của tác giả.

    Nguồn: ModernTCN-short-term/models/ModernTCN.py, lớp Block và lớp
    ReparamLargeKernelConv.

    LUỒNG TRONG KHỐI

        vào ──┬─ dw_large  ─┐
              │             ├─ cộng ─ BatchNorm ─ pw1 ─ GELU ─ pw2 ─┐
              ├─ dw_small  ─┘                                       │
              └───────────────── nối tắt ──────────────────────────(+)─ ra

    CHỈ MỘT NỐI TẮT, ÔM CẢ KHỐI

    Dễ tưởng có hai: một quanh phần tích chập, một quanh phần FFN, như khối
    Transformer quen thuộc. Mã gốc không vậy — nó giữ `input = x` ở đầu
    `forward` rồi mới `x = input + x` ở dòng cuối cùng, sau khi đã đi hết cả
    tích chập lẫn FFN.

    HAI NHÁNH KERNEL, KHÔNG PHẢI MỘT

    Nhánh rộng 31 bắt phụ thuộc xa, nhánh hẹp 5 bắt chi tiết gần, cộng thẳng
    vào nhau. Đây là lối tái tham số hoá cấu trúc: lúc suy luận hai nhánh gộp
    lại thành đúng một phép tích chập, nhưng lúc huấn luyện phải để rời.

    Mỗi nhánh tự mang BatchNorm riêng, và tích chập KHÔNG có bias — vì
    BatchNorm ngay sau đó có sẵn tham số dịch, thêm bias là thừa.

    KHÔNG CÓ ConvFFN2

    Mã gốc khai báo đủ bộ `ffn2pw1`, `ffn2act`, `ffn2pw2`, `ffn2drop1`,
    `ffn2drop2` trong `__init__` nhưng `forward` KHÔNG gọi cái nào. Chúng nằm
    trong model, được đếm vào số tham số, mà không tham gia dự báo.

    Ở đây bỏ hẳn. Giữ lại thì số tham số báo cáo sẽ phồng lên vì phần chết,
    khiến so sánh cùng ngân sách tham số với LSTM-67 mất ý nghĩa.

    Khối đó vốn để trộn thông tin giữa các BIẾN. Chuỗi ở đây chỉ có MỘT biến,
    nên dù có gọi cũng không trộn được gì.
    """

    def __init__(self, channels, kernel_large=31, kernel_small=5, ffn_ratio=2,
                 dropout=0.0):
        super().__init__()
        self.dw_large = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_large, padding=kernel_large // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels))
        self.dw_small = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_small, padding=kernel_small // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels))
        self.norm = nn.BatchNorm1d(channels)

        wide = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, wide, 1),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Conv1d(wide, channels, 1),
            nn.Dropout(dropout))

    def forward(self, x):
        residual = x
        x = self.norm(self.dw_large(x) + self.dw_small(x))
        return residual + self.ffn(x)


class ModernTCN(nn.Module):
    """Chia chuỗi thành đoạn rồi xử lý bằng tích chập kernel lớn.

    Luo, Wang (2024), ICLR — "ModernTCN: A Modern Pure Convolution Structure
    for General Time Series Analysis".

    Viết theo mã của tác giả, nhánh dự báo ngắn hạn:
    ModernTCN-short-term/models/ModernTCN.py

    LUỒNG DỮ LIỆU

        (batch, 200)                    200 mẫu quá khứ
          -> lặp giá trị cuối 4 lần     -> 204
          -> Conv1d(1, 32, k=8, s=4)    -> (batch, 32, 50)
          -> BatchNorm1d(32)
          -> 3 khối ModernTCN           -> (batch, 32, 50)
          -> trải phẳng                 -> (batch, 1600)
          -> Linear(1600, 25)           -> (batch, 25)

    CHIA ĐOẠN CHỒNG LẤN

    Bề rộng đoạn 8, bước dịch 4: hai đoạn liền nhau dùng chung 4 mẫu. Chuỗi
    200 mẫu thành 50 đoạn.

    ĐỆM BẰNG CÁCH LẶP GIÁ TRỊ CUỐI, KHÔNG PHẢI ĐỆM 0

    Mã gốc: `pad = x[:, :, -1:].repeat(1, 1, pad_len)` rồi nối vào cuối, với
    `pad_len = patch_size - patch_stride`. Đệm 0 sẽ bịa ra một bậc nhảy giả ở
    cuối cửa sổ — đúng chỗ model cần nhìn kỹ nhất để dự báo bước kế tiếp.

    Số đệm là hằng 4, KHÔNG phải phần dư của phép chia. Chỗ này tôi từng tính
    sai thành 0, ra 49 đoạn thay vì 50; model vẫn chạy, vẫn ra điểm, chỉ là
    sai kiến trúc. check_model.py kiểm đúng con số 50 để chặn.

    KHÁC HAI KIẾN TRÚC ĐÃ CHẠY Ở ĐÂU

        TCN (Bai)     từng mẫu một, kernel 3, độ giãn 1-2-4-8-16-32 để phủ xa
        ModernTCN     chia đoạn trước, kernel 31 phủ xa ngay trong một lớp

        CNN-LSTM      rút 200 xuống 50 rồi đưa vào LSTM xử lý tuần tự
        ModernTCN     rút 200 xuống 50 rồi xử lý cả 50 bước song song

    PHẦN LỚN THAM SỐ NẰM Ở LỚP RA

    Linear(1600, 25) tốn 40.025 tham số, khoảng 70% tổng số. Ba khối tích chập
    chỉ chiếm chưa tới 17.000. Đây là hệ quả của việc trải phẳng toàn bộ 50
    đoạn, vốn có trong thiết kế gốc, không phải lỗi cài đặt.

    HAI CHỖ CỐ Ý KHÔNG THEO MÃ GỐC, CÓ LÝ DO

    1. Bỏ ConvFFN2, phần mã gốc khai báo mà không gọi — xem ModernTCNBlock.

    2. Bỏ mọi thao tác reshape theo trục biến. Mã gốc mang theo chiều M cho
       chuỗi nhiều biến; ở đây M luôn bằng 1 nên các phép reshape đó là phép
       đồng nhất. Bỏ đi cho đọc được, không đổi phép tính.

    Hai thứ trong bài gốc CHƯA bật ở bản đầu: RevIN, và tách xu thế khỏi mùa
    vụ. Bật thêm là đổi thêm biến, để dành lần sau.
    """

    def __init__(self, channels=32, patch_size=8, patch_stride=4, n_blocks=3,
                 kernel_large=31, kernel_small=5, ffn_ratio=2, dropout=0.0,
                 revin=False):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pad_len = patch_size - patch_stride

        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, channels, patch_size, stride=patch_stride),
            nn.BatchNorm1d(channels))
        self.blocks = nn.Sequential(*[
            ModernTCNBlock(channels, kernel_large, kernel_small, ffn_ratio, dropout)
            for _ in range(n_blocks)])

        n_patch = mv.HISTORY_LENGTH // patch_stride
        self.output_linear = nn.Linear(channels * n_patch, mv.FUTURE_LENGTH)
        self.revin = RevIN() if revin else None

    def forward(self, x):
        if self.revin is not None:
            x = self.revin.normalize(x)

        x = x.unsqueeze(1)                                  # (b, 1, 200)
        tail = x[:, :, -1:].repeat(1, 1, self.pad_len)      # lặp mẫu cuối
        x = torch.cat([x, tail], dim=-1)                    # (b, 1, 204)
        x = self.patch_embed(x)                             # (b, 32, 50)
        x = self.blocks(x)
        y = self.output_linear(x.flatten(1))                # (b, 25)

        if self.revin is not None:
            y = self.revin.denormalize(y)
        return y


class GRU(nn.Module):
    """Hồi quy có cổng, ít cổng hơn LSTM. Dự báo đa bước.

    Cho GRU đọc hết 200 mẫu rồi lấy trạng thái ở bước cuối để bắn thẳng ra 25
    mẫu. Giống hệt LSTM baseline, đổi đúng một thứ: loại tế bào hồi quy.

    KHÁC LSTM Ở ĐÂU

        LSTM   3 cổng (quên, vào, ra) + một ô nhớ riêng tách khỏi trạng thái ẩn
        GRU    2 cổng (đặt lại, cập nhật), KHÔNG có ô nhớ riêng

    Vì bỏ ô nhớ, mỗi đơn vị GRU chỉ tốn 3 khối trọng số thay vì 4, nên cùng số
    tham số thì GRU chứa được nhiều đơn vị ẩn hơn: 77 so với 67 của LSTM.

    Ở ĐÂY output[:, -1, :] LÀ ĐÚNG, KHÁC VỚI BiLSTM

    GRU này một chiều, nên bước cuối của chuỗi cũng là bước cuối cùng nó xử lý —
    trạng thái tại đó đã đọc đủ 200 mẫu. Với BiLSTM thì không, và đó là lý do
    lớp BiLSTM phải lấy từ h_n; xem docstring của lớp đó.
    """

    def __init__(self, hidden_size, num_layers, future_len):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, future_len)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        output, _ = self.gru(x)
        return self.linear(output[:, -1, :])


class MixLinear(nn.Module):
    """Ghép nhánh thời gian với nhánh tần số, cực ít tham số.

    Nguồn: github.com/aitianma/MixLinear, models/MixLinear.py, lớp Model.

    LUỒNG DỮ LIỆU

        (batch, 200)
          -> trừ trung bình của chính cửa sổ
          -> Conv1d(1, 1, kernel=11) rồi cộng nối tắt      làm mượt
          -> chia thành 20 đoạn, mỗi đoạn 10 mẫu
               |
               +-- nhánh THỜI GIAN: đệm 20 lên 25, xếp thành lưới 5x5,
               |   ép hai chiều của lưới bằng TLinear1 rồi TLinear2
               |
               +-- nhánh TẦN SỐ: FFT dọc trục 20 đoạn, giữ 5 hệ số tần thấp,
                   FLinear1 rồi FLinear2 (trọng số SỐ PHỨC), rồi FFT ngược
               |
          -> trộn: nhánh_thời_gian * 0,5 + nhánh_tần_số * 0,5 + trung bình
          -> lấy 25 mẫu đầu

    period_len = 10 là cỡ chia đoạn BÊN TRONG mạng, không phải khẳng định một
    nhịp thở dài 10 mẫu. Ở 50 Hz thì 10 mẫu là 0,2 giây, còn một nhịp thở
    khoảng 4 giây tức 200 mẫu — đúng bằng cả cửa sổ. FFT chạy dọc trục 20 đoạn,
    tức nhìn chu kỳ ở thang 0,2 giây trải trên 4 giây; lpf = 5 giữ 5 hệ số tần
    thấp nhất.

    VÌ SAO CHỈ 63 THAM SỐ

    Không có lớp nào ánh xạ 200 chiều xuống 25 chiều. Mọi phép nén đều làm trên
    trục ĐOẠN (20 hoặc 5 phần tử), còn 10 mẫu trong mỗi đoạn thì đi song song
    dùng chung trọng số. Vì vậy số tham số gần như không phụ thuộc độ dài chuỗi.

    ĐẾM THAM SỐ: 47 HAY 63

    Hai lớp FLinear có trọng số SỐ PHỨC. Một số phức là hai con số thật, cả hai
    đều được cập nhật khi train, nhưng `numel()` của PyTorch đếm nó là một.

        TLinear1, TLinear2, conv1d        31 số thật
        FLinear1, FLinear2   16 số phức = 32 số thật
        numel() báo 47, số thật là 63

    `count_params` trong tệp này nhân đôi phần phức nên trả về 63.

    BA CHỖ CỐ Ý LỆCH MÃ TÁC GIẢ

    1. Bỏ hai lệnh `print("shape", ...)` trong `forward`. Mã gốc bỏ quên chúng.
       292.708 cửa sổ x 20 epoch x 4 fold thì ngập màn hình và chậm hẳn.

    2. `torch.fft.ifft(...).float()` đổi thành `.real`. Bản gốc ném cảnh báo
       "Casting complex values to real discards the imaginary part" mỗi lượt
       gọi. Đã đối chiếu: hai cách cho ra ĐÚNG CÙNG giá trị.

    3. Bỏ đối tượng `configs`, nhận tham số rời, và thêm phần đổi hình dạng
       (batch, 200) sang (batch, 200, 1) rồi ngược lại ở đầu ra.

    GIỮ NGUYÊN PHẦN TRỪ TRUNG BÌNH, VÀ ĐÓ LÀ ĐIỀU MAY

    Mã gốc trừ trung bình của cửa sổ rồi cộng trả lại ở cuối. Nó KHÔNG chia cho
    độ lệch chuẩn, nên BIÊN ĐỘ được giữ nguyên. Ở TN2, RevIN chia độ lệch chuẩn
    đã kéo DS-TCN xuống 0,0124 vì xoá mất biên độ — thứ mà đo đạc cho thấy có
    tương quan 0,53 với chất lượng kênh. MixLinear không dính bẫy đó.
    """

    def __init__(self, seq_len=None, pred_len=None, period_len=10, lpf=5,
                 mix_alpha=0.5):
        super().__init__()
        self.seq_len = seq_len or mv.HISTORY_LENGTH
        self.pred_len = pred_len or mv.FUTURE_LENGTH
        self.period_len = period_len
        self.lpf = lpf
        self.alpha = mix_alpha

        # period_len LẺ thì kernel của conv1d thành CHẴN, và lớp đó trả về
        # 199 mẫu thay vì 200, vỡ ở bước cộng nối tắt ngay sau. Đây là giới hạn
        # có sẵn trong mã tác giả (kernel = period_len + 1, padding =
        # period_len // 2), không phải lỗi chép lại. Báo sớm cho dễ hiểu, thay
        # vì để torch ném "shape [-1, 1, 200] is invalid for input of size 796".
        if period_len % 2 != 0:
            raise ValueError(
                "period_len phải CHẴN, nhận %d. Số lẻ làm kernel tích chập "
                "thành chẵn nên chuỗi ra ngắn hơn một mẫu." % period_len)
        if self.seq_len % period_len != 0:
            raise ValueError(
                "period_len phải chia hết %d, nhận %d." % (self.seq_len, period_len))

        self.seg_num_y = math.ceil(self.pred_len / period_len)
        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / period_len))

        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x,
                                  math.ceil(math.sqrt(self.pred_len / period_len)),
                                  bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x,
                                  math.ceil(math.sqrt(self.pred_len / period_len)),
                                  bias=False)
        self.conv1d = nn.Conv1d(1, 1, period_len + 1, stride=1,
                                padding=period_len // 2, padding_mode="zeros",
                                bias=False)
        self.FLinear1 = nn.Linear(lpf, 2, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)                              # (b, 200, 1)
        batch = x.shape[0]

        mean = x.mean(dim=1).unsqueeze(1)
        x = (x - mean).permute(0, 2, 1)                      # (b, 1, 200)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
            -1, 1, self.seq_len) + x
        x = x.reshape(batch, 1, -1, self.period_len).permute(0, 1, 3, 2)

        time_branch = self._time_domain(x, batch)
        freq_branch = self._freq_domain(x, batch)

        y = (time_branch[:, :self.pred_len, :] * self.alpha
             + freq_branch[:, :self.pred_len, :] * (1 - self.alpha)
             + mean)
        return y.squeeze(-1)                                 # (b, 25)

    def _time_domain(self, x, batch):
        """Xếp 20 đoạn thành lưới 5x5, ép hai chiều của lưới xuống còn 2x2.

        Mẹo của tác giả: thay vì một lớp 20 -> 4 tốn 80 trọng số, xếp 20 đoạn
        thành lưới vuông rồi ép từng chiều bằng lớp 5 -> 2. Hai lớp như vậy
        tốn 10 + 10 = 20 trọng số mà vẫn trộn được mọi đoạn với nhau.

        10 mẫu bên trong mỗi đoạn đi song song, dùng chung trọng số — đó là
        lý do số tham số không phụ thuộc độ dài chuỗi.

        Cột bên phải là hình dạng tensor sau mỗi dòng, với batch 4.
        """
        side = self.sqrt_seg_num_x                              # 5
        # Lưới 5x5 chứa 25 ô nhưng chỉ có 20 đoạn, đệm 5 ô cuối bằng 0.
        x = nn.functional.pad(x, (0, side ** 2 - x.shape[-1], 0, 0, 0, 0))
        #                                                       (4, 1, 10, 25)
        x = x.reshape(batch, 1, self.period_len, side, side)  # (4, 1, 10, 5, 5)
        # permute đổi chỗ hai chiều cuối, để lớp sau ép nốt chiều còn lại.
        x = self.TLinear1(x).permute(0, 1, 2, 4, 3)           # (4, 1, 10, 2, 5)
        x = self.TLinear2(x).permute(0, 1, 2, 4, 3)           # (4, 1, 10, 2, 2)

        x = x.reshape(batch, 1, self.period_len, -1)          # (4, 1, 10, 4)
        x = x.permute(0, 1, 3, 2)                             # (4, 1, 4, 10)
        x = x.reshape(batch, 1, -1)                           # (4, 1, 40)
        return x.permute(0, 2, 1)                             # (4, 40, 1)

    def _freq_domain(self, x, batch):
        """Giữ 5 hệ số tần thấp, biến đổi bằng trọng số phức, rồi FFT ngược.

        FFT chạy dọc trục 20 ĐOẠN, không phải dọc 200 mẫu. Nó hỏi "biên độ của
        đoạn thay đổi tuần hoàn thế nào qua 4 giây", đúng thang của nhịp thở.

        Giữ 5 hệ số đầu là giữ 5 tần số THẤP nhất, tức bỏ dao động nhanh và
        giữ dao động chậm. Nhịp thở là dao động chậm.

        Đây là chỗ duy nhất có trọng số số phức, vì đầu ra của FFT là số phức.
        """
        # Cắt lấy lpf hệ số đầu.                                (4, 1, 10, 5) phức
        spectrum = torch.fft.fft(x, dim=3)[:, :, :, :self.lpf]
        spectrum = self.FLinear1(spectrum)                    # (4, 1, 10, 2) phức
        spectrum = self.FLinear2(spectrum)                    # (4, 1, 10, 3) phức
        spectrum = spectrum.reshape(batch, 1, self.period_len, -1)

        # `.real` thay cho `.float()` của mã gốc — cùng giá trị, không cảnh báo.
        wave = torch.fft.ifft(spectrum, dim=3).real           # (4, 1, 10, 3)
        wave = wave.permute(0, 1, 3, 2)                       # (4, 1, 3, 10)
        wave = wave.reshape(batch, 1, -1)                     # (4, 1, 30)
        return wave.permute(0, 2, 1)                          # (4, 30, 1)


def build_model(name, revin=False, **kwargs):
    """Dựng model theo tên, để notebook chỉ cần truyền chuỗi.

        build_model("lstm")
        build_model("bilstm", hidden=41)
        build_model("gru", hidden=77)
        build_model("mix_linear", period_len=10, lpf=5)
        build_model("ds_tcn", revin=True, channels=96)
    """
    if name == "lstm":
        # RevIN không áp cho baseline. hidden mặc định là 352 của MobiVital;
        # các tham số riêng của TCN (kernel_size, n_blocks...) không dùng ở đây.
        return mv.new_lstm(kwargs.get("hidden"))

    if name == "bilstm":
        # Cùng số tầng và độ dài dự báo với LSTM, chỉ đổi chiều đọc.
        hidden = kwargs.get("hidden") or mv.LSTM_HIDDEN_SIZE
        return BiLSTM(hidden, mv.LSTM_NUM_LAYERS, mv.FUTURE_LENGTH)

    if name == "gru":
        # Cùng số tầng và độ dài dự báo với LSTM, chỉ đổi loại tế bào hồi quy.
        hidden = kwargs.get("hidden") or mv.LSTM_HIDDEN_SIZE
        return GRU(hidden, mv.LSTM_NUM_LAYERS, mv.FUTURE_LENGTH)

    if name == "mix_linear":
        return MixLinear(period_len=kwargs.get("period_len", 10),
                         lpf=kwargs.get("lpf", 5),
                         mix_alpha=kwargs.get("mix_alpha", 0.5))

    if name == "modern_tcn":
        return ModernTCN(channels=kwargs.get("channels", 32),
                         n_blocks=kwargs.get("n_blocks", 3),
                         kernel_large=kwargs.get("kernel_large", 31),
                         kernel_small=kwargs.get("kernel_small", 5),
                         dropout=kwargs.get("dropout", 0.0),
                         revin=revin)

    if name == "cnn_lstm":
        return CNNLSTM(hidden_size=kwargs.get("hidden") or 58,
                       conv_channels=kwargs.get("conv_channels", 32),
                       conv_kernel=kwargs.get("conv_kernel", 5),
                       num_layers=mv.LSTM_NUM_LAYERS,
                       future_len=mv.FUTURE_LENGTH)

    # LSTM không nhận các tham số riêng của TCN; lọc bớt để build_model dùng
    # được chung một bộ đối số cho mọi model.
    kwargs.pop("hidden", None)
    kwargs.pop("conv_channels", None)
    kwargs.pop("conv_kernel", None)
    kwargs.pop("kernel_large", None)
    kwargs.pop("kernel_small", None)
    kwargs.pop("period_len", None)
    kwargs.pop("lpf", None)
    kwargs.pop("mix_alpha", None)

    if name == "tcn":
        return TCN(separable=False, revin=revin, **kwargs)

    if name == "ds_tcn":
        return TCN(separable=True, revin=revin, **kwargs)

    raise ValueError("không biết model tên " + name)


def count_params(model):
    """Số CON SỐ THẬT phải học. Trả lời câu "tốt hơn vì kiến trúc hay vì to hơn".

    MỘT SỐ PHỨC ĐẾM LÀ HAI

    `numel()` của PyTorch đếm một số phức là một, nhưng số phức gồm phần thực và
    phần ảo, và khi train thì CẢ HAI đều được cập nhật. Đếm là một thì báo thiếu.

    Chỉ MixLinear có tham số phức. Mọi model khác không đổi số so với trước.

        MixLinear   numel() báo 47, số thật là 63
    """
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.numel() * (2 if p.is_complex() else 1)
    return total
