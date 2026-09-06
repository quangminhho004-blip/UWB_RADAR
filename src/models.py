"""Các model dự báo. Tất cả cùng một giao diện.

    from src import models
    model = models.build_model("ds_tcn", revin=True)
    pred = model(torch.randn(64, 200))     # -> (64, 25)

GIAO DIỆN BẮT BUỘC

    vào   (batch, 200)   200 mẫu quá khứ
    ra    (batch,  25)   25 mẫu tiếp theo

Đúng như LSTMMultiStep của MobiVital. Sai shape là hỏng cả chuỗi: bộ chọn kênh
gọi model 52 lần mỗi ứng viên, rồi so từng cửa sổ 25 mẫu.

BA MODEL

    lstm      LSTMMultiStep của MobiVital, làm mốc so sánh
    tcn       tích chập nhân quả, giãn dần
    ds_tcn    như trên nhưng tách depthwise + pointwise, ít tham số hơn nhiều

Số tham số đo được, mặc định kernel=3, n_blocks=6, hai tầng conv mỗi khối:

    model      kênh   tham số      so LSTM
    lstm        352   1,502,713    --
    tcn          64     151,513    -90%
    ds_tcn       64      56,281    -96%
    tcn         200   1,452,625    -3%     ngang tham số LSTM
    ds_tcn      352   1,525,945    +2%     ngang tham số LSTM

RevIN là lớp bọc, dùng được với cả tcn lẫn ds_tcn.

TÀI LIỆU THAM CHIẾU

    Bai, Kolter & Koltun (2018), arXiv:1803.01271 -- kiến trúc TCN
    Howard et al. (2017), arXiv:1704.04861        -- depthwise separable

Trích dẫn từng tham số: xem docs/THAM_CHIEU.md
"""

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
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + 1e-5
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
        dem = conv_kernel // 2          # giữ độ dài chia đôi đúng khi stride 2

        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, conv_kernel, stride=2, padding=dem),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel,
                      stride=2, padding=dem),
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


def build_model(name, revin=False, **kwargs):
    """Dựng model theo tên, để notebook chỉ cần truyền chuỗi.

        build_model("lstm")
        build_model("bilstm", hidden=41)
        build_model("tcn")
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

    if name == "tcn":
        return TCN(separable=False, revin=revin, **kwargs)

    if name == "ds_tcn":
        return TCN(separable=True, revin=revin, **kwargs)

    raise ValueError("không biết model tên " + name)


def count_params(model):
    """Số tham số học được. Để trả lời câu "tốt hơn vì kiến trúc hay vì to hơn"."""
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.numel()
    return total
