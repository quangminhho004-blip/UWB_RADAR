"""Các model dự báo. Tất cả cùng một giao diện.

    from src import models
    model = models.build_model("ds_tcn", channels=64)
    pred = model(torch.randn(64, 200))     # -> (64, 25)

GIAO DIỆN BẮT BUỘC

    vào   (batch, 200)   200 mẫu quá khứ
    ra    (batch,  25)   25 mẫu tiếp theo

Đúng như LSTMMultiStep của MobiVital. Sai shape là hỏng cả chuỗi: bộ chọn kênh
gọi model 52 lần mỗi ứng viên, rồi so từng cửa sổ 25 mẫu.

PHẠM VI CỦA NHÁNH NÀY

Nhánh nộp cuối giữ đúng mã cần để dựng lại năm thực nghiệm TN0 tới TN4. Các
kiến trúc khác từng thử trong quá trình chọn, và nhánh C192 (đối chứng dung
lượng lớn), không nằm ở đây.

    hồi quy       đọc lần lượt từng mẫu, mang trạng thái đi theo
      lstm        LSTMMultiStep của MobiVital, làm mốc so sánh
      cnn_lstm    tích chập rút đặc trưng rồi mới đưa vào LSTM

    tích chập     nhìn cả cửa sổ một lúc, không mang trạng thái
      tcn         nhân quả, giãn dần
      ds_tcn      như trên, tách depthwise + pointwise, ít tham số hơn nhiều

BỐN CẤU HÌNH CỦA TN1 — chọn kiến trúc

    cấu hình             tham số   CV macro (4 fold x 3 seed)
    lstm-352           1.502.713   0,756998 +- 0,004141   baseline MobiVital
    lstm-67               56.908   0,753208 +- 0,001966
    cnn_lstm-58           55.667   0,752666 +- 0,003749
    ds_tcn-64 k3n4        37.081   0,760878 +- 0,003095   được chọn

MÔ HÌNH CUỐI — sau TN2 (tầm nhìn) và TN3 (hàm loss)

    ds_tcn-64 k3n4, loss lai alpha 0,6      37.081 tham số
    trên G H I J: macro 0,803590 +- 0,015350

    Mốc lstm-352 trên cùng tập đó: 0,810302 +- 0,015402. Chênh 0,0067 nhỏ hơn
    dao động seed, nên phát biểu đúng là NGANG ĐIỂM với ít hơn 40,5 lần tham
    số — không phải "tốt hơn". Xem docs/BANG_TCN.md mục 5.

Bảng đầy đủ: docs/BANG_TCN.md
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
    """Một khối tích chập nhân quả giãn, có nhánh tắt cộng thẳng.

    Đây là thiết kế của đồ án, không phải bản tái lập một kiến trúc có sẵn nào.
    Mọi con số trong bảng kết quả đều sinh ra từ đúng khối này.

    CẤU TRÚC

        vào ──┬─ tầng 1 ─ tầng 2 ──(+)── ra
              └───────────────────┘

        một tầng = đệm trái → tích chập giãn → chuẩn hoá → ReLU → dropout

    Hai tầng trong cùng một khối dùng CÙNG một độ giãn. Độ giãn tăng gấp đôi
    khi sang khối sau, việc đó do lớp TCN bên ngoài lo.

    NHÂN QUẢ

    Đầu ra ở thời điểm t chỉ được nhìn mẫu t trở về trước. Cài bằng cách đệm
    thêm (kernel_size - 1) * dilation mẫu vào bên TRÁI rồi để tích chập tự cắt
    phần thừa bên phải — không dùng padding đối xứng.

    "Giãn" là lấy mẫu cách quãng: tầm nhìn một tầng là (kernel_size - 1) * d
    trong khi số tham số không đổi.

    KHÔNG CÓ PHI TUYẾN SAU PHÉP CỘNG

    `forward` trả về `x + residual` rồi thôi. Nhiều thiết kế residual đặt thêm
    một ReLU sau phép cộng; ở đây không có. Đường tắt vì thế là đường thẳng
    hoàn toàn từ đầu vào khối tới đầu ra khối.

    Đây là lựa chọn cố định của mọi cấu hình trong đồ án, nên các phép so giữa
    tcn và ds_tcn vẫn cân — hai bên dùng chung đúng lớp này.

    KHÔNG CÓ NHÁNH 1x1 TRÊN ĐƯỜNG TẮT

    Mọi khối giữ nguyên số kênh từ đầu tới cuối, nên đầu vào cộng thẳng được
    với đầu ra, không cần lớp chiếu cho khớp chiều.

    DEPTHWISE (separable=True)

    Thay một tích chập thường bằng hai phép nối tiếp:

        depthwise   mỗi kênh một bộ lọc riêng, KHÔNG trộn kênh  (groups=channels)
        pointwise   kernel 1, CHỈ trộn kênh

    Đếm thật ở C=64, k=3: thường 12.352 tham số, tách ra còn 4.416 — bằng
    0,357 lần. Chuẩn hoá và ReLU đặt MỘT lần sau cả hai phép, không chèn vào
    giữa.
    """

    def __init__(self, channels, kernel_size, dilation, dropout, separable,
                 norm="batch", dropout_kind="channel"):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation

        # Hai tầng giống hệt nhau, cùng độ giãn.
        self.layer_one = self._one_layer(channels, kernel_size, dilation,
                                         dropout, separable, norm, dropout_kind)
        self.layer_two = self._one_layer(channels, kernel_size, dilation,
                                         dropout, separable, norm, dropout_kind)

    def _one_layer(self, channels, kernel_size, dilation, dropout, separable,
                   norm, dropout_kind="channel"):
        """Một tầng: tích chập giãn -> chuẩn hoá -> ReLU -> dropout."""
        if separable:
            # Depthwise: mỗi kênh một bộ lọc riêng, không trộn kênh.
            # Pointwise: kernel 1, chỉ trộn kênh.
            # Ở C kênh, kernel k: tích chập thường tốn C*C*k tham số, tách ra
            # còn C*k + C*C. Với C=64, k=3 là 12.352 xuống 4.416.
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
        elif norm == "none":
            # Không đặt lớp chuẩn hoá nào. Dùng cho cấu hình DS-TCN mà
            # nhóm tối ưu riêng, vốn không có chuẩn hoá.
            norm_layer = nn.Identity()
        else:
            raise ValueError("norm phải là 'batch', 'weight' hoặc 'none', "
                             "nhận " + str(norm))

        return nn.ModuleDict({
            "conv": conv,
            "norm": norm_layer,
            "act": nn.ReLU(),
            "drop": (nn.Dropout1d(dropout) if dropout_kind == "channel"
                     else nn.Dropout(dropout)),
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
        return x + residual        # không có phi tuyến sau phép cộng


class TCN(nn.Module):
    """Chồng nhiều khối TCNBlock, độ giãn gấp đôi mỗi khối.

    RÀNG BUỘC CHÍNH: TẦM NHÌN PHẢI PHỦ HẾT CỬA SỔ VÀO

    Cửa sổ vào 200 mẫu. Nhịp thở khoảng 0,25 Hz, lấy mẫu 50 Hz, nên một nhịp
    thở cũng đúng 200 mẫu. Mỗi khối có hai tầng nên tầm nhìn là

        (k - 1) * 2 * sum(2^i, i = 0..n-1) + 1

        k=3, n=6  ->  253  >= 200   đủ phủ một nhịp thở
        k=3, n=5  ->  125  <  200   thiếu
        k=3, n=4  ->   61              cấu hình được chọn

    Mặc định n_blocks = 6 là để thoả ràng buộc trên. Nhưng cấu hình cho điểm
    cao nhất lại là n_blocks = 4, tầm nhìn 61 — chưa tới một phần ba nhịp thở.
    Đây là kết quả đo được, không phải suy ra từ ràng buộc; xem docs/BANG_TCN.md.

    dropout mặc định 0,0 vì LSTM của MobiVital cũng không dùng dropout. Thực
    nghiệm 1 so KIẾN TRÚC, nên regularization phải giống nhau giữa các cấu hình,
    không thêm biến thứ hai. Cấu hình được chọn bật dropout 0,2 theo phần tử —
    đó là một phần của chính cấu hình đó, đã ghi trong tên.

    norm mặc định "batch". Cấu hình được chọn dùng "none" — không có lớp chuẩn
    hoá nào. Tuỳ chọn "weight" viết lại trọng số của lớp tích chập thay vì thêm
    lớp vào luồng dữ liệu; xem apply_weight_norm.

    channels mặc định 64. Thu nhỏ model là mục tiêu của đồ án, nên không chọn
    số kênh theo cách cho model to ngang mốc hồi quy đem so.
    """

    def __init__(self, channels=64, kernel_size=3, n_blocks=6,
                 dropout=0.0, separable=False, norm="batch",
                 dropout_kind="channel"):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNBlock(channels, kernel_size, 2 ** i,
                                   dropout, separable, norm, dropout_kind))
        self.blocks = nn.Sequential(*blocks)

        self.output_linear = nn.Linear(channels, mv.FUTURE_LENGTH)

    def forward(self, x):
        x = x.unsqueeze(1)          # (batch, 200) -> (batch, 1, 200)
        x = self.input_conv(x)
        x = self.blocks(x)
        x = x[:, :, -1]             # chỉ lấy mẫu cuối cùng
        return self.output_linear(x)      # -> (batch, 25)


class CNNLSTM(nn.Module):
    """Tích chập rút đặc trưng cục bộ rồi đưa vào LSTM. Dự báo đa bước.

    KHÔNG PHẢI ConvLSTM

    ConvLSTM là kiến trúc khác hẳn: nó đưa tích chập vào BÊN TRONG ô LSTM, thay
    phép nhân ma trận bằng tích chập. Ở đây tích chập đứng TRƯỚC, rút đặc trưng
    xong mới đưa vào một LSTM thường. Tên đúng là CNN-LSTM.

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
    `output[:, -1, :]` là LSTM hai chiều, vì với chiều ngược thì bước cuối lại
    chính là mẫu nó xử lý đầu tiên.

    ĐƯỢC GÌ SO VỚI LSTM THUẦN

    Mỗi bước LSTM nhận 32 số mô tả một ĐOẠN sóng, thay vì một mẫu đơn lẻ. Và
    chuỗi ngắn đi bốn lần, từ 200 bước tuần tự xuống 50 — đây chính là nút thắt
    tốc độ của LSTM, vì 200 bước không song song hoá được.

    GIỚI HẠN PHẢI GHI KHI BÁO CÁO

    Kiến trúc này đổi ĐỒNG THỜI hai thứ: cách trích đặc trưng, và độ dài chuỗi
    đưa vào LSTM. Nếu nó thắng thì chưa biết nhờ cái nào. Đối chứng rẻ để tách
    hai nguyên nhân: thay hai tầng tích chập bằng AvgPool 200 xuống 50 rồi đưa
    vào LSTM — nếu bản AvgPool cũng thắng thì công là của việc rút ngắn chuỗi,
    không phải của đặc trưng tích chập. Đối chứng này CHƯA chạy.

    THAM SỐ MẶC ĐỊNH

        conv_channels 32, conv_kernel 5, hai tầng stride 2, hidden 58
        -> 55.667 tham số, xấp xỉ lstm-67 (56.908) để so cùng ngân sách

    Chọn 32 kênh để phần tích chập nhẹ, dồn ngân sách cho LSTM. kernel 5 phủ
    0,1 giây ở tần số lấy mẫu 50 Hz, cỡ một đoạn dốc của sóng thở.
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


def build_model(name, **kwargs):
    """Dựng model theo tên, để notebook chỉ cần truyền chuỗi.

        build_model("lstm")                  -> 1.502.713 tham số
        build_model("lstm", hidden=67)       ->    56.908
        build_model("cnn_lstm", hidden=58)   ->    55.667
        build_model("ds_tcn", channels=64, kernel_size=3, n_blocks=4,
                    dropout=0.2, norm="none", dropout_kind="element")
                                             ->    37.081
    """
    if name == "lstm":
        # hidden mặc định là 352 của MobiVital. Các tham số riêng của TCN
        # (kernel_size, n_blocks...) không dùng ở đây.
        return mv.new_lstm(kwargs.get("hidden"))

    if name == "cnn_lstm":
        return CNNLSTM(hidden_size=kwargs.get("hidden", 58),
                       conv_channels=kwargs.get("conv_channels", 32),
                       conv_kernel=kwargs.get("conv_kernel", 5),
                       future_len=mv.FUTURE_LENGTH)

    # Họ tích chập không nhận `hidden` của lstm, cũng không nhận tham số riêng
    # của cnn_lstm. Lọc bớt để mọi script gọi build_model bằng chung một bộ
    # đối số cho mọi model.
    for rieng in ("hidden", "conv_channels", "conv_kernel"):
        kwargs.pop(rieng, None)

    if name == "tcn":
        return TCN(separable=False, **kwargs)

    if name == "ds_tcn":
        return TCN(separable=True, **kwargs)

    raise ValueError("không biết model tên " + name)


def count_params(model):
    """Số CON SỐ THẬT phải học. Trả lời câu "tốt hơn vì kiến trúc hay vì to hơn".

    MỘT SỐ PHỨC ĐẾM LÀ HAI

    `numel()` của PyTorch đếm một số phức là một, nhưng số phức gồm phần thực và
    phần ảo, và khi train thì CẢ HAI đều được cập nhật. Đếm là một thì báo thiếu.

    Các model trên nhánh này không có tham số phức, nên nhánh đó không đổi số
    của model nào. Giữ lại để công thức đúng chung, không chỉ đúng ở đây.
    """
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.numel() * (2 if p.is_complex() else 1)
    return total
