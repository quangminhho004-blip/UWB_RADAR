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

Số tham số đo được:

    lstm     1,502,713
    tcn         76,633    -95%
    ds_tcn      29,017    -98%

RevIN là lớp bọc, dùng được với cả tcn lẫn ds_tcn.
"""

import torch
import torch.nn as nn

from src import mobivital_reference as mv


class RevIN(nn.Module):
    """Chuẩn hoá theo từng mẫu, rồi trả lại thang đo cũ ở đầu ra.

    Mỗi cửa sổ 200 mẫu có mức nền và biên độ riêng: người thở sâu hay nông,
    ngồi gần hay xa radar. Model phải học vừa hình dạng vừa mấy thứ đó.

    RevIN gỡ phần đó ra: trừ trung bình, chia độ lệch chuẩn, cho model chỉ lo
    hình dạng. Xong thì nhân lại và cộng lại vào đầu ra.
    """

    def normalize(self, x):
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean


class TCNBlock(nn.Module):
    """Một khối tích chập nhân quả.

    "Nhân quả" nghĩa là mẫu thứ t chỉ được nhìn các mẫu <= t, không nhìn tương
    lai. Làm bằng cách đệm thêm bên TRÁI rồi cắt bỏ phần thừa bên phải.

    "Giãn" (dilation) là bỏ cách quãng khi lấy mẫu: giãn 8 thì lấy mẫu cách nhau
    8 bước. Nhờ vậy chồng vài khối là nhìn được xa mà không tốn tham số.
    """

    def __init__(self, channels, kernel_size, dilation, dropout, separable):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation

        if separable:
            # Depthwise: mỗi kênh một bộ lọc riêng, không trộn kênh.
            # Pointwise: kernel 1, chỉ trộn kênh.
            # Hai bước rời nhau tốn ít tham số hơn nhiều so với làm một lần.
            self.conv = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=dilation, groups=channels),
                nn.Conv1d(channels, channels, 1))
        else:
            self.conv = nn.Conv1d(channels, channels, kernel_size,
                                  dilation=dilation)

        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = nn.functional.pad(x, (self.left_pad, 0))    # đệm bên trái
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual                              # nối tắt


class TCN(nn.Module):
    """Chồng nhiều khối TCN, giãn gấp đôi mỗi khối.

    Giãn 1, 2, 4, 8, 16, 32 với kernel 3 thì nhìn được 127 mẫu quá khứ. Nhịp thở
    khoảng 0.25 Hz, lấy mẫu 50 Hz, nên một nhịp thở dài khoảng 200 mẫu.
    """

    def __init__(self, channels=64, kernel_size=3, n_blocks=6,
                 dropout=0.1, separable=False, revin=False):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNBlock(channels, kernel_size, 2 ** i,
                                   dropout, separable))
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


def build_model(name, revin=False, **kwargs):
    """Dựng model theo tên, để notebook chỉ cần truyền chuỗi.

        build_model("lstm")
        build_model("tcn")
        build_model("ds_tcn", revin=True, channels=96)
    """
    if name == "lstm":
        return mv.new_lstm()          # RevIN không áp cho baseline

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
