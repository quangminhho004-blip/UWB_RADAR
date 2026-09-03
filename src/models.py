"""Các model dự báo. Tất cả cùng một giao diện.

    from src import models
    model = models.tao_model("ds_tcn", revin=True)
    du_bao = model(torch.randn(64, 200))     # -> (64, 25)

GIAO DIỆN BẮT BUỘC

    vào   (batch, 200)   200 mẫu quá khứ
    ra    (batch,  25)   25 mẫu tiếp theo

Đúng như LSTMMultiStep của MobiVital. Sai shape là hỏng cả chuỗi: bộ chọn kênh
gọi model 52 lần mỗi ứng viên, rồi so từng cửa sổ 25 mẫu.

BA MODEL

    lstm      LSTMMultiStep của MobiVital, làm mốc so sánh
    tcn       tích chập nhân quả, giãn dần
    ds_tcn    như trên nhưng tách depthwise + pointwise, ít tham số hơn nhiều

RevIN là lớp bọc, dùng được với cả tcn lẫn ds_tcn.
"""

import torch
import torch.nn as nn

from src import mobivital_reference as mv


class RevIN(nn.Module):
    """Chuẩn hoá theo từng mẫu, rồi trả lại thang đo cũ ở đầu ra.

    Mỗi cửa sổ 200 mẫu có mức nền và biên độ riêng: người thở sâu hay nông,
    ngồi gần hay xa radar. Model phải học vừa hình dạng vừa mấy thứ đó.

    RevIN gỡ phần đó ra: trừ trung bình, chia độ lệch chuẩn, cho model chỉ
    lo hình dạng. Xong thì nhân lại và cộng lại vào đầu ra.
    """

    def forward_chuan_hoa(self, x):
        self.trung_binh = x.mean(dim=1, keepdim=True)
        self.do_lech = x.std(dim=1, keepdim=True) + 1e-5
        return (x - self.trung_binh) / self.do_lech

    def forward_tra_lai(self, y):
        return y * self.do_lech + self.trung_binh


class KhoiTCN(nn.Module):
    """Một khối tích chập nhân quả.

    "Nhân quả" nghĩa là mẫu thứ t chỉ được nhìn các mẫu <= t, không nhìn tương
    lai. Làm bằng cách đệm thêm bên TRÁI rồi cắt bỏ phần thừa bên phải.

    "Giãn" (dilation) là bỏ cách quãng khi lấy mẫu: giãn 8 thì lấy mẫu cách
    nhau 8 bước. Nhờ vậy chồng vài khối là nhìn được xa mà không tốn tham số.
    """

    def __init__(self, so_kenh, kernel_size, gian, dropout, tach_roi):
        super().__init__()
        self.dem_trai = (kernel_size - 1) * gian

        if tach_roi:
            # Depthwise: mỗi kênh một bộ lọc riêng, không trộn kênh.
            # Pointwise: kernel 1, chỉ trộn kênh.
            # Hai bước rời nhau tốn ít tham số hơn nhiều so với làm một lần.
            self.conv = nn.Sequential(
                nn.Conv1d(so_kenh, so_kenh, kernel_size,
                          dilation=gian, groups=so_kenh),
                nn.Conv1d(so_kenh, so_kenh, 1))
        else:
            self.conv = nn.Conv1d(so_kenh, so_kenh, kernel_size, dilation=gian)

        self.chuan_hoa = nn.BatchNorm1d(so_kenh)
        self.kich_hoat = nn.ReLU()
        self.bo_bot = nn.Dropout(dropout)

    def forward(self, x):
        goc = x
        x = nn.functional.pad(x, (self.dem_trai, 0))   # đệm bên trái
        x = self.conv(x)
        x = self.chuan_hoa(x)
        x = self.kich_hoat(x)
        x = self.bo_bot(x)
        return x + goc                                  # nối tắt


class TCN(nn.Module):
    """Chồng nhiều khối TCN, giãn gấp đôi mỗi khối.

    Giãn 1, 2, 4, 8, 16, 32 với kernel 3 thì nhìn được 127 mẫu quá khứ.
    Nhịp thở khoảng 0.25 Hz, lấy mẫu 50 Hz, nên một nhịp thở dài ~200 mẫu.
    """

    def __init__(self, so_kenh=64, kernel_size=3, so_khoi=6,
                 dropout=0.1, tach_roi=False, revin=False):
        super().__init__()
        self.dau_vao = nn.Conv1d(1, so_kenh, 1)

        cac_khoi = []
        for i in range(so_khoi):
            cac_khoi.append(KhoiTCN(so_kenh, kernel_size, 2 ** i,
                                    dropout, tach_roi))
        self.cac_khoi = nn.Sequential(*cac_khoi)

        self.dau_ra = nn.Linear(so_kenh, mv.FUTURE_LENGTH)
        self.revin = RevIN() if revin else None

    def forward(self, x):
        if self.revin is not None:
            x = self.revin.forward_chuan_hoa(x)

        x = x.unsqueeze(1)          # (batch, 200) -> (batch, 1, 200)
        x = self.dau_vao(x)
        x = self.cac_khoi(x)
        x = x[:, :, -1]             # chỉ lấy mẫu cuối cùng
        y = self.dau_ra(x)          # -> (batch, 25)

        if self.revin is not None:
            y = self.revin.forward_tra_lai(y)
        return y


def tao_model(ten, revin=False, **tham_so):
    """Dựng model theo tên, để notebook chỉ cần truyền chuỗi.

        tao_model("lstm")
        tao_model("tcn")
        tao_model("ds_tcn", revin=True, so_kenh=96)
    """
    if ten == "lstm":
        return mv.new_lstm()          # RevIN không áp cho baseline

    if ten == "tcn":
        return TCN(tach_roi=False, revin=revin, **tham_so)

    if ten == "ds_tcn":
        return TCN(tach_roi=True, revin=revin, **tham_so)

    raise ValueError("không biết model tên " + ten)


def dem_tham_so(model):
    """Số tham số học được. Để trả lời câu 'tốt hơn vì kiến trúc hay vì to hơn'."""
    tong = 0
    for p in model.parameters():
        if p.requires_grad:
            tong = tong + p.numel()
    return tong
