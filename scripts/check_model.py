"""Kiểm một kiến trúc trước khi đem đi train.

    python scripts/check_model.py --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4
    python scripts/check_model.py --model tcn --channels 64 --norm weight
    python scripts/check_model.py --model lstm --hidden 67 --compare-with lstm
    python scripts/check_model.py --model cnn_lstm --hidden 58 --compare-with lstm --compare-hidden 67

VÌ SAO CẦN

Một lần chạy CV mất một tới ba giờ. Bản cài đặt sai một chi tiết thì model vẫn
chạy, vẫn ra số, rồi cho kết luận "kiến trúc này thua" trong khi thật ra là code
hỏng. Kiểm trước mất vài giây.

KIỂM GÌ

    chung        shape vào ra, giá trị hữu hạn, gradient lan ngược được,
                 số tham số, lưu và nạp lại state_dict
    tcn/ds_tcn   norm=weight thì BatchNorm bị gỡ hẳn và WeightNorm được áp;
                 norm=none thì không có lớp chuẩn hoá nào; đúng loại
                 dropout; và tầm nhìn so với cửa sổ 200 mẫu
    cnn_lstm     độ dài chuỗi vào LSTM đúng 50, và LSTM là MỘT chiều

Model `lstm` là bản của MobiVital, chỉ chạy phần kiểm chung.

Không kiểm chất lượng dự báo — việc đó là của run_cv.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn

from src import mobivital_reference as mv
from src import models


def check(condition, description):
    """In một dòng kết quả. Sai thì dừng cả script."""
    print("   %-58s %s" % (description, "đạt" if condition else "KHÔNG ĐẠT"))
    if not condition:
        raise SystemExit("\nDỪNG — bản cài đặt có vấn đề, đừng đem đi train.")


def check_common(model, name):
    print("\n1. Hình dạng vào ra và giá trị")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print("   vào %s  ->  ra %s" % (tuple(x.shape), tuple(y.shape)))
    check(y.shape == (4, mv.FUTURE_LENGTH),
         "ra đúng (4, %d)" % mv.FUTURE_LENGTH)
    check(bool(torch.isfinite(y).all()), "mọi giá trị hữu hạn")

    print("\n2. Gradient")
    model.train()
    model(x).sum().backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = len(list(model.parameters()))
    print("   %d/%d tham số nhận được gradient" % (n_with_grad, total))
    check(n_with_grad == total, "mọi tham số đều lan ngược tới")

    print("\n3. Số tham số")
    n = models.count_params(model)
    print("   %d" % n)
    return n


def check_save_load(model, rebuild):
    """Lưu rồi nạp lại phải ra đúng số cũ — điều kiện để checkpoint dùng được."""
    print("\n4. Lưu và nạp lại state_dict")
    x = torch.randn(2, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        before = model(x)

    tmp_path = "/tmp/check_model_tam.pth"
    torch.save(model.state_dict(), tmp_path)
    reloaded = rebuild()
    reloaded.load_state_dict(torch.load(tmp_path, map_location="cpu"))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(x)
    os.remove(tmp_path)

    check(torch.equal(before, after), "nạp lại cho ra đúng đầu ra cũ")


def check_cnn_lstm(model):
    """Hai chỗ dễ sai của CNN-LSTM, kiểm bằng số chứ không đọc mã."""
    print("\n5. Riêng CNN-LSTM")

    # Hai tầng stride 2 phải rút 200 mẫu xuống đúng 50. Sai padding một mẫu là
    # ra 49 hoặc 51 mà model vẫn chạy bình thường, không báo gì.
    x = torch.randn(4, 1, mv.HISTORY_LENGTH)
    with torch.no_grad():
        feat = model.conv(x)
    print("   tích chập: (%d, %d, %d) -> (%d, %d, %d)"
          % (tuple(x.shape) + tuple(feat.shape)))
    check(feat.shape[2] == mv.HISTORY_LENGTH // 4,
          "chuỗi rút đúng 4 lần, còn %d bước cho LSTM" % (mv.HISTORY_LENGTH // 4))

    # LSTM phải MỘT chiều. Nếu ai đó bật bidirectional thì output[:, -1, :]
    # lấy phải nửa chiều ngược mới đọc đúng một mẫu — hỏng ngầm, không báo lỗi.
    check(not model.lstm.bidirectional,
          "LSTM một chiều, nên output[:, -1, :] đã đọc hết chuỗi")
    check(model.lstm.input_size == feat.shape[1],
          "số kênh tích chập khớp input_size của LSTM")


def check_tcn(model, norm, dropout_kind="channel"):
    print("\n5. Riêng TCN")
    n_batchnorm = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    param_names = [t for t, _ in model.named_parameters()]
    has_weightnorm = any("parametrizations" in t or t.endswith("_g")
                        for t in param_names)
    print("   %d lớp BatchNorm1d, WeightNorm: %s" % (n_batchnorm, has_weightnorm))

    if norm == "weight":
        check(n_batchnorm == 0, "BatchNorm bị gỡ hẳn khi bật WeightNorm")
        check(has_weightnorm, "WeightNorm thật sự được áp lên trọng số")
    elif norm == "none":
        check(n_batchnorm == 0, "KHÔNG có lớp chuẩn hoá nào, đúng cấu hình đã chọn")
        check(not has_weightnorm, "cũng không có WeightNorm")
    else:
        check(n_batchnorm > 0, "có BatchNorm như mong đợi")
        check(not has_weightnorm, "không có WeightNorm")

    print("\n6. Loại dropout")
    n_chan = sum(1 for m in model.modules() if isinstance(m, nn.Dropout1d))
    n_elem = sum(1 for m in model.modules() if type(m) is nn.Dropout)
    print("   %d lớp Dropout1d (xoá cả kênh), %d lớp Dropout (xoá từng phần tử)"
          % (n_chan, n_elem))
    if dropout_kind == "element":
        check(n_elem > 0 and n_chan == 0,
              "dùng nn.Dropout, xoá từng phần tử")
    else:
        check(n_chan > 0 and n_elem == 0,
              "dùng nn.Dropout1d, spatial dropout theo Bai mục 3.4")

    print("\n7. Tầm nhìn so với cửa sổ vào")
    k = model.blocks[0].layer_one["conv"]
    k = k[0].kernel_size[0] if isinstance(k, nn.Sequential) else k.kernel_size[0]
    n = len(model.blocks)
    rf = (k - 1) * 2 * sum(2 ** i for i in range(n)) + 1
    print("   kernel %d, %d khối -> tầm nhìn %d, cửa sổ vào %d"
          % (k, n, rf, mv.HISTORY_LENGTH))
    if rf >= mv.HISTORY_LENGTH:
        check(True, "phủ trọn cửa sổ")
    else:
        print("   CHÚ Ý: chỉ thấy %d/%d mẫu gần nhất, mất %.0f%% đầu cửa sổ"
              % (rf, mv.HISTORY_LENGTH, 100 * (1 - rf / mv.HISTORY_LENGTH)))
        check(True, "tầm nhìn ngắn hơn cửa sổ — có chủ ý, không phải lỗi")


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="lstm | tcn | ds_tcn")
parser.add_argument("--hidden", type=int, default=mv.LSTM_HIDDEN_SIZE)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--n_blocks", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--norm", default="batch",
                    choices=["batch", "weight", "none"])
parser.add_argument("--dropout_kind", default="channel",
                    choices=["channel", "element"])
parser.add_argument("--conv_channels", type=int, default=32)
parser.add_argument("--conv_kernel", type=int, default=5)
parser.add_argument("--compare-with", dest="compare_with", default=None,
                    help="tên model đem so số tham số, ví dụ lstm")
parser.add_argument("--compare-hidden", dest="compare_hidden", type=int, default=None,
                    help="hidden của model đem so")
args = parser.parse_args()


def build(name, hidden):
    return models.build_model(name,
                              hidden=hidden,
                              conv_channels=args.conv_channels,
                              conv_kernel=args.conv_kernel,
                              channels=args.channels,
                              kernel_size=args.kernel_size,
                              n_blocks=args.n_blocks,
                              dropout=args.dropout,
                              norm=args.norm,
                              dropout_kind=args.dropout_kind)


print("Kiểm model:", args.model)
model = build(args.model, args.hidden)

n_params = check_common(model, args.model)
check_save_load(model, lambda: build(args.model, args.hidden))

if args.model in ("tcn", "ds_tcn"):
    check_tcn(model, args.norm, args.dropout_kind)

if args.model == "cnn_lstm":
    check_cnn_lstm(model)

if args.compare_with:
    print("\nSO SỐ THAM SỐ với %s" % args.compare_with)
    other = build(args.compare_with, args.compare_hidden or mv.LSTM_HIDDEN_SIZE)
    n_other = models.count_params(other)
    gap_percent = 100 * (n_params / n_other - 1)
    print("   %-12s %d" % (args.model, n_params))
    print("   %-12s %d   (lệch %+.1f%%)" % (args.compare_with, n_other, gap_percent))
    check(abs(gap_percent) < 5.0,
         "lệch dưới 5%, so được ở cùng ngân sách tham số")

print("\nTẤT CẢ ĐẠT — bản cài đặt dùng được.")
