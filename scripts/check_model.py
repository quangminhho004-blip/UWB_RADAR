"""Kiểm một kiến trúc trước khi đem đi train.

    python scripts/check_model.py --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4
    python scripts/check_model.py --model tcn --channels 64 --norm weight
    python scripts/check_model.py --model lstm --hidden 67 --compare-with lstm

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
    revin        không thêm tham số học được, đảo ngược được, và có
                 đổi đầu ra thật

Model `lstm` là bản của MobiVital, chỉ chạy phần kiểm chung.

Các phép kiểm của bilstm, gru, cnn_lstm, modern_tcn và mix_linear nằm ở nhánh
`main` cùng mã của chúng.

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


def check_revin(model, build_without_revin):
    print("\n6. Riêng RevIN")
    check(model.revin is not None, "RevIN được gắn vào model")

    n_with = models.count_params(model)
    n_without = models.count_params(build_without_revin())
    print("   có RevIN %d tham số, không RevIN %d" % (n_with, n_without))
    check(n_with == n_without,
          "số tham số KHÔNG đổi — RevIN không thêm tham số học được")

    print("\n7. RevIN đảo ngược được")
    x = torch.randn(4, mv.HISTORY_LENGTH) * 5 + 3      # cố ý lệch thang đo
    model.eval()
    with torch.no_grad():
        normed = model.revin.normalize(x)
        restored = model.revin.denormalize(normed)
    print("   sau chuẩn hoá: trung bình %.4f, độ lệch %.4f"
          % (normed.mean().item(), normed.std().item()))
    check(abs(normed.mean().item()) < 0.01, "trung bình về gần 0")
    check(abs(normed.std().item() - 1) < 0.05, "độ lệch chuẩn về gần 1")
    check(torch.allclose(restored, x, atol=1e-3),
          "denormalize trả lại đúng đầu vào ban đầu")

    print("\n8. Đầu ra đổi khi bật RevIN")
    plain = build_without_revin()
    plain.load_state_dict(model.state_dict(), strict=False)
    plain.eval()
    with torch.no_grad():
        check(not torch.allclose(model(x), plain(x)),
              "cùng trọng số nhưng đầu ra khác — RevIN có tác dụng thật")


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
parser.add_argument("--revin", default="false")
parser.add_argument("--compare-with", dest="compare_with", default=None,
                    help="tên model đem so số tham số, ví dụ lstm")
parser.add_argument("--compare-hidden", dest="compare_hidden", type=int, default=None,
                    help="hidden của model đem so")
args = parser.parse_args()


def build(name, hidden):
    return models.build_model(name,
                              revin=args.revin.lower() == "true",
                              hidden=hidden,
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
    if args.revin.lower() == "true":
        check_revin(model, lambda: models.build_model(
            args.model, revin=False, channels=args.channels,
            kernel_size=args.kernel_size, n_blocks=args.n_blocks,
            dropout=args.dropout, norm=args.norm))

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
