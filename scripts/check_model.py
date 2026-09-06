"""Kiểm một kiến trúc trước khi đem đi train.

    python scripts/check_model.py --model bilstm --hidden 41
    python scripts/check_model.py --model tcn --channels 64 --norm weight
    python scripts/check_model.py --model lstm --hidden 67 --compare-with lstm

VÌ SAO CẦN

Một lần chạy CV mất một tới ba giờ. Bản cài đặt sai một chi tiết thì model vẫn
chạy, vẫn ra số, rồi cho kết luận "kiến trúc này thua" trong khi thật ra là code
hỏng. Kiểm trước mất vài giây.

Đây là chỗ dễ sai nhất từng gặp: với LSTM hai chiều, `output[:, -1, :]` cho nửa
chiều xuôi đã đọc hết 200 mẫu, nhưng nửa chiều ngược mới đọc một mẫu — vì với
chiều ngược thì bước cuối chính là bước đầu tiên nó xử lý. Phải lấy từ `h_n`.

KIỂM GÌ

    chung        shape vào ra, giá trị hữu hạn, gradient lan ngược được,
                 số tham số, lưu và nạp lại state_dict
    bilstm       hai chiều thật, Linear nhận đủ 2*hidden, và chứng minh
                 h[-2],h[-1] KHÁC output[:,-1,:]
    cnn_lstm     một chiều, chuỗi rút từ 200 xuống 50, hai tầng conv
                 stride 2, không có pooling thêm
    tcn/ds_tcn   norm=weight thì BatchNorm bị gỡ hẳn và WeightNorm được áp
    revin        không thêm tham số học được, đảo ngược được, và có
                 đổi đầu ra thật
    modern_tcn   chia đúng 50 đoạn, đệm bằng cách lặp mẫu cuối, hai
                 nhánh kernel đều có tác dụng, chỉ một nối tắt, và
                 không sót phần khai báo mà không gọi

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


def check_bilstm(model, hidden):
    print("\n5. Riêng BiLSTM")
    check(model.lstm.bidirectional, "nn.LSTM bật bidirectional")
    check(model.lstm.num_layers == mv.LSTM_NUM_LAYERS,
         "đúng %d tầng như LSTM gốc" % mv.LSTM_NUM_LAYERS)
    check(model.linear.in_features == 2 * hidden,
         "Linear nhận %d chiều, gấp đôi hidden" % (2 * hidden))

    print("\n6. h[-2],h[-1] phải KHÁC output[:,-1,:]")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        out, (h, _) = model.lstm(x.unsqueeze(-1))
        correct_way = torch.cat([h[-2], h[-1]], dim=1)
        wrong_way = out[:, -1, :]

    forward_same = torch.allclose(correct_way[:, :hidden], wrong_way[:, :hidden])
    backward_same = torch.allclose(correct_way[:, hidden:], wrong_way[:, hidden:])
    print("   chênh lệch lớn nhất: %.6f"
          % (correct_way - wrong_way).abs().max().item())
    check(forward_same, "nửa chiều xuôi giống nhau, đúng như mong đợi")
    check(not backward_same,
         "nửa chiều ngược KHÁC nhau — đây là chỗ dễ dùng nhầm")

    print("\n7. forward dùng đúng h[-2],h[-1]")
    with torch.no_grad():
        check(torch.allclose(model(x), model.linear(correct_way)),
             "đầu ra khớp với cách lấy từ h_n")


def check_cnn_lstm(model, hidden, conv_channels):
    print("\n5. Riêng CNN-LSTM")
    check(not model.lstm.bidirectional,
          "LSTM MỘT chiều, nên output[:, -1, :] là đúng")
    check(model.lstm.input_size == conv_channels,
          "LSTM nhận %d chiều, bằng số kênh tích chập" % conv_channels)
    check(model.lstm.hidden_size == hidden, "hidden đúng %d" % hidden)

    print("\n6. Khối tích chập rút ngắn chuỗi 200 -> 50")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        after_conv = model.conv(x.unsqueeze(1))
    print("   (4, 1, %d)  ->  %s" % (mv.HISTORY_LENGTH, tuple(after_conv.shape)))
    check(after_conv.shape == (4, conv_channels, mv.HISTORY_LENGTH // 4),
          "ra đúng (4, %d, %d)" % (conv_channels, mv.HISTORY_LENGTH // 4))

    print("\n7. Hai tầng tích chập, cả hai stride 2")
    convs = [m for m in model.conv if isinstance(m, nn.Conv1d)]
    print("   %d lớp Conv1d, stride %s"
          % (len(convs), [c.stride[0] for c in convs]))
    check(len(convs) == 2, "đúng hai tầng tích chập")
    check(all(c.stride[0] == 2 for c in convs), "cả hai đều stride 2")
    check(not any(isinstance(m, (nn.MaxPool1d, nn.AvgPool1d))
                  for m in model.modules()),
          "không có tầng pooling nào thêm")


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


def check_tcn(model, norm):
    print("\n5. Riêng TCN")
    n_batchnorm = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    param_names = [t for t, _ in model.named_parameters()]
    has_weightnorm = any("parametrizations" in t or t.endswith("_g")
                        for t in param_names)
    print("   %d lớp BatchNorm1d, WeightNorm: %s" % (n_batchnorm, has_weightnorm))

    if norm == "weight":
        check(n_batchnorm == 0, "BatchNorm bị gỡ hẳn khi bật WeightNorm")
        check(has_weightnorm, "WeightNorm thật sự được áp lên trọng số")
    else:
        check(n_batchnorm > 0, "có BatchNorm như mong đợi")
        check(not has_weightnorm, "không có WeightNorm")


def check_moderntcn(model, channels):
    print("\n5. Riêng ModernTCN — chia đoạn")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        z = x.unsqueeze(1)
        padded = torch.cat([z, z[:, :, -1:].repeat(1, 1, model.pad_len)], dim=-1)
        patched = model.patch_embed(padded)
    print("   (4, 1, %d) -> đệm %s -> đoạn %s"
          % (mv.HISTORY_LENGTH, tuple(padded.shape), tuple(patched.shape)))
    check(model.pad_len == model.patch_size - model.patch_stride,
          "đệm %d = patch_size - patch_stride, hằng số" % model.pad_len)
    check(patched.shape == (4, channels, 50), "ra đúng 50 đoạn, không phải 49")

    print("\n6. Đệm bằng cách LẶP giá trị cuối, không phải đệm 0")
    tail = padded[:, :, -model.pad_len:]
    last = z[:, :, -1:].expand_as(tail)
    print("   %d giá trị đệm, lệch lớn nhất so với mẫu cuối: %.6f"
          % (model.pad_len, (tail - last).abs().max().item()))
    check(torch.equal(tail, last), "mọi giá trị đệm bằng đúng mẫu cuối")
    check(tail.abs().sum().item() > 0, "không phải đệm 0")

    print("\n7. Cả hai nhánh kernel đều nối vào forward")
    block = model.blocks[0]
    with torch.no_grad():
        base = model(x).clone()
        for ten in ("dw_large", "dw_small"):
            conv = getattr(block, ten)[0]
            saved = conv.weight.detach().clone()
            conv.weight.zero_()
            doi = not torch.allclose(model(x), base)
            conv.weight.copy_(saved)
            check(doi, "xoá %s làm đổi đầu ra — nhánh này có tác dụng" % ten)

    print("\n8. Đúng MỘT nối tắt, ôm cả khối")
    with torch.no_grad():
        u = torch.randn(4, channels, 50)
        thu_cong = u + block.ffn(block.norm(block.dw_large(u) + block.dw_small(u)))
        check(torch.allclose(block(u), thu_cong, atol=1e-6),
              "khối = vào + ffn(norm(lớn + nhỏ)), không có nối tắt thứ hai")

    print("\n9. Không có phần khai báo mà không dùng")
    ten_module = [t for t, _ in model.named_modules()]
    print("   %d module, không có tên nào chứa 'ffn2'" % len(ten_module))
    check(not any("ffn2" in t for t in ten_module),
          "ConvFFN2 bị bỏ hẳn — mã gốc khai báo nhưng forward không gọi")
    check(not any(isinstance(m, nn.Conv1d) and m.bias is not None
                  for m in [block.dw_large[0], block.dw_small[0]]),
          "hai nhánh depthwise không có bias, vì BatchNorm ngay sau")


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="lstm | bilstm | cnn_lstm | tcn | ds_tcn | modern_tcn")
parser.add_argument("--hidden", type=int, default=mv.LSTM_HIDDEN_SIZE)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--n_blocks", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--norm", default="batch", choices=["batch", "weight"])
parser.add_argument("--conv_channels", type=int, default=32)
parser.add_argument("--conv_kernel", type=int, default=5)
parser.add_argument("--kernel_large", type=int, default=31)
parser.add_argument("--kernel_small", type=int, default=5)
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
                              conv_channels=args.conv_channels,
                              conv_kernel=args.conv_kernel,
                              kernel_large=args.kernel_large,
                              kernel_small=args.kernel_small)


print("Kiểm model:", args.model)
model = build(args.model, args.hidden)

n_params = check_common(model, args.model)
check_save_load(model, lambda: build(args.model, args.hidden))

if args.model == "bilstm":
    check_bilstm(model, args.hidden)
elif args.model == "cnn_lstm":
    check_cnn_lstm(model, args.hidden, args.conv_channels)
elif args.model == "modern_tcn":
    check_moderntcn(model, args.channels)
elif args.model in ("tcn", "ds_tcn"):
    check_tcn(model, args.norm)
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
