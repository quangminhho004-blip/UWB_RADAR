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
    tcn/ds_tcn   norm=weight thì BatchNorm bị gỡ hẳn và WeightNorm được áp;
                 norm=none thì không có lớp chuẩn hoá nào; đúng loại
                 dropout; và tầm nhìn so với cửa sổ 200 mẫu
    revin        không thêm tham số học được, đảo ngược được, và có
                 đổi đầu ra thật
    modern_tcn   chia đúng 50 đoạn, đệm bằng cách lặp mẫu cuối, hai
                 nhánh kernel đều có tác dụng, chỉ một nối tắt, và
                 không sót phần khai báo mà không gọi
    gru          một chiều nên output[:, -1, :] là đúng, Linear KHÔNG
                 gấp đôi hidden như BiLSTM
    low_rank_linear   hai lớp không phi tuyến hợp lại thành một phép affine
                 hạng tối đa bằng chiều giữa
    mix_linear_linear / mix_linear_mlp
                 gradient tới cả nền lẫn nhánh phụ, xoá nhánh phụ thì
                 khớp đúng MixLinear nền, trung bình chỉ cộng một lần,
                 và hàm kích hoạt đúng loại mong đợi
    mix_linear   hai cách đếm tham số (47 numel / 63 số thật), tham số
                 phức nhận gradient và Adam đổi được, forward không còn
                 lệnh print của mã gốc, hai nhánh đều tác động tới đầu
                 ra, và mix_alpha đúng là trọng số trộn

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
        check(n_batchnorm == 0, "KHÔNG có lớp chuẩn hoá nào, đúng thí nghiệm cũ")
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
              "dùng nn.Dropout, đúng loại của thí nghiệm cũ")
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
        for name in ("dw_large", "dw_small"):
            conv = getattr(block, name)[0]
            saved = conv.weight.detach().clone()
            conv.weight.zero_()
            changed = not torch.allclose(model(x), base)
            conv.weight.copy_(saved)
            check(changed, "xoá %s làm đổi đầu ra — nhánh này có tác dụng" % name)

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


def check_gru(model, hidden):
    print("\n5. Riêng GRU")
    check(not model.gru.bidirectional,
          "GRU MỘT chiều, nên output[:, -1, :] là đúng")
    check(model.gru.num_layers == mv.LSTM_NUM_LAYERS,
          "đúng %d tầng như LSTM gốc" % mv.LSTM_NUM_LAYERS)
    check(model.gru.hidden_size == hidden, "hidden đúng %d" % hidden)
    check(model.linear.in_features == hidden,
          "Linear nhận %d chiều — KHÔNG gấp đôi như BiLSTM" % hidden)

    print("\n6. forward dùng đúng output[:, -1, :]")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        output, _ = model.gru(x.unsqueeze(-1))
        check(torch.allclose(model(x), model.linear(output[:, -1, :])),
              "đầu ra khớp với cách lấy bước cuối")

    print("\n7. Mỗi đơn vị GRU tốn 3 khối trọng số, LSTM tốn 4")
    n_gru = sum(p.numel() for p in model.gru.parameters())
    first_layer = 3 * (hidden * 1 + hidden * hidden + 2 * hidden)
    print("   tầng đầu tính tay %d, cả %d tầng %d"
          % (first_layer, model.gru.num_layers, n_gru))
    check(n_gru > first_layer, "còn tầng thứ hai nữa")


def check_mix_linear(model):
    print("\n5. Riêng MixLinear — hai cách đếm tham số")
    numel = sum(p.numel() for p in model.parameters())
    n_real = models.count_params(model)
    n_complex = sum(p.numel() for p in model.parameters() if p.is_complex())
    print("   numel() báo %d, số thật %d, trong đó %d số phức"
          % (numel, n_real, n_complex))
    check(n_real == numel + n_complex, "số thật = numel cộng thêm phần ảo")
    check(n_complex > 0, "có tham số phức thật — đúng thiết kế nhánh tần số")

    print("\n6. Tham số phức nhận gradient và Adam cập nhật được")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.train()
    model(x).sum().backward()
    complex_params = [p for p in model.parameters() if p.is_complex()]
    check(all(p.grad is not None for p in complex_params), "mọi tham số phức có gradient")
    before = complex_params[0].detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), torch.randn(4, mv.FUTURE_LENGTH)).backward()
        opt.step()
    check(not torch.equal(before, complex_params[0]), "Adam đổi được tham số phức")
    check(bool(torch.isfinite(complex_params[0]).all()), "giá trị vẫn hữu hạn sau 3 bước")

    print("\n7. forward không còn lệnh print của mã gốc")
    import inspect
    source = inspect.getsource(type(model).forward)
    source += inspect.getsource(type(model)._time_domain)
    source += inspect.getsource(type(model)._freq_domain)
    check("print(" not in source,
          "không có print — mã tác giả bỏ quên hai lệnh, đã gỡ")

    print("\n8. Cả nhánh thời gian lẫn nhánh tần số đều tác động tới đầu ra")
    model.eval()
    with torch.no_grad():
        base = model(x).clone()
        for name in ("TLinear1", "FLinear1"):
            w = getattr(model, name).weight
            saved = w.detach().clone()
            w.zero_()
            changed = not torch.allclose(model(x), base)
            w.copy_(saved)
            check(changed, "xoá %s làm đổi đầu ra" % name)

    print("\n9. mix_alpha đúng là trọng số trộn hai nhánh")
    with torch.no_grad():
        z = x.unsqueeze(-1)
        mean = z.mean(dim=1).unsqueeze(1)
        u = (z - mean).permute(0, 2, 1)
        u = model.conv1d(u.reshape(-1, 1, model.seq_len)).reshape(
            -1, 1, model.seq_len) + u
        u = u.reshape(4, 1, -1, model.period_len).permute(0, 1, 3, 2)
        time_part = model._time_domain(u, 4)[:, :model.pred_len, :]
        freq_part = model._freq_domain(u, 4)[:, :model.pred_len, :]
        mixed = (time_part * model.alpha + freq_part * (1 - model.alpha)
                 + mean).squeeze(-1)
        check(torch.allclose(model(x), mixed, atol=1e-5),
              "đầu ra = thời_gian*%.2f + tần_số*%.2f + trung bình"
              % (model.alpha, 1 - model.alpha))

    print("\n10. Chia đoạn đúng số")
    n_seg = model.seq_len // model.period_len
    print("   %d mẫu / đoạn %d = %d đoạn, lưới %dx%d"
          % (model.seq_len, model.period_len, n_seg,
             model.sqrt_seg_num_x, model.sqrt_seg_num_x))
    check(n_seg == 20, "đúng 20 đoạn")
    check(model.sqrt_seg_num_x == 5, "lưới 5x5, đủ chứa 20 đoạn")


def check_low_rank_linear(model, hidden):
    print("\n5. Riêng LowRankLinear")
    layers = [m for m in model.branch if isinstance(m, nn.Linear)]
    check(len(layers) == 2, "đúng hai lớp Linear")
    check(layers[0].out_features == hidden,
          "lớp đầu ép %d chiều xuống %d" % (mv.HISTORY_LENGTH, hidden))
    check(all(l.bias is not None for l in layers), "cả hai lớp đều có bias")

    print("\n6. Hai lớp không phi tuyến = MỘT phép affine hạng <= %d" % hidden)
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.eval()
    with torch.no_grad():
        W = layers[1].weight @ layers[0].weight
        rank = torch.linalg.matrix_rank(W).item()
    print("   tích hai ma trận %s, hạng %d" % (tuple(W.shape), rank))
    check(rank <= hidden, "hạng không vượt %d" % hidden)
    print("   Linear(%d, %d) đầy đủ sẽ tốn %d tham số, ở đây %d"
          % (mv.HISTORY_LENGTH, mv.FUTURE_LENGTH,
             mv.HISTORY_LENGTH * mv.FUTURE_LENGTH + mv.FUTURE_LENGTH,
             models.count_params(model)))


def check_mix_linear_plus(model, hidden, has_gelu):
    print("\n5. Riêng MixLinearPlus")
    check(isinstance(model.base, models.MixLinear), "nền đúng là MixLinear")
    layers = [m for m in model.correction if isinstance(m, nn.Linear)]
    check(len(layers) == 2, "nhánh phụ có đúng hai lớp Linear")
    check(layers[0].out_features == hidden, "chiều giữa đúng %d" % hidden)
    act = [m for m in model.correction if not isinstance(m, nn.Linear)][0]
    print("   hàm kích hoạt giữa hai lớp: %s" % type(act).__name__)
    check(isinstance(act, nn.GELU) == has_gelu,
          "đúng %s như mong đợi" % ("GELU" if has_gelu else "Identity"))

    print("\n6. Gradient tới CẢ nền lẫn nhánh phụ")
    x = torch.randn(4, mv.HISTORY_LENGTH)
    model.train()
    model(x).sum().backward()
    n_base = sum(1 for p in model.base.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    n_corr = sum(1 for p in model.correction.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    print("   nền %d/%d tham số có gradient khác 0, nhánh phụ %d/%d"
          % (n_base, len(list(model.base.parameters())),
             n_corr, len(list(model.correction.parameters()))))
    check(n_base > 0, "nền thật sự được học, không nằm chết")
    check(n_corr > 0, "nhánh phụ thật sự được học")

    print("\n7. Ghép là phép CỘNG THUẦN, không gate hay hệ số ẩn")
    model.eval()
    with torch.no_grad():
        saved = [p.detach().clone() for p in model.correction.parameters()]
        for p in model.correction.parameters():
            p.zero_()
        same_as_base = torch.equal(model(x), model.base(x))
        for p, v in zip(model.correction.parameters(), saved):
            p.copy_(v)
    check(same_as_base, "xoá nhánh phụ thì đầu ra khớp ĐÚNG MixLinear nền")

    print("\n8. Trung bình chỉ được cộng MỘT lần")
    z = torch.randn(4, mv.HISTORY_LENGTH) + 50.0
    with torch.no_grad():
        out_mean = model(z).mean().item()
    print("   vào trung bình %.2f  ->  ra trung bình %.2f"
          % (z.mean().item(), out_mean))
    check(abs(out_mean - z.mean().item()) < 5.0,
          "ra bám mức nền đầu vào, không lệch gấp đôi")

    print("\n9. GELU có tạo khác biệt thật không")
    with torch.no_grad():
        u = torch.randn(64, hidden) * 3
        activated = act(u)
        changed = not torch.allclose(activated, u)
    if has_gelu:
        check(changed, "GELU đổi giá trị — phi tuyến thật sự hoạt động")
    else:
        check(not changed, "Identity giữ nguyên giá trị, đúng vai trò đối chứng")


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="lstm | bilstm | gru | cnn_lstm | tcn | ds_tcn | "
                         "modern_tcn | mix_linear")
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
parser.add_argument("--kernel_large", type=int, default=31)
parser.add_argument("--kernel_small", type=int, default=5)
parser.add_argument("--period_len", type=int, default=10)
parser.add_argument("--lpf", type=int, default=5)
parser.add_argument("--mix_alpha", type=float, default=0.5)
parser.add_argument("--mix_hidden", type=int, default=2)
parser.add_argument("--correction_hidden", type=int, default=4)
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
                              dropout_kind=args.dropout_kind,
                              conv_channels=args.conv_channels,
                              conv_kernel=args.conv_kernel,
                              kernel_large=args.kernel_large,
                              kernel_small=args.kernel_small,
                              period_len=args.period_len,
                              lpf=args.lpf,
                              mix_alpha=args.mix_alpha,
                              mix_hidden=args.mix_hidden,
                              correction_hidden=args.correction_hidden)


print("Kiểm model:", args.model)
model = build(args.model, args.hidden)

n_params = check_common(model, args.model)
check_save_load(model, lambda: build(args.model, args.hidden))

if args.model == "bilstm":
    check_bilstm(model, args.hidden)
elif args.model == "gru":
    check_gru(model, args.hidden)
elif args.model == "mix_linear":
    check_mix_linear(model)
elif args.model == "low_rank_linear":
    check_low_rank_linear(model, args.correction_hidden)
elif args.model in ("mix_linear_linear", "mix_linear_mlp"):
    check_mix_linear_plus(model, args.correction_hidden,
                          args.model == "mix_linear_mlp")
elif args.model == "cnn_lstm":
    check_cnn_lstm(model, args.hidden, args.conv_channels)
elif args.model == "modern_tcn":
    check_moderntcn(model, args.channels)
elif args.model in ("tcn", "ds_tcn"):
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
