"""Train đủ ABCDEFKL rồi test đúng một lần trên GHIJ.

    python scripts/run_final_test.py --model ds_tcn --revin true

Đây là SỐ CÔNG BỐ. Chỉ chạy sau khi đã chốt cấu hình bằng scripts/run_cv.py —
G H I J không được nhìn vào lúc chọn cấu hình (docs/PROTOCOL.md mục 1).

LUỒNG

    windows/final_train  ->  training.train  ->  final.pth
                                                     |
                         scoring.score_all(GHIJ)  <--+
                                 |
                                 +--> .txt   bảng lựa chọn kênh, 537 dòng
                                 +--> .csv   điểm từng buổi ghi
                                 +--> runs/summary.csv   một dòng

Cửa sổ đọc từ `windows/final_train/`, cắt gộp cả 8 người theo đúng thứ tự
MobiVital, nên số ra so thẳng được với TN0. Chạy scripts/make_windows.py trước.

ĐỐI CHIẾU VỚI PIPELINE GỐC

Script này làm đúng việc mà bốn lệnh của MobiVital làm:

    prep_breath_final.py + autoreg_training.py  ->  src/training.py
    mobivital_gen.py                            ->  src/scoring.py
    evaluate.py                                 ->  src/results.py

Khác đúng một chỗ: nhận model bất kỳ. `mobivital_gen.py` dòng 152 ghi cứng
`LSTMMultiStep(...)` nên không nhét TCN vào được. Bằng chứng hai bên tương
đương: notebooks/TN0.ipynb mục 7, lựa chọn kênh trùng 537/537.
"""

import argparse
import os
import subprocess
import sys
import time

# Bộ nhớ GPU vỡ vụn ở bước chấm điểm: mỗi buổi ghi đẩy 6708 chuỗi một lô, xin
# rồi trả liên tục 537 lần. Không có cờ này thì TN0 tràn ở buổi ghi 27/1874 dù
# tổng chỗ trống vẫn còn — chỉ là không còn khối liền đủ lớn.
# Phải đặt TRƯỚC "import torch", vì torch đọc biến này lúc nạp.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src import mobivital_reference as mv
from src import models
from src import results
from src import scoring
from src import training


# ===================== CÀI ĐẶT — sửa ở đây =====================

TEST_USERS = ["G", "H", "I", "J"]

WINDOWS_DIR = "data/processed/windows/final_train"
SUMMARY_FILE = "runs/summary.csv"

# EXP_DIR đặt theo --experiment: MỘT thư mục cho cả checkpoint lẫn điểm,
# mỗi thực nghiệm một thư mục riêng. Xem phần argparse bên dưới.

# ===============================================================


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ds_tcn",
                    help="lstm | bilstm | gru | cnn_lstm | tcn | ds_tcn | modern_tcn | "
                         "mix_linear | mix_linear_linear | mix_linear_mlp | "
                         "low_rank_linear")
parser.add_argument("--revin", default="false", help="true | false")
parser.add_argument("--loss", default="mse", help="mse | mse_pearson")
parser.add_argument("--alpha", type=float, default=1.0, help="trọng số MSE khi loss=mse_pearson")
parser.add_argument("--corr", type=float, default=mv.CORR_THRESHOLD)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=mv.EPOCHS)
parser.add_argument("--channels", type=int, default=64,
                    help="số kênh ẩn của TCN. Bai et al. mục A.1 chọn sao cho "
                         "model to xấp xỉ model đem so; đồ án cố ý thu nhỏ")
parser.add_argument("--kernel_size", type=int, default=3,
                    help="bề rộng bộ lọc. Bai mục 3.3: tầm nhìn một tầng = (k-1)*d")
parser.add_argument("--n_blocks", type=int, default=6,
                    help="số khối. Phải đủ để tầm nhìn phủ 200 mẫu vào "
                         "(Bai mục 5 và A.1). k=3, n=6, khối hai tầng -> 253")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="spatial dropout (Bai mục 3.4). Mặc định 0.0 cho khớp "
                         "LSTM của MobiVital, để TN1 chỉ đổi đúng một biến")
parser.add_argument("--hidden", type=int, default=mv.LSTM_HIDDEN_SIZE,
                    help="số chiều ẩn của LSTM. Mặc định 352 là cấu hình "
                         "MobiVital công bố; 67 cho ~56k tham số, ngang DS-TCN-64")
parser.add_argument("--conv_channels", type=int, default=32,
                    help="số kênh của khối tích chập trong cnn_lstm")
parser.add_argument("--conv_kernel", type=int, default=5,
                    help="bề rộng bộ lọc tích chập trong cnn_lstm. "
                         "kernel 5 phủ 0,1 giây ở tần số lấy mẫu 50 Hz")
parser.add_argument("--kernel_large", type=int, default=31,
                    help="ModernTCN: kernel nhánh rộng")
parser.add_argument("--kernel_small", type=int, default=5,
                    help="ModernTCN: kernel nhánh hẹp")
parser.add_argument("--period_len", type=int, default=10,
                    help="MixLinear: cỡ chia đoạn bên trong mạng")
parser.add_argument("--lpf", type=int, default=5,
                    help="MixLinear: số hệ số FFT tần thấp được giữ")
parser.add_argument("--mix_alpha", type=float, default=0.5,
                    help="MixLinear: trọng số nhánh thời gian khi trộn hai nhánh. "
                         "KHÁC --alpha của loss mse_pearson")
parser.add_argument("--mix_hidden", type=int, default=2,
                    help="MixLinear: chiều trung gian nhánh tần số. "
                         "Quá 3 là vô ích, xem docstring lớp MixLinear")
parser.add_argument("--correction_hidden", type=int, default=4,
                    help="chiều giữa của nhánh phụ trong mix_linear_linear, "
                         "mix_linear_mlp và low_rank_linear. KHÁC --mix_hidden")
parser.add_argument("--norm", default="batch", choices=["batch", "weight"],
                    help="chuẩn hoá trong khối TCN. batch là mặc định của đồ án; "
                         "weight là bản đúng chuẩn Bai et al. mục 3.4")
parser.add_argument("--experiment", required=True,
                    help="tên thực nghiệm, ví dụ tn7 — quyết định thư mục runs/<tên>/")
args = parser.parse_args()

# Mỗi thực nghiệm một thư mục riêng.
EXP_DIR = "runs/" + args.experiment

revin = args.revin.lower() == "true"

# Tên cấu hình phải chứa channels: TCN-64 và TCN-200 cùng model, cùng loss,
# cùng seed — không đưa channels vào thì hai cấu hình ra CÙNG một tên, ghi đè
# kết quả của nhau. Model lstm không có channels nên bỏ qua.
# Hậu tố CHỈ thêm khi giá trị khác mặc định, nhờ đó tên của mọi lần chạy cũ
# không đổi khi thêm tuỳ chọn mới. Ví dụ lstm_mse_corr0.9_seed0 giữ nguyên dù
# về sau có thêm --hidden, --norm hay tuỳ chọn nào nữa.
if args.model in ("lstm", "bilstm"):
    # Họ hồi quy: chỉ có hidden, không có tham số nào của TCN.
    arch_tag = "" if args.hidden == mv.LSTM_HIDDEN_SIZE else "_h%d" % args.hidden
elif args.model == "cnn_lstm":
    # Không có "cấu hình gốc" nào để lấy làm mặc định, nên ghi đủ ba con số
    # quyết định kiến trúc. Đổi bất kỳ cái nào là ra tên khác, không đè kết quả.
    arch_tag = "_h%d_c%d_k%d" % (args.hidden, args.conv_channels, args.conv_kernel)
elif args.model == "modern_tcn":
    # Mặc định --channels 64 và --n_blocks 6 là của TCN, KHÔNG phải của
    # ModernTCN (32 và 3). Ghi cả hai vào tên để quên cờ thì lộ ra ngay.
    arch_tag = "_c%d_n%d" % (args.channels, args.n_blocks)
    if args.kernel_large != 31 or args.kernel_small != 5:
        arch_tag += "_kl%d_ks%d" % (args.kernel_large, args.kernel_small)
elif args.model == "gru":
    # Luôn ghi hidden. Khác lstm/bilstm ở chỗ GRU không có cấu hình gốc nào của
    # MobiVital để lấy làm mặc định, nên "khác mặc định mới ghi" là vô nghĩa.
    arch_tag = "_h%d" % args.hidden
elif args.model == "low_rank_linear":
    arch_tag = "_h%d" % args.correction_hidden
elif args.model in ("mix_linear_linear", "mix_linear_mlp"):
    # Ghi cả tham số của nền lẫn chiều nhánh phụ, vì cả hai đổi kiến trúc.
    arch_tag = "_p%d_lpf%d_h%d" % (args.period_len, args.lpf,
                                   args.correction_hidden)
elif args.model == "mix_linear":
    # Hai con số này quyết định cả kiến trúc lẫn số tham số, luôn ghi.
    arch_tag = "_p%d_lpf%d" % (args.period_len, args.lpf)
    if args.mix_alpha != 0.5:
        arch_tag += "_a%g" % args.mix_alpha
    if args.mix_hidden != 2:
        arch_tag += "_r%d" % args.mix_hidden
else:
    # Họ tích chập: channels luôn ghi, vì TCN-64 và TCN-200 phải khác tên nhau.
    arch_tag = "_c%d" % args.channels
    if args.kernel_size != 3 or args.n_blocks != 6:
        arch_tag += "_k%d_n%d" % (args.kernel_size, args.n_blocks)
    if args.norm != "batch":
        arch_tag += "_" + args.norm

if args.dropout != 0.0:
    arch_tag += "_do%g" % args.dropout

run_id = "%s%s%s_%s_corr%s_seed%d" % (
    args.model, arch_tag, "_revin" if revin else "",
    args.loss, args.corr, args.seed)

run_dir = EXP_DIR + "/" + run_id
os.makedirs(EXP_DIR, exist_ok=True)

print("thực nghiệm", args.experiment, " ->", EXP_DIR + "/")
print("run_id  ", run_id)
print("thiết bị", results.device_name())
print()

# Lần chạy đã có kết quả thì dừng ngay, đừng train lại 15 phút rồi mới đụng
# phải phép chặn trùng ở results.add_summary.
done = results.find_run(SUMMARY_FILE, args.experiment, run_id)
if done is not None:
    print("đã có kết quả macro %s — không chạy lại." % done["score_macro"])
    print("Muốn chạy lại thì đổi --experiment, hoặc xoá dòng cũ trong",
          SUMMARY_FILE)
    raise SystemExit(0)


# --- Train trên đủ 8 người ---

X, y = training.load_windows(["train"], args.corr, folder=WINDOWS_DIR)
print(X.shape[0], "cửa sổ train")

training.set_seed(args.seed)
model = models.build_model(args.model, revin=revin,
                                   hidden=args.hidden,
                                   channels=args.channels,
                                   kernel_size=args.kernel_size,
                                   n_blocks=args.n_blocks,
                                   dropout=args.dropout,
                                   norm=args.norm,
                                   conv_channels=args.conv_channels,
                                   conv_kernel=args.conv_kernel,
                                   kernel_large=args.kernel_large,
                                   kernel_small=args.kernel_small,
                                   period_len=args.period_len,
                                   lpf=args.lpf,
                                   mix_hidden=args.mix_hidden,
                                   correction_hidden=args.correction_hidden,
                                   mix_alpha=args.mix_alpha)
n_params = models.count_params(model)
print(n_params, "tham số")
print()

train_result = training.train(model,
                              training.make_loader(X, y),
                              None,
                              run_dir,
                              epochs=args.epochs,
                              loss_name=args.loss,
                              alpha=args.alpha)

results.save_curve(run_dir + "/curve.csv", train_result["curve"])


# --- Test một lần trên GHIJ ---

print()
print("chấm điểm G H I J...")
started = time.time()

model.eval()
rows = scoring.score_all(TEST_USERS, model)

minutes_score = (time.time() - started) / 60

scoring.write_txt(rows, EXP_DIR + "/" + run_id + ".txt")
results.save_sessions(EXP_DIR + "/scores_" + run_id + ".csv", rows)

by_user = scoring.mean_by_user(rows)
macro = float(np.mean([by_user[u] for u in TEST_USERS]))
micro = float(np.mean([row["pearson"] for row in rows]))
n_negative = sum(1 for row in rows if row["pearson"] < 0)

results.add_summary({"run_id": run_id,
                     "experiment": args.experiment,
                     "model": args.model,
                     "revin": int(revin),
                     "loss": args.loss,
                     "alpha": args.alpha,
                     "corr_threshold": args.corr,
                     "seed": args.seed,
                     "n_params": n_params,
                     "n_train_windows": X.shape[0],
                     "epochs": args.epochs,
                     "train_mse": train_result["train_mse"],
                     "train_pearson": train_result["train_pearson"],
                     "train_loss": train_result["train_loss"],
                     "minutes_train": train_result["minutes_train"],
                     "resumed": train_result["resumed"],
                     "score_macro": macro,
                     "score_micro": micro,
                     "n_sessions": len(rows),
                     "n_negative": n_negative,
                     "minutes_score": round(minutes_score, 2),
                     "test_ghij_macro": macro}, SUMMARY_FILE)


print()
print("=" * 58)
for user in TEST_USERS:
    print("   người %s   %.4f" % (user, by_user[user]))
print("   macro    %.10f   <- số công bố" % macro)
print("   micro    %.10f" % micro)
print("=" * 58)
print(EXP_DIR + "/" + run_id + ".txt")
print(EXP_DIR + "/scores_" + run_id + ".csv")
print(SUMMARY_FILE)


# --- Nén rồi chép sang Drive ---
#
# Colab xoá sạch /content khi ngắt phiên. Một lần chạy ở đây mất khoảng 15 phút
# (train trên đủ 8 người rồi chấm 537 buổi ghi), và notebook thường gọi ba lần
# liên tiếp cho ba seed — không lưu ngay là mất cả ba.
#
# Tên tệp nén chứa cấu hình nên không đè zip của lần chạy nào khác.
print()
subprocess.run([sys.executable, "scripts/save_results.py", args.experiment,
                "--out", args.experiment + "_" + run_id],
               check=False)
