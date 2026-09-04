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
import sys
import time

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
parser.add_argument("--model", default="ds_tcn", help="lstm | tcn | ds_tcn")
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
parser.add_argument("--experiment", required=True,
                    help="tên thực nghiệm, ví dụ tn7 — quyết định thư mục runs/<tên>/")
args = parser.parse_args()

# Mỗi thực nghiệm một thư mục riêng.
EXP_DIR = "runs/" + args.experiment

revin = args.revin.lower() == "true"

run_id = "%s%s_%s_corr%s_seed%d" % (
    args.model, "_revin" if revin else "", args.loss, args.corr, args.seed)

run_dir = EXP_DIR + "/" + run_id
os.makedirs(EXP_DIR, exist_ok=True)

print("thực nghiệm", args.experiment, " ->", EXP_DIR + "/")
print("run_id  ", run_id)
print("thiết bị", results.device_name())
print()


# --- Train trên đủ 8 người ---

X, y = training.load_windows(["train"], args.corr, folder=WINDOWS_DIR)
print(X.shape[0], "cửa sổ train")

training.set_seed(args.seed)
model = models.build_model(args.model, revin=revin,
                                   channels=args.channels,
                                   kernel_size=args.kernel_size,
                                   n_blocks=args.n_blocks,
                                   dropout=args.dropout)
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
