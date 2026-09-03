"""BƯỚC 9 — TN0.1: chạy bộ chọn kênh của mình, rồi so với code MobiVital.

    python scripts/9_tn0_ours.py

VÌ SAO CẦN BƯỚC NÀY

Code MobiVital chỉ chạy được LSTM: `mobivital_gen.py` dòng 152 ghi cứng
`LSTMMultiStep(...)`, không nạp được TCN. Nên phải viết bộ chọn kênh riêng —
`src/scoring.py` — nhận model bất kỳ.

Đổi lại phải chứng minh viết đúng. Cách chứng minh: nạp đúng checkpoint LSTM mà
MobiVital phát hành, chạy bộ chọn kênh của mình trên 537 buổi ghi GHIJ, rồi đối
chiếu với bảng do chính code họ sinh ra ở bước 8.

    checkpoint LSTM cua MobiVital
            |
            +--> mobivital_gen.py cua HO   -> results/TN0b.txt      buoc 8
            |
            +--> src/scoring.py   cua MINH -> results/TN0_1.txt     buoc nay
                            |
                  buoc 10 doi chieu

Cùng checkpoint, cùng dữ liệu, cùng thuật toán, không có gì ngẫu nhiên → phải ra
y hệt. Lệch một dòng là bộ chọn kênh viết sai.

CẦN CHẠY TRƯỚC

    scripts/2_make_npz.py       -> data/processed/by_user/*.npz
    scripts/8_tn0_mobivital.py  -> results/TN0b.txt
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src import mobivital_reference as mv
from src import results
from src import scoring


# ===================== CÀI ĐẶT — sửa ở đây =====================

TEST_USERS = ["G", "H", "I", "J"]

CHECKPOINT = "external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth"
OUT_DIR = "results"

# ===============================================================


device = "cuda" if torch.cuda.is_available() else "cpu"

print("thiết bị:", device)
if device == "cuda":
    print("        ", torch.cuda.get_device_name(0))
print()

model = mv.new_lstm()
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model = model.to(device)
model.eval()

print("nạp", CHECKPOINT)
print("số tham số:", format(sum(p.numel() for p in model.parameters()), ","))
print()

print("chạy bộ chọn kênh trên", len(TEST_USERS), "người...")
started = time.time()
rows = scoring.score_all(TEST_USERS, model)
minutes = (time.time() - started) / 60

os.makedirs(OUT_DIR, exist_ok=True)
scoring.write_txt(rows, OUT_DIR + "/TN0_1.txt")
results.save_sessions(OUT_DIR + "/scores_TN0_1.csv", rows)

by_user = scoring.mean_by_user(rows)
micro = float(np.mean([r["pearson"] for r in rows]))
macro = float(np.mean([by_user[u] for u in sorted(by_user)]))

print("xong sau %.0f phút, %d buổi ghi" % (minutes, len(rows)))
print()
for user in sorted(by_user):
    n = sum(1 for r in rows if r["user"] == user)
    print("   %s  %3d buổi ghi   %.4f" % (user, n, by_user[user]))

print()
print("micro (537 buổi ghi) : %.6f" % micro)
print("macro (4 người)      : %.6f   <- điểm chính thức, PROTOCOL mục 4" % macro)
print()
print("ứng viên sống sót sau bộ lọc lộn ngược: trung bình %.0f / 240"
      % np.mean([r["n_candidates_kept"] for r in rows]))
print()
print("ghi ra", OUT_DIR + "/TN0_1.txt", "và", OUT_DIR + "/scores_TN0_1.csv")

if len(rows) != 537:
    raise RuntimeError("phải ra 537 buổi ghi, ra " + str(len(rows)))
