"""Đo tốc độ train của từng kiến trúc trước khi chạy thí nghiệm thật.

    python scripts/do_toc_do.py

VÌ SAO ĐO TRƯỚC

Một thí nghiệm CV mất vài giờ. Đo một batch của từng kiến trúc mất chưa tới một
phút, đổi lại biết trước tổng thời gian, tránh cam kết nhiều giờ rồi mới phát
hiện cần nhiều hơn hẳn.

Cũng là câu trả lời cho "TCN ít tham số hơn thì có nhanh hơn không". Không hiển
nhiên: tích chập tách depthwise ít phép nhân hơn nhưng nghẽn băng thông bộ nhớ.
Đo ở máy CPU cho ds_tcn chậm gấp 13 lần tcn; trên GPU chỉ chậm 1.2 lần — thư
viện tối ưu cho hai loại phần cứng rất khác nhau, nên phải đo trên đúng máy sẽ
chạy thật.

CÁCH ĐO

Dữ liệu ngẫu nhiên đúng kích thước thật — tốc độ chỉ phụ thuộc hình dạng tensor,
không phụ thuộc giá trị. Mỗi bước gồm forward, backward và optimizer, đúng như
lúc train. Chạy vài lượt khởi động trước để cuDNN chọn xong thuật toán.

Số cửa sổ mỗi fold lấy thẳng từ data/processed/windows/dev_cv nếu có; không có
thì dùng số đã đo sẵn ghi trong CÀI ĐẶT.
"""

import os
import sys
import time
from glob import glob

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src import models
from src import mobivital_reference as mv


# ===================== CÀI ĐẶT — sửa ở đây =====================

WINDOWS_DIR = "data/processed/windows/dev_cv"

FOLDS = [("val_AB", "AB"), ("val_CE", "CE"), ("val_DF", "DF"), ("val_KL", "KL")]
DEV_USERS = "ABCDEFKL"

# Dùng khi chưa có thư mục cửa sổ. Đo được ngày 2026-09-04.
FALLBACK_WINDOW_COUNTS = [210964, 218920, 230152, 218088]

# Các kiến trúc đem so. None nghĩa là dùng mặc định của model.
CONFIGS = [("lstm", None), ("tcn", 64), ("ds_tcn", 64),
            ("tcn", 200), ("ds_tcn", 352)]

N_TIMED_STEPS = 12          # số bước đo, sau khi đã khởi động
N_WARMUP_STEPS = 3

# Ước thêm thời gian chấm điểm mỗi cấu hình, đo ở TN0: khoảng 1.1 giây một buổi
# ghi, bốn fold phủ đủ 1289 buổi của tám người dev.
SECONDS_PER_SESSION = 1.1
N_DEV_SESSIONS = 1289

# ===============================================================


def windows_per_fold():
    """Số cửa sổ train của từng fold. Đọc từ đĩa nếu có."""
    files = glob(WINDOWS_DIR + "/*.npz")
    if not files:
        print("chưa có", WINDOWS_DIR, "— dùng số đã đo sẵn\n")
        return FALLBACK_WINDOW_COUNTS

    n = {}
    for f in files:
        n[os.path.basename(f)[0]] = np.load(f)["X"].shape[0]

    return [sum(n[u] for u in DEV_USERS if u not in val) for _, val in FOLDS]


def seconds_per_batch(model, device):
    """Thời gian một bước train thật: forward, backward, optimizer."""
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=mv.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(mv.BATCH_SIZE, mv.HISTORY_LENGTH, device=device)
    y = torch.randn(mv.BATCH_SIZE, mv.FUTURE_LENGTH, device=device)

    def one_step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(N_WARMUP_STEPS):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    started_at = time.time()
    for _ in range(N_TIMED_STEPS):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    return (time.time() - started_at) / N_TIMED_STEPS


device = "cuda" if torch.cuda.is_available() else "cpu"
print("thiết bị:", device,
      torch.cuda.get_device_name(0) if device == "cuda" else "")
print()

windows = windows_per_fold()
batches_per_epoch = sum(n // mv.BATCH_SIZE for n in windows)

print("Khối lượng một lần chạy CV:")
for (name, val), n in zip(FOLDS, windows):
    print("   %-8s train %s = %7d cửa sổ" % (name, "".join(u for u in DEV_USERS if u not in val), n))
print("   %-8s %28d cửa sổ, %d epoch" % ("tổng", sum(windows), mv.EPOCHS))
print()

scoring_hours = N_DEV_SESSIONS * SECONDS_PER_SESSION / 3600

print("%-9s %-6s %11s %10s %12s %11s %10s"
      % ("model", "kênh", "tham số", "s/batch", "giờ train", "giờ chấm", "TỔNG"))
print("-" * 78)

total_hours = 0.0
for name, channels in CONFIGS:
    if channels is None:
        model = models.build_model(name)
        width = mv.LSTM_HIDDEN_SIZE
    else:
        model = models.build_model(name, channels=channels)
        width = channels

    seconds = seconds_per_batch(model, device)
    train_hours = seconds * batches_per_epoch * mv.EPOCHS / 3600
    total_hours += train_hours + scoring_hours

    print("%-9s %-6d %11d %9.4fs %11.1f %11.1f %9.1f"
          % (name, width, models.count_params(model), seconds,
             train_hours, scoring_hours, train_hours + scoring_hours))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

print()
print("Giờ chấm là ước từ TN0: %.1f giây một buổi ghi x %d buổi. Phần này khoảng"
      % (SECONDS_PER_SESSION, N_DEV_SESSIONS))
print("65% là numpy chạy CPU nên gần như không đổi theo model — xem docs/TOC_DO.md.")
print()
print("Chạy cả %d cấu hình: %.1f giờ. TN1 chỉ chạy 4 nên ít hơn."
      % (len(CONFIGS), total_hours))
