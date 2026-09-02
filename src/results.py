"""Ghi kết quả ra file CSV. Chỉ ghi, không tính toán gì.

    from src import results
    results.add_summary({"run_id": "tn1_fold1_seed0", "score_macro": 0.84, ...})

BA MỨC CHI TIẾT

    runs/summary.csv            MỘT dòng mỗi lần chạy, cho cả đồ án
                                -> bảng trong luận văn lấy thẳng từ đây

    runs/<tn>/sessions.csv      MỘT dòng mỗi buổi ghi
                                -> so được hai lần chạy bất kỳ: thắng/hoà/thua

    runs/<tn>/curves/<run>.csv  MỘT dòng mỗi epoch
                                -> vẽ đường hội tụ

CỘT NÀO LÀ LÚC TRAIN, CỘT NÀO LÚC CHẤM

    train_*   đo trên CỬA SỔ cắt sẵn, bằng MSE
    score_*   đo trên BUỔI GHI thô, bằng Pearson, qua bộ chọn kênh

Hai thước đo khác nhau, không quy đổi cho nhau được. train_mse thấp không
đảm bảo score_macro cao — đó chính là lý do có TN3.

VỀ CỘT minutes_*

Chỉ để tính giờ Colab, KHÔNG dùng làm bằng chứng tốc độ trong luận văn:
phần cứng Colab đổi giữa các phiên (T4 hôm nay, L4 hôm sau), lại dùng chung
nên bị bóp tuỳ lúc. Muốn so tốc độ thì đo riêng, hai model trong cùng một
phiên. Còn câu "TCN hơn vì kiến trúc hay vì to hơn" thì cột n_params trả lời
được, và nó không phụ thuộc phần cứng.
"""

import csv
import os
import subprocess
import time


COT_SUMMARY = [
    # nhận dạng
    "run_id", "timestamp", "git_commit", "device",
    # cấu hình đang thử
    "experiment", "model", "revin", "loss", "alpha",
    "corr_threshold", "seed", "fold", "val_users",
    # lúc train
    "n_params", "n_train_windows", "epochs",
    "train_mse", "train_pearson", "train_loss", "minutes_train", "resumed",
    # lúc chấm
    "score_macro", "score_micro", "n_sessions", "n_negative", "minutes_score",
    # chỉ để nhìn, không được dùng để chọn cấu hình
    "test_ghij_macro",
]

COT_SESSIONS = ["run_id", "user", "session_file", "bin", "method",
                "n_candidates_kept", "pearson"]

COT_CURVE = ["epoch", "train_mse", "train_pearson", "train_loss",
             "val_mse", "val_pearson", "minutes"]


def git_commit():
    """Mã commit đang chạy. Để sau này biết dòng số đó ra từ bản code nào."""
    try:
        ket_qua = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True)
        return ket_qua.stdout.strip()
    except Exception:
        return ""


def ten_thiet_bi():
    """Tên GPU đang dùng, hoặc 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def ghi_mot_dong(path, cot, fields):
    """Thêm một dòng vào file CSV. Tự tạo header nếu file chưa có."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    can_header = not os.path.exists(path)

    opened_file = open(path, "a", newline="")
    writer = csv.writer(opened_file)
    if can_header:
        writer.writerow(cot)
    writer.writerow([fields.get(ten, "") for ten in cot])
    opened_file.close()


def add_summary(fields, path):
    """Thêm một dòng vào summary.csv. Tự điền timestamp, git_commit, device."""
    fields = dict(fields)
    fields.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M"))
    fields.setdefault("git_commit", git_commit())
    fields.setdefault("device", ten_thiet_bi())

    ghi_mot_dong(path, COT_SUMMARY, fields)


def save_sessions(path, rows):
    """Ghi điểm từng buổi ghi. rows là danh sách dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    opened_file = open(path, "w", newline="")
    writer = csv.writer(opened_file)
    writer.writerow(COT_SESSIONS)
    for row in rows:
        writer.writerow([row.get(ten, "") for ten in COT_SESSIONS])
    opened_file.close()


def save_curve(path, rows):
    """Ghi loss từng epoch. rows là danh sách dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    opened_file = open(path, "w", newline="")
    writer = csv.writer(opened_file)
    writer.writerow(COT_CURVE)
    for row in rows:
        writer.writerow([row.get(ten, "") for ten in COT_CURVE])
    opened_file.close()
