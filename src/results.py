"""Ghi kết quả ra file CSV. Chỉ ghi, không tính toán gì.

    from src import results
    results.add_summary({"run_id": "tn1_fold1_seed0", "score_macro": 0.84}, path)

BA MỨC CHI TIẾT

    runs/summary.csv            MỘT dòng mỗi lần chạy, cho cả đồ án
                                -> bảng trong luận văn lấy thẳng từ đây

    runs/<tn>/sessions.csv      MỘT dòng mỗi buổi ghi
                                -> so được hai lần chạy: thắng / hoà / thua

    runs/<tn>/curves/<run>.csv  MỘT dòng mỗi epoch
                                -> vẽ đường hội tụ

CỘT NÀO LÀ LÚC TRAIN, CỘT NÀO LÚC CHẤM

    train_*   đo trên CỬA SỔ cắt sẵn, bằng MSE
    score_*   đo trên BUỔI GHI thô, bằng Pearson, qua bộ chọn kênh

Hai thước đo khác nhau, không quy đổi cho nhau. `train_mse` thấp không đảm bảo
`score_macro` cao — đó chính là lý do có TN3.

VỀ CỘT minutes_*

Chỉ để tính giờ Colab, KHÔNG dùng làm bằng chứng tốc độ trong luận văn: phần
cứng Colab đổi giữa các phiên (T4 hôm nay, L4 hôm sau), lại dùng chung nên bị
bóp tuỳ lúc. Muốn so tốc độ thì đo riêng, hai model trong cùng một phiên. Còn
câu "TCN hơn vì kiến trúc hay vì to hơn" thì cột n_params trả lời được, và nó
không phụ thuộc phần cứng.
"""

import csv
import os
import subprocess


SUMMARY_COLUMNS = [
    # nhận dạng
    "run_id", "timestamp", "git_commit", "device",
    # cấu hình đang thử
    "experiment", "model", "revin", "loss", "alpha",
    "corr_threshold", "seed", "fold", "val_users",
    # lúc train
    "n_params", "n_train_windows", "epochs",
    "train_mse", "train_pearson", "train_loss", "minutes_train", "resumed",
    # lúc chấm
    "score_macro", "score_micro", "score_std", "n_sessions", "n_negative",
    "minutes_score",
    # chỉ để nhìn, không được dùng để chọn cấu hình
    "test_ghij_macro",
]

SESSION_COLUMNS = ["run_id", "user", "session_file", "bin", "method",
                   "n_candidates_kept", "pearson"]

CURVE_COLUMNS = ["epoch", "train_mse", "train_pearson", "train_loss",
                 "val_mse", "val_pearson", "minutes"]


def git_commit():
    """Mã commit đang chạy. Để sau này biết dòng số đó ra từ bản code nào."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def device_name():
    """Tên GPU đang dùng, hoặc 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def append_row(path, columns, fields):
    """Thêm một dòng vào file CSV. Tự tạo header nếu file chưa có.

    Nếu file đã có mà header khác `columns` thì DỪNG. Ghi tiếp vào file có
    header khác sẽ làm lệch cột: dòng mới ít hơn một giá trị thì mọi giá trị
    phía sau trượt sang ô bên cạnh, và đọc lại bằng DictReader không báo lỗi
    gì — điểm số rơi vào cột tên model, cột điểm thành rỗng.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_header = not os.path.exists(path)

    if not need_header:
        existing_header = next(csv.reader(open(path)), None)
        if existing_header is not None and existing_header != columns:
            raise ValueError(
                "header của %s khác với cột đang ghi — ghi tiếp sẽ lệch cột.\n"
                "  file có : %s\n  đang ghi: %s"
                % (path, existing_header, columns))

    opened_file = open(path, "a", newline="")
    writer = csv.writer(opened_file)
    if need_header:
        writer.writerow(columns)
    writer.writerow([fields.get(name, "") for name in columns])
    opened_file.close()


def find_run(path, experiment, run_id):
    """Dòng đã có của một lần chạy, hoặc None. Khoá là (experiment, run_id).

    Cần cả hai vì `run_id` chỉ duy nhất TRONG một thực nghiệm — tn1 và tn2 đều
    có thể có `lstm_mse_corr0.9_seed0_val_AB`.
    """
    if not os.path.exists(path):
        return None

    for row in csv.DictReader(open(path)):
        if row.get("experiment") == experiment and row.get("run_id") == run_id:
            return row
    return None


def write_rows(path, columns, rows):
    """Ghi đè cả file CSV bằng danh sách dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    opened_file = open(path, "w", newline="")
    writer = csv.writer(opened_file)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(name, "") for name in columns])
    opened_file.close()


def add_summary(fields, path):
    """Thêm một dòng vào summary.csv. Tự điền git_commit và device.

    DỪNG nếu đã có dòng cùng (experiment, run_id). Trước đây chạy lại một lần
    chạy đã xong thì nối thêm dòng thứ hai, và `compare_cv.py --final` đếm nó
    thành một seed nữa — ba seed báo thành bốn, độ lệch chuẩn bị bóp nhỏ vì có
    giá trị lặp.

    Cột `timestamp` giữ lại để không lệch với các file đã ghi từ trước, nhưng
    không điền nữa: `git_commit` chỉ ra bản code chính xác hơn.
    """
    fields = dict(fields)
    fields.setdefault("git_commit", git_commit())
    fields.setdefault("device", device_name())

    old = find_run(path, fields.get("experiment"), fields.get("run_id"))
    if old is not None:
        raise ValueError(
            "đã có kết quả cho (%s, %s) trong %s — không ghi đè.\n"
            "  điểm đã lưu: %s\n"
            "  Muốn chạy lại thì đổi --experiment, hoặc xoá dòng cũ trước."
            % (fields.get("experiment"), fields.get("run_id"), path,
               old.get("score_macro")))

    append_row(path, SUMMARY_COLUMNS, fields)


def save_sessions(path, rows):
    """Ghi điểm từng buổi ghi."""
    write_rows(path, SESSION_COLUMNS, rows)


def save_curve(path, rows):
    """Ghi loss từng epoch."""
    write_rows(path, CURVE_COLUMNS, rows)
