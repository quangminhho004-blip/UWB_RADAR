"""TN0 — chạy pipeline đồ án và đối chiếu với pipeline MobiVital.

    python scripts/run_tn0.py --case a       tính điểm từ tệp lựa chọn kênh có sẵn
    python scripts/run_tn0.py --case b       chọn kênh bằng tệp trọng số của tác giả
    python scripts/run_tn0.py --case c       train lại LSTM bằng pipeline đồ án
    python scripts/run_tn0.py --compare      in bảng ĐẠT / KHÔNG ĐẠT

BA KIỂM TRA

    TN0a  Tệp lựa chọn kênh tác giả cung cấp có tái hiện điểm công bố không?
    TN0b  Cùng tệp trọng số, pipeline đồ án có chọn đúng 537/537 kênh giống
          pipeline MobiVital không?
    TN0c  Train lại LSTM từ đầu thì đạt mức nào? Không bắt buộc giống tuyệt đối
          tệp trọng số tác giả phát hành — hai vòng train khác nhau ở thứ tự
          xáo trộn dữ liệu.

Mỗi `--case` thêm đúng một bộ phận của pipeline đồ án, nên bộ phận nào sai thì
lộ ra ở đúng kiểm tra đó:

    a   chỉ hàm tính điểm            src/scoring.py: score_from_txt
    b   thêm bộ chọn kênh            src/scoring.py: score_all
    c   thêm vòng train              src/training.py: train

VÌ SAO CẦN PIPELINE RIÊNG

Code MobiVital chỉ chạy LSTM — `inference/mobivital_gen.py` dòng 152 ghi cứng
`LSTMMultiStep(...)`, không hỗ trợ TCN. Muốn thử TCN thì phải có bộ chọn kênh
riêng, rồi chứng minh nó cho ra đúng kết quả code gốc. Đó là việc của TN0.

CẦN CHẠY TRƯỚC

    mục 2 và mục 3 của notebooks/TN0.ipynb — dựng dữ liệu và chạy pipeline
    MobiVital, sinh results/TN0a.txt, TN0b.txt, TN0c.txt và scores_TN0*.csv
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src import mobivital_reference as mv
from src import results, scoring, training


# ===================== CÀI ĐẶT — sửa ở đây =====================

TEST_USERS = ["G", "H", "I", "J"]

RESULTS_DIR = "results"
WINDOWS_DIR = "data/processed/windows/final_train"
RUN_DIR = "runs/tn0/ours_c"

RELEASED_PTH = "external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth"
MOBIVITAL_DIR = "external/mobivital"

SEED = 1234        # cùng số MobiVital ghi cứng ở inference/mobivital_gen.py dòng 21

# Ngưỡng coi là khớp. Hai bên cùng dữ liệu, cùng thuật toán, không có gì ngẫu
# nhiên, nên chênh lệch chỉ đến từ thứ tự cộng số dấu phẩy động.
TOLERANCE = 1e-9

# ===============================================================


def load_lstm(path, device):
    """Nạp tệp trọng số .pth vào LSTM của MobiVital, đặt sẵn ở chế độ chấm."""
    model = mv.new_lstm()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def report(name, rows):
    """Ghi điểm từng buổi ghi ra CSV, in điểm trung bình, trả về điểm đó."""
    path = RESULTS_DIR + "/scores_" + name + ".csv"
    results.save_sessions(path, rows)

    score = float(np.mean([r["pearson"] for r in rows]))
    print("%-9s %d buổi ghi   điểm trung bình %.10f" % (name, len(rows), score))
    print("          ->", path)
    return score


# ---------------------------------------------------------------
# Ba kiểm tra
# ---------------------------------------------------------------

def case_a(device):
    """Chỉ hàm tính điểm: đọc tệp lựa chọn kênh có sẵn, không chạy model."""
    print("TN0a — tính điểm từ", RESULTS_DIR + "/TN0a.txt", "(không chạy model)")
    rows = scoring.score_from_txt(RESULTS_DIR + "/TN0a.txt", TEST_USERS)
    report("ours_a", rows)


def case_b(device):
    """Thêm bộ chọn kênh: cùng tệp trọng số tác giả phát hành."""
    print("TN0b — chọn kênh bằng", RELEASED_PTH)
    rows = scoring.score_all(TEST_USERS, load_lstm(RELEASED_PTH, device))
    scoring.write_txt(rows, RESULTS_DIR + "/ours_b.txt")
    report("ours_b", rows)
    print("          ->", RESULTS_DIR + "/ours_b.txt")


def case_c(device):
    """Thêm vòng train: train lại LSTM từ đầu bằng src/training.py."""
    print("TN0c — train lại LSTM, cấu hình MobiVital công bố:")
    print("       %d epoch, Adam lr %g, batch %d, MSE"
          % (mv.EPOCHS, mv.LEARNING_RATE, mv.BATCH_SIZE))

    X, y = training.load_windows(["train"], folder=WINDOWS_DIR)
    print("      ", X.shape[0], "cửa sổ train")

    training.set_seed(SEED)
    train_result = training.train(mv.new_lstm(), training.make_loader(X, y), None,
                                  RUN_DIR, loss_name="mse")
    results.save_curve(RUN_DIR + "/curve.csv", train_result["curve"])
    print("       train xong sau %.0f phút" % train_result["minutes_train"])

    rows = scoring.score_all(TEST_USERS, load_lstm(train_result["final_path"], device))
    scoring.write_txt(rows, RESULTS_DIR + "/ours_c.txt")
    report("ours_c", rows)
    print("          ->", RESULTS_DIR + "/ours_c.txt")


# ---------------------------------------------------------------
# Đối chiếu
# ---------------------------------------------------------------

def scores_mobivital(name):
    """evaluate.py ghi: cột 0 tên tệp CSV, cột 1 điểm."""
    rows = list(csv.reader(open(RESULTS_DIR + "/" + name)))[1:]
    return {r[0]: float(r[1]) for r in rows}


def scores_project(name):
    """src/results.py ghi theo SESSION_COLUMNS."""
    return {r["session_file"]: float(r["pearson"])
            for r in csv.DictReader(open(RESULTS_DIR + "/" + name))}


def picks(name):
    """Tệp lựa chọn kênh -> {tên tệp CSV: (kênh khoảng cách, phép biến đổi)}."""
    return {r[0]: (r[1], r[2]) for r in csv.reader(open(RESULTS_DIR + "/" + name))}


def verdict(ok):
    return "ĐẠT" if ok else "KHÔNG ĐẠT"


def compare():
    """In bảng đối chiếu. Trả về True nếu tất cả kiểm tra bắt buộc đều đạt."""
    a_mob = float(np.mean(list(scores_mobivital("scores_TN0a.csv").values())))
    b_mob = float(np.mean(list(scores_mobivital("scores_TN0b.csv").values())))
    c_mob = float(np.mean(list(scores_mobivital("scores_TN0c.csv").values())))

    a_prj = float(np.mean(list(scores_project("scores_ours_a.csv").values())))
    b_prj = float(np.mean(list(scores_project("scores_ours_b.csv").values())))
    c_prj = float(np.mean(list(scores_project("scores_ours_c.csv").values())))

    # TN0b: đếm số buổi ghi hai bên chọn cùng kênh và cùng phép biến đổi
    pick_mob, pick_prj = picks("TN0b.txt"), picks("ours_b.txt")
    same = sum(1 for f in pick_mob if pick_mob[f] == pick_prj.get(f))
    total = len(pick_mob)

    # TN0b: chênh lệch điểm lớn nhất trên từng buổi ghi
    s_mob = scores_mobivital("scores_TN0b.csv")
    s_prj = scores_project("scores_ours_b.csv")
    gap_b = max(abs(s_mob[f] - s_prj[f]) for f in s_mob)
    gap_a = abs(a_mob - a_prj)

    git_status = subprocess.run("git -C " + MOBIVITAL_DIR + " status --porcelain",
                                shell=True, capture_output=True, text=True).stdout.strip()

    ok_a = gap_a < TOLERANCE
    ok_b = (same == total) and (gap_b < TOLERANCE)
    ok_git = git_status == ""

    print()
    print("%-28s %-13s %-15s %s"
          % ("Kiểm tra", "MobiVital", "Pipeline đồ án", "Kết luận"))
    print("-" * 76)
    print("%-28s %-13.6f %-15.6f %s"
          % ("TN0a  điểm từ TXT tác giả", a_mob, a_prj, verdict(ok_a)))
    print("%-28s %-13.6f %-15.6f %s"
          % ("TN0b  cùng tệp trọng số", b_mob, b_prj,
             "%d/%d kênh — %s" % (same, total, verdict(ok_b))))
    print("%-28s %-13.6f %-15.6f %s"
          % ("TN0c  train lại LSTM", c_mob, c_prj, "thông tin tham khảo"))
    print("-" * 76)
    print("chênh lệch điểm TN0a          : %.2e" % gap_a)
    print("chênh lệch điểm TN0b lớn nhất : %.2e" % gap_b)
    print("repo MobiVital không bị sửa   : %s" % verdict(ok_git))
    if not ok_git:
        print(git_status)
    print()

    if ok_a and ok_b and ok_git:
        print("TN0 ĐẠT — pipeline đồ án cho ra đúng kết quả pipeline MobiVital.")
        return True

    print("TN0 KHÔNG ĐẠT — xem bảng trên.")
    return False


# ---------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["a", "b", "c"], help="chạy một kiểm tra")
parser.add_argument("--compare", action="store_true", help="in bảng đối chiếu")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="thiết bị cho bước chọn kênh")
args = parser.parse_args()

if not args.case and not args.compare:
    parser.error("cần --case a|b|c hoặc --compare")

os.makedirs(RESULTS_DIR, exist_ok=True)

if args.case:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("thiết bị:", device,
          torch.cuda.get_device_name(0) if device == "cuda" else "")
    print()

    {"a": case_a, "b": case_b, "c": case_c}[args.case](device)

if args.compare:
    if not compare():
        sys.exit(1)
