"""TN0 — chạy pipeline ĐỒ ÁN và đối chiếu với pipeline MobiVital.

CÓ HAI TỆP TÊN run_tn0.py, ĐỪNG NHẦM:

    scripts/run_tn0.py             <- TỆP NÀY. Chạy code CỦA ĐỒ ÁN trong src/:
                                      scoring.py, training.py, results.py.
                                      Còn in bảng ĐẠT/KHÔNG ĐẠT với --compare
    scripts/mobivital/run_tn0.py      Chạy code CỦA TÁC GIẢ

Phân biệt bằng thư mục: nằm trong mobivital/ nghĩa là chạy code của tác giả.

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
    MobiVital, sinh runs/tn0/TN0a.txt, TN0b.txt, TN0c.txt và scores_TN0*.csv
"""

import argparse
import csv
import os
import subprocess
import sys

# Phải đặt TRƯỚC khi import torch, vì torch đọc biến này lúc khởi tạo CUDA.
# Bước chọn kênh đưa cả một buổi ghi vào LSTM một lần (khoảng 6700 chuỗi), mà lô
# to nhỏ khác nhau giữa các buổi nên bộ cấp phát mặc định giữ lại nhiều vùng đã
# dành mà bỏ không. Đo trên pipeline MobiVital: chết ở buổi 27/1874 vì 9.36 GB bị
# giữ mà không dùng; bật cờ này thì chạy trọn.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

sys.path.insert(0, os.path.abspath("scripts/mobivital"))
import check_repo

from src import mobivital_reference as mv
from src import results, scoring, training


# ===================== CÀI ĐẶT — sửa ở đây =====================

TEST_USERS = ["G", "H", "I", "J"]

RESULTS_DIR = "runs/tn0"
WINDOWS_DIR = "data/processed/windows/final_train"
RUN_DIR = "runs/tn0/ours_c"

RELEASED_PTH = "external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth"
MOBIVITAL_DIR = "external/mobivital"

SEED = 1234        # cùng số MobiVital ghi cứng ở inference/mobivital_gen.py dòng 21

# Ngưỡng coi là khớp. Hai bên cùng dữ liệu, cùng thuật toán, không có gì ngẫu
# nhiên, nên chênh lệch chỉ đến từ thứ tự cộng số dấu phẩy động.
TOLERANCE = 1e-9

# Điểm MobiVital công bố ở Bảng 4 bài báo (arXiv:2503.11064).
# TN0a phải tái hiện được con số này, nếu không thì dữ liệu tải về đã sai —
# hai pipeline cùng ra 0.700 vẫn "khớp nhau" nhưng vô nghĩa.
PAPER_SCORE = 0.819
PAPER_TOLERANCE = 0.001

# Số buổi ghi của G H I J. Hai bên cùng chấm 500/500 vẫn phải bị coi là trượt —
# thiếu 37 buổi nghĩa là dữ liệu hoặc bảng lựa chọn kênh bị cụt.
N_TEST_SESSIONS = 537

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


def max_session_gap(mobivital_csv, project_csv):
    """Chênh lệch điểm lớn nhất trên TỪNG buổi ghi giữa hai pipeline.

    So từng buổi chứ không so điểm trung bình: hai bảng điểm khác hẳn nhau vẫn
    có thể cho cùng một trung bình. Trả về (chênh lệch lớn nhất, số buổi ghi).
    """
    a = scores_mobivital(mobivital_csv)
    b = scores_project(project_csv)

    if set(a) != set(b):
        only_mobivital = sorted(set(a) - set(b))[:3]
        only_project = sorted(set(b) - set(a))[:3]
        sys.exit("hai bên chấm khác tập buổi ghi: %d và %d\n"
                 "  chỉ MobiVital có: %s\n  chỉ đồ án có: %s"
                 % (len(a), len(b), only_mobivital, only_project))

    if len(a) != N_TEST_SESSIONS:
        sys.exit("chấm %d buổi ghi, phải đúng %d (G H I J)\n  %s"
                 % (len(a), N_TEST_SESSIONS, mobivital_csv))

    return max(abs(a[f] - b[f]) for f in a), len(a)


def verdict(ok):
    return "ĐẠT" if ok else "KHÔNG ĐẠT"


def compare():
    """In bảng đối chiếu. Trả về True nếu tất cả kiểm tra bắt buộc đều đạt."""
    required = ["scores_TN0a.csv", "scores_TN0b.csv", "scores_TN0c.csv",
                "scores_ours_a.csv", "scores_ours_b.csv", "scores_ours_c.csv",
                "TN0b.txt", "ours_b.txt"]
    missing = [f for f in required if not os.path.exists(RESULTS_DIR + "/" + f)]
    if missing:
        sys.exit("Chưa đủ kết quả để đối chiếu, thiếu trong %s/:\n  %s\n"
                 "Chạy hết mục 3 và mục 4 của notebooks/TN0.ipynb trước."
                 % (RESULTS_DIR, "\n  ".join(missing)))

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

    # TN0a và TN0b: so TỪNG buổi ghi, không chỉ so điểm trung bình.
    # Trung bình bằng nhau vẫn có thể che hai bảng điểm khác hẳn nhau.
    gap_a, n_a = max_session_gap("scores_TN0a.csv", "scores_ours_a.csv")
    gap_b, n_b = max_session_gap("scores_TN0b.csv", "scores_ours_b.csv")

    # TN0a còn phải tái hiện đúng con số bài báo công bố.
    paper_gap = abs(a_mob - PAPER_SCORE)
    ok_paper = paper_gap < PAPER_TOLERANCE

    ok_git, git_description = check_repo.check_patched_only(MOBIVITAL_DIR)

    ok_a = (gap_a < TOLERANCE) and ok_paper
    ok_b = (total == N_TEST_SESSIONS) and (same == total) and (gap_b < TOLERANCE)

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
    print("TN0a  so với bài báo %.3f     : lệch %.5f  %s"
          % (PAPER_SCORE, paper_gap, verdict(ok_paper)))
    print("TN0a  chênh lệch lớn nhất trên %d buổi ghi : %.2e" % (n_a, gap_a))
    print("TN0b  chênh lệch lớn nhất trên %d buổi ghi : %.2e" % (n_b, gap_b))
    print("repo MobiVital                : %s — %s" % (verdict(ok_git), git_description))
    print()

    # Ghi bảng ra tệp, không chỉ in màn hình — để save_results.py gói theo và
    # để dựng bảng trong luận văn mà không phải chép tay.
    table = [
        {"kiem_tra": "TN0a  điểm từ TXT tác giả", "mobivital": a_mob,
         "pipeline_do_an": a_prj, "ket_luan": verdict(ok_a),
         "detail": "lệch bài báo %.5f, chênh lệch từng buổi %.2e" % (paper_gap, gap_a)},
        {"kiem_tra": "TN0b  cùng tệp trọng số", "mobivital": b_mob,
         "pipeline_do_an": b_prj, "ket_luan": verdict(ok_b),
         "detail": "%d/%d kênh trùng, chênh lệch từng buổi %.2e" % (same, total, gap_b)},
        {"kiem_tra": "TN0c  train lại LSTM", "mobivital": c_mob,
         "pipeline_do_an": c_prj, "ket_luan": "thông tin tham khảo",
         "detail": "hai lần train độc lập, không yêu cầu giống tuyệt đối"},
        {"kiem_tra": "repo MobiVital", "mobivital": "",
         "pipeline_do_an": "", "ket_luan": verdict(ok_git),
         "detail": git_description.replace("\n", " ; ")},
    ]
    results.write_rows(RESULTS_DIR + "/compare.csv",
                       ["kiem_tra", "mobivital", "pipeline_do_an", "ket_luan", "detail"],
                       table)
    print("bảng trên đã ghi ra", RESULTS_DIR + "/compare.csv")
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
