"""So các cấu hình đã chạy trong một thực nghiệm CV.

    python scripts/compare_cv.py --experiment tn1
    python scripts/compare_cv.py --experiment tn1 --baseline lstm

HAI BẢNG

    1. cv_score  — điểm trung bình 4 fold của từng cấu hình, kèm chi tiết
                   từng fold và độ lệch chuẩn
    2. thắng/hoà/thua — so TỪNG buổi ghi giữa hai cấu hình

VÌ SAO ĐẾM THẮNG / HOÀ / THUA

Hai cấu hình chênh nhau 0.003 điểm trung bình có thể là: một cấu hình tốt hơn
đều trên mọi buổi ghi, hoặc thắng đậm vài buổi mà thua nhẹ phần lớn. Trung bình
không phân biệt được. Đếm số buổi thắng thì thấy ngay.

Hoà = chênh lệch dưới 1e-6, tức nhỏ hơn nhiễu số dấu phẩy động.

ĐỌC TỪ ĐÂU

    runs/summary.csv                  dòng fold=TONG là cv_score của cả cấu hình
    runs/<thực nghiệm>/scores_*.csv   điểm từng buổi ghi, mỗi fold một tệp;
                                      bốn fold gộp lại phủ đủ 1289 buổi của
                                      tám người A B C D E F K L

CẦN CHẠY TRƯỚC

    scripts/run_cv.py --experiment <tên> ...   ít nhất hai cấu hình
"""

import argparse
import csv
import os
import re
import sys
from glob import glob

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

RUNS_DIR = "runs"
SUMMARY_FILE = "runs/summary.csv"

# Chênh lệch dưới ngưỡng này coi là hoà — nhỏ hơn nhiễu số dấu phẩy động.
TIE_MARGIN = 1e-6

# ===============================================================


def read_summary(experiment):
    """Các dòng của thực nghiệm này trong bảng metric chung."""
    if not os.path.exists(SUMMARY_FILE):
        sys.exit("không thấy " + SUMMARY_FILE + " — chưa chạy run_cv.py lần nào?")

    rows = [r for r in csv.DictReader(open(SUMMARY_FILE))
            if r["experiment"] == experiment]
    if not rows:
        sys.exit("không có dòng nào của thực nghiệm '%s' trong %s"
                 % (experiment, SUMMARY_FILE))
    return rows


def session_scores(experiment, config_id):
    """Gộp bốn tệp fold thành một bảng {tên buổi ghi: điểm}.

    Mỗi người nằm ở đúng một fold validation, nên bốn tệp không đè nhau.
    """
    scores = {}
    for path in sorted(glob("%s/%s/scores_%s_*.csv"
                            % (RUNS_DIR, experiment, config_id))):
        for r in csv.DictReader(open(path)):
            scores[r["session_file"]] = float(r["pearson"])
    return scores


def win_tie_loss(a, b):
    """Đếm số buổi ghi cấu hình a hơn / bằng / kém cấu hình b."""
    shared = sorted(set(a) & set(b))
    wins = sum(1 for f in shared if a[f] - b[f] > TIE_MARGIN)
    losses = sum(1 for f in shared if b[f] - a[f] > TIE_MARGIN)
    return wins, len(shared) - wins - losses, losses, len(shared)


def print_final_table(rows, experiment):
    """Chế độ test GHIJ: gộp các seed của cùng một cấu hình thành mean ± std."""
    # run_id của run_final_test.py kết thúc bằng _seed<N>. Bỏ phần đó đi thì
    # các lần chạy khác seed của cùng một cấu hình gom về một nhóm.
    groups = {}
    for r in rows:
        name = re.sub(r"_seed\d+$", "", r["run_id"])
        groups.setdefault(name, []).append(r)

    def group_mean(name):
        return np.mean([float(x["score_macro"]) for x in groups[name]])

    print()
    print("TEST GHIJ — thực nghiệm %s" % experiment)
    print("Train đủ 8 người A B C D E F K L, test 537 buổi ghi của G H I J.")
    print()
    print("%-34s %10s %6s %11s %10s   %s"
          % ("cấu hình", "tham số", "seed", "mean", "std", "từng seed"))
    print("-" * 100)

    for name in sorted(groups, key=lambda k: -group_mean(k)):
        seed_rows = sorted(groups[name], key=lambda x: int(x["seed"]))
        scores = [float(x["score_macro"]) for x in seed_rows]
        detail = "  ".join("s%s %.4f" % (x["seed"], s)
                           for x, s in zip(seed_rows, scores))
        print("%-34s %10s %6d %11.6f %10.6f   %s"
              % (name, seed_rows[0]["n_params"], len(scores),
                 float(np.mean(scores)), float(np.std(scores)), detail))

    print()
    print("std là độ lệch chuẩn giữa các seed — cho biết chênh lệch giữa hai cấu")
    print("hình có lớn hơn nhiễu ngẫu nhiên hay không.")
    print()


def print_cv_table(rows, totals, experiment):
    """Bảng 1: cv_score của từng cấu hình, kèm điểm từng fold."""
    print()
    print("BẢNG 1 — cv_score, thực nghiệm %s" % experiment)
    print("%-32s %10s %10s %9s   %s"
          % ("cấu hình", "tham số", "cv_score", "cv_std", "từng fold"))
    print("-" * 100)

    for r in totals:
        config_id = r["run_id"][:-len("_tong")]
        fold_rows = sorted((x for x in rows if x["fold"] not in ("", "TONG")
                            and x["run_id"].startswith(config_id + "_")),
                           key=lambda x: x["fold"])
        detail = "  ".join("%s %.4f" % (x["fold"].replace("val_", ""),
                                        float(x["score_macro"]))
                           for x in fold_rows)
        print("%-32s %10s %10.6f %9.6f   %s"
              % (config_id, r["n_params"], float(r["score_macro"]),
                 float(r["score_std"] or 0), detail))

    print()
    print("cv_score = trung bình điểm macro của 4 fold. Điểm macro = trung bình theo")
    print("NGƯỜI, không theo buổi ghi — mỗi người có số buổi khác nhau.")


def print_win_tie_loss_table(totals, experiment, baseline_model):
    """Bảng 2: đếm thắng/hoà/thua trên từng buổi ghi so với một cấu hình mốc."""
    baseline_rows = [r for r in totals if r["model"] == baseline_model]
    if not baseline_rows:
        sys.exit("không thấy cấu hình nào dùng model '%s'" % baseline_model)

    baseline_id = baseline_rows[0]["run_id"][:-len("_tong")]
    baseline_scores = session_scores(experiment, baseline_id)
    if not baseline_scores:
        sys.exit("không đọc được điểm từng buổi ghi của " + baseline_id)

    print()
    print("BẢNG 2 — thắng / hoà / thua trên TỪNG buổi ghi, mốc là %s" % baseline_id)
    print("%-32s %8s %7s %7s %9s   %s"
          % ("cấu hình", "thắng", "hoà", "thua", "tổng", "chênh lệch điểm"))
    print("-" * 100)

    for r in totals:
        config_id = r["run_id"][:-len("_tong")]
        if config_id == baseline_id:
            continue
        scores = session_scores(experiment, config_id)
        wins, ties, losses, total = win_tie_loss(scores, baseline_scores)
        gap = float(r["score_macro"]) - float(baseline_rows[0]["score_macro"])
        print("%-32s %8d %7d %7d %9d   %+.6f"
              % (config_id, wins, ties, losses, total, gap))

    print()
    print("Hoà = chênh lệch dưới %g. Tổng phải là 1289 buổi ghi của 8 người dev."
          % TIE_MARGIN)


# ---------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, help="tên thực nghiệm, ví dụ tn1")
parser.add_argument("--baseline", default=None,
                    help="tên model làm mốc để đếm thắng/hoà/thua, ví dụ lstm")
parser.add_argument("--final", action="store_true",
                    help="chế độ test GHIJ: gộp nhiều seed thành mean +- std "
                         "thay vì in bảng cv_score")
args = parser.parse_args()

rows = read_summary(args.experiment)

if args.final:
    print_final_table(rows, args.experiment)
    raise SystemExit(0)

totals = [r for r in rows if r["fold"] == "TONG"]
if not totals:
    sys.exit("chưa có dòng fold=TONG nào. Chạy lại run_cv.py bằng bản mới nhất.")

totals.sort(key=lambda r: -float(r["score_macro"]))

print_cv_table(rows, totals, args.experiment)

if args.baseline:
    print_win_tie_loss_table(totals, args.experiment, args.baseline)

print()
