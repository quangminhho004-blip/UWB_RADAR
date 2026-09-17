"""So các cấu hình đã chạy trong một thực nghiệm.

    python scripts/compare_cv.py --experiment tn1
    python scripts/compare_cv.py --experiment tn4 --final

VIỆC DUY NHẤT SCRIPT NÀY LÀM

Biến `runs/<thực nghiệm>/summary.csv` — bảng 26 cột cho máy đọc — thành một
bảng điểm cho người đọc. Không tính lại gì từ dữ liệu thô; mọi con số đã nằm
sẵn trong summary.csv, script chỉ gom theo cấu hình, gộp các seed, rồi xếp hạng.

HAI CHẾ ĐỘ

    mặc định   bảng cv_score: trung bình 4 fold của từng cấu hình, kèm điểm
               từng seed và hai loại độ lệch chuẩn
    --final    bảng test cuối: không chia fold, gộp nhiều seed thành
               mean +- std mẫu

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
    """Các dòng của thực nghiệm này.

    Đọc runs/<thực nghiệm>/summary.csv trước, vì đó là bản đã commit vào repo.
    Chỉ khi không có mới quay sang runs/summary.csv — bảng chung do runner ghi
    trong phiên đang chạy, không có sẵn khi mới clone về.
    """
    rieng = "%s/%s/summary.csv" % (RUNS_DIR, experiment)
    nguon = rieng if os.path.exists(rieng) else SUMMARY_FILE

    if not os.path.exists(nguon):
        sys.exit("không thấy %s, cũng không thấy %s — chưa chạy lần nào?"
                 % (rieng, SUMMARY_FILE))

    rows = [r for r in csv.DictReader(open(nguon))
            if r["experiment"] == experiment]
    if not rows:
        sys.exit("không có dòng nào của thực nghiệm '%s' trong %s"
                 % (experiment, nguon))
    return rows


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
                 float(np.mean(scores)),
                 # ddof=1 cho khớp cột seed_std của bảng CV ở trên. Với 3 seed,
                 # ddof=0 cho số nhỏ hơn khoảng 18% — hai bảng sẽ đá nhau.
                 float(np.std(scores, ddof=1)), detail))

    print()
    print("std là độ lệch chuẩn mẫu của điểm giữa các seed, tính trên 3 giá trị.")
    print("Nó mô tả mức tản của chính cấu hình đó qua các lần khởi tạo — KHÔNG")
    print("phải kiểm định. Bảng này báo chênh lệch quan sát được; muốn nói hai")
    print("cấu hình khác nhau hay không thì cần thiết kế kiểm định riêng.")
    print()


def strip_seed(config_id):
    """Bỏ hậu tố _seed<N>. Nhiều seed của cùng một cấu hình gom về một tên."""
    return re.sub(r"_seed\d+$", "", config_id)


def print_cv_table(rows, totals, experiment):
    """Bảng 1: một dòng mỗi CẤU HÌNH, gộp các seed thành cv_mean ± std.

    HAI LOẠI ĐỘ LỆCH CHUẨN, ĐỪNG LẪN

        seed_std   đổi hạt giống thì cv_score dao động bao nhiêu
        fold_std   trong MỘT seed, bốn fold lệch nhau bao nhiêu

    fold_std lớn hơn seed_std nhiều lần vì đổi người test tác động mạnh hơn đổi
    hạt giống. Chỉ seed_std mới dùng để nói hai cấu hình có khác nhau thật hay
    không; fold_std nói dữ liệu giữa các người khác nhau ra sao.

    Trước đây mỗi seed in thành một dòng riêng, xếp hạng lẫn với nhau — đọc
    bảng dễ nhặt nhầm seed may nhất của một cấu hình rồi tưởng nó thắng.
    """
    groups = {}
    for r in totals:
        groups.setdefault(strip_seed(r["run_id"][:-len("_tong")]), []).append(r)

    print()
    print("BẢNG 1 — cv_score, thực nghiệm %s" % experiment)
    print("%-30s %9s %5s %10s %9s %9s   %s"
          % ("cấu hình", "tham số", "seed", "cv_mean", "seed_std", "fold_std",
             "từng seed"))
    print("-" * 110)

    def group_mean(name):
        return np.mean([float(r["score_macro"]) for r in groups[name]])

    for name in sorted(groups, key=lambda k: -group_mean(k)):
        seed_rows = sorted(groups[name], key=lambda r: int(r["seed"]))
        scores = [float(r["score_macro"]) for r in seed_rows]
        fold_stds = [float(r["score_std"] or 0) for r in seed_rows]

        # Độ lệch chuẩn MẪU (ddof=1) vì ba seed là mẫu rút từ vô số seed có thể
        # có. Một seed thì không có gì để so, in N/A thay vì số 0 gây hiểu nhầm
        # là "không dao động".
        seed_std = "%9.6f" % np.std(scores, ddof=1) if len(scores) > 1 else "      N/A"
        detail = "  ".join("s%s %.4f" % (r["seed"], s)
                           for r, s in zip(seed_rows, scores))

        print("%-30s %9s %5d %10.6f %s %9.6f   %s"
              % (name, seed_rows[0]["n_params"], len(scores),
                 float(np.mean(scores)), seed_std, float(np.mean(fold_stds)),
                 detail))

    print()
    print("cv_mean  = trung bình cv_score của các seed. cv_score của một seed là")
    print("           trung bình điểm macro 4 fold; macro = trung bình theo NGƯỜI.")
    print("seed_std = độ lệch chuẩn mẫu của cv_score giữa các seed, trên 3 giá trị.")
    print("           Mô tả mức tản của chính cấu hình đó, KHÔNG phải kiểm định.")
    print("           Chênh lệch nhỏ hơn số này thì chưa kết luận được gì.")
    print("fold_std = dao động giữa 4 fold, trung bình trên các seed. Nói dữ liệu")
    print("           giữa các người khác nhau ra sao, KHÔNG dùng để so cấu hình.")

    seed_counts = {name: len(g) for name, g in groups.items()}
    if len(set(seed_counts.values())) > 1:
        print()
        print("CHÚ Ý: các cấu hình KHÔNG cùng số seed — %s."
              % ", ".join("%s:%d" % (n, k) for n, k in sorted(seed_counts.items())))
        print("So cv_mean của 3 seed với cv_mean của 1 seed là so hai đại lượng khác")
        print("nhau. Chạy đủ cùng tập seed rồi hãy xếp hạng.")


# ---------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, help="tên thực nghiệm, ví dụ tn1")
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

print()
