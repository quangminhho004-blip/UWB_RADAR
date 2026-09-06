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
    print("seed_std = dao động giữa các seed. Chênh lệch giữa hai cấu hình nhỏ hơn")
    print("           số này thì chưa kết luận được.")
    print("fold_std = dao động giữa 4 fold, trung bình trên các seed. Nói dữ liệu")
    print("           giữa các người khác nhau ra sao, KHÔNG dùng để so cấu hình.")

    seed_counts = {name: len(g) for name, g in groups.items()}
    if len(set(seed_counts.values())) > 1:
        print()
        print("CHÚ Ý: các cấu hình KHÔNG cùng số seed — %s."
              % ", ".join("%s:%d" % (n, k) for n, k in sorted(seed_counts.items())))
        print("So cv_mean của 3 seed với cv_mean của 1 seed là so hai đại lượng khác")
        print("nhau. Chạy đủ cùng tập seed rồi hãy xếp hạng.")


def print_win_tie_loss_table(totals, experiment, baseline_model):
    """Bảng 2: đếm thắng/hoà/thua trên từng buổi ghi so với một cấu hình mốc.

    So CÙNG SEED với CÙNG SEED. Lấy seed 1 của cấu hình này so seed 0 của mốc
    là trộn hai nguồn chênh lệch — kiến trúc và hạt giống — vào một con số.

    Chỉ đếm những seed mà CẢ HAI bên đều có. Bên nào thiếu seed thì ghi ra, để
    không âm thầm so 3 seed với 1 seed.
    """
    # {seed: dòng TONG} cho mốc và cho từng cấu hình khác.
    by_config = {}
    for r in totals:
        config_id = r["run_id"][:-len("_tong")]
        by_config.setdefault(strip_seed(config_id), {})[int(r["seed"])] = r

    baseline_name = None
    for r in totals:
        if r["model"] == baseline_model:
            baseline_name = strip_seed(r["run_id"][:-len("_tong")])
            break
    if baseline_name is None:
        sys.exit("không thấy cấu hình nào dùng model '%s'" % baseline_model)

    baseline_seeds = by_config[baseline_name]

    print()
    print("BẢNG 2 — thắng / hoà / thua trên TỪNG buổi ghi, mốc là %s" % baseline_name)
    print("%-30s %5s %8s %7s %7s %9s   %s"
          % ("cấu hình", "seed", "thắng", "hoà", "thua", "tổng", "chênh cv_mean"))
    print("-" * 110)

    baseline_mean = np.mean([float(r["score_macro"]) for r in baseline_seeds.values()])

    for name in sorted(by_config):
        if name == baseline_name:
            continue

        shared_seeds = sorted(set(by_config[name]) & set(baseline_seeds))
        if not shared_seeds:
            print("%-30s   không có seed nào trùng với mốc" % name)
            continue

        wins = ties = losses = total = 0
        counted_seeds = []
        for seed in shared_seeds:
            a = session_scores(experiment, "%s_seed%d" % (name, seed))
            b = session_scores(experiment, "%s_seed%d" % (baseline_name, seed))
            if not a or not b:
                continue
            w, t, l, n = win_tie_loss(a, b)
            wins += w
            ties += t
            losses += l
            total += n
            counted_seeds.append(seed)

        # Không có tệp điểm thì báo hẳn ra. In dãy số 0 sẽ trông như "hoà tất
        # cả" trong khi thật ra là thiếu dữ liệu.
        if not counted_seeds:
            print("%-30s   thiếu tệp scores_*.csv, không đếm được" % name)
            continue

        gap = np.mean([float(by_config[name][s]["score_macro"])
                       for s in by_config[name]]) - baseline_mean
        print("%-30s %5d %8d %7d %7d %9d   %+.6f"
              % (name, len(counted_seeds), wins, ties, losses, total, gap))

        unmatched = sorted(set(by_config[name]) ^ set(baseline_seeds))
        if unmatched:
            print("%-30s   bỏ qua seed %s vì chỉ một bên có"
                  % ("", ", ".join(str(x) for x in unmatched)))

    print()
    print("Hoà = chênh lệch dưới %g." % TIE_MARGIN)
    print("Mỗi seed phủ 1289 buổi ghi của 8 người dev, nên cột tổng là")
    print("1289 x số seed — đây là số CẶP seed-buổi ghi, không phải số buổi ghi")
    print("độc lập. Đừng dùng nó làm cỡ mẫu cho bất kỳ phép tính nào.")


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
