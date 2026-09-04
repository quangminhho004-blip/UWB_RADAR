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
import sys
from glob import glob

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

RUNS_DIR = "runs"
SUMMARY_FILE = "runs/summary.csv"

# Chênh lệch dưới ngưỡng này coi là hoà — nhỏ hơn nhiễu số dấu phẩy động.
HOA = 1e-6

# ===============================================================


def doc_summary(experiment):
    """Các dòng của thực nghiệm này trong bảng metric chung."""
    if not os.path.exists(SUMMARY_FILE):
        sys.exit("không thấy " + SUMMARY_FILE + " — chưa chạy run_cv.py lần nào?")

    rows = [r for r in csv.DictReader(open(SUMMARY_FILE))
            if r["experiment"] == experiment]
    if not rows:
        sys.exit("không có dòng nào của thực nghiệm '%s' trong %s"
                 % (experiment, SUMMARY_FILE))
    return rows


def diem_tung_buoi(experiment, config_id):
    """Gộp bốn tệp fold thành một bảng {tên buổi ghi: điểm}.

    Mỗi người nằm ở đúng một fold validation, nên bốn tệp không đè nhau.
    """
    diem = {}
    for path in sorted(glob("%s/%s/scores_%s_*.csv" % (RUNS_DIR, experiment, config_id))):
        for r in csv.DictReader(open(path)):
            diem[r["session_file"]] = float(r["pearson"])
    return diem


def thang_hoa_thua(a, b):
    """Đếm số buổi ghi cấu hình a hơn / bằng / kém cấu hình b."""
    chung = sorted(set(a) & set(b))
    thang = sum(1 for f in chung if a[f] - b[f] > HOA)
    thua = sum(1 for f in chung if b[f] - a[f] > HOA)
    return thang, len(chung) - thang - thua, thua, len(chung)


# ---------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, help="tên thực nghiệm, ví dụ tn1")
parser.add_argument("--baseline", default=None,
                    help="tên model làm mốc để đếm thắng/hoà/thua, ví dụ lstm")
parser.add_argument("--final", action="store_true",
                    help="chế độ test GHIJ: gộp nhiều seed thành mean +- std "
                         "thay vì in bảng cv_score")
args = parser.parse_args()

rows = doc_summary(args.experiment)


# --- Chế độ test GHIJ: gộp các seed ---

if args.final:
    import re

    # run_id của run_final_test.py kết thúc bằng _seed<N>. Bỏ phần đó đi thì
    # các lần chạy khác seed của cùng một cấu hình gom về một nhóm.
    nhom = {}
    for r in rows:
        ten = re.sub(r"_seed\d+$", "", r["run_id"])
        nhom.setdefault(ten, []).append(r)

    print()
    print("TEST GHIJ — thực nghiệm %s" % args.experiment)
    print("Train đủ 8 người A B C D E F K L, test 537 buổi ghi của G H I J.")
    print()
    print("%-34s %10s %6s %11s %10s   %s"
          % ("cấu hình", "tham số", "seed", "mean", "std", "từng seed"))
    print("-" * 100)

    for ten in sorted(nhom, key=lambda k: -np.mean([float(x["score_macro"]) for x in nhom[k]])):
        rs = sorted(nhom[ten], key=lambda x: int(x["seed"]))
        diem = [float(x["score_macro"]) for x in rs]
        chi_tiet = "  ".join("s%s %.4f" % (x["seed"], d) for x, d in zip(rs, diem))
        print("%-34s %10s %6d %11.6f %10.6f   %s"
              % (ten, rs[0]["n_params"], len(diem),
                 float(np.mean(diem)), float(np.std(diem)), chi_tiet))

    print()
    print("std là độ lệch chuẩn giữa các seed — cho biết chênh lệch giữa hai cấu")
    print("hình có lớn hơn nhiễu ngẫu nhiên hay không.")
    print()
    raise SystemExit(0)


# --- Bảng 1: cv_score ---

tong = [r for r in rows if r["fold"] == "TONG"]
if not tong:
    sys.exit("chưa có dòng fold=TONG nào. Chạy lại run_cv.py bằng bản mới nhất.")

tong.sort(key=lambda r: -float(r["score_macro"]))

print()
print("BẢNG 1 — cv_score, thực nghiệm %s" % args.experiment)
print("%-32s %10s %10s %9s   %s"
      % ("cấu hình", "tham số", "cv_score", "cv_std", "từng fold"))
print("-" * 100)

for r in tong:
    config_id = r["run_id"][:-len("_tong")]
    fold_rows = sorted((x for x in rows if x["fold"] not in ("", "TONG")
                        and x["run_id"].startswith(config_id + "_")),
                       key=lambda x: x["fold"])
    chi_tiet = "  ".join("%s %.4f" % (x["fold"].replace("val_", ""),
                                      float(x["score_macro"])) for x in fold_rows)
    print("%-32s %10s %10.6f %9.6f   %s"
          % (config_id, r["n_params"], float(r["score_macro"]),
             float(r["score_std"] or 0), chi_tiet))

print()
print("cv_score = trung bình điểm macro của 4 fold. Điểm macro = trung bình theo")
print("NGƯỜI, không theo buổi ghi — mỗi người có số buổi khác nhau.")


# --- Bảng 2: thắng / hoà / thua ---

if args.baseline:
    moc = [r for r in tong if r["model"] == args.baseline]
    if not moc:
        sys.exit("không thấy cấu hình nào dùng model '%s'" % args.baseline)
    moc_id = moc[0]["run_id"][:-len("_tong")]
    diem_moc = diem_tung_buoi(args.experiment, moc_id)

    if not diem_moc:
        sys.exit("không đọc được điểm từng buổi ghi của " + moc_id)

    print()
    print("BẢNG 2 — thắng / hoà / thua trên TỪNG buổi ghi, mốc là %s" % moc_id)
    print("%-32s %8s %7s %7s %9s   %s"
          % ("cấu hình", "thắng", "hoà", "thua", "tổng", "chênh lệch điểm"))
    print("-" * 100)

    for r in tong:
        config_id = r["run_id"][:-len("_tong")]
        if config_id == moc_id:
            continue
        diem = diem_tung_buoi(args.experiment, config_id)
        t, h, th, n = thang_hoa_thua(diem, diem_moc)
        lech = float(r["score_macro"]) - float(moc[0]["score_macro"])
        print("%-32s %8d %7d %7d %9d   %+.6f" % (config_id, t, h, th, n, lech))

    print()
    print("Hoà = chênh lệch dưới %g. Tổng phải là 1289 buổi ghi của 8 người dev." % HOA)
print()
