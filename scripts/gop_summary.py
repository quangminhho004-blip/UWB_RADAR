"""Chèn các dòng metric mới vào runs/<thực nghiệm>/summary.csv.

    python scripts/gop_summary.py <tệp.csv> --dich runs/tn1/summary.csv

Dùng khi lấy kết quả của một lượt chạy cũ từ tệp nén trên Drive về: giải nén ra
được phần metric riêng, script này ghép nó vào bảng chung của thực nghiệm.

Khoá của một dòng là `run_id`. Dòng đã có thì GIỮ NGUYÊN bản trong repo và báo
ra, không đè — để một lần chạy lại không lặng lẽ thay số đã công bố.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.results import SUMMARY_COLUMNS


parser = argparse.ArgumentParser()
parser.add_argument("nguon", help="tệp csv chứa các dòng cần chèn")
parser.add_argument("--dich", default="runs/tn1/summary.csv")
args = parser.parse_args()

if not os.path.exists(args.nguon):
    sys.exit("không thấy " + args.nguon)

dang_co = []
if os.path.exists(args.dich):
    dang_co = list(csv.DictReader(open(args.dich)))
da_co = {r["run_id"] for r in dang_co}

them, trung = [], []
for r in csv.DictReader(open(args.nguon)):
    rid = r["run_id"]
    if rid in da_co:
        trung.append(rid)
        continue
    # Cập nhật da_co NGAY, không đợi hết vòng lặp: tệp nguồn có thể tự chứa hai
    # dòng cùng run_id (gộp từ nhiều tệp nén trùng nhau). Không làm vậy thì cả
    # hai cùng được chèn, và summary.csv có hai dòng cho một lần chạy.
    da_co.add(rid)
    them.append(rid)
    dang_co.append({k: r.get(k, "") for k in SUMMARY_COLUMNS})

dang_co.sort(key=lambda r: (r["model"], int(r["n_params"] or 0),
                            int(r["seed"] or 0), r["fold"]))

with open(args.dich, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, restval="",
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(dang_co)

print("chèn %d dòng, bỏ qua %d dòng đã có" % (len(them), len(trung)))
for rid in trung:
    print("   đã có, giữ bản trong repo:", rid)
print("%s giờ có %d dòng" % (args.dich, len(dang_co)))
