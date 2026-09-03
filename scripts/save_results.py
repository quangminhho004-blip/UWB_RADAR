"""Nén toàn bộ kết quả của MỘT thực nghiệm thành một tệp .zip.

    python scripts/save_results.py tn0     ->  runs/tn0.zip
    python scripts/save_results.py tn1     ->  runs/tn1.zip

Chạy xong tự tải `runs/<tên>.zip` về rồi tự đưa lên Drive. Giải nén ngược lại
vào đúng chỗ cũ:

    unzip tn0.zip -d runs/        ->  runs/tn0/...

MỖI THỰC NGHIỆM MỘT THƯ MỤC

Mọi thứ của một thực nghiệm nằm chung một chỗ — checkpoint, đường cong loss,
bảng lựa chọn kênh, điểm từng buổi ghi, metric:

    runs/tn0/
      TN0a.txt  TN0b.txt  TN0c.txt      tệp lựa chọn kênh, pipeline MobiVital
      ours_b.txt  ours_c.txt            tệp lựa chọn kênh, pipeline đồ án
      scores_*.csv                      điểm từng buổi ghi
      compare.csv                       bảng ĐẠT / KHÔNG ĐẠT
      summary.csv                       metric, do script này lọc ra
      README.txt                        sinh lúc nào, commit nào
      ours_c/final.pth  ours_c/curve.csv    trọng số và loss từng epoch

    runs/tn1/
      <cấu hình>_val_AB/final.pth  curve.csv
      scores_<cấu hình>_val_AB.csv
      summary.csv  README.txt

VÌ SAO CHÉP summary.csv VÀO TỪNG THƯ MỤC

`runs/summary.csv` là bảng metric chung của cả đồ án, mỗi lần chạy một dòng, dùng
để dựng bảng trong luận văn. Nhưng tệp nén của một thực nghiệm phải **tự chứa
đủ** — tải riêng về vẫn đọc được metric mà không cần bảng chung. Nên script lọc
đúng các dòng có cột `experiment` bằng tên thực nghiệm rồi chép vào.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time


# ===================== CÀI ĐẶT — sửa ở đây =====================

RUNS_DIR = "runs"
SUMMARY_FILE = "runs/summary.csv"

# ===============================================================


def summary_rows(name):
    """Các dòng của thực nghiệm này trong bảng metric chung."""
    if not os.path.exists(SUMMARY_FILE):
        return None, []

    reader = csv.reader(open(SUMMARY_FILE))
    header = next(reader)
    where = header.index("experiment")
    return header, [r for r in reader if len(r) > where and r[where] == name]


parser = argparse.ArgumentParser()
parser.add_argument("name", help="tên thực nghiệm, ví dụ tn0 hoặc tn1")
args = parser.parse_args()

name = args.name
folder = RUNS_DIR + "/" + name

if not os.path.isdir(folder):
    sys.exit("không thấy " + folder + " — thực nghiệm chưa chạy?")


# --- Metric: lọc đúng các dòng của thực nghiệm này vào ngay thư mục của nó ---

header, rows = summary_rows(name)
if header:
    with open(folder + "/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --- Ghi chú kèm theo, để mở tệp nén ra là biết nó từ đâu ---

commit = subprocess.run("git rev-parse --short HEAD", shell=True,
                        capture_output=True, text=True).stdout.strip()
with open(folder + "/README.txt", "w") as f:
    f.write("Thực nghiệm : %s\n" % name)
    f.write("Sinh lúc    : %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    f.write("Commit code : %s\n" % commit)
    f.write("Số dòng metric trong summary.csv: %d\n\n" % len(rows))
    f.write("Giải nén:  unzip %s.zip -d runs/\n" % name)


# --- Nén ---

archive = RUNS_DIR + "/" + name + ".zip"
if os.path.exists(archive):
    os.remove(archive)

shutil.make_archive(RUNS_DIR + "/" + name, "zip", root_dir=RUNS_DIR, base_dir=name)

print("%s/  ->  %s   (%.1f MB)" % (folder, archive, os.path.getsize(archive) / 1e6))
print("   %d dòng metric trong summary.csv" % len(rows))
print()
print("Bên trong:", flush=True)
subprocess.run("unzip -l " + archive + " | tail -n +4 | head -40", shell=True)
sys.stdout.flush()
print()
print("Tải %s về rồi đưa lên Drive. Giải nén: unzip %s.zip -d runs/"
      % (archive, name))
