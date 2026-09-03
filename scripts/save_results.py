"""Gói toàn bộ kết quả của MỘT thực nghiệm thành một tệp nén, cất lên Drive.

    python scripts/save_results.py tn0
    python scripts/save_results.py tn1
    python scripts/save_results.py tn1 --out /tmp        # thử ở máy, không cần Drive

`.tar.gz` giống `.zip`: `tar` gộp cả thư mục thành một tệp, `gz` nén lại.

BÊN TRONG TỆP NÉN CÓ GÌ

    tn1/
      summary.csv       metric — các dòng của tn1 trong runs/summary.csv
      README.txt        sinh lúc nào, commit nào, máy nào
      *.txt             tệp lựa chọn kênh
      scores_*.csv      điểm từng buổi ghi
      runs/             checkpoint .pth và đường cong loss từng epoch

Bung ra xem, đúng chỗ cũ:

    tar -xzf tn1.tar.gz -C results/     ->  results/tn1/...

VÌ SAO PHẢI CHÉP summary.csv VÀO ĐÂY

`runs/summary.csv` là bảng metric chung của cả đồ án, mỗi lần chạy một dòng, dùng
để dựng bảng trong luận văn. Nhưng tệp nén của một thực nghiệm phải **tự chứa
đủ** — tải riêng về vẫn đọc được metric mà không cần bảng chung. Nên script lọc
đúng các dòng có cột `experiment` bằng tên thực nghiệm rồi chép vào.

MỖI THỰC NGHIỆM MỘT THƯ MỤC, KHÔNG DÙNG CHUNG

    results/tn0/  runs/tn0/   <- notebooks/TN0.ipynb
    results/tn1/  runs/tn1/   <- scripts/run_cv.py --experiment tn1
    results/tn7/  runs/tn7/   <- scripts/run_final_test.py --experiment tn7
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time


# ===================== CÀI ĐẶT — sửa ở đây =====================

RESULTS_DIR = "results"
RUNS_DIR = "runs"
SUMMARY_FILE = "runs/summary.csv"

DRIVE = "/content/drive/MyDrive/mobivital"

# ===============================================================


def summary_rows(name):
    """Các dòng của thực nghiệm này trong bảng metric chung. Trả về (header, rows)."""
    if not os.path.exists(SUMMARY_FILE):
        return None, []

    reader = csv.reader(open(SUMMARY_FILE))
    header = next(reader)
    where = header.index("experiment")
    return header, [r for r in reader if len(r) > where and r[where] == name]


parser = argparse.ArgumentParser()
parser.add_argument("name", help="tên thực nghiệm, ví dụ tn0 hoặc tn1")
parser.add_argument("--out", default=DRIVE,
                    help="thư mục chứa tệp nén, mặc định là thư mục Drive")
args = parser.parse_args()

name = args.name
source = RESULTS_DIR + "/" + name

if not os.path.isdir(source):
    sys.exit("không thấy " + source + " — thực nghiệm chưa chạy?")

if not os.path.isdir(args.out):
    sys.exit("không thấy thư mục đích " + args.out +
             "\nChưa mount Drive thì chạy drive.mount(\"/content/drive\") trước, "
             "hoặc thêm --out <thư mục khác>.")


# --- Dựng thư mục tạm đúng cấu trúc muốn có trong tệp nén ---

stage_root = "/tmp/save_results"
stage = stage_root + "/" + name
shutil.rmtree(stage_root, ignore_errors=True)
shutil.copytree(source, stage)

# runs/ trên Colab là lối tắt vào Drive, nên chép nội dung chứ không chép lối tắt.
# ignore_dangling_symlinks: bỏ qua lối tắt trỏ vào tệp đã xoá, không làm chết cả
# lệnh gói kết quả.
run_folder = RUNS_DIR + "/" + name
if os.path.isdir(run_folder):
    shutil.copytree(run_folder, stage + "/runs",
                    symlinks=False, ignore_dangling_symlinks=True)

# Metric: lọc đúng các dòng của thực nghiệm này.
header, rows = summary_rows(name)
if header:
    with open(stage + "/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

commit = subprocess.run("git rev-parse --short HEAD", shell=True,
                        capture_output=True, text=True).stdout.strip()
with open(stage + "/README.txt", "w") as f:
    f.write("Thực nghiệm : %s\n" % name)
    f.write("Sinh lúc    : %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    f.write("Commit code : %s\n" % commit)
    f.write("Số dòng metric trong summary.csv: %d\n\n" % len(rows))
    f.write("Bung ra:  tar -xzf %s.tar.gz -C results/\n" % name)


# --- Nén ---

archive = os.path.join(args.out, name + ".tar.gz")
cmd = "tar -czf %s -C %s %s" % (archive, stage_root, name)
if subprocess.run(cmd, shell=True).returncode != 0:
    sys.exit("DỪNG — lệnh lỗi: " + cmd)

shutil.rmtree(stage_root, ignore_errors=True)

print("%s  ->  %s   (%.1f MB)"
      % (source + "/", archive, os.path.getsize(archive) / 1e6))
print("   %d dòng metric trong summary.csv" % len(rows))
print()
subprocess.run("tar -tzf " + archive, shell=True)
