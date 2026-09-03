"""Nén kết quả một thực nghiệm rồi cất lên Google Drive.

    python scripts/save_results.py tn0

Mỗi thực nghiệm có thư mục riêng, nén thành một tệp riêng:

    results/tn0/    -> Drive/mobivital/tn0.tar.gz
    results/cv/     -> Drive/mobivital/cv.tar.gz
    results/final/  -> Drive/mobivital/final.tar.gz

Bung ở máy vào đúng chỗ cũ:

    tar -xzf ~/Downloads/tn0.tar.gz -C /Users/udnb/Desktop/THESIS_GRADUATE/

Không nén runs/ vì nó đã là lối tắt trỏ thẳng vào Drive.
"""

import os
import subprocess
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

RESULTS_DIR = "results"
DRIVE = "/content/drive/MyDrive/mobivital"

# ===============================================================


if len(sys.argv) != 2:
    sys.exit("dùng: python scripts/save_results.py tn0")

name = sys.argv[1]
folder = RESULTS_DIR + "/" + name

if not os.path.isdir(folder):
    sys.exit("không thấy " + folder)

if not os.path.isdir(DRIVE):
    sys.exit("chưa mount Drive — chạy drive.mount(\"/content/drive\") trước")

archive = DRIVE + "/" + name + ".tar.gz"
cmd = "tar -czf %s -C %s %s" % (archive, RESULTS_DIR, name)

if subprocess.run(cmd, shell=True).returncode != 0:
    sys.exit("DỪNG — lệnh lỗi: " + cmd)

print("%s  ->  %s  (%.1f MB)" % (folder, archive, os.path.getsize(archive) / 1e6))
print()
subprocess.run("ls -la " + folder, shell=True)
