"""Tải bộ dữ liệu tripod từ Zenodo và giải nén vào thư mục MobiVital.

    python scripts/download_dataset.py

Giải nén thẳng vào `external/mobivital/dataset/mobivital/tripod/` — đúng đường
dẫn tương đối `prep_breath_final.py` dòng 18 đòi. Chỉ giữ MỘT bản CSV; `data/`
chỉ chứa thứ pipeline đồ án sinh ra.

Tệp zip đã có sẵn thư mục `tripod/` bên trong nên giải nén vào `.../mobivital/`,
không vào `.../mobivital/tripod/`, nếu không sẽ lồng hai tầng.

Dùng `aria2c` 16 luồng. Đo thật trên Colab: `wget` một luồng 0.7 MB/s mất 2.3
giờ, `aria2c -x16` 25-56 MB/s mất 2-4 phút — Zenodo bóp băng thông mỗi kết nối.

Chạy lại được: đủ 1874 tệp CSV thì bỏ qua, không tải lại.
"""

import os
import subprocess
import sys
from glob import glob


# ===================== CÀI ĐẶT — sửa ở đây =====================

ZIP_PATH = "/content/tripod.zip"
ZIP_SIZE = 5700705593      # byte, lấy từ Zenodo API
ZIP_URL = "https://zenodo.org/api/records/15022885/files/tripod.zip/content"

UNZIP_DIR = "external/mobivital/dataset/mobivital"
CSV_DIR = UNZIP_DIR + "/tripod"
N_CSV = 1874

# ===============================================================


def run(cmd):
    """Chạy lệnh, dừng hẳn nếu lỗi."""
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit("Lệnh lỗi: " + cmd)


if len(glob(CSV_DIR + "/*.csv")) == N_CSV:
    print("đã đủ", N_CSV, "tệp CSV trong", CSV_DIR, "— bỏ qua")
    sys.exit(0)

if not (os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) == ZIP_SIZE):
    run("apt-get install -qq -y aria2")
    run("aria2c -x16 -s16 -k5M --summary-interval=0 --console-log-level=warn "
        "-d %s -o %s %s" % (os.path.dirname(ZIP_PATH),
                            os.path.basename(ZIP_PATH), ZIP_URL))

size = os.path.getsize(ZIP_PATH)
if size != ZIP_SIZE:
    sys.exit("tệp zip sai kích thước: %d, cần %d" % (size, ZIP_SIZE))
print("tệp zip %.2f GB, đúng kích thước Zenodo công bố" % (size / 1e9))

os.makedirs(UNZIP_DIR, exist_ok=True)
run("unzip -q -o %s -d %s/" % (ZIP_PATH, UNZIP_DIR))

n = len(glob(CSV_DIR + "/*.csv"))
if n != N_CSV:
    sys.exit("giải nén ra %d tệp CSV, cần %d" % (n, N_CSV))

print(n, "tệp CSV trong", CSV_DIR)
