"""Lấy dữ liệu đã xử lý từ Google Drive về, thay vì dựng lại từ đầu.

    python scripts/restore_processed_data_on_drive.py

`notebooks/DATA_PREPARE.ipynb` chạy một lần rồi cất hai tệp lên Drive:

    by_user.tar       2.6 GB   data/processed/by_user/*.npz
    windows.tar.gz    106 MB   data/processed/windows/

Script này bung chúng về đúng chỗ. Tiết kiệm khoảng 15 phút so với chạy lại
`make_npz.py` và `make_windows.py`.

CÁI GÌ KHÔNG LẤY ĐƯỢC TỪ DRIVE

CSV thô 13 GB không cất lên Drive (quá nặng, tải lại Zenodo chỉ mất 4 phút), mà
pipeline MobiVital đọc thẳng CSV — `inference/mobivital_gen.py` dòng 122 duyệt
thư mục `dataset/mobivital/tripod/`. Nên `scripts/download_dataset.py` vẫn phải
chạy dù có Drive hay không.

CHẠY LẠI ĐƯỢC

Thư mục đích đã đủ tệp thì bỏ qua, không bung đè. Drive chưa mount hoặc chưa có
tệp thì báo rồi thoát bình thường — bước sau (`make_npz.py`, `make_windows.py`)
tự dựng lại.
"""

import os
import subprocess
import sys
from glob import glob


# ===================== CÀI ĐẶT — sửa ở đây =====================

DRIVE = "/content/drive/MyDrive/mobivital"
PROCESSED = "data/processed"

USERS = "ABCDEFGHIJKL"

# ===============================================================


def run(cmd):
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit("DỪNG — lệnh lỗi: " + cmd)


if not os.path.isdir(DRIVE):
    print("chưa mount Drive (hoặc chưa có %s) — bỏ qua, các bước sau sẽ tự dựng lại"
          % DRIVE)
    raise SystemExit(0)

os.makedirs(PROCESSED, exist_ok=True)

# --- by_user ---

have = sorted(os.path.basename(p) for p in glob(PROCESSED + "/by_user/*.npz"))
want = sorted(u + ".npz" for u in USERS)
archive = DRIVE + "/by_user.tar"

if have == want:
    print("by_user   : đã đủ 12 tệp, bỏ qua")
elif os.path.exists(archive):
    print("by_user   : bung", archive, "...")
    run("tar -xf %s -C %s" % (archive, PROCESSED))
    n = len(glob(PROCESSED + "/by_user/*.npz"))
    print("            %d tệp" % n)
else:
    print("by_user   : Drive không có by_user.tar — make_npz.py sẽ tự dựng")

# --- windows ---

archive = DRIVE + "/windows.tar.gz"
final_train = PROCESSED + "/windows/final_train/train_corr0.9_h200_f25.npz"

if os.path.exists(final_train) and len(glob(PROCESSED + "/windows/dev_cv/*.npz")) == 8:
    print("windows   : đã đủ, bỏ qua")
elif os.path.exists(archive):
    print("windows   : bung", archive, "...")
    run("tar -xzf %s -C %s" % (archive, PROCESSED))
    print("            dev_cv %d tệp, final_train %s"
          % (len(glob(PROCESSED + "/windows/dev_cv/*.npz")),
             "có" if os.path.exists(final_train) else "thiếu"))
else:
    print("windows   : Drive không có windows.tar.gz — make_windows.py sẽ tự dựng")

print()
run("du -sh %s/* 2>/dev/null || true" % PROCESSED)
