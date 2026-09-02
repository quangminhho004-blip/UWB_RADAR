"""BƯỚC 1 — Gom file CSV vào thư mục theo từng người.

    python scripts/1_organize_raw.py

Trước:  data/raw/tripod/240502_userA_tripod_04_8.csv   (1874 file nằm chung)
Sau:    data/raw/A/240502_userA_tripod_04_8.csv
        data/raw/B/...
        ...

File được DI CHUYỂN (không copy) nên không tốn thêm dung lượng.
Sau khi chạy xong, thư mục data/raw/tripod/ còn lại rỗng.
"""

import glob
import os
import shutil

SRC_DIR = "data/raw/tripod"
DST_DIR = "data/raw"

csv_files = glob.glob(SRC_DIR + "/*.csv")
print("Tìm thấy", len(csv_files), "file CSV")

moved = 0
for path in csv_files:
    file_name = os.path.basename(path)

    # Tên file dạng "..._userA_...". Lấy chữ cái ngay sau "user".
    user = file_name.split("user")[1][0].upper()

    # Tạo thư mục data/raw/A/ nếu chưa có, rồi chuyển file vào.
    user_dir = DST_DIR + "/" + user
    os.makedirs(user_dir, exist_ok=True)
    shutil.move(path, user_dir + "/" + file_name)
    moved = moved + 1

print("Đã chuyển", moved, "file")
