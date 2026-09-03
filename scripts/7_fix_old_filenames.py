"""BƯỚC 7 — Vá 52 tên file lỗi thời trong bảng kết quả MobiVital.

    python scripts/7_fix_old_filenames.py

VẤN ĐỀ

MobiVital commit sẵn bảng kết quả của họ trong repo:

    external/mobivital/inference/methods/tripod_mobivital_pre_invert_0.9.txt

537 dòng, mỗi dòng một buổi ghi:

    231003_userI_tripod_01_0.csv , 24 , phase , 0
          tên file CSV             bin   phép   cờ lật

Bảng này sinh ra TRƯỚC khi dataset được đổi tên đưa lên Zenodo. 52 dòng trong đó
ghi mốc ngày tháng 10, còn bản Zenodo hiện tại là tháng 12:

    trong bảng : 231003_userI_tripod_01_0.csv
    trên đĩa   : 231203_userI_tripod_01_0.csv
                     ^^

`evaluate.py` dòng 28 mở file theo tên trong bảng, gặp 52 tên đó là chết giữa
chừng. Không chấm được TN0a, tức mất luôn con số đối chiếu với bài báo.

CÁCH VÁ

Tạo lối tắt (symlink) mang tên cũ, trỏ vào file thật. Code MobiVital mở tên cũ,
hệ điều hành dẫn nó tới file thật. Không sửa bảng, không sửa code họ.

VÌ SAO PHẢI DỰNG HAI THƯ MỤC

52 tên cũ đó đều thuộc G H I J. `mobivital_gen.py` dòng 123 duyệt thư mục bằng
`os.listdir()` rồi lọc lấy GHIJ — nhét lối tắt vào chung thì nó đếm thành 589
buổi ghi thay vì 537, tính hai lần cùng một bản ghi.

    csv_flat/       1874 file, TÊN THẬT     -> mobivital_gen.py dùng
    csv_old_names/  1874 + 52               -> CHỈ evaluate.py chấm bảng của họ

`evaluate.py` có cờ `-d` nên trỏ sang thư mục thứ hai được, và nó đọc theo danh
sách trong bảng chứ không `listdir` nên thừa file cũng không sao.

BẰNG CHỨNG 52 TÊN ĐÓ LÀ CÙNG BUỔI GHI, CHỈ ĐỔI TÊN

Đổi hai chữ số tháng 10 thành 12 thì khớp 1-1 đủ 52 file, không thiếu không thừa.
Và chấm điểm 52 buổi ghi đó qua lối tắt cho 0.8859, cao hơn 485 buổi ghi còn lại
(0.8124) — trỏ nhầm buổi ghi thì điểm phải rải quanh 0.
"""

import csv
import glob
import os


# ===================== CÀI ĐẶT — sửa ở đây =====================

RAW_DIR = "data/raw"
MOBIVITAL_TXT = ("external/mobivital/inference/methods/"
                 "tripod_mobivital_pre_invert_0.9.txt")

FLAT_DIR = "runs/tn0/csv_flat"
OLD_NAMES_DIR = "runs/tn0/csv_old_names"

# Mốc ngày trong tên file: 6 chữ số đầu, dạng YYMMDD.
# Tháng nằm ở vị trí thứ 3 và 4.
MONTH_START = 2
MONTH_END = 4
OLD_MONTH = "10"
NEW_MONTH = "12"

# ===============================================================


def link_all(csv_paths, target_dir):
    """Tạo lối tắt cho mọi file CSV vào một thư mục phẳng. Trả về số cái đã tạo."""
    os.makedirs(target_dir, exist_ok=True)

    created = 0
    for real_path in csv_paths:
        link = target_dir + "/" + os.path.basename(real_path)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(real_path), link)
            created = created + 1
    return created


def to_new_name(old_name):
    """Đổi tháng 10 thành 12 trong tên file."""
    return old_name[:MONTH_START] + NEW_MONTH + old_name[MONTH_END:]


csv_paths = glob.glob(RAW_DIR + "/*/*.csv")
if len(csv_paths) == 0:
    raise RuntimeError("không thấy CSV nào trong " + RAW_DIR + "/*/ — chạy script 1 trước")

print("tìm thấy", len(csv_paths), "file CSV trong", RAW_DIR)
print()


# --- Thư mục 1: tên thật, cho mobivital_gen.py ---

print("1. dựng", FLAT_DIR, "— tên thật")
link_all(csv_paths, FLAT_DIR)
print("   ", len(os.listdir(FLAT_DIR)), "file")


# --- Thư mục 2: thêm 52 lối tắt tên cũ, cho evaluate.py ---

print()
print("2. dựng", OLD_NAMES_DIR, "— thêm lối tắt tên cũ")
link_all(csv_paths, OLD_NAMES_DIR)

names_in_txt = []
for row in csv.reader(open(MOBIVITAL_TXT)):
    names_in_txt.append(row[0])

patched = 0
for old_name in names_in_txt:
    link = OLD_NAMES_DIR + "/" + old_name
    if os.path.lexists(link):
        continue

    real_path = OLD_NAMES_DIR + "/" + to_new_name(old_name)
    if not os.path.exists(real_path):
        raise RuntimeError("không tìm được file thật cho " + old_name)

    os.symlink(os.path.realpath(real_path), link)
    patched = patched + 1

print("   vá", patched, "tên cũ ->", len(os.listdir(OLD_NAMES_DIR)), "file")


# --- Kiểm tra: mọi tên trong bảng đều mở được ---

print()
print("3. kiểm tra")

missing = 0
for name in names_in_txt:
    if not os.path.exists(OLD_NAMES_DIR + "/" + name):
        missing = missing + 1

print("   tên trong bảng      :", len(names_in_txt))
print("   mở không được       :", missing, " <- phải là 0")
print("   thư mục cho gen     :", len(os.listdir(FLAT_DIR)), " <- phải là 1874")

if missing > 0:
    raise RuntimeError("còn " + str(missing) + " tên mở không được")

print()
print("Xong.")
