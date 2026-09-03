"""Dọn chỗ để chạy ĐÚNG lệnh trong README của MobiVital.

    python scripts/mobivital/setup_dataset.py

MỤC ĐÍCH

README của MobiVital ghi chạy như sau, từ trong thư mục repo của họ:

    python dataset_preparation/prep_breath_final.py
    python -m training.autoreg_training
    python -m inference.mobivital_gen
    python -m inference.evaluate -m YOUR_METHOD.txt

Bốn lệnh đó chạy nguyên bản, không qua lớp bọc. Chỉ thiếu đúng một thứ:
`./dataset/mobivital/tripod/` phải nằm ngay trong repo họ, vì
`prep_breath_final.py` dòng 18 đọc đường dẫn tương đối đó. Script này dựng nó
bằng lối tắt, rồi ghi vào `.git/info/exclude` — file loại trừ CỤC BỘ, không
thuộc repo, không bị track — nên `git status` của họ vẫn trống.

`./data_final/*.npy` thì KHÔNG dựng ở đây: chính `prep_breath_final.py` sinh ra
nó. Bên mình không làm hộ việc đó nữa.

VIỆC KHÓ NHẤT: 52 TÊN FILE LỖI THỜI

Bảng kết quả MobiVital commit sẵn ra đời TRƯỚC khi dataset đổi tên đưa lên
Zenodo. 52 trong 537 dòng ghi mốc ngày tháng 10, bản Zenodo hiện tại là tháng 12:

    trong bảng : 231003_userI_tripod_01_0.csv
    trên đĩa   : 231203_userI_tripod_01_0.csv
                     ^^

`evaluate.py` dòng 28 mở file theo tên trong bảng, gặp 52 tên đó là chết giữa
chừng — mất luôn con số đối chiếu với bài báo.

Cách vá: tạo lối tắt mang tên cũ, trỏ vào file thật. Không sửa bảng, không sửa
code họ.

VÌ SAO PHẢI DỰNG HAI THƯ MỤC CSV

Hai script của MobiVital chọn buổi ghi theo hai cách khác nhau:

    evaluate.py:23        duyệt từng dòng của bảng .txt   -> chấm đúng tên trong bảng
    mobivital_gen.py:122  os.listdir() rồi lọc G H I J    -> chấm mọi file trong thư mục

52 tên cũ đó đều thuộc G H I J. Nhét chung một thư mục thì `mobivital_gen.py`
đếm thành 589 buổi ghi thay vì 537, tính hai lần cùng một bản ghi.

    dataset/mobivital/tripod/            1874 file, tên thật
                                         -> mobivital_gen.py dùng (mặc định)
    dataset/mobivital/tripod_old_names/  1874 + 52
                                         -> CHỈ evaluate.py chấm bảng của họ, qua cờ -d

BẰNG CHỨNG 52 TÊN ĐÓ LÀ CÙNG BUỔI GHI

Đổi hai chữ số tháng 10 thành 12 thì khớp 1-1 đủ 52 file, không thiếu không thừa.
Chấm 52 buổi ghi đó qua lối tắt cho 0.8859, cao hơn 485 buổi còn lại (0.8124) —
trỏ nhầm buổi ghi thì điểm phải rải quanh 0.

CẦN CHẠY TRƯỚC

    scripts/prepare_raw.py   -> data/raw/A..L
"""

import csv
import glob
import os
import shutil


# ===================== CÀI ĐẶT — sửa ở đây =====================

RAW_DIR = "data/raw"
MOBIVITAL_DIR = "external/mobivital"
OUT_DIR = "results"

# Bảng kết quả MobiVital commit sẵn. Sao ra results/ trước khi làm gì, vì
# mobivital_gen.py ghi đè đúng cái tên này.
THEIR_TXT = "tripod_mobivital_pre_invert_0.9.txt"

# Mốc ngày trong tên file có dạng YYMMDD, tháng nằm ở vị trí thứ 3 và 4.
MONTH_START = 2
MONTH_END = 4
NEW_MONTH = "12"

# ===============================================================


def link_all(csv_paths, target_dir):
    """Tạo lối tắt cho mọi file CSV vào một thư mục phẳng."""
    os.makedirs(target_dir, exist_ok=True)
    for real_path in csv_paths:
        link = target_dir + "/" + os.path.basename(real_path)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(real_path), link)


def to_new_name(old_name):
    """Đổi tháng 10 thành 12 trong tên file."""
    return old_name[:MONTH_START] + NEW_MONTH + old_name[MONTH_END:]


def exclude_from_git(repo_dir, patterns):
    """Thêm mẫu vào .git/info/exclude — loại trừ cục bộ, không thuộc repo."""
    path = repo_dir + "/.git/info/exclude"
    already = open(path).read() if os.path.exists(path) else ""

    opened_file = open(path, "a")
    for pattern in patterns:
        if pattern not in already:
            opened_file.write(pattern + "\n")
    opened_file.close()


csv_paths = glob.glob(RAW_DIR + "/*/*.csv")
if len(csv_paths) == 0:
    raise RuntimeError("không thấy CSV nào trong " + RAW_DIR + "/*/ — chạy script 1 trước")

print("tìm thấy", len(csv_paths), "file CSV")
print()


# --- 1. Sao lưu bảng kết quả của họ trước khi bị đè ---

os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy(MOBIVITAL_DIR + "/inference/methods/" + THEIR_TXT,
            OUT_DIR + "/TN0a.txt")
print("1. sao lưu bảng của họ ->", OUT_DIR + "/TN0a.txt",
      "(" + str(len(open(OUT_DIR + "/TN0a.txt").readlines())) + " dòng)")


# --- 2. Thư mục CSV tên thật, cho mobivital_gen.py ---

flat_dir = MOBIVITAL_DIR + "/dataset/mobivital/tripod"
link_all(csv_paths, flat_dir)
print("2.", flat_dir, "->", len(os.listdir(flat_dir)), "file")


# --- 3. Thư mục CSV thêm 52 tên cũ, cho evaluate.py chấm bảng của họ ---

old_dir = MOBIVITAL_DIR + "/dataset/mobivital/tripod_old_names"
link_all(csv_paths, old_dir)

names_in_txt = []
for row in csv.reader(open(OUT_DIR + "/TN0a.txt")):
    names_in_txt.append(row[0])

patched = 0
for old_name in names_in_txt:
    link = old_dir + "/" + old_name
    if os.path.lexists(link):
        continue

    real_path = old_dir + "/" + to_new_name(old_name)
    if not os.path.exists(real_path):
        raise RuntimeError("không tìm được file thật cho " + old_name)

    os.symlink(os.path.realpath(real_path), link)
    patched = patched + 1

print("3.", old_dir, "-> vá", patched, "tên cũ,",
      len(os.listdir(old_dir)), "file")


# --- 4. Giấu khỏi git của họ ---

exclude_from_git(MOBIVITAL_DIR, ["dataset/", "data_final/", "inference/methods/scores*.csv"])
print("4. thêm dataset/ data_final/ scores*.csv vào .git/info/exclude")


# --- 5. Kiểm tra ---

missing = 0
for name in names_in_txt:
    if not os.path.exists(old_dir + "/" + name):
        missing = missing + 1

print()
print("KIỂM TRA")
print("   tên trong bảng của họ :", len(names_in_txt))
print("   mở không được         :", missing, " <- phải là 0")
print("   thư mục cho gen       :", len(os.listdir(flat_dir)), " <- phải là 1874")

if missing > 0:
    raise RuntimeError("còn " + str(missing) + " tên mở không được")

print()
print("Xong. Giờ chạy được đúng lệnh trong README của MobiVital:")
print("   cd " + MOBIVITAL_DIR)
print("   python dataset_preparation/prep_breath_final.py")
print("   python -m inference.evaluate -m " + THEIR_TXT
      + " -d ./dataset/mobivital/tripod_old_names")
print("   python -m inference.mobivital_gen")
print("   python -m training.autoreg_training --model_name lstm_retrained")
