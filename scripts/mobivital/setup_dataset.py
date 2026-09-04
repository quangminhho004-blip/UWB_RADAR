"""Dọn chỗ để chạy ĐÚNG lệnh trong README của MobiVital.

    python scripts/mobivital/setup_dataset.py

MỤC ĐÍCH

README của MobiVital ghi chạy như sau, từ trong thư mục repo đó:

    python dataset_preparation/prep_breath_final.py
    python -m training.autoreg_training
    python -m inference.mobivital_gen
    python -m inference.evaluate -m YOUR_METHOD.txt

Bốn lệnh chạy nguyên bản, không qua lớp bọc nào. CSV giải nén thẳng vào
`dataset/mobivital/tripod/` — đúng đường dẫn code MobiVital đòi — nên chúng chạy được
ngay. Script này chỉ thêm hai thứ MobiVital không có:

    1. tripod_old_names/   vá 52 tên file lỗi thời, cho evaluate.py
    2. .git/info/exclude   giấu dữ liệu khỏi git của MobiVital

MỘT BẢN CSV DUY NHẤT

    external/mobivital/dataset/mobivital/tripod/   1874 CSV thật  <- ở đây
    data/                                          chỉ dữ liệu đồ án sinh ra

Hai pipeline đọc chung một bản CSV. Nhờ vậy khi `scripts/check_data.py` báo sai
lệch bằng 0 thì không ai cãi được là do hai bản dữ liệu khác nhau.

52 TÊN FILE LỖI THỜI

Bảng kết quả MobiVital commit sẵn ra đời TRƯỚC khi dataset đổi tên đưa lên
Zenodo. 52 trong 537 dòng ghi mốc ngày tháng 10, bản Zenodo hiện tại là tháng 12:

    trong bảng : 231003_userI_tripod_01_0.csv
    trên đĩa   : 231203_userI_tripod_01_0.csv
                     ^^

`evaluate.py` dòng 28 mở file theo tên trong bảng, gặp 52 tên đó là chết giữa
chừng — mất luôn con số đối chiếu với bài báo. Cách vá: thư mục lối tắt mang tên
cũ, trỏ vào file thật. Không sửa bảng, không sửa code tác giả.

VÌ SAO PHẢI LÀ THƯ MỤC RIÊNG

Hai script của MobiVital chọn buổi ghi theo hai cách khác nhau:

    evaluate.py:23        duyệt từng dòng của bảng .txt   -> chấm đúng tên trong bảng
    mobivital_gen.py:122  os.listdir() rồi lọc G H I J    -> chấm mọi file trong thư mục

52 tên cũ đó đều thuộc G H I J. Nhét chung một thư mục thì `mobivital_gen.py`
đếm thành 589 buổi ghi thay vì 537, tính hai lần cùng một bản ghi.

    tripod/            1874 file thật     -> prep_breath_final.py, mobivital_gen.py
    tripod_old_names/  1874 + 52 lối tắt  -> CHỈ evaluate.py chấm bảng tác giả, qua cờ -d

BẰNG CHỨNG 52 TÊN ĐÓ LÀ CÙNG BUỔI GHI

Đổi hai chữ số tháng 10 thành 12 thì khớp 1-1 đủ 52 file, không thiếu không thừa.
Chấm 52 buổi ghi đó qua lối tắt cho 0.8859, cao hơn 485 buổi còn lại (0.8124) —
trỏ nhầm buổi ghi thì điểm phải rải quanh 0.

CẦN CHẠY TRƯỚC

    giải nén tripod.zip vào external/mobivital/dataset/mobivital/
"""

import csv
import glob
import os
import shutil


# ===================== CÀI ĐẶT — sửa ở đây =====================

MOBIVITAL_DIR = "external/mobivital"
CSV_DIR = MOBIVITAL_DIR + "/dataset/mobivital/tripod"
OLD_NAMES_DIR = MOBIVITAL_DIR + "/dataset/mobivital/tripod_old_names"
OUT_DIR = "runs/tn0"

# Bảng kết quả MobiVital commit sẵn. Sao ra runs/tn0/ trước khi làm gì, vì
# mobivital_gen.py ghi đè đúng cái tên này.
THEIR_TXT = "tripod_mobivital_pre_invert_0.9.txt"

# Mốc ngày trong tên file có dạng YYMMDD, tháng nằm ở vị trí thứ 3 và 4.
MONTH_START = 2
MONTH_END = 4
NEW_MONTH = "12"

# ===============================================================


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


csv_paths = sorted(glob.glob(CSV_DIR + "/*.csv"))
if len(csv_paths) == 0:
    raise RuntimeError("không thấy CSV nào trong " + CSV_DIR
                       + " — giải nén tripod.zip vào " + MOBIVITAL_DIR
                       + "/dataset/mobivital/ trước")

print(len(csv_paths), "file CSV trong", CSV_DIR)
print()


# --- 1. Sao lưu bảng kết quả của tác giả trước khi bị đè ---

os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy(MOBIVITAL_DIR + "/inference/methods/" + THEIR_TXT,
            OUT_DIR + "/TN0a.txt")
print("1. sao lưu bảng tác giả ->", OUT_DIR + "/TN0a.txt",
      "(" + str(len(open(OUT_DIR + "/TN0a.txt").readlines())) + " dòng)")


# --- 2. Thư mục lối tắt, thêm 52 tên cũ, cho evaluate.py ---

os.makedirs(OLD_NAMES_DIR, exist_ok=True)
for real_path in csv_paths:
    link = OLD_NAMES_DIR + "/" + os.path.basename(real_path)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(real_path), link)

names_in_txt = []
for row in csv.reader(open(OUT_DIR + "/TN0a.txt")):
    names_in_txt.append(row[0])

patched = 0
for old_name in names_in_txt:
    link = OLD_NAMES_DIR + "/" + old_name
    if os.path.lexists(link):
        continue

    real_path = CSV_DIR + "/" + to_new_name(old_name)
    if not os.path.exists(real_path):
        raise RuntimeError("không tìm được file thật cho " + old_name)

    os.symlink(os.path.abspath(real_path), link)
    patched = patched + 1

print("2.", OLD_NAMES_DIR, "-> vá", patched, "tên cũ,",
      len(os.listdir(OLD_NAMES_DIR)), "lối tắt")


# --- 3. Giấu dữ liệu khỏi git của MobiVital ---

exclude_from_git(MOBIVITAL_DIR,
                 ["dataset/", "data_final/",
                  "inference/methods/scores*.csv",
                  "inference/methods/TN0*.txt",
                  "checkpoints/lstm_retrained*"])
print("3. thêm dataset/ data_final/ scores*.csv TN0*.txt checkpoints/lstm_retrained* vào .git/info/exclude")


# --- 4. Kiểm tra ---

missing = 0
for name in names_in_txt:
    if not os.path.exists(OLD_NAMES_DIR + "/" + name):
        missing = missing + 1

print()
print("KIỂM TRA")
print("   tên trong bảng tác giả:", len(names_in_txt))
print("   mở không được         :", missing, " <- phải là 0")
print("   thư mục CSV thật      :", len(csv_paths), " <- phải là 1874")

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
