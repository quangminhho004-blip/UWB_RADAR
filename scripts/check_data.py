"""Chứng minh dữ liệu của mình đúng bằng dữ liệu MobiVital dùng.

    python scripts/check_data.py

Phải in ra:

    ABCDEFKL   1289/1289 buổi ghi khớp TỪNG BYTE   = training_breath_tripod_data.npy
    GHIJ        537/537  buổi ghi khớp TỪNG BYTE   = testing_breath_tripod_data.npy

Không ra thế thì dừng, mọi so sánh về sau vô nghĩa.

HAI ĐƯỜNG ĐỌC, MỘT BỘ CSV

    external/mobivital/dataset/mobivital/tripod/*.csv     1874 file, bản duy nhất
            |
            +--> prep_breath_final.py cua HO   --> data_final/*.npy
            |
            +--> scripts/make_npz.py cua MINH  --> by_user/*.npz

Khớp thì mọi thí nghiệm sau chỉ cần đọc `by_user/*.npz`, bỏ được CSV thô 13 GB.

SO BẰNG BYTE CHỨ KHÔNG SO BẰNG SAI SỐ

`gt` và `uwb` đều là `float32`/`complex64`, đọc từ cùng một file CSV bằng cùng
công thức, nên phải giống nhau tuyệt đối chứ không phải "gần bằng". So bằng byte
thì không cần chọn ngưỡng, và lệch một chữ số cuối cũng lộ ngay.

CÁCH GHÉP CẶP

Hai bên xếp buổi ghi theo thứ tự khác nhau: MobiVital dùng `os.listdir()`, mình
dùng `sorted()`. Nên không so theo vị trí, mà lấy chuỗi byte của `gt` làm khoá —
1500 số float32, không hai buổi ghi nào trùng nhau.

CHẠY TỪNG BÊN RỒI GIẢI PHÓNG

`training_breath_tripod_data.npy` nặng 1.9 GB, mỗi file `.npz` nặng 200-300 MB.
Nạp file .npy, băm từng buổi ghi thành chữ ký ngắn rồi giải phóng ngay, sau đó
mới mở .npz. Không bao giờ giữ cả hai cùng lúc.
"""

import hashlib

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

BY_USER_DIR = "data/processed/by_user"
DATA_FINAL_DIR = "external/mobivital/data_final"

DEV_USERS = ["A", "B", "C", "D", "E", "F", "K", "L"]
TEST_USERS = ["G", "H", "I", "J"]

# ===============================================================


def read_mobivital_npy(path):
    """Đọc file .npy của MobiVital.

    File của họ chứa hai mảng ghi nối tiếp nhau, nên phải gọi np.load hai lần
    trên cùng một file đang mở.
    """
    opened_file = open(path, "rb")
    uwb = np.load(opened_file)
    gt = np.load(opened_file)
    opened_file.close()
    return uwb, gt


def fingerprint(array):
    """Chữ ký 32 ký tự của một mảng. Đổi một byte là đổi chữ ký."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def index_mobivital(path):
    """Đọc .npy của MobiVital, trả về {chữ ký gt: chữ ký uwb}, rồi giải phóng."""
    uwb_all, gt_all = read_mobivital_npy(path)
    print("   ", path)
    print("    uwb", uwb_all.shape, uwb_all.dtype,
          "| gt", gt_all.shape, gt_all.dtype)

    table = {}
    for i in range(len(gt_all)):
        table[fingerprint(gt_all[i])] = fingerprint(uwb_all[i])

    del uwb_all, gt_all
    return table


def compare(label, users, npy_name):
    """So từng buổi ghi của các người này với file .npy tương ứng."""
    print()
    print(label)
    print("-" * 66)

    table = index_mobivital(DATA_FINAL_DIR + "/" + npy_name)

    matched = 0
    total = 0
    gt_missing = 0
    uwb_differs = 0

    for user in users:
        data = np.load(BY_USER_DIR + "/" + user + ".npz")
        uwb_all = data["uwb"]
        gt_all = data["gt"]

        for i in range(len(gt_all)):
            total = total + 1
            key = fingerprint(gt_all[i])

            if key not in table:
                gt_missing = gt_missing + 1
            elif table[key] != fingerprint(uwb_all[i]):
                uwb_differs = uwb_differs + 1
            else:
                matched = matched + 1

        del data, uwb_all, gt_all

    print("    của mình  ", total, "buổi ghi")
    print("    MobiVital ", len(table), "buổi ghi")
    print("    khớp TỪNG BYTE %d/%d" % (matched, total))

    if gt_missing > 0:
        print("    gt không tìm thấy bên MobiVital:", gt_missing)
    if uwb_differs > 0:
        print("    gt khớp nhưng uwb lệch        :", uwb_differs)

    return matched, total, len(table)


print("Đối chiếu by_user/*.npz  với  data_final/*.npy")

dev = compare("A B C D E F K L", DEV_USERS, "training_breath_tripod_data.npy")
test = compare("G H I J", TEST_USERS, "testing_breath_tripod_data.npy")


print()
print("=" * 66)
print("ABCDEFKL  %4d/%-4d buổi ghi khớp TỪNG BYTE   = training_breath_tripod_data.npy"
      % (dev[0], dev[1]))
print("GHIJ      %4d/%-4d buổi ghi khớp TỪNG BYTE   = testing_breath_tripod_data.npy"
      % (test[0], test[1]))
print("=" * 66)

if dev[0] != dev[1] or dev[1] != dev[2]:
    raise RuntimeError("ABCDEFKL không khớp — dừng")
if test[0] != test[1] or test[1] != test[2]:
    raise RuntimeError("GHIJ không khớp — dừng")

print("Từ đây mọi thí nghiệm chỉ đọc by_user/*.npz.")
