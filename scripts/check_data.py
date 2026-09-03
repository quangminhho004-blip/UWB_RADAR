"""Xem lại dữ liệu đã tạo có đúng không.

    python scripts/check_data.py

Script này chỉ ĐỌC dữ liệu rồi IN ra màn hình. Không sửa gì cả.

In ba phần:
    1. 12 file .npz  (pipeline của mình)
    2. 2 file .npy   (pipeline MobiVital)
    3. So hai bên xem có giống nhau không

Dòng quan trọng nhất là dòng cuối:

    Khac nhau nhieu nhat tren 1500 mau: 0.0

Đó là bằng chứng dữ liệu mình tự đọc giống hệt dữ liệu MobiVital dùng. Nhờ nó
mà mọi thí nghiệm sau chỉ cần đọc by_user/*.npz, bỏ được CSV thô 13 GB.
"""

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

BY_USER_DIR = "data/processed/by_user"
MOBIVITAL_DIR = "external/mobivital/data_final"

# 8 người dùng để phát triển model
DEV_USERS = ["A", "B", "C", "D", "E", "F", "K", "L"]

# 4 người để dành, chỉ chấm điểm ở bước cuối cùng
TEST_USERS = ["G", "H", "I", "J"]

# ===============================================================


def print_one_user(user):
    """Đọc file .npz của một người, in ra màn hình, trả về số session."""
    path = BY_USER_DIR + "/" + user + ".npz"
    data = np.load(path)

    uwb = data["uwb"]
    gt = data["gt"]

    n_sessions = len(gt)

    print("nguoi", user,
          "|", n_sessions, "session",
          "| uwb", uwb.shape,
          "| gt", gt.shape,
          "| gt tu", gt.min(), "den", gt.max())

    return n_sessions


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


# ---------------------------------------------------------------
# PHẦN 1 — đọc 12 file .npz của mình
# ---------------------------------------------------------------

print("PHAN 1 — 12 file .npz cua minh")
print("-" * 60)

total_dev = 0
for user in DEV_USERS:
    total_dev = total_dev + print_one_user(user)

print()

total_test = 0
for user in TEST_USERS:
    total_test = total_test + print_one_user(user)

print()
print("8 nguoi DEV  cong lai =", total_dev, "session")
print("4 nguoi TEST cong lai =", total_test, "session")


# ---------------------------------------------------------------
# PHẦN 2 — đọc 2 file .npy của MobiVital
# ---------------------------------------------------------------

print()
print("PHAN 2 — 2 file .npy cua MobiVital")
print("-" * 60)

uwb_dev, gt_dev = read_mobivital_npy(
    MOBIVITAL_DIR + "/training_breath_tripod_data.npy")

uwb_test, gt_test = read_mobivital_npy(
    MOBIVITAL_DIR + "/testing_breath_tripod_data.npy")

print("file DEV  | uwb", uwb_dev.shape, "| gt", gt_dev.shape,
      "| gt tu", gt_dev.min(), "den", gt_dev.max())

print("file TEST | uwb", uwb_test.shape, "| gt", gt_test.shape,
      "| gt tu", gt_test.min(), "den", gt_test.max())


# ---------------------------------------------------------------
# PHẦN 3 — so hai bên
# ---------------------------------------------------------------

print()
print("PHAN 3 — So hai ben")
print("-" * 60)

mobivital_dev = len(gt_dev)
mobivital_test = len(gt_test)

print("DEV :  cua minh =", total_dev, "| MobiVital =", mobivital_dev)
print("TEST:  cua minh =", total_test, "| MobiVital =", mobivital_test)

if total_dev != mobivital_dev:
    print("-> SO SESSION DEV KHAC NHAU, phai xem lai!")
elif total_test != mobivital_test:
    print("-> SO SESSION TEST KHAC NHAU, phai xem lai!")
else:
    print("-> So session GIONG NHAU")


# So từng con số. Lấy session đầu tiên của người A, đi tìm nó bên MobiVital.
# Hai bên xếp session theo thứ tự khác nhau nên phải dò từng dòng.

print()
print("Tim session dau cua nguoi A trong file MobiVital...")

data_a = np.load(BY_USER_DIR + "/A.npz")
first_session_a = data_a["gt"][0]

found_row = -1

for i in range(mobivital_dev):
    largest_diff = np.abs(gt_dev[i] - first_session_a).max()

    if largest_diff < 0.001:
        found_row = i
        print("Tim thay o dong", i)
        print("Khac nhau nhieu nhat tren 1500 mau:", largest_diff)
        break

if found_row == -1:
    print("KHONG tim thay — hai pipeline khac nhau, phai xem lai!")
