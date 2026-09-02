"""BƯỚC 4 — Xem lại dữ liệu đã tạo có đúng không.

    python scripts/4_check_data.py

Script này chỉ ĐỌC dữ liệu rồi IN ra màn hình. Không sửa gì cả.

In 3 phần:
    1. 12 file .npz  (pipeline của mình)
    2. 2 file .npy   (pipeline MobiVital)
    3. So hai bên xem có giống nhau không
"""

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

BY_USER_DIR = "data/processed/by_user"
MOBIVITAL_DIR = "data/processed/mobivital_original"

# 8 người dùng để phát triển model
NGUOI_DEV = ["A", "B", "C", "D", "E", "F", "K", "L"]

# 4 người để dành, chỉ chấm điểm ở bước cuối cùng
NGUOI_TEST = ["G", "H", "I", "J"]

# ===============================================================


def in_thong_tin_mot_nguoi(user):
    """Đọc file .npz của một người, in ra màn hình, trả về số session."""
    duong_dan = BY_USER_DIR + "/" + user + ".npz"
    du_lieu = np.load(duong_dan)

    uwb = du_lieu["uwb"]
    gt = du_lieu["gt"]

    so_session = len(gt)
    gt_nho_nhat = gt.min()
    gt_lon_nhat = gt.max()

    print("nguoi", user,
          "|", so_session, "session",
          "| uwb", uwb.shape,
          "| gt", gt.shape,
          "| gt tu", gt_nho_nhat, "den", gt_lon_nhat)

    return so_session


def doc_file_mobivital(duong_dan):
    """Đọc file .npy của MobiVital.

    File của họ chứa 2 mảng ghi nối tiếp nhau, nên phải gọi np.load hai lần
    trên cùng một file đang mở.
    """
    file_dang_mo = open(duong_dan, "rb")
    uwb = np.load(file_dang_mo)
    gt = np.load(file_dang_mo)
    file_dang_mo.close()

    return uwb, gt


# ---------------------------------------------------------------
# PHẦN 1 — đọc 12 file .npz của mình
# ---------------------------------------------------------------

print("PHAN 1 — 12 file .npz cua minh")
print("-" * 60)

tong_session_dev = 0

for user in NGUOI_DEV:
    so_session = in_thong_tin_mot_nguoi(user)
    tong_session_dev = tong_session_dev + so_session

print()

tong_session_test = 0

for user in NGUOI_TEST:
    so_session = in_thong_tin_mot_nguoi(user)
    tong_session_test = tong_session_test + so_session

print()
print("8 nguoi DEV  cong lai =", tong_session_dev, "session")
print("4 nguoi TEST cong lai =", tong_session_test, "session")


# ---------------------------------------------------------------
# PHẦN 2 — đọc 2 file .npy của MobiVital
# ---------------------------------------------------------------

print()
print("PHAN 2 — 2 file .npy cua MobiVital")
print("-" * 60)

uwb_dev, gt_dev = doc_file_mobivital(
    MOBIVITAL_DIR + "/training_breath_tripod_data.npy")

uwb_test, gt_test = doc_file_mobivital(
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

so_session_mobivital_dev = len(gt_dev)
so_session_mobivital_test = len(gt_test)

print("DEV :  cua minh =", tong_session_dev,
      "| MobiVital =", so_session_mobivital_dev)

print("TEST:  cua minh =", tong_session_test,
      "| MobiVital =", so_session_mobivital_test)

if tong_session_dev != so_session_mobivital_dev:
    print("-> SO SESSION DEV KHAC NHAU, phai xem lai!")
elif tong_session_test != so_session_mobivital_test:
    print("-> SO SESSION TEST KHAC NHAU, phai xem lai!")
else:
    print("-> So session GIONG NHAU")


# So từng con số. Lấy session đầu tiên của người A, đi tìm nó bên MobiVital.
# Hai bên xếp session theo thứ tự khác nhau nên phải dò từng dòng.

print()
print("Tim session dau cua nguoi A trong file MobiVital...")

du_lieu_cua_A = np.load(BY_USER_DIR + "/A.npz")
session_dau_cua_A = du_lieu_cua_A["gt"][0]

dong_tim_thay = -1

for i in range(so_session_mobivital_dev):
    session_ben_mobivital = gt_dev[i]
    khac_nhau_nhieu_nhat = np.abs(session_ben_mobivital - session_dau_cua_A).max()

    if khac_nhau_nhieu_nhat < 0.001:
        dong_tim_thay = i
        print("Tim thay o dong", i)
        print("Khac nhau nhieu nhat tren 1500 mau:", khac_nhau_nhieu_nhat)
        break

if dong_tim_thay == -1:
    print("KHONG tim thay — hai pipeline khac nhau, phai xem lai!")
