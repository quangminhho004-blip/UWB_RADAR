"""BƯỚC 2 — Đọc CSV của từng người, lưu thành một file .npz.

    python scripts/2_make_npz.py

Trước:  data/raw/A/*.csv
Sau:    data/processed/by_user/A.npz   gồm 2 mảng:
            uwb  -- tín hiệu radar dạng số phức, (số_session, 1500, 120)
            gt   -- respiration ground truth,     (số_session, 1500), đã chuẩn hoá [-1, 1]

Mỗi file CSV là một session, có 1500 dòng.
    cột  12..131  = phần thực của UWB   (120 kênh)
    cột 132..251  = phần ảo  của UWB
    cột áp chót   = respiration ground truth

Đây là PIPELINE TRAIN/VALIDATE (xem docs/PROTOCOL.md mục 1).
"""

import csv
import glob
import os

import numpy as np

RAW_DIR = "data/raw"
OUT_DIR = "data/processed/by_user"

users = ["A", "B", "C", "D", "E", "F",
         "G", "H", "I", "J", "K", "L"]


def self_normalize(mat):
    """Kéo giãn dãy số về khoảng [-1, 1]. Dãy hằng số thì trả về toàn 0.

    Chép nguyên công thức của MobiVital (hàm self_normalize trong
    prep_breath_final.py) để hai pipeline lưu ra giá trị giống hệt nhau.
    """
    max_val = np.amax(mat)
    min_val = np.amin(mat)

    if max_val == min_val:
        return np.zeros(mat.shape)

    return (mat - min_val) / (max_val - min_val) * 2 - 1


def read_one_csv(path):
    """Đọc một file CSV. Trả về (uwb, gt) hoặc None nếu file không đúng 1500 dòng.

    Đọc y hệt cách MobiVital làm trong prep_breath_final.py: dùng csv.reader rồi
    ép sang float32 NGAY, sau đó mới chuẩn hoá. Nếu đọc ở float64 rồi mới ép
    xuống thì kết quả lệch với họ ở chữ số thứ 8 -- không ảnh hưởng gì tới model,
    nhưng làm giống hệt để khỏi phải giải trình.
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader)).astype(np.float32)

    if data.ndim != 2 or len(data) != 1500 or data.shape[1] < 254:
        return None

    real = data[:, 12:132]
    imag = data[:, 132:252]
    uwb = real + 1j * imag

    # Chuẩn hoá về [-1, 1], giống MobiVital. Giá trị thô là đơn vị cảm biến
    # tuỳ tiện (khoảng 1465..4738), không mang ý nghĩa gì.
    gt = self_normalize(data[:, -2])

    return uwb.astype(np.complex64), gt.astype(np.float32)


os.makedirs(OUT_DIR, exist_ok=True)

for user in users:
    csv_files = sorted(glob.glob(RAW_DIR + "/" + user + "/*.csv"))

    uwb_list = [] # raw signal dạng số phức (complex-valued I/Q samples)
    gt_list = [] # Ground truth
    for path in csv_files:
        result = read_one_csv(path)
        if result is None:
            continue
        uwb, gt = result
        uwb_list.append(uwb)
        gt_list.append(gt)

    uwb_all = np.stack(uwb_list)
    gt_all = np.stack(gt_list)

    out_file = OUT_DIR + "/" + user + ".npz"
    np.savez(out_file, uwb=uwb_all, gt=gt_all)

    print("user", user, "->", out_file,
          "| uwb", uwb_all.shape, "| gt", gt_all.shape)

print("Xong")
