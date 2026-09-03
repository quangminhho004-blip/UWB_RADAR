"""Cắt sẵn cửa sổ để train, lưu ra file.

    python scripts/make_windows.py

Luồng:  đọc dữ liệu -> gọi generate_dataset() của MobiVital -> lấy X, y -> lưu .npz

Chạy hai lần, cho hai pipeline (xem docs/PROTOCOL.md mục 1):

    dev_cv/      cắt RIÊNG từng người A B C D E F K L, để ghép fold tuỳ ý
    final_train/ cắt GỘP cả 8 người, đúng thứ tự MobiVital, cho số công bố

CHỈ CẮT PHẦN TRAIN. Lúc chấm điểm, cửa sổ được cắt tại chỗ bên trong hàm
get_best_sequence của MobiVital vì lúc đó phải quét cả 120 bin (không phải
9 bin như lúc train), khoảng 5500 cửa sổ mỗi session. Cắt sẵn cho GHIJ sẽ
tốn 2.5 GB, to hơn cả dữ liệu thô.

LƯU Ý KHI DÙNG CHO CV: trong một fold, chỉ dùng file cửa sổ của 6 người
train. Cửa sổ của 2 người validation KHÔNG được dùng để chấm điểm, vì
việc chọn sóng ở đây có nhìn nhịp thở thật (ngưỡng corr > 0.9). Chấm điểm
phải đọc session thô của 2 người đó rồi cho model tự chọn sóng, không nhìn
nhịp thở thật. Nhìn vào là rò rỉ, cấu hình chọn ra sẽ sai.

generate_dataset của MobiVital làm 2 việc:
    1. Chọn sóng đáng học: quét bin 20..28, mỗi bin 4 phép biến đổi
       (abs, real, imag, phase) = 36 ứng viên, giữ sóng nào giống nhịp thở
       thật hơn ngưỡng. Luôn thêm cả sóng nhịp thở thật của session đó vào.
    2. Cắt mỗi sóng đã chọn: 200 mẫu vào -> 25 mẫu đáp án, trượt 25,
       ra 52 cửa sổ mỗi sóng.

    Chú ý: 4 phép là lúc TRAIN (training/utils/model_utils.py:29).
    Lúc INFERENCE chỉ có 2 phép abs và phase (utils/model_utils.py:30).
    Hai file trùng tên nhưng nội dung khác nhau.
"""

import os
import sys
import time

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

MOBIVITAL_DIR = "external/mobivital"

BY_USER_DIR = "data/processed/by_user"
MOBIVITAL_NPY = "external/mobivital/data_final/training_breath_tripod_data.npy"

OUT_DIR = "data/processed/windows"

# 8 người dùng để phát triển model. G H I J để dành, không train trên đó.
USERS = ["A", "B", "C", "D", "E", "F", "K", "L"]

# Ngưỡng lọc sóng đáng học. 0.9 là giá trị MobiVital công bố.
# TN4 sẽ quét thêm: thêm số vào danh sách này rồi chạy lại.
THRESHOLDS = [0.9]

HISTORY_LENGTH = 200      # số mẫu đưa vào model
FUTURE_LENGTH = 25        # số mẫu model phải đoán

# ===============================================================


# Cho Python biết tìm code MobiVital ở đâu, rồi mượn hàm cắt cửa sổ của họ.
sys.path.append(os.path.abspath(MOBIVITAL_DIR))
from training.utils.model_utils import generate_dataset


def file_name(prefix, threshold):
    """Tên file ghi đủ điều kiện cắt, để sau này đổi tham số không bị dùng nhầm.

    Ví dụ:  A_corr0.9_h200_f25.npz
    """
    return (prefix
            + "_corr" + str(threshold)
            + "_h" + str(HISTORY_LENGTH)
            + "_f" + str(FUTURE_LENGTH)
            + ".npz")


def cut_windows(uwb, gt, threshold):
    """Cắt một mớ session thành cửa sổ. Trả về hai mảng X và y.

    uwb  -- tín hiệu radar dạng số phức, (số_session, 1500, 120)
    gt   -- nhịp thở thật,               (số_session, 1500)
    """
    # Số 64 là batch_size. Hàm của MobiVital đòi tham số này nhưng
    # bên trong không dùng tới, nên điền gì cũng được.
    dataset = generate_dataset(uwb, gt, 64,
                               HISTORY_LENGTH, FUTURE_LENGTH, threshold)
    return dataset.x.numpy(), dataset.y.numpy()


def save(path, X, y):
    """Ghi hai mảng ra file .npz rồi in một dòng cho biết."""
    opened_file = open(path, "wb")
    np.savez(opened_file, X=X, y=y)
    opened_file.close()

    size_mb = os.path.getsize(path) / 1024 / 1024
    print("   ", path, "| X", X.shape, "| %.0f MB" % size_mb)


def read_mobivital_npy(path):
    """Đọc file .npy của MobiVital.

    File của họ chứa hai mảng ghi nối tiếp nhau trong cùng một file,
    nên phải gọi np.load hai lần trên cùng một file đang mở.
    """
    opened_file = open(path, "rb")
    uwb = np.load(opened_file)
    gt = np.load(opened_file)
    opened_file.close()
    return uwb, gt


def make_dev_cv_windows():
    """Pipeline DEV: cắt riêng từng người, để ghép 4 fold tuỳ ý.

        fold 1  train = C+D+E+F+K+L   val = A+B
        fold 2  train = A+B+D+F+K+L   val = C+E
        fold 3  train = A+B+C+E+K+L   val = D+F
        fold 4  train = A+B+C+D+E+F   val = K+L

    Ghép được vì hàm của MobiVital xử lý từng session độc lập: cắt riêng
    8 người rồi ghép lại cho ra đúng tập cửa sổ như cắt một lần.
    """
    out = OUT_DIR + "/dev_cv"
    os.makedirs(out, exist_ok=True)

    print("PHẦN 1 — pipeline DEV, cắt riêng từng người")
    print("-" * 70)

    for threshold in THRESHOLDS:
        print("ngưỡng", threshold)

        for user in USERS:
            start = time.time()

            data = np.load(BY_USER_DIR + "/" + user + ".npz")
            X, y = cut_windows(data["uwb"], data["gt"], threshold)

            save(out + "/" + file_name(user, threshold), X, y)
            print("      mất %.0f giây" % (time.time() - start))


def make_final_train_windows():
    """Pipeline GỐC: cắt gộp cả 8 người, cho bảng kết quả công bố.

    Đọc thẳng file .npy do chính `dataset_preparation/prep_breath_final.py`
    của MobiVital sinh ra, nên thứ tự session giống hệt lúc MobiVital chạy,
    không phải ghép lại từ 8 file ở phần trên.
    """
    out = OUT_DIR + "/final_train"
    os.makedirs(out, exist_ok=True)

    print()
    print("PHẦN 2 — pipeline GỐC, cắt gộp 8 người")
    print("-" * 70)

    uwb_all, gt_all = read_mobivital_npy(MOBIVITAL_NPY)
    print("đọc", MOBIVITAL_NPY, "->", len(gt_all), "session")

    for threshold in THRESHOLDS:
        start = time.time()

        X, y = cut_windows(uwb_all, gt_all, threshold)

        save(out + "/" + file_name("train", threshold), X, y)
        print("      mất %.0f giây" % (time.time() - start))


def main():
    make_dev_cv_windows()
    make_final_train_windows()
    print()
    print("Xong. Kết quả ở", OUT_DIR)


if __name__ == "__main__":
    main()
