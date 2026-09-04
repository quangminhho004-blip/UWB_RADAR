"""Ghi đè các tệp MobiVital đã sửa sẵn, chạy ngay sau khi clone repo họ.

    python scripts/mobivital/apply_patched_files.py

ĐÂY LÀ CHỖ DUY NHẤT ĐỒ ÁN SỬA CODE CỦA TÁC GIẢ. Đúng MỘT tệp, MỘT dòng lệnh.

    scripts/mobivital/patched/mobivital_gen.py
        -> external/mobivital/inference/mobivital_gen.py

Bản trong `patched/` là bản gốc của tác giả cộng thêm một dòng `model.eval()`,
kèm khối chú thích `=== SỬA BỞI ĐỒ ÁN ===` bao quanh để nhìn là thấy ngay.

VÌ SAO PHẢI SỬA

`inference/mobivital_gen.py` nạp trọng số rồi gọi `model.to(device)` mà KHÔNG
gọi `model.eval()`. LSTM ở chế độ train nên cuDNN cấp thêm vùng nhớ dự trữ cho
backward, dù cả hàm nằm trong `torch.no_grad()`.

Bước chọn kênh đưa cả một buổi ghi vào LSTM một lần: 129 ứng viên sống sót qua
`invert_detector` x 52 cửa sổ = 6708 chuỗi, gấp 105 lần batch lúc train (64).

    dự trữ backward = 2 lớp x 4 cổng x 6708 x 200 x 352 x 4 byte = 15.1 GB

Đo thật trên Colab, cùng buổi ghi đầu tiên:

    T4  15 GB               -> tràn, xin 16.95 GB
    L4  22 GB               -> tràn, xin 21.00 GB
    L4 + expandable_segments-> vẫn tràn, cần thật ~33 GB
    L4 + ghi đè tệp này     -> chạy trọn 1874 buổi ghi trong 17 phút

Bài báo ghi tác giả dùng GTX 1080 Ti, chỉ 11 GB VRAM — còn ít hơn L4 — mà vẫn
chạy được. `requirements.txt` của họ ghim `torch==2.3.0`; bản đó bỏ qua vùng dự
trữ khi đang trong `torch.no_grad()`. PyTorch hiện hành cấp theo cờ
`model.training`, không quan tâm `no_grad`. Không cài lại `torch==2.3.0` được vì
bản đó không có wheel cho Python 3.13.

Thời gian bước chọn kênh: CPU 1 giờ 40 -> GPU sau khi sửa 17 phút.

CÓ ĐỔI KẾT QUẢ KHÔNG — KHÔNG

`LSTMMultiStep` chỉ gồm `nn.LSTM(dropout=0)` và `nn.Linear`. Không Dropout, không
BatchNorm. Hai chế độ train và eval cho forward GIỐNG HỆT. `.eval()` chỉ đổi cách
cuDNN xin bộ nhớ.

KIỂM TRƯỚC KHI GHI ĐÈ

Script so mã băm tệp gốc với mã đã ghi lại. Khác thì dừng hẳn, không ghi đè mù —
upstream đổi code thì bản `patched/` không còn dùng được.
"""

import hashlib
import os
import shutil
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

# Bản gốc của tác giả tại commit 4319731d. Ghi đè lên bản khác là sai.
PATCHES = {
    "scripts/mobivital/patched/mobivital_gen.py":
        ("external/mobivital/inference/mobivital_gen.py",
         "c312481ab7088be8685ee7f20e33377c"),
}

DAU_HIEU = "SỬA BỞI ĐỒ ÁN"

# ===============================================================


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


for nguon, (dich, md5_goc) in PATCHES.items():
    if not os.path.exists(dich):
        sys.exit("không thấy " + dich + " — chạy scripts/setup_colab.py trước")
    if not os.path.exists(nguon):
        sys.exit("không thấy " + nguon)

    if DAU_HIEU in open(dich).read():
        print("đã ghi đè rồi, bỏ qua:", dich)
        continue

    hien_tai = md5(dich)
    if hien_tai != md5_goc:
        sys.exit("KHÔNG ghi đè " + dich + ":\n"
                 "  mã băm hiện tại %s\n"
                 "  mã băm dự kiến  %s\n"
                 "Upstream đã đổi tệp này. Phải dựng lại bản trong patched/ rồi "
                 "cập nhật mã băm trong script này." % (hien_tai, md5_goc))

    shutil.copy(nguon, dich)
    print("ghi đè", dich)
    print("   nguồn:", nguon)

    lines = open(dich).read().split("\n")
    for i, line in enumerate(lines):
        if DAU_HIEU in line:
            print()
            print("   Phần sửa:")
            for l in lines[i - 1:i + 9]:
                print("     " + l)
            break
