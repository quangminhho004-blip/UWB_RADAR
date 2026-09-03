"""VÁ CODE MOBIVITAL: thêm `model.eval()` vào inference/mobivital_gen.py.

    python scripts/mobivital/patch_eval.py

ĐÂY LÀ CHỖ DUY NHẤT ĐỒ ÁN SỬA CODE CỦA TÁC GIẢ. Sửa đúng MỘT dòng.

VÌ SAO PHẢI VÁ

`inference/mobivital_gen.py` nạp trọng số rồi gọi `model.to(device)` mà KHÔNG gọi
`model.eval()`. LSTM ở chế độ train nên cuDNN cấp thêm vùng nhớ dự trữ cho
backward, dù cả hàm nằm trong `torch.no_grad()`.

Bước chọn kênh đưa cả một buổi ghi vào LSTM một lần: 129 ứng viên sống sót qua
`invert_detector` x 52 cửa sổ = 6708 chuỗi, gấp 105 lần batch lúc train (64).

    dự trữ backward = 2 lớp x 4 cổng x 6708 x 200 x 352 x 4 byte = 15.1 GB
    + tensor đầu ra 1.89 GB + workspace                          ~ 17-21 GB

Đo thật trên Colab, cùng buổi ghi đầu tiên:

    T4  15 GB -> CUDA out of memory, xin 16.95 GB
    L4  22 GB -> CUDA out of memory, xin 21.00 GB
    cờ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> vẫn tràn

Ở chế độ eval, cuDNN KHÔNG cấp vùng dự trữ đó — tổng còn khoảng 3 GB.

VÌ SAO TÁC GIẢ CHẠY ĐƯỢC MÀ MÌNH KHÔNG

Bài báo ghi họ dùng GTX 1080 Ti, chỉ 11 GB VRAM — còn ít hơn L4. `requirements.txt`
của họ ghim `torch==2.3.0`; bản đó bỏ qua vùng dự trữ khi đang trong
`torch.no_grad()`. PyTorch hiện hành trên Colab (Python 3.13) cấp theo cờ
`model.training`, không quan tâm `no_grad`. Không cài lại `torch==2.3.0` được vì
bản đó không có wheel cho Python 3.13.

VÁ NÀY CÓ ĐỔI KẾT QUẢ KHÔNG — KHÔNG

`LSTMMultiStep` chỉ gồm `nn.LSTM(dropout=0)` và `nn.Linear`. Không Dropout, không
BatchNorm. Hai chế độ train và eval cho forward GIỐNG HỆT nhau. `.eval()` chỉ đổi
cách cuDNN xin bộ nhớ.

CHẠY LẠI ĐƯỢC

Đã vá rồi thì bỏ qua. Nội dung gốc khác dự kiến thì dừng hẳn, không vá mù.
"""

import os
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

TARGET = "external/mobivital/inference/mobivital_gen.py"

# Đoạn gốc của tác giả, phải khớp từng ký tự thì mới vá.
BEFORE = """    model.to(device)
    inference(model)"""

# Đoạn sau khi vá. Chỉ thêm model.eval() và chú thích nói rõ vì sao.
AFTER = """    model.to(device)
    # === SỬA BỞI ĐỒ ÁN — thêm đúng dòng model.eval() dưới đây ===
    # Bản gốc không gọi .eval() nên LSTM ở chế độ train; cuDNN cấp thêm vùng nhớ
    # dự trữ cho backward (15.1 GB cho lô 6708 chuỗi), tràn VRAM mọi GPU Colab.
    # Model này chỉ có nn.LSTM(dropout=0) + nn.Linear nên train và eval cho
    # forward giống hệt nhau — vá này không đổi kết quả, chỉ đổi cách xin bộ nhớ.
    # Chi tiết: scripts/mobivital/patch_eval.py
    model.eval()
    # === HẾT PHẦN SỬA ===
    inference(model)"""

# ===============================================================


if not os.path.exists(TARGET):
    sys.exit("không thấy " + TARGET + " — chạy scripts/setup_colab.py trước")

source = open(TARGET).read()

if "SỬA BỞI ĐỒ ÁN" in source:
    print("đã vá rồi, bỏ qua:", TARGET)
    raise SystemExit(0)

if BEFORE not in source:
    sys.exit("KHÔNG vá được " + TARGET + ":\n"
             "không tìm thấy đoạn gốc dự kiến. Có thể upstream đã đổi.\n"
             "Kiểm tra lại rồi sửa BEFORE/AFTER trong script này.")

open(TARGET, "w").write(source.replace(BEFORE, AFTER, 1))

print("đã vá", TARGET)
print("  thêm model.eval() ngay sau model.to(device) trong hàm main()")
print()
print("Đoạn sau khi vá:")
for i, line in enumerate(open(TARGET).read().split("\n")):
    if "model.to(device)" in line or "SỬA BỞI ĐỒ ÁN" in line:
        start = i
        break
lines = open(TARGET).read().split("\n")
for line in lines[start:start + 10]:
    print("   ", line)
