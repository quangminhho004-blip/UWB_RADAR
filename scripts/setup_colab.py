"""Chuẩn bị môi trường Colab SAU KHI repo đồ án đã được clone.

    !git clone -q https://github.com/quangminhho004-blip/UWB_RADAR.git /content/UWB_RADAR
    %cd /content/UWB_RADAR
    !python scripts/setup_colab.py

VÌ SAO SCRIPT NÀY KHÔNG TỰ CLONE REPO ĐỒ ÁN

Nó nằm bên trong chính repo đó. Colab mới mở chỉ có /content, chưa có
/content/UWB_RADAR/scripts/setup_colab.py — chưa tải repo về thì chưa gọi được
script. Nên hai lệnh `git clone` và `%cd` phải nằm trong notebook, trước lệnh
gọi script này.

`%cd` cũng bắt buộc phải ở notebook: `os.chdir()` bên trong script chỉ đổi thư
mục của tiến trình con, ô lệnh sau vẫn đứng ở chỗ cũ.

BA VIỆC SCRIPT NÀY LÀM

    1. clone repo MobiVital rồi GHIM đúng commit dùng cho mọi số liệu.
       Repo họ không có LICENSE nên không chép vào repo đồ án, phải clone riêng
       mỗi phiên Colab.
    2. cài thư viện thiếu (einops)

runs/ là thư mục THẬT, không phải lối tắt vào Drive. Lý do: thư mục lối tắt thì
git không track được tệp bên trong, mà bằng chứng thực nghiệm (bảng lựa chọn
kênh, điểm từng buổi ghi) cần commit lên GitHub cho hội đồng xem. Muốn giữ kết
quả sau khi Colab ngắt phiên thì chạy `python scripts/save_results.py <tên>` rồi
tải tệp .zip về.

Lệnh nào lỗi là dừng hẳn, không chạy tiếp sang bước sau.
"""

import os
import subprocess
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

MOBIVITAL_DIR = "external/mobivital"
MOBIVITAL_URL = "https://github.com/nesl/mobivital-public.git"
MOBIVITAL_COMMIT = "4319731d2769d4134c92088dd846666e262f18e9"

# ===============================================================


def run(cmd):
    """Chạy lệnh, dừng hẳn nếu lỗi."""
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit("DỪNG — lệnh lỗi: " + cmd)


def grab(cmd):
    """Chạy lệnh, trả về output. Dừng hẳn nếu lỗi."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("DỪNG — lệnh lỗi: " + cmd + "\n" + r.stderr)
    return r.stdout.strip()


def grab_soft(cmd):
    """Như grab nhưng không dừng — dùng cho thông tin phụ, ví dụ tên GPU."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


# Phải đứng ở thư mục gốc repo đồ án.
if not os.path.isdir("scripts") or not os.path.isdir("src"):
    sys.exit("Phải chạy từ thư mục gốc repo đồ án.\n"
             "  !git clone -q <repo> /content/UWB_RADAR\n"
             "  %cd /content/UWB_RADAR\n"
             "  !python scripts/setup_colab.py")


# --- 1. Mã nguồn MobiVital, ghim đúng commit ---

if not os.path.isdir(MOBIVITAL_DIR + "/.git"):
    run("git clone -q " + MOBIVITAL_URL + " " + MOBIVITAL_DIR)

run("git -C %s checkout -q %s" % (MOBIVITAL_DIR, MOBIVITAL_COMMIT))


# --- 2. Thư viện ---

run("pip install -q einops")


os.makedirs("runs", exist_ok=True)


# --- Thông tin phiên chạy ---

print()
print("thư mục làm việc :", os.getcwd())
print("commit đồ án     :", grab("git rev-parse --short HEAD"))
print("commit MobiVital :", grab("git -C %s rev-parse --short HEAD" % MOBIVITAL_DIR),
      "(đã ghim)")
print("GPU              :",
      grab_soft("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
      or "không có (chạy CPU)")
