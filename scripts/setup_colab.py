"""Dựng môi trường Colab: lấy mã nguồn, ghim commit MobiVital, nối runs/ vào Drive.

    python scripts/setup_colab.py

Chạy sau khi đã `drive.mount("/content/drive")` trong notebook.

BA VIỆC

    1. clone hoặc cập nhật repo đồ án
    2. clone repo MobiVital rồi GHIM đúng commit dùng cho mọi số liệu.
       Repo họ không có LICENSE nên không chép vào repo đồ án, phải clone riêng
       mỗi phiên Colab.
    3. biến runs/ thành lối tắt vào Drive — Colab hay ngắt phiên, checkpoint ghi
       vào đó thì phiên sau chạy tiếp được (xem src/training.py).

Lệnh nào lỗi là dừng hẳn, không chạy tiếp sang bước sau.
"""

import os
import subprocess
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

REPO = "/content/UWB_RADAR"
REPO_URL = "https://github.com/quangminhho004-blip/UWB_RADAR.git"

MOBIVITAL_URL = "https://github.com/nesl/mobivital-public.git"
MOBIVITAL_COMMIT = "4319731d2769d4134c92088dd846666e262f18e9"

DRIVE = "/content/drive/MyDrive/mobivital"

# ===============================================================


def run(cmd, cwd=None):
    """Chạy lệnh, dừng hẳn nếu lỗi."""
    if subprocess.run(cmd, shell=True, cwd=cwd).returncode != 0:
        sys.exit("DỪNG — lệnh lỗi: " + cmd)


def grab(cmd, cwd=None):
    """Chạy lệnh, trả về output."""
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("DỪNG — lệnh lỗi: " + cmd + "\n" + r.stderr)
    return r.stdout.strip()


# --- 1. Mã nguồn đồ án ---

if os.path.isdir(REPO + "/.git"):
    run("git pull -q origin main", cwd=REPO)
else:
    run("git clone -q " + REPO_URL + " " + REPO)

os.chdir(REPO)


# --- 2. Mã nguồn MobiVital, ghim đúng commit ---

mobivital = "external/mobivital"
if not os.path.isdir(mobivital + "/.git"):
    run("git clone -q " + MOBIVITAL_URL + " " + mobivital)

run("git -C %s checkout -q %s" % (mobivital, MOBIVITAL_COMMIT))
run("pip install -q einops")


# --- 3. runs/ trỏ vào Drive ---

if os.path.isdir(DRIVE):
    os.makedirs(DRIVE + "/runs", exist_ok=True)
    if not os.path.islink("runs"):
        run("rm -rf runs")
        os.symlink(DRIVE + "/runs", "runs")
    print("runs/ -> " + os.readlink("runs"))
else:
    os.makedirs("runs", exist_ok=True)
    print("chưa mount Drive — runs/ nằm trong máy ảo, mất khi Colab ngắt phiên")


# --- Thông tin phiên chạy ---

print()
print("thư mục làm việc :", os.getcwd())
print("commit đồ án     :", grab("git rev-parse --short HEAD"))
print("commit MobiVital :", grab("git -C %s rev-parse --short HEAD" % mobivital),
      "(đã ghim)")
print("GPU              :",
      grab("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
      or "không có")
