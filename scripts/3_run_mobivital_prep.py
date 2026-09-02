"""BƯỚC 3 — Chạy script chuẩn bị dữ liệu CỦA MOBIVITAL, nguyên bản, 0 dòng sửa.

    python scripts/3_run_mobivital_prep.py

Đây là PIPELINE TEST (xem docs/PROTOCOL.md mục 1). Kết quả dùng cho TN0
(reproduce LSTM) và cho bảng so sánh cuối.

Script MobiVital dùng đường dẫn tương đối:
    rootdir     = "./dataset/mobivital/tripod/"     <- đọc từ đây
    save_folder = "./data_final/"                   <- ghi vào đây
nên chỉ cần dựng đúng cấu trúc đó rồi đứng trong thư mục làm việc mà gọi. Không
đụng một dòng nào trong file của MobiVital.

Việc script này làm:
    1. Dựng thư mục tạm _workdir/ có dataset/mobivital/tripod/ chứa SYMLINK
       (lối tắt) tới toàn bộ CSV trong data/raw/A/ ... data/raw/L/.
       Symlink gần như không tốn dung lượng; script MobiVital cần một thư mục
       phẳng vì nó gọi os.listdir().
    2. Đứng trong _workdir/ và chạy prep_breath_final.py của MobiVital.
    3. Chuyển 2 file .npy nó sinh ra lên thư mục kết quả, xoá thư mục tạm.

Kết quả (định dạng của MobiVital, khác .npz của mình):
    data/processed/mobivital_original/training_breath_tripod_data.npy   8 người ABCDEFKL
    data/processed/mobivital_original/testing_breath_tripod_data.npy    4 người GHIJ

Mỗi file .npy chứa HAI mảng ghi nối tiếp nhau: X_uwb (số phức) rồi y_breath
(đã chuẩn hoá về [-1, 1]). Đọc lại phải gọi np.load hai lần trên cùng file.
"""

import glob
import os
import shutil
import subprocess
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

RAW_DIR = "data/raw"
OUT_DIR = "data/processed/mobivital_original"
MOBIVITAL_SCRIPT = "external/mobivital/dataset_preparation/prep_breath_final.py"

# ===============================================================


OUTPUT_NAMES = ["training_breath_tripod_data.npy",
                "testing_breath_tripod_data.npy"]

# Thư mục tạm để chạy script MobiVital. Xoá sau khi xong.
work_dir = OUT_DIR + "/_workdir"
flat_dir = work_dir + "/dataset/mobivital/tripod"


# --- Bước 1: dựng thư mục phẳng bằng symlink ---

csv_files = glob.glob(RAW_DIR + "/*/*.csv")

if len(csv_files) == 0:
    print("LỖI: không tìm thấy file CSV nào trong", RAW_DIR + "/*/")
    print("Chạy scripts/1_organize_raw.py trước.")
    sys.exit(1)

print("Tìm thấy", len(csv_files), "file CSV")

os.makedirs(flat_dir, exist_ok=True)

for path in csv_files:
    file_name = os.path.basename(path)
    link_path = flat_dir + "/" + file_name

    if os.path.lexists(link_path):
        continue

    # Dùng đường dẫn tuyệt đối để lối tắt không phụ thuộc chỗ đang đứng.
    real_path = os.path.abspath(path)
    os.symlink(real_path, link_path)

print("Đã dựng", len(csv_files), "symlink trong", flat_dir)


# --- Bước 2: chạy script MobiVital, nguyên bản ---

if not os.path.isfile(MOBIVITAL_SCRIPT):
    print("LỖI: không thấy", MOBIVITAL_SCRIPT)
    print("Clone upstream trước:")
    print("  git clone https://github.com/nesl/mobivital-public.git external/mobivital")
    sys.exit(1)

mobivital_script_abs = os.path.abspath(MOBIVITAL_SCRIPT)

print()
print("Chạy script MobiVital (không sửa dòng nào):")
print("  cd", work_dir)
print("  python", MOBIVITAL_SCRIPT)
print()

# cwd=work_dir để các đường dẫn "./..." trong script MobiVital trỏ đúng chỗ.
subprocess.run([sys.executable, mobivital_script_abs], cwd=work_dir, check=True)


# --- Bước 3: chuyển kết quả ra ngoài, dọn thư mục tạm ---

print()
print("Xong. Kết quả:")

for name in OUTPUT_NAMES:
    made_by_mobivital = work_dir + "/data_final/" + name
    final_path = OUT_DIR + "/" + name

    shutil.move(made_by_mobivital, final_path)

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print("  " + final_path, "(%.0f MB)" % size_mb)

shutil.rmtree(work_dir)
print()
print("Đã xoá thư mục tạm", work_dir)
