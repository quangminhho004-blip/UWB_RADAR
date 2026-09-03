"""Cất tệp lựa chọn kênh mà `mobivital_gen.py` vừa sinh, rồi khôi phục tệp gốc.

    cd external/mobivital
    python ../../scripts/mobivital/save_txt.py TN0b

VÌ SAO CẦN

`mobivital_gen.py` dòng 121 luôn ghi ra đúng một tên:

    inference/methods/tripod_mobivital_pre_invert_0.9.txt

Tên đó trùng với một tệp **đã có sẵn trong repo tác giả** (`git ls-files` thấy
nó). Chạy `mobivital_gen.py` là ghi đè lên tệp của họ. Chạy lần hai (TN0c) lại
đè tiếp lần một (TN0b).

Script này làm ba việc, chạy ngay sau mỗi lần `mobivital_gen.py`:

    1. chép tệp vừa sinh -> inference/methods/<tên>.txt   (để evaluate.py đọc)
    2. chép tệp vừa sinh -> results/<tên>.txt             (để đối chiếu về sau)
    3. git checkout khôi phục tệp gốc của tác giả

Nhờ vậy `git -C external/mobivital status` vẫn trống — kiểm ở cuối mục 3 của
notebooks/TN0.ipynb. Tệp `<tên>.txt` tạm trong `inference/methods/` đã được
`scripts/mobivital/setup_dataset.py` cho vào `.git/info/exclude`.

CHẠY TỪ ĐÂU

Từ trong `external/mobivital/` — đúng chỗ các lệnh của tác giả chạy.
"""

import os
import shutil
import subprocess
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

# Tên cố định mà mobivital_gen.py luôn ghi ra.
GENERATED = "inference/methods/tripod_mobivital_pre_invert_0.9.txt"

METHODS_DIR = "inference/methods"
RESULTS_DIR = "../../results"

# ===============================================================


if len(sys.argv) != 2:
    sys.exit("dùng: python ../../scripts/mobivital/save_txt.py TN0b")

name = sys.argv[1]

if not os.path.exists(GENERATED):
    sys.exit("không thấy " + GENERATED + " — chạy mobivital_gen.py trước, "
             "và phải đứng trong external/mobivital/")

n_lines = len(open(GENERATED).readlines())

shutil.copy(GENERATED, METHODS_DIR + "/" + name + ".txt")
shutil.copy(GENERATED, RESULTS_DIR + "/" + name + ".txt")

# Khôi phục tệp gốc của tác giả, không để lại vết sửa nào.
if subprocess.run("git checkout -- " + GENERATED, shell=True).returncode != 0:
    sys.exit("không khôi phục được " + GENERATED)

print("%s.txt  %d dòng  ->  %s/  và  results/" % (name, n_lines, METHODS_DIR))
print("đã khôi phục", GENERATED)
