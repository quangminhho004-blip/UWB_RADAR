"""Băm nội dung dữ liệu đã xử lý, để đối chiếu giữa các máy.

    python scripts/checksums.py

VÌ SAO KHÔNG BĂM THẲNG FILE

File `.npz` thực chất là file ZIP, mà ZIP nhúng **thời điểm ghi** vào từng mục
bên trong. Hai file có nội dung mảng giống hệt nhau nhưng ghi lúc khác nhau thì
md5 khác nhau.

Đã đo: cùng bộ dữ liệu, chạy ở máy cá nhân và trên Colab cho ra file **cùng kích
thước từng byte** (323.929.826 byte cho A.npz cả hai bên) nhưng md5 khác hoàn
toàn, cả 21/21 file.

Nên script này nạp mảng ra rồi băm `tobytes()` — chỉ băm đúng những con số,
không băm vỏ ZIP.

DÙNG ĐỂ LÀM GÌ

Dữ liệu thô nằm trên Zenodo (DOI 10.5281/zenodo.15022885), 13 GB, không đưa lên
GitHub được. Ai muốn kiểm chứng thì chạy `notebooks/DATA_PREPARE.ipynb` rồi chạy
script này, đối chiếu với `results/checksums.txt` trong repo.

ĐO ĐƯỢC GÌ KHI CHẠY Ở HAI MÁY

    by_user/*.npz     12/12 giống TỪNG SỐ
    windows/*.npz     số cửa sổ giống hệt, giá trị lệch ~2e-8

Khác nhau vì phép tính khác nhau:

    by_user   đọc CSV + (x-min)/(max-min)*2-1     chỉ + - x :      chính xác tuyệt đối
    windows   np.angle, np.unwrap, np.corrcoef    hàm siêu việt    lệch chữ số cuối

`np.angle` gọi `arctan2`, chữ số cuối phụ thuộc thư viện toán của từng máy.
Lệch 2e-8 nằm đúng cỡ sai số làm tròn của float32 (1.2e-7), tức nhỏ hơn cả sai
số biểu diễn số — không ảnh hưởng model, cùng cỡ với việc đổi seed.

Nên chỉ bắt buộc `by_user` trùng tuyệt đối. Đó mới là dữ liệu gốc; cửa sổ chỉ là
thứ cắt ra từ nó, và cắt lại lúc nào cũng được.
"""

import glob
import hashlib
import os

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

FOLDERS = ["data/processed/by_user",
           "data/processed/windows/dev_cv",
           "data/processed/windows/final_train"]

OUT_FILE = "results/checksums.txt"

# ===============================================================


def hash_content(path):
    """Băm nội dung các mảng trong file .npz, bỏ qua vỏ ZIP.

    Băm cả tên mảng lẫn dữ liệu, theo thứ tự tên đã sắp xếp, để hai file có
    cùng số liệu nhưng khác thứ tự lưu vẫn ra cùng mã băm.
    """
    data = np.load(path)
    h = hashlib.md5()

    for name in sorted(data.files):
        array = data[name]
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(np.ascontiguousarray(array).tobytes())

    return h.hexdigest()


paths = []
for folder in FOLDERS:
    paths = paths + sorted(glob.glob(folder + "/*.npz"))

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

lines = []
for path in paths:
    one_line = "%s  %s" % (hash_content(path), path)
    lines.append(one_line)
    print(one_line)

opened_file = open(OUT_FILE, "w")
opened_file.write("\n".join(lines) + "\n")
opened_file.close()

print()
print("Xong.", len(lines), "file ->", OUT_FILE)
