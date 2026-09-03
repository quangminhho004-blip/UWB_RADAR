"""BƯỚC 6 — Băm nội dung dữ liệu đã xử lý, để đối chiếu giữa các máy.

    python scripts/6_checksums.py

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

Trùng nghĩa là họ dựng lại được đúng bộ dữ liệu mà các thí nghiệm đã dùng.
"""

import glob
import hashlib
import os

import numpy as np


# ===================== CÀI ĐẶT — sửa ở đây =====================

THU_MUC = ["data/processed/by_user",
           "data/processed/windows/dev_cv",
           "data/processed/windows/final_train"]

FILE_RA = "results/checksums.txt"

# ===============================================================


def bam_noi_dung(path):
    """Băm nội dung các mảng trong file .npz, bỏ qua vỏ ZIP.

    Băm cả tên mảng lẫn dữ liệu, theo thứ tự tên đã sắp xếp, để hai file có
    cùng số liệu nhưng khác thứ tự lưu vẫn ra cùng mã băm.
    """
    du_lieu = np.load(path)
    h = hashlib.md5()

    for ten in sorted(du_lieu.files):
        mang = du_lieu[ten]
        h.update(ten.encode())
        h.update(str(mang.shape).encode())
        h.update(str(mang.dtype).encode())
        h.update(np.ascontiguousarray(mang).tobytes())

    return h.hexdigest()


duong_dan = []
for thu_muc in THU_MUC:
    duong_dan = duong_dan + sorted(glob.glob(thu_muc + "/*.npz"))

os.makedirs(os.path.dirname(FILE_RA), exist_ok=True)

dong = []
for path in duong_dan:
    mot_dong = "%s  %s" % (bam_noi_dung(path), path)
    dong.append(mot_dong)
    print(mot_dong)

opened_file = open(FILE_RA, "w")
opened_file.write("\n".join(dong) + "\n")
opened_file.close()

print()
print("Xong.", len(dong), "file ->", FILE_RA)
