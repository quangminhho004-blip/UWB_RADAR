"""Kiểm repo MobiVital chỉ mang đúng một vá đã biết, không gì khác.

Dùng chung bởi `scripts/mobivital/run_tn0.py` và `scripts/run_tn0.py`.

Đồ án sửa code tác giả đúng MỘT dòng: thêm `model.eval()` vào
`inference/mobivital_gen.py`, xem `scripts/mobivital/apply_patched_files.py`. Hàm dưới
xác nhận:

    1. chỉ đúng tệp đó bị đổi, không tệp nào khác
    2. mọi dòng thêm vào đều là chú thích, trừ đúng một dòng `model.eval()`

Trả về (đạt hay không, câu mô tả).
"""

import subprocess


MOBIVITAL_DIR = "external/mobivital"
PATCHED_FILE = "inference/mobivital_gen.py"


def check_patched_only(mobivital_dir=MOBIVITAL_DIR):
    """Trả về (ok, mô tả). ok=False khi repo tác giả bị đụng ngoài dự kiến."""
    dirty = subprocess.run("git status --porcelain", shell=True, cwd=mobivital_dir,
                           capture_output=True, text=True).stdout

    # Không strip cả chuỗi: porcelain để mã trạng thái ở hai cột đầu, dòng chưa
    # đưa vào chỉ mục bắt đầu bằng dấu cách.
    changed = [line[3:].strip() for line in dirty.split("\n") if line.strip()]

    if not changed:
        return True, "sạch, không sửa dòng nào"

    if changed != [PATCHED_FILE]:
        return False, "bị đụng ngoài dự kiến:\n" + dirty

    diff = subprocess.run("git diff -- " + PATCHED_FILE, shell=True, cwd=mobivital_dir,
                          capture_output=True, text=True).stdout

    added = [l for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]

    ngoai_khoi = []
    for l in added:
        noi_dung = l[1:].strip()
        if noi_dung.startswith("#") or noi_dung == "model.eval()":
            continue
        ngoai_khoi.append(l)

    if ngoai_khoi:
        return False, ("tệp đã vá còn thay đổi ngoài khối đánh dấu:\n"
                       + "\n".join(ngoai_khoi))

    return True, ("chỉ %s bị sửa, %d dòng thêm, toàn bộ trong khối đánh dấu "
                  "(đúng một dòng lệnh model.eval())" % (PATCHED_FILE, len(added)))
