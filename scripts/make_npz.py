"""Đọc CSV của từng người, lưu thành một file .npz.

    python scripts/make_npz.py

Trước:  external/mobivital/dataset/mobivital/tripod/*.csv   (1874 file, một bản duy nhất)
Sau:    data/processed/by_user/A.npz   gồm 3 mảng:
            uwb   -- tín hiệu radar dạng số phức, (số_session, 1500, 120)
            gt    -- respiration ground truth,     (số_session, 1500), đã chuẩn hoá [-1, 1]
            files -- tên file CSV của từng session, (số_session,)

Lưu tên file để sau này ghi ra bảng kết quả đối chiếu được với bảng của
MobiVital, vốn ghi theo tên file.

Mỗi file CSV là một session, có 1500 dòng.
    cột  12..131  = phần thực của UWB   (120 kênh)
    cột 132..251  = phần ảo  của UWB
    cột áp chót   = respiration ground truth

Đây là PIPELINE TRAIN/VALIDATE (xem docs/PROTOCOL.md mục 1).

VÌ SAO ĐỌC CSV TRONG THƯ MỤC MOBIVITAL

Chỉ giữ MỘT bản CSV, đặt đúng chỗ `prep_breath_final.py` của MobiVital đòi. Hai
pipeline đọc chung một bản đó, nên khi `scripts/check_data.py` báo sai lệch bằng
0 thì không ai cãi được là do hai bản dữ liệu khác nhau.

`data/` từ đây chỉ chứa thứ pipeline của đồ án sinh ra.

Lọc người theo tên file, đúng cách `prep_breath_final.py` dòng 27-31 làm:
tên `240502_userA_tripod_04_8.csv` thuộc người A.
"""

import csv
import glob
import os

import numpy as np

CSV_DIR = "external/mobivital/dataset/mobivital/tripod"
OUT_DIR = "data/processed/by_user"

users = ["A", "B", "C", "D", "E", "F",
         "G", "H", "I", "J", "K", "L"]


def self_normalize(mat):
    """Kéo giãn dãy số về khoảng [-1, 1]. Dãy hằng số thì trả về toàn 0.

    Chép nguyên công thức của MobiVital (hàm self_normalize trong
    prep_breath_final.py) để hai pipeline lưu ra giá trị giống hệt nhau.
    """
    max_val = np.amax(mat)
    min_val = np.amin(mat)

    if max_val == min_val:
        return np.zeros(mat.shape)

    return (mat - min_val) / (max_val - min_val) * 2 - 1


def read_one_csv(path):
    """Đọc một file CSV. Trả về (uwb, gt) hoặc None nếu file không đúng 1500 dòng.

    Đọc y hệt cách MobiVital làm trong prep_breath_final.py: dùng csv.reader rồi
    ép sang float32 NGAY, sau đó mới chuẩn hoá. Nếu đọc ở float64 rồi mới ép
    xuống thì kết quả lệch với MobiVital ở chữ số thứ 8 -- không ảnh hưởng gì tới model,
    nhưng làm giống hệt để khỏi phải giải trình.
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader)).astype(np.float32)

    if data.ndim != 2 or len(data) != 1500 or data.shape[1] < 254:
        return None

    real = data[:, 12:132]
    imag = data[:, 132:252]
    uwb = real + 1j * imag

    # Chuẩn hoá về [-1, 1], giống MobiVital. Giá trị thô là đơn vị cảm biến
    # tuỳ tiện (khoảng 1465..4738), không mang ý nghĩa gì.
    gt = self_normalize(data[:, -2])

    return uwb.astype(np.complex64), gt.astype(np.float32)


os.makedirs(OUT_DIR, exist_ok=True)

# Đã đủ 12 tệp thì bỏ qua — chạy lại notebook không mất thời gian làm lại.
# Một tệp dở dang thì thiếu, và cả 12 tệp được cắt lại từ đầu.
done = sorted(os.path.basename(p) for p in glob.glob(OUT_DIR + "/*.npz"))
if done == sorted(u + ".npz" for u in users):
    print("đã đủ 12 tệp trong", OUT_DIR, "— bỏ qua")
    raise SystemExit(0)

all_csv = sorted(glob.glob(CSV_DIR + "/*.csv"))
if len(all_csv) == 0:
    raise RuntimeError("không thấy CSV nào trong " + CSV_DIR
                       + " — giải nén tripod.zip vào đó trước")
print("tìm thấy", len(all_csv), "file CSV")

for user in users:
    csv_files = [p for p in all_csv if ("user" + user) in os.path.basename(p)]

    uwb_list = [] # raw signal dạng số phức (complex-valued I/Q samples)
    gt_list = [] # Ground truth
    file_list = [] # tên file CSV, để đối chiếu về sau
    for path in csv_files:
        result = read_one_csv(path)
        if result is None:
            continue
        uwb, gt = result
        uwb_list.append(uwb)
        gt_list.append(gt)
        file_list.append(os.path.basename(path))

    uwb_all = np.stack(uwb_list)
    gt_all = np.stack(gt_list)
    files_all = np.array(file_list)

    out_file = OUT_DIR + "/" + user + ".npz"
    np.savez(out_file, uwb=uwb_all, gt=gt_all, files=files_all)

    print("user", user, "->", out_file,
          "| uwb", uwb_all.shape, "| gt", gt_all.shape,
          "| files", files_all.shape)

print("Xong")
