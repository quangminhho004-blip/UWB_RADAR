"""BƯỚC 9 — Đối chiếu pipeline của mình với pipeline gốc.

    python scripts/9_tn0_compare.py

So hai bảng, cùng checkpoint LSTM của MobiVital:

    results/TN0b.txt   do mobivital_gen.py cua HO sinh    (notebook muc 3)
    results/TN0_1.txt  do src/scoring.py  cua MINH sinh   (script 8)

Hai thứ phải khớp:

    1. lua chon kenh   537/537 dong
    2. diem tung buoi ghi   lech duoi 1e-12

Khớp cả hai thì chứng minh cùng lúc ba điều:

    du lieu by_user/*.npz dung   scoring.py doc .npz, mobivital_gen.py doc CSV.
                                 Khac du lieu thi kenh chon ra da lech
    bo chon kenh dung            537/537
    ham cham diem dung           537 diem khop toi chu so 15

Từ đây thay LSTM bằng TCN, mọi khâu còn lại giữ nguyên.

LƯU Ý: hai bảng có thứ tự dòng khác nhau — code MobiVital duyệt bằng
`os.listdir`, mình đọc `by_user` theo thứ tự sắp xếp. Nên ghép theo TÊN FILE,
không theo số dòng.
"""

import csv


# ===================== CÀI ĐẶT — sửa ở đây =====================

THEIRS_TXT = "results/TN0b.txt"
OURS_TXT = "results/TN0_1.txt"

THEIRS_CSV = "results/scores_TN0b.csv"
OURS_CSV = "results/scores_TN0_1.csv"

# Cho phép lệch tới mức này ở điểm từng buổi ghi. float64 chỉ chính xác tới
# khoảng 1e-16, nên 1e-12 là rộng rãi mà vẫn bắt được lỗi thật.
TOLERANCE = 1e-12

# ===============================================================


def read_choices(path):
    """Đọc bảng lựa chọn kênh. Trả về {tên_file: (bin, phép)}."""
    table = {}
    for file_name, bin_number, method, invert_flag in csv.reader(open(path)):
        table[file_name] = (bin_number, method)
    return table


def read_scores_theirs(path):
    """Đọc scores.csv do evaluate.py của MobiVital ghi: cột 0 tên file, cột 1 điểm."""
    table = {}
    rows = list(csv.reader(open(path)))
    for row in rows[1:]:
        table[row[0]] = float(row[1])
    return table


def read_scores_ours(path):
    """Đọc sessions.csv do src/results.py ghi, lấy cột session_file và pearson."""
    table = {}
    reader = csv.DictReader(open(path))
    for row in reader:
        table[row["session_file"]] = float(row["pearson"])
    return table


# ---------------------------------------------------------------
# 1. Lựa chọn kênh
# ---------------------------------------------------------------

theirs = read_choices(THEIRS_TXT)
ours = read_choices(OURS_TXT)

print("1. LỰA CHỌN KÊNH")
print("   số dòng   MobiVital:", len(theirs), " | mình:", len(ours))
print("   cùng tập tên file  :", set(theirs) == set(ours))

same = 0
different = []
for file_name in theirs:
    if ours.get(file_name) == theirs[file_name]:
        same = same + 1
    else:
        different.append((file_name, theirs[file_name], ours.get(file_name)))

print("   TRÙNG %d / %d" % (same, len(theirs)))
for row in different[:10]:
    print("      LỆCH", row)

print()
print("   vài dòng bất kỳ:")
print("   %-32s %-12s %-12s" % ("buổi ghi", "MobiVital", "mình"))
for file_name in list(theirs)[:5]:
    a = theirs[file_name]
    b = ours[file_name]
    print("   %-32s %-12s %-12s %s"
          % (file_name[:32], "%s,%s" % a, "%s,%s" % b, "OK" if a == b else "LỆCH"))


# ---------------------------------------------------------------
# 2. Điểm từng buổi ghi
# ---------------------------------------------------------------

score_theirs = read_scores_theirs(THEIRS_CSV)
score_ours = read_scores_ours(OURS_CSV)

gaps = []
for file_name in score_theirs:
    gaps.append(abs(score_theirs[file_name] - score_ours[file_name]))

largest = max(gaps)
average = sum(gaps) / len(gaps)
over = sum(1 for g in gaps if g > TOLERANCE)

print()
print("2. ĐIỂM TỪNG BUỔI GHI")
print("   số buổi ghi so được :", len(gaps))
print("   lệch lớn nhất       : %.2e" % largest)
print("   lệch trung bình     : %.2e" % average)
print("   số buổi lệch > %.0e : %d" % (TOLERANCE, over))

mean_theirs = sum(score_theirs.values()) / len(score_theirs)
mean_ours = sum(score_ours.values()) / len(score_ours)

print()
print("   điểm trung bình MobiVital : %r" % mean_theirs)
print("   điểm trung bình của mình  : %r" % mean_ours)
print("   chênh                     : %.2e" % abs(mean_theirs - mean_ours))


# ---------------------------------------------------------------
# 3. Kết luận
# ---------------------------------------------------------------

print()
print("=" * 62)

if set(theirs) != set(ours):
    raise RuntimeError("hai bảng không cùng tập tên file")
if same != len(theirs):
    raise RuntimeError("lựa chọn kênh chỉ trùng %d/%d" % (same, len(theirs)))
if over > 0:
    raise RuntimeError("%d buổi ghi có điểm lệch quá %.0e" % (over, TOLERANCE))

print("ĐẠT — pipeline của mình cho ra đúng kết quả pipeline gốc")
print("      lựa chọn kênh      %d / %d" % (same, len(theirs)))
print("      điểm từng buổi ghi lệch tối đa %.2e" % largest)
print("=" * 62)
