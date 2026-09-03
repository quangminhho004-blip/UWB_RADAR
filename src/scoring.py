"""Bộ chọn kênh và hàm chấm điểm. Dùng chung cho LSTM, TCN, DS-TCN.

    from src import scoring
    diem = scoring.score_users(["G", "H", "I", "J"], model)

VIỆC NÀY LÀM GÌ

Radar UWB đo phản hồi ở 120 khoảng cách khác nhau. Người ngồi ở đâu đó
trong 120 khoảng cách đó, không biết trước. Nên với mỗi buổi ghi phải
tìm ra kênh nào bắt được nhịp thở rõ nhất.

Cách MobiVital làm, và đây cũng là điểm chính của bài báo: cho model dự
báo trước 25 mẫu tiếp theo của từng kênh, kênh nào model đoán chuẩn nhất
thì chọn. Sóng thở đều đặn nên dễ đoán, sóng nhiễu lộn xộn nên đoán trật.
Cách này KHÔNG cần nhìn nhịp thở thật, nên dùng được ngoài đời khi không
có cảm biến đo nhịp thở.

VÌ SAO VIẾT LẠI THAY VÌ GỌI CODE HỌ

Hàm get_best_sequence nằm trong inference/mobivital_gen.py, mà file đó
đọc dòng lệnh ngay lúc nạp nên phải giả lập sys.argv mới import được.
Viết lại thì code sạch, và quan trọng hơn là nhận được cả TCN lẫn LSTM.

Đổi lại phải chứng minh viết đúng. Bằng chứng ở notebooks/TN0.ipynb mục 7:
nạp đúng checkpoint LSTM của MobiVital, chạy hàm này trên 537 buổi ghi
của G H I J, phải chọn ra cùng kênh trên cả 537 và cho cùng điểm số với
kết quả code gốc đã chạy ở TN0.

CHÉP CHÍNH XÁC TỪNG BƯỚC

Phần pick_channel bên dưới bám sát mobivital_gen.py dòng 56-101, kể cả
kiểu số ở từng bước, vì bước cuối là argmax: hai kênh gần bằng điểm nhau
mà lệch ở chữ số cuối là đảo thứ hạng, không đối chiếu được nữa.

    float32   sóng sau biến đổi, và toàn bộ đường đi qua model
    float64   phép tính tương quan và tổng (np.corrcoef tự nâng lên)
"""

import csv

import numpy as np
import torch
from einops import rearrange

from src import mobivital_reference as mv


def pick_channel(uwb, model):
    """Chọn một kênh trong 240 ứng viên của một buổi ghi.

    uwb    -- (1500, 120) số phức, tín hiệu radar thô của một buổi ghi
    model  -- nhận (batch, 200) trả (batch, 25)

    Trả về (sóng_đã_chọn, chỉ_số, số_ứng_viên_sống_sót). Chỉ số chạy 0..239:
        bin  = chỉ_số // 2
        phép = "abs" nếu chỉ_số chẵn, "phase" nếu lẻ
    """
    history = mv.HISTORY_LENGTH
    future = mv.FUTURE_LENGTH

    # --- Bước 1: dựng 240 ứng viên (120 kênh x 2 phép biến đổi) ---
    sequences = []
    for one_channel in rearrange(uwb, "t c -> c t"):
        transformed = mv.sequence_transforms(one_channel)
        sequences.append(transformed[0])      # abs
        sequences.append(transformed[-1])     # phase

    # --- Bước 2: bỏ những sóng bị lộn ngược ---
    # invert_detector trả về 0 hoặc 1, không phải xác suất. MobiVital so
    # với 0.8 nhưng số đó không quan trọng: mọi ngưỡng trong (0, 1] đều
    # cho cùng kết quả. Giữ nguyên 0.8 để chép đúng.
    kept = []
    original_index = []
    for i in range(len(sequences)):
        if mv.invert_detector(sequences[i]) < 0.8:
            kept.append(sequences[i])
            original_index.append(i)

    kept = np.array(kept)

    # --- Bước 3: cắt mỗi ứng viên thành 52 cửa sổ ---
    X = []
    y = []
    for i in range(len(kept)):
        one_sequence = kept[i]
        start = 0
        while start + history + future <= len(one_sequence):
            X.append(one_sequence[start: start + history])
            y.append(one_sequence[start + history: start + history + future])
            start = start + future

    X = np.array(X)
    y = np.array(y)

    # --- Bước 4: model dự báo ---
    device = next(model.parameters()).device
    with torch.no_grad():
        predicted = model(torch.from_numpy(X).to(device).float())
    predicted = predicted.cpu().numpy()

    # --- Bước 5: chấm từng cửa sổ bằng Pearson, cộng theo từng ứng viên ---
    # Không dùng nhịp thở thật ở đây. So dự báo với 25 mẫu thật của
    # chính ứng viên đó.
    scores = []
    for i in range(len(y)):
        scores.append(np.corrcoef(predicted[i], y[i])[0, 1])

    scores = np.array(scores)
    scores = rearrange(scores, "(c l) -> c l", l=mv.WINDOWS_PER_SEQUENCE)

    best = np.argmax(np.sum(scores, axis=1))
    return kept[best], original_index[best], len(kept)


def index_to_name(index):
    """Đổi chỉ số 0..239 thành (bin, tên phép biến đổi)."""
    if index % 2 == 0:
        return index // 2, "abs"
    return index // 2, "phase"


def score_from_txt(txt_path, users, by_user_dir=None):
    """Chấm điểm theo bảng lựa chọn CÓ SẴN — không chạy model.

    Đây đúng là việc evaluate.py của MobiVital làm: đọc từng dòng bảng, lấy
    đúng kênh và phép đã ghi, rồi tính Pearson với nhịp thở thật.

    Xử lý từng người một rồi giải phóng, không nạp cả bốn người cùng lúc:
    mỗi người nặng khoảng 200 MB.

    Bảng MobiVital commit sẵn có 52 tên lỗi thời (tháng 10 thay vì 12) nên
    hàm này dò cả hai tên. Xem scripts/7_setup_mobivital.py.
    """
    if by_user_dir is None:
        by_user_dir = mv.PROJECT_DIR + "/data/processed/by_user"

    # Đọc bảng, gom theo người.
    picks = {}
    for file_name, bin_number, method, invert_flag in csv.reader(open(txt_path)):
        user = file_name.split("_")[1][-1]
        picks.setdefault(user, []).append((file_name, int(bin_number), method))

    rows = []
    for user in users:
        data = np.load(by_user_dir + "/" + user + ".npz")
        uwb_all = data["uwb"]
        gt_all = data["gt"]
        files = data["files"]

        where = {}
        for i in range(len(files)):
            where[str(files[i])] = i

        for file_name, bin_number, method in picks[user]:
            if file_name in where:
                i = where[file_name]
            else:
                # 52 tên lỗi thời: đổi tháng 10 thành 12
                i = where[file_name[:2] + "12" + file_name[4:]]

            sequence = mv.transform(uwb_all[i][:, bin_number], method)

            rows.append({"user": user,
                         "session_file": file_name,
                         "bin": bin_number,
                         "method": method,
                         "n_candidates_kept": "",
                         "pearson": float(np.corrcoef(
                             mv.self_normalize(gt_all[i]), sequence)[0, 1])})

        del data, uwb_all, gt_all

    return rows


def score_all(users, model, by_user_dir=None):
    """Chấm điểm mọi buổi ghi của nhiều người.

    Trả về danh sách dict, mỗi buổi ghi một dòng:

        user               "G"
        session_file       "240409_userG_tripod_02_3.csv"
        bin                24
        method             "phase"
        n_candidates_kept  137      số ứng viên sống sót sau bộ lọc lộn ngược
        pearson            0.9312

    Đây là dữ liệu thô. Từ đây tính ra được điểm theo người, điểm chung, và
    so được thắng/hoà/thua giữa hai lần chạy bất kỳ.
    """
    if by_user_dir is None:
        by_user_dir = mv.PROJECT_DIR + "/data/processed/by_user"

    rows = []
    for user in users:
        data = np.load(by_user_dir + "/" + user + ".npz")
        uwb_all = data["uwb"]
        gt_all = data["gt"]
        files = data["files"]

        for i in range(len(gt_all)):
            sequence, index, n_kept = pick_channel(uwb_all[i], model)

            # gt trong .npz đã chuẩn hoá một lần lúc đọc CSV. MobiVital chuẩn
            # hoá thêm một lần nữa ở evaluate.py dòng 56, nên làm y hệt.
            gt = mv.self_normalize(gt_all[i])

            bin_number, method = index_to_name(index)
            rows.append({"user": user,
                         "session_file": str(files[i]),
                         "bin": bin_number,
                         "method": method,
                         "n_candidates_kept": n_kept,
                         "pearson": float(np.corrcoef(gt, sequence)[0, 1])})
    return rows


def mean_by_user(rows):
    """Điểm trung bình của từng người, từ dữ liệu score_all trả về.

    Điểm chính thức của đồ án là trung bình các số này (macro theo người),
    không phải trung bình trên toàn bộ buổi ghi — xem docs/PROTOCOL.md mục 4.
    Lý do: mỗi người có số buổi ghi khác nhau, tính gộp thì người ghi nhiều
    buổi bị tính nặng ký hơn một cách vô lý.
    """
    total = {}
    count = {}
    for row in rows:
        user = row["user"]
        total[user] = total.get(user, 0) + row["pearson"]
        count[user] = count.get(user, 0) + 1

    result = {}
    for user in sorted(total):
        result[user] = total[user] / count[user]
    return result


def write_txt(rows, path):
    """Ghi bảng lựa chọn kênh, đúng định dạng của MobiVital.

        240409_userG_tripod_02_3.csv,24,phase,0

    Cột cuối là cờ lật ngược sóng. MobiVital ghi cứng 0 ở mobivital_gen.py
    dòng 139, không bao giờ bằng 1, nên ở đây cũng ghi 0.
    """
    lines = []
    for row in rows:
        lines.append(row["session_file"] + "," + str(row["bin"]) + ","
                     + row["method"] + ",0")

    opened_file = open(path, "w")
    opened_file.write("\n".join(lines) + "\n")
    opened_file.close()
