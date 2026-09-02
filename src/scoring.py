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

Đổi lại phải chứng minh viết đúng. Bằng chứng ở notebooks/tn0_1.ipynb:
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

import numpy as np
import torch
from einops import rearrange

from src import mobivital_reference as mv


def pick_channel(uwb, model):
    """Chọn một kênh trong 240 ứng viên của một buổi ghi.

    uwb    -- (1500, 120) số phức, tín hiệu radar thô của một buổi ghi
    model  -- nhận (batch, 200) trả (batch, 25)

    Trả về (sóng_đã_chọn, chỉ_số). Chỉ số chạy 0..239:
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
    return kept[best], original_index[best]


def index_to_name(index):
    """Đổi chỉ số 0..239 thành (bin, tên phép biến đổi)."""
    if index % 2 == 0:
        return index // 2, "abs"
    return index // 2, "phase"


def score_one_user(user, model, by_user_dir):
    """Chấm điểm mọi buổi ghi của một người.

    Trả về (danh sách điểm, danh sách tên file, danh sách chỉ số kênh).
    """
    data = np.load(by_user_dir + "/" + user + ".npz")
    uwb_all = data["uwb"]
    gt_all = data["gt"]
    files = data["files"]

    scores = []
    indexes = []
    for i in range(len(gt_all)):
        sequence, index = pick_channel(uwb_all[i], model)

        # gt trong .npz đã chuẩn hoá một lần lúc đọc CSV. MobiVital chuẩn
        # hoá thêm một lần nữa ở evaluate.py dòng 56, nên làm y hệt.
        # Phép này lặp lại không đổi giá trị, nhưng chép đúng cho chắc.
        gt = mv.self_normalize(gt_all[i])

        scores.append(np.corrcoef(gt, sequence)[0, 1])
        indexes.append(index)

    return scores, list(files), indexes


def score_users(users, model, by_user_dir=None):
    """Chấm điểm nhiều người. Trả về điểm trung bình của từng người.

    Điểm chính thức của đồ án là trung bình các số này (macro theo người),
    không phải trung bình trên toàn bộ buổi ghi — xem docs/PROTOCOL.md
    mục 4. Lý do: mỗi người có số buổi ghi khác nhau, tính gộp thì người
    ghi nhiều buổi bị tính nặng ký hơn một cách vô lý.
    """
    if by_user_dir is None:
        by_user_dir = mv.PROJECT_DIR + "/data/processed/by_user"

    result = {}
    for user in users:
        scores, files, indexes = score_one_user(user, model, by_user_dir)
        result[user] = float(np.mean(scores))
    return result


def write_txt(users, model, path, by_user_dir=None):
    """Ghi bảng lựa chọn kênh ra file, đúng định dạng của MobiVital.

        240409_userG_tripod_02_3.csv,24,phase,0

    Cột cuối là cờ lật ngược sóng. MobiVital ghi cứng 0 ở mobivital_gen.py
    dòng 139, không bao giờ bằng 1, nên ở đây cũng ghi 0.

    Dùng để đối chiếu với bảng do code gốc sinh ra.
    """
    if by_user_dir is None:
        by_user_dir = mv.PROJECT_DIR + "/data/processed/by_user"

    lines = []
    all_scores = []
    for user in users:
        scores, files, indexes = score_one_user(user, model, by_user_dir)
        for i in range(len(files)):
            bin_number, method = index_to_name(indexes[i])
            lines.append(files[i] + "," + str(bin_number) + "," + method + ",0")
        all_scores = all_scores + scores

    opened_file = open(path, "w")
    opened_file.write("\n".join(lines) + "\n")
    opened_file.close()

    return float(np.mean(all_scores))
