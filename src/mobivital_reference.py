"""Mượn lại các thành phần ổn định của MobiVital. Không sửa file nào của họ.

    from src import mobivital_reference as mv
    windows = mv.generate_dataset(uwb, gt, 64, 200, 25, 0.9)

RANH GIỚI

Chỉ mượn những hàm nhỏ, thuần tuý tính toán, không có tác dụng phụ. Không
nạp hai file script mobivital_gen.py và autoreg_training.py, vì hai file đó
đọc dòng lệnh ngay lúc nạp (argparse) nên phải giả lập sys.argv mới nạp
được — khó giải thích và dễ hỏng.

Mượn từ MobiVital                                   Tự viết ở dự án này
--------------------------------------------------  ---------------------------
generate_dataset      cắt cửa sổ train               TCN, DS-TCN, RevIN
sequence_transforms   abs / real / imag / phase      loss MSE + Pearson
transform             một phép biến đổi              vòng train, checkpoint, resume
self_normalize        kéo về [-1, 1]                 4-fold CV trên ABCDEFKL
invert_detector       phát hiện sóng lộn ngược       bộ chọn kênh dùng chung
LSTMMultiStep         model baseline                 cho cả LSTM lẫn TCN

evaluate.py không nạp vào đây. Nó là script chấm điểm, gọi bằng subprocess
ở bước cuối để ra số công bố, giữ nguyên bản.

HAI FILE model_utils.py TRÙNG TÊN NHƯNG KHÁC NHAU

    training/utils/model_utils.py:29   ['abs', 'real', 'imag', 'phase']   4 phép
    utils/model_utils.py:30            ['abs', 'phase']                   2 phép

MobiVital dùng bản 4 phép lúc train, bản 2 phép lúc chấm điểm. File này giữ
đúng như vậy: generate_dataset kéo theo bản 4 phép, còn sequence_transforms
lấy từ bản 2 phép để bộ chọn kênh khớp với lúc chấm.
"""

import os
import sys


# ===================== CÀI ĐẶT — sửa ở đây =====================

# Thư mục gốc dự án. Trên Colab khác với ở máy nên phải hỏi.
if os.path.exists("/content"):
    PROJECT_DIR = "/content/THESIS_GRADUATE"
else:
    PROJECT_DIR = "/Users/udnb/Desktop/THESIS_GRADUATE"

# Cấu hình MobiVital công bố, chép từ checkpoints/optimal_params.json.
HISTORY_LENGTH = 200      # số mẫu đưa vào model
FUTURE_LENGTH = 25        # số mẫu model phải đoán
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0001

CORR_THRESHOLD = 0.9      # lọc sóng đáng học lúc TRAIN, CÓ nhìn nhịp thở thật

# Riêng cho model LSTM baseline
LSTM_HIDDEN_SIZE = 352
LSTM_NUM_LAYERS = 2

# ===============================================================


MOBIVITAL_DIR = PROJECT_DIR + "/external/mobivital"

sys.path.append(MOBIVITAL_DIR)

# Cắt cửa sổ để train. Bản 4 phép biến đổi.
from training.utils.model_utils import generate_dataset

# Biến đổi số phức thành sóng thật. Bản 2 phép, khớp với lúc chấm điểm.
from utils.model_utils import self_normalize, sequence_transforms, transform

# Phát hiện sóng bị lộn ngược. Trả về 0 hoặc 1, không phải xác suất.
from utils.peak_width_inverter import invert_detector

# Model baseline của MobiVital, để so sánh.
from utils.models import LSTMMultiStep


# Số cửa sổ cắt được từ một sóng dài 1500 mẫu.
#     bắt đầu ở 0, 25, 50, ... miễn là còn đủ 200 + 25 mẫu phía sau
WINDOWS_PER_SEQUENCE = (1500 - HISTORY_LENGTH) // FUTURE_LENGTH      # 52

# Số ứng viên mỗi session lúc chấm điểm: 120 kênh radar x 2 phép biến đổi.
CANDIDATES_PER_SESSION = 120 * 2                                     # 240


def new_lstm():
    """Dựng model LSTM baseline của MobiVital, đúng cấu hình họ công bố."""
    return LSTMMultiStep(LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, FUTURE_LENGTH)


def info():
    """In cấu hình đang dùng, để dán vào báo cáo."""
    print("thư mục dự án :", PROJECT_DIR)
    print("cửa sổ        :", HISTORY_LENGTH, "vào ->", FUTURE_LENGTH, "ra,",
          WINDOWS_PER_SEQUENCE, "cửa sổ mỗi sóng")
    print("ngưỡng train  : corr >", CORR_THRESHOLD)
    print("ứng viên/chấm :", CANDIDATES_PER_SESSION)
