"""Chạy 4-fold trên ABCDEFKL để chọn cấu hình. KHÔNG đụng G H I J.

    python scripts/run_cv.py --model cnn_lstm --hidden 58

Số ra là `cv_score`. Muốn so hai cấu hình thì chạy script này hai lần rồi so
`cv_score`. G H I J để nguyên, không dùng để chọn cấu hình.

BỐN FOLD — CỐ ĐỊNH cho mọi thí nghiệm

    fold      train 6 người      chấm điểm 2 người
    val_AB    C D E F K L        A B
    val_CE    A B D F K L        C E
    val_DF    A B C E K L        D F
    val_KL    A B C D E F        K L

Ghép cặp một người nhiều dữ liệu với một người ít dữ liệu, để bốn fold có lượng
train xấp xỉ nhau — lệch ~9%, so với ~35% nếu ghép theo bảng chữ cái.

VÌ SAO CHẤM ĐIỂM ĐỌC BUỔI GHI THÔ CHỨ KHÔNG ĐỌC CỬA SỔ

Cửa sổ trong `windows/dev_cv/` chọn sóng bằng ngưỡng `corr(sóng, nhịp thở thật)
> 0.9` — tức đã nhìn đáp án. Chấm điểm trên đó là rò rỉ, cấu hình chọn ra sẽ
sai. Nên chấm bằng `scoring.score_all`, đọc buổi ghi thô rồi để model tự chọn
kênh, đúng như lúc chạy thật.

Cửa sổ của 2 người validation không được dùng vào việc gì trong fold đó.
"""

import argparse
import os
import subprocess
import sys

# Bộ nhớ GPU vỡ vụn ở bước chấm điểm: mỗi buổi ghi đẩy 6708 chuỗi một lô, xin
# rồi trả liên tục 1289 lần. Không có cờ này thì TN0 tràn ở buổi ghi 27/1874 dù
# tổng chỗ trống vẫn còn — chỉ là không còn khối liền đủ lớn.
# Phải đặt TRƯỚC "import torch", vì torch đọc biến này lúc nạp.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src import mobivital_reference as mv
from src import models
from src import results
from src import scoring
from src import training


# ===================== CÀI ĐẶT — sửa ở đây =====================

FOLDS = [("val_AB", ["A", "B"]),
         ("val_CE", ["C", "E"]),
         ("val_DF", ["D", "F"]),
         ("val_KL", ["K", "L"])]

DEV_USERS = ["A", "B", "C", "D", "E", "F", "K", "L"]

WINDOWS_DIR = "data/processed/windows/dev_cv"
SUMMARY_FILE = "runs/summary.csv"

# EXP_DIR đặt theo --experiment: MỘT thư mục cho cả checkpoint lẫn điểm,
# mỗi thực nghiệm một thư mục riêng. Xem phần argparse bên dưới.

# ===============================================================


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ds_tcn",
                    help="lstm | cnn_lstm | tcn | ds_tcn")
parser.add_argument("--loss", default="mse", help="mse | mse_pearson")
parser.add_argument("--alpha", type=float, default=1.0, help="trọng số MSE khi loss=mse_pearson")
parser.add_argument("--corr", type=float, default=mv.CORR_THRESHOLD)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=mv.EPOCHS)
parser.add_argument("--channels", type=int, default=64,
                    help="số kênh ẩn của TCN. Thu nhỏ model là mục tiêu của "
                         "đồ án nên không chọn theo cỡ của mốc đem so")
parser.add_argument("--kernel_size", type=int, default=3,
                    help="bề rộng bộ lọc. Tầm nhìn một tầng = (k-1)*dilation")
parser.add_argument("--n_blocks", type=int, default=6,
                    help="số khối. k=3, n=6, khối hai tầng -> tầm nhìn 253, "
                         "phủ hết 200 mẫu vào. Cấu hình được chọn dùng n=4")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="tỉ lệ dropout. Mặc định 0.0 cho khớp LSTM của "
                         "MobiVital, để TN1 chỉ đổi đúng một biến")
parser.add_argument("--hidden", type=int, default=mv.LSTM_HIDDEN_SIZE,
                    help="số chiều ẩn của lstm và cnn_lstm. Mặc định 352 là "
                         "cấu hình MobiVital công bố; 67 cho ~56k tham số")
parser.add_argument("--conv_channels", type=int, default=32,
                    help="số kênh của hai tầng tích chập trong cnn_lstm")
parser.add_argument("--conv_kernel", type=int, default=5,
                    help="bề rộng bộ lọc của cnn_lstm. 5 phủ 0,1 giây ở 50 Hz")
parser.add_argument("--dropout_kind", default="channel",
                    choices=["channel", "element"],
                    help="channel = nn.Dropout1d, xoá cả một kênh, mặc định của "
                         "đồ án. element = nn.Dropout, xoá từng phần tử")
parser.add_argument("--norm", default="batch",
                    choices=["batch", "weight", "none"],
                    help="chuẩn hoá trong khối TCN. batch là mặc định của đồ án; "
                         "none là cấu hình được chọn")
parser.add_argument("--folds", default="all",
                    help="all = chạy đủ 4 fold. Hoặc liệt kê cách nhau bằng dấu "
                         "phẩy, ví dụ val_KL. Chạy một fold là VÒNG SÀNG LỌC, "
                         "không phải vòng kết luận")
parser.add_argument("--experiment", required=True,
                    help="tên thực nghiệm, ví dụ tn1 — quyết định thư mục runs/<tên>/")
args = parser.parse_args()

# Mỗi thực nghiệm một thư mục riêng.
EXP_DIR = "runs/" + args.experiment

# Tên cấu hình phải chứa channels: TCN-64 và TCN-200 cùng model, cùng loss,
# cùng seed — không đưa channels vào thì hai cấu hình ra CÙNG một tên, ghi đè
# kết quả của nhau. Model lstm không có channels nên bỏ qua.
# Hậu tố CHỈ thêm khi giá trị khác mặc định, nhờ đó tên của mọi lần chạy cũ
# không đổi khi thêm tuỳ chọn mới. Ví dụ lstm_mse_corr0.9_seed0 giữ nguyên dù
# về sau có thêm --hidden, --norm hay tuỳ chọn nào nữa.
if args.model == "lstm":
    arch_tag = "" if args.hidden == mv.LSTM_HIDDEN_SIZE else "_h%d" % args.hidden
elif args.model == "cnn_lstm":
    # Không có "cấu hình gốc" nào để lấy làm mặc định, nên ghi đủ ba con số
    # quyết định kiến trúc. Đổi bất kỳ cái nào là ra tên khác, không đè kết quả.
    arch_tag = "_h%d_c%d_k%d" % (args.hidden, args.conv_channels,
                                 args.conv_kernel)
else:
    # Họ tích chập: channels luôn ghi, vì TCN-64 và TCN-200 phải khác tên nhau.
    arch_tag = "_c%d" % args.channels
    if args.kernel_size != 3 or args.n_blocks != 6:
        arch_tag += "_k%d_n%d" % (args.kernel_size, args.n_blocks)
    if args.norm != "batch":
        arch_tag += "_" + args.norm

if args.dropout != 0.0:
    arch_tag += "_do%g" % args.dropout
if args.dropout_kind != "channel":
    # Chỉ có tác dụng khi dropout > 0, nhưng vẫn ghi để hai lần chạy
    # khác loại dropout không đè tên nhau.
    arch_tag += "_dp" + args.dropout_kind[:2]

# alpha PHẢI nằm trong tên. Không có nó thì --alpha 0.3, 0.5, 0.7 ra CÙNG
# một config_id, và cơ chế bỏ qua fold đã xong sẽ nuốt luôn hai lần chạy
# sau — bảng in ra ba dòng giống hệt nhau, trông như đã chạy đủ.
# Chỉ thêm khi loss là mse_pearson, nên mọi tên cũ dùng mse giữ nguyên.
loss_tag = args.loss
if args.loss == "mse_pearson":
    loss_tag += "_a%g" % args.alpha

config_id = "%s%s_%s_corr%s_seed%d" % (
    args.model, arch_tag, loss_tag, args.corr, args.seed)

os.makedirs(EXP_DIR, exist_ok=True)

print("thực nghiệm", args.experiment, " ->", EXP_DIR + "/")
print("cấu hình", config_id)
print("thiết bị ", results.device_name())
print()


def run_one_fold(fold_name, val_users):
    """Train trên 6 người, chấm điểm 2 người còn lại. Trả về điểm macro.

    Fold đã có kết quả thì bỏ qua, không train lại. Nhờ vậy Colab ngắt phiên
    giữa chừng thì chạy lại lệnh cũ là đi tiếp từ fold còn thiếu, thay vì làm
    lại từ fold 1.
    """
    train_users = [u for u in DEV_USERS if u not in val_users]
    run_id = config_id + "_" + fold_name

    print("-" * 58)
    print(fold_name, " train", "".join(train_users), " chấm", "".join(val_users))

    done = results.find_run(SUMMARY_FILE, args.experiment, run_id)
    if done is not None:
        print("   đã có kết quả %.4f — bỏ qua, không train lại"
              % float(done["score_macro"]))
        return float(done["score_macro"])

    X, y = training.load_windows(train_users, args.corr, folder=WINDOWS_DIR)
    print(X.shape[0], "cửa sổ train")

    training.set_seed(args.seed)
    model = models.build_model(args.model,
                               hidden=args.hidden,
                               conv_channels=args.conv_channels,
                               conv_kernel=args.conv_kernel,
                               channels=args.channels,
                               kernel_size=args.kernel_size,
                               n_blocks=args.n_blocks,
                               dropout=args.dropout,
                               norm=args.norm,
                               dropout_kind=args.dropout_kind)

    run_dir = EXP_DIR + "/" + run_id
    train_result = training.train(model,
                                  training.make_loader(X, y),
                                  None,
                                  run_dir,
                                  epochs=args.epochs,
                                  loss_name=args.loss,
                                  alpha=args.alpha)
    results.save_curve(run_dir + "/curve.csv", train_result["curve"])

    model.eval()
    rows = scoring.score_all(val_users, model)
    results.save_sessions(EXP_DIR + "/scores_" + run_id + ".csv", rows)

    by_user = scoring.mean_by_user(rows)
    macro = float(np.mean([by_user[u] for u in val_users]))

    results.add_summary({"run_id": run_id,
                         "experiment": args.experiment,
                         "model": args.model,
                         "loss": args.loss,
                         "alpha": args.alpha,
                         "corr_threshold": args.corr,
                         "seed": args.seed,
                         "fold": fold_name,
                         "val_users": "".join(val_users),
                         "n_params": models.count_params(model),
                         "n_train_windows": X.shape[0],
                         "epochs": args.epochs,
                         "train_mse": train_result["train_mse"],
                         "train_pearson": train_result["train_pearson"],
                         "train_loss": train_result["train_loss"],
                         "minutes_train": train_result["minutes_train"],
                         "resumed": train_result["resumed"],
                         "score_macro": macro,
                         "score_micro": float(np.mean([r["pearson"] for r in rows])),
                         "n_sessions": len(rows),
                         "n_negative": sum(1 for r in rows if r["pearson"] < 0)},
                        SUMMARY_FILE)

    for user in val_users:
        print("   người %s   %.4f" % (user, by_user[user]))
    print("   fold      %.4f" % macro)
    return macro


def snapshot():
    """Nén kết quả rồi chép sang Drive. Gọi sau mỗi fold.

    Bốn fold mất vài giờ, Colab hay ngắt phiên và xoá sạch /content. Không lưu
    dọc đường thì mất trắng. Tên tệp nén chứa cấu hình nên hai phiên chạy song
    song không đè zip của nhau.
    """
    subprocess.run([sys.executable, "scripts/save_results.py", args.experiment,
                    "--out", args.experiment + "_" + config_id],
                   check=False)


fold_scores = []
# Lọc fold theo --folds. Tên sai thì báo ngay, đừng để chạy xong mới biết
# thiếu fold.
if args.folds == "all":
    chosen = FOLDS
else:
    want = [x.strip() for x in args.folds.split(",")]
    known = [f for f, _ in FOLDS]
    for x in want:
        if x not in known:
            raise SystemExit("không có fold tên %s. Có: %s" % (x, ", ".join(known)))
    chosen = [(f, u) for f, u in FOLDS if f in want]
    print("CHỈ chạy %d/%d fold: %s" % (len(chosen), len(FOLDS), ", ".join(want)))
    print("Đây là vòng sàng lọc. Điểm KHÔNG so được với cv_score đủ 4 fold.")

for fold_name, val_users in chosen:
    fold_scores.append(run_one_fold(fold_name, val_users))
    snapshot()

cv_score = float(np.mean(fold_scores))
cv_std = float(np.std(fold_scores))

# Một dòng TỔNG cho cả cấu hình, ngoài 4 dòng của 4 fold. Nhờ nó chọn cấu hình
# chỉ cần lọc summary.csv theo fold == "TONG", không phải tự cộng trung bình.
#
# Chạy lại lệnh cũ sau khi đã xong đủ 4 fold thì dòng này đã có; ghi nữa là
# trùng. Bỏ qua để lệnh vẫn chạy được và vẫn in ra bảng tổng kết.
# Dòng TONG chỉ có nghĩa khi chạy ĐỦ 4 fold. Chạy một phần mà vẫn ghi thì
# compare_cv đọc phải một "cv_score" tính từ một fold — số đó cao hơn hẳn và
# đảo cả thứ hạng. Bốn dòng fold vẫn được ghi bình
# thường, nên chạy nốt các fold còn lại thì dòng TONG tự có.
du_bon_fold = len(chosen) == len(FOLDS)
if not du_bon_fold:
    print("\nCHƯA đủ 4 fold nên KHÔNG ghi dòng TONG vào summary.csv.")
    print("Chạy nốt các fold còn lại thì dòng đó tự có.")
if du_bon_fold and results.find_run(
        SUMMARY_FILE, args.experiment, config_id + "_tong") is None:
    results.add_summary({"run_id": config_id + "_tong",
                         "experiment": args.experiment,
                         "model": args.model,
                         "loss": args.loss,
                         "alpha": args.alpha,
                         "corr_threshold": args.corr,
                         "seed": args.seed,
                         "fold": "TONG",
                         "val_users": "".join(DEV_USERS),
                         "n_params": models.count_params(models.build_model(args.model,
                               hidden=args.hidden,
                               conv_channels=args.conv_channels,
                               conv_kernel=args.conv_kernel,
                               channels=args.channels,
                               kernel_size=args.kernel_size,
                               n_blocks=args.n_blocks,
                               dropout=args.dropout,
                               norm=args.norm,
                               dropout_kind=args.dropout_kind)),
                         "epochs": args.epochs,
                         "score_macro": cv_score,
                         "score_std": cv_std,
                         "n_sessions": 4}, SUMMARY_FILE)

print()
print("=" * 58)
for (fold_name, _), diem in zip(chosen, fold_scores):
    print("   %-8s %.4f" % (fold_name, diem))
if du_bon_fold:
    print("   cv_score %.6f   <- số dùng để chọn cấu hình" % cv_score)
    print("   cv_std   %.6f   <- chỉ để báo cáo" % cv_std)
else:
    print("   trung bình %d fold %.6f   <- VÒNG SÀNG LỌC, không phải cv_score"
          % (len(chosen), cv_score))
print("=" * 58)
print(SUMMARY_FILE)
