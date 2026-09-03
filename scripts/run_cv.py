"""Chạy 4-fold trên ABCDEFKL để chọn cấu hình. KHÔNG đụng G H I J.

    python scripts/run_cv.py --model ds_tcn --revin true

Số ra là `cv_score`. Muốn so hai cấu hình thì chạy script này hai lần rồi so
`cv_score`. Chốt xong mới chạy scripts/run_final_test.py một lần duy nhất.

BỐN FOLD (docs/PROTOCOL.md mục 2, CỐ ĐỊNH cho mọi thí nghiệm)

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
import sys

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
RUNS_DIR = "runs/cv"
RESULTS_DIR = "results/cv"
SUMMARY_FILE = "runs/summary.csv"

# ===============================================================


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ds_tcn", help="lstm | tcn | ds_tcn")
parser.add_argument("--revin", default="false", help="true | false")
parser.add_argument("--loss", default="mse", help="mse | mse_pearson")
parser.add_argument("--alpha", type=float, default=1.0, help="trọng số MSE khi loss=mse_pearson")
parser.add_argument("--corr", type=float, default=mv.CORR_THRESHOLD)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=mv.EPOCHS)
parser.add_argument("--experiment", default="cv", help="tên thí nghiệm, ghi vào summary.csv")
args = parser.parse_args()

revin = args.revin.lower() == "true"

config_id = "%s%s_%s_corr%s_seed%d" % (
    args.model, "_revin" if revin else "", args.loss, args.corr, args.seed)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("cấu hình", config_id)
print("thiết bị ", results.device_name())
print()


def run_one_fold(fold_name, val_users):
    """Train trên 6 người, chấm điểm 2 người còn lại. Trả về điểm macro."""
    train_users = [u for u in DEV_USERS if u not in val_users]
    run_id = config_id + "_" + fold_name

    print("-" * 58)
    print(fold_name, " train", "".join(train_users), " chấm", "".join(val_users))

    X, y = training.load_windows(train_users, args.corr, folder=WINDOWS_DIR)
    print(X.shape[0], "cửa sổ train")

    training.set_seed(args.seed)
    model = models.build_model(args.model, revin=revin)

    run_dir = RUNS_DIR + "/" + run_id
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
    results.save_sessions(RESULTS_DIR + "/scores_" + run_id + ".csv", rows)

    by_user = scoring.mean_by_user(rows)
    macro = float(np.mean([by_user[u] for u in val_users]))

    results.add_summary({"run_id": run_id,
                         "experiment": args.experiment,
                         "model": args.model,
                         "revin": int(revin),
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


fold_scores = []
for fold_name, val_users in FOLDS:
    fold_scores.append(run_one_fold(fold_name, val_users))

cv_score = float(np.mean(fold_scores))
cv_std = float(np.std(fold_scores))

print()
print("=" * 58)
for i in range(len(FOLDS)):
    print("   %-8s %.4f" % (FOLDS[i][0], fold_scores[i]))
print("   cv_score %.6f   <- số dùng để chọn cấu hình" % cv_score)
print("   cv_std   %.6f   <- chỉ để báo cáo" % cv_std)
print("=" * 58)
print(SUMMARY_FILE)
