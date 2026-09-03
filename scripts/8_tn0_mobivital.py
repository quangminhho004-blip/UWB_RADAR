"""BƯỚC 8 — TN0: dựng lại kết quả MobiVital bằng chính code của họ.

    python scripts/8_tn0_mobivital.py

Chạy `mobivital_gen.py` và `evaluate.py` bản gốc, **0 dòng sửa**, ba lần:

    TN0a   chấm bảng kết quả MobiVital commit sẵn        -> so với bài báo 0.819
    TN0b   checkpoint của họ  -> tự sinh bảng -> chấm    -> mốc cho TN0.1
    TN0c   train lại từ đầu   -> tự sinh bảng -> chấm    -> tái lập công thức train

Mỗi bậc thêm đúng một việc do mình tự làm. Bậc nào lệch đầu tiên thì lỗi nằm ở
đúng cái vừa thêm.

CÁCH CHẠY CODE HỌ MÀ KHÔNG SỬA GÌ

Code MobiVital dùng đường dẫn tương đối — `./dataset/mobivital/tripod/`,
`./data_final/`, `checkpoints/`. Script này dựng một thư mục tạm có đúng những
cái tên đó, bên trong toàn lối tắt, rồi `cd` vào đó mà gọi. Code họ tưởng nó
đang nằm trong repo của nó.

Checkpoint phải COPY chứ không lối tắt: TN0c ghi file .pth mới, lối tắt sẽ ghi
xuyên qua và đè mất bản gốc của họ.

KHOÁ CHỐNG GHI ĐÈ

`mobivital_gen.py` dòng 121 luôn ghi ra đúng một tên. Chạy hai lần mà quên đổi
tên là mất kết quả lần trước. Script này đổi tên ngay sau mỗi lần chạy, và dừng
nếu file đích đã tồn tại.

CẦN CHẠY TRƯỚC

    scripts/3_run_mobivital_prep.py   -> data/processed/mobivital_original/*.npy
    scripts/7_fix_old_filenames.py    -> runs/tn0/csv_flat, csv_old_names
"""

import os
import shutil
import subprocess
import sys
import time


# ===================== CÀI ĐẶT — sửa ở đây =====================

MOBIVITAL_DIR = "external/mobivital"
NPY_DIR = "data/processed/mobivital_original"

WORK_DIR = "runs/tn0/work"
FLAT_DIR = "runs/tn0/csv_flat"
OLD_NAMES_DIR = "runs/tn0/csv_old_names"
OUT_DIR = "results"

# Tên file .pth mà autoreg_training.py sẽ ghi ra ở TN0c.
RETRAINED_NAME = "lstm_retrained"

# ===============================================================


# mobivital_gen.py luôn ghi ra đúng cái tên này, không đổi được.
GEN_OUTPUT = "tripod_mobivital_pre_invert_0.9.txt"


def run(command, cwd):
    """Chạy một lệnh, in ra dòng cuối cùng. Lỗi thì dừng hẳn."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(MOBIVITAL_DIR)

    result = subprocess.run(command, cwd=cwd, env=env, shell=True,
                            capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()

    if result.returncode != 0:
        print(output[-2000:])
        raise RuntimeError("lệnh lỗi: " + command)

    # Bỏ thanh tiến trình của tqdm, chỉ giữ dòng chữ.
    lines = []
    for line in output.replace("\r\n", "\n").split("\n"):
        text = line.split("\r")[-1].strip()
        if text and "it/s]" not in text and "s/it]" not in text:
            lines.append(text)
    return lines


def python_cmd(script, extra=""):
    """Lệnh gọi một script của MobiVital bằng đúng trình Python đang chạy."""
    return (sys.executable + " " + os.path.abspath(MOBIVITAL_DIR) + "/" + script
            + " " + extra)


def score(methods_file, data_folder):
    """Chạy evaluate.py của MobiVital. Trả về điểm trung bình 537 buổi ghi.

    Mỗi lần dùng một --save_file riêng. Lý do: evaluate.py dòng 67 gán Series
    vào DataFrame đã có, pandas căn theo index cũ và vứt key lạ — dồn chung một
    file thì những lần sau bị cắt mất dòng.
    """
    name = methods_file.replace(".txt", "")
    lines = run(python_cmd("inference/evaluate.py",
                           "-m " + methods_file
                           + " -d " + data_folder
                           + " --save_file scores_" + name + ".csv"), WORK_DIR)
    return float(lines[-1])


def keep_result(new_name):
    """Đổi tên bảng mobivital_gen.py vừa sinh ra. Không cho ghi đè."""
    made = WORK_DIR + "/inference/methods/" + GEN_OUTPUT
    target = WORK_DIR + "/inference/methods/" + new_name

    if os.path.exists(target):
        raise RuntimeError(target + " đã có rồi, xoá đi nếu thật sự muốn chạy lại")

    shutil.move(made, target)
    return len(open(target).readlines())


# ---------------------------------------------------------------
# Dựng sân cho code MobiVital
# ---------------------------------------------------------------

print("Dựng thư mục tạm", WORK_DIR)

for folder in ["dataset/mobivital", "data_final", "checkpoints", "inference/methods"]:
    os.makedirs(WORK_DIR + "/" + folder, exist_ok=True)

# mobivital_gen.py ghi cứng ./dataset/mobivital/tripod/ nên trỏ nó vào csv_flat.
tripod_link = WORK_DIR + "/dataset/mobivital/tripod"
if not os.path.lexists(tripod_link):
    os.symlink(os.path.abspath(FLAT_DIR), tripod_link)

# autoreg_training.py đọc ./data_final/
for name in ["training_breath_tripod_data.npy", "testing_breath_tripod_data.npy"]:
    link = WORK_DIR + "/data_final/" + name
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(NPY_DIR + "/" + name), link)

# Copy thật, không lối tắt — TN0c sẽ ghi .pth mới vào đây.
for name in ["lstm_pred_tripod_0.9.pth", "optimal_params.json"]:
    shutil.copy(MOBIVITAL_DIR + "/checkpoints/" + name,
                WORK_DIR + "/checkpoints/" + name)

# Bảng kết quả MobiVital commit sẵn, đặt tên riêng để lát nữa không bị đè.
shutil.copy(MOBIVITAL_DIR + "/inference/methods/" + GEN_OUTPUT,
            WORK_DIR + "/inference/methods/TN0a.txt")

print("  ", len(os.listdir(FLAT_DIR)), "lối tắt CSV")
print("   cấu hình:", open(WORK_DIR + "/checkpoints/optimal_params.json").read().strip())


# ---------------------------------------------------------------
# TN0a — chấm bảng MobiVital commit sẵn
# ---------------------------------------------------------------

print()
print("TN0a — chấm bảng MobiVital commit sẵn")

tn0a = score("TN0a.txt", "../csv_old_names")
print("   điểm:", repr(tn0a))


# ---------------------------------------------------------------
# TN0b — checkpoint của họ, mình tự sinh bảng
# ---------------------------------------------------------------

print()
print("TN0b — checkpoint của họ, tự sinh bảng")

started = time.time()
run(python_cmd("inference/mobivital_gen.py"), WORK_DIR)
print("   sinh bảng xong sau %.0f phút, %d dòng"
      % ((time.time() - started) / 60, keep_result("TN0b.txt")))

tn0b = score("TN0b.txt", "./dataset/mobivital/tripod")
print("   điểm:", repr(tn0b))


# ---------------------------------------------------------------
# TN0c — train lại từ đầu
# ---------------------------------------------------------------

print()
print("TN0c — train lại từ đầu bằng autoreg_training.py")

started = time.time()
lines = run(python_cmd("training/autoreg_training.py",
                       "--model_name " + RETRAINED_NAME), WORK_DIR)
print("   train xong sau %.0f phút" % ((time.time() - started) / 60))
for line in lines:
    if "Epoch" in line or "Finished" in line:
        print("   ", line)

run(python_cmd("inference/mobivital_gen.py",
               "--model_name " + RETRAINED_NAME), WORK_DIR)
print("   sinh bảng xong,", keep_result("TN0c.txt"), "dòng")

tn0c = score("TN0c.txt", "./dataset/mobivital/tripod")
print("   điểm:", repr(tn0c))


# ---------------------------------------------------------------
# Cất kết quả, kiểm tra không đụng code MobiVital
# ---------------------------------------------------------------

print()
print("Cất kết quả vào", OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)
for name in ["TN0b.txt", "TN0c.txt",
             "scores_TN0b.csv", "scores_TN0c.csv", "scores_TN0a.csv"]:
    shutil.copy(WORK_DIR + "/inference/methods/" + name, OUT_DIR + "/" + name)
    print("   ", OUT_DIR + "/" + name)

diff = subprocess.run("git -C " + MOBIVITAL_DIR + " status --short",
                      shell=True, capture_output=True, text=True).stdout.strip()

print()
print("KIỂM TRA không sửa gì trong repo MobiVital:")
print("  ", diff if diff else "git diff trống — không sửa dòng nào")

print()
print("=" * 62)
print("TN0a  chấm bảng họ commit sẵn      %.6f" % tn0a)
print("TN0b  checkpoint họ, tự sinh bảng  %.6f" % tn0b)
print("TN0c  train lại từ đầu             %.6f" % tn0c)
print("=" * 62)
