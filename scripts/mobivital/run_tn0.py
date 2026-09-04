"""TN0 — chạy pipeline MobiVital bằng đúng lệnh trong README của tác giả.

    python scripts/mobivital/run_tn0.py --case prep   sinh data_final/*.npy
    python scripts/mobivital/run_tn0.py --case a      tính điểm từ tệp có sẵn
    python scripts/mobivital/run_tn0.py --case b      chọn kênh bằng tệp trọng số tác giả
    python scripts/mobivital/run_tn0.py --case c      train lại LSTM từ đầu

KHÔNG SỬA MỘT DÒNG NÀO TRONG CODE MOBIVITAL

Script này chỉ `cd` vào thư mục repo của họ rồi gọi lệnh, đúng như README ghi.
Mỗi lệnh được **in nguyên văn ra màn hình trước khi chạy**, nên đọc output là
thấy đủ chuỗi lệnh của tác giả:

    $ python dataset_preparation/prep_breath_final.py
    $ python -m training.autoreg_training --model_name lstm_retrained
    $ python -m inference.mobivital_gen
    $ python -m inference.evaluate -m TN0b.txt --save_file scores_TN0b.csv

VIỆC DUY NHẤT LÀM THÊM: GIỮ REPO TÁC GIẢ SẠCH

`inference/mobivital_gen.py` dòng 121 luôn ghi tệp lựa chọn kênh ra đúng một tên

    inference/methods/tripod_mobivital_pre_invert_0.9.txt

mà tên đó trùng một tệp **đã có sẵn trong repo tác giả**. Chạy là ghi đè lên tệp
của họ, chạy lần hai lại đè lần một. Nên ngay sau mỗi lần chạy, script:

    1. chép tệp vừa sinh -> inference/methods/<tên>.txt   (để evaluate.py đọc)
    2. chép tệp vừa sinh -> runs/tn0/<tên>.txt         (để đối chiếu về sau)
    3. git checkout khôi phục tệp gốc của tác giả
    4. xoá tệp tạm <tên>.txt

Cuối mỗi lần chạy có đụng repo họ, script kiểm `git status --porcelain` phải
trống và dừng hẳn nếu không.

MỘT DÒNG SỬA TRONG CODE TÁC GIẢ

`inference/mobivital_gen.py` nạp trọng số xong không gọi `model.eval()`. LSTM ở
chế độ train nên cuDNN cấp thêm vùng nhớ dự trữ cho backward — 15.1 GB cho lô
6708 chuỗi của một buổi ghi. Đo thật: T4 và L4 đều tràn ngay ở buổi ghi 1/1874.

`scripts/mobivital/apply_patched_files.py` thêm đúng một dòng `model.eval()`, có chú thích
đánh dấu rõ trong file. Model chỉ gồm `nn.LSTM(dropout=0)` và `nn.Linear` nên
train và eval cho forward giống hệt — vá này không đổi kết quả, chỉ đổi cách xin
bộ nhớ. Sau khi vá, chọn kênh chạy trên GPU mất vài phút thay vì 1 giờ 20 CPU.

`check_clean()` vì vậy chấp nhận ĐÚNG một tệp bị sửa là
`inference/mobivital_gen.py`, và kiểm phần thêm vào chỉ nằm trong khối đánh dấu.
Bất kỳ tệp nào khác bị đụng là dừng hẳn.

DỪNG NGAY KHI LỖI

Mọi lệnh chạy qua `run()`; lệnh nào trả mã khác 0 là dừng cả script. Nhờ vậy
không bao giờ có chuyện `mobivital_gen.py` chết mà bước sau vẫn cất nhầm tệp cũ.

CẦN CHẠY TRƯỚC

    scripts/download_dataset.py         -> 1874 tệp CSV
    scripts/mobivital/setup_dataset.py  -> vá 52 tên tệp lỗi thời
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_repo


# ===================== CÀI ĐẶT — sửa ở đây =====================

MOBIVITAL_DIR = "external/mobivital"
RESULTS_DIR = "runs/tn0"

# Tên cố định mà mobivital_gen.py luôn ghi ra.
GENERATED = "inference/methods/tripod_mobivital_pre_invert_0.9.txt"

# Thư mục có thêm 52 lối tắt mang tên cũ, chỉ evaluate.py dùng tới.
OLD_NAMES = "./dataset/mobivital/tripod_old_names"

NPY_FILES = ["data_final/training_breath_tripod_data.npy",   # ABCDEFKL
             "data_final/testing_breath_tripod_data.npy"]    # GHIJ

# ===============================================================


def run(cmd):
    """Chạy một lệnh trong thư mục MobiVital. In nguyên văn rồi chạy.

    Đặt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Sau khi vá .eval(),
    một buổi ghi chỉ cần khoảng 2.3 GB, nhưng lô to nhỏ khác nhau giữa các buổi
    (số ứng viên sống sót qua invert_detector dao động) nên bộ cấp phát mặc định
    giữ lại nhiều vùng đã dành mà không dùng. Đo thật: chết ở buổi ghi 27/1874,
    xin 11.35 GB trong khi 9.36 GB đang bị giữ mà bỏ không. Bật cờ này thì chạy
    trọn 1874 buổi trong 17 phút.

    Dừng cả script nếu lệnh trả mã khác 0 — không để bước sau chạy tiếp trên
    dữ liệu của lần trước.
    """
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print()
    print("$ " + cmd, flush=True)
    if subprocess.run(cmd, shell=True, cwd=MOBIVITAL_DIR, env=env).returncode != 0:
        sys.exit("DỪNG — lệnh trên trả mã lỗi")


def inside(path):
    """Đường dẫn bên trong repo MobiVital, tính từ thư mục gốc đồ án."""
    return MOBIVITAL_DIR + "/" + path


def save_txt(name):
    """Cất tệp lựa chọn kênh vừa sinh, rồi khôi phục tệp gốc của tác giả."""
    src = inside(GENERATED)
    if not os.path.exists(src):
        sys.exit("không thấy " + src + " — mobivital_gen.py chưa sinh ra tệp")

    n_lines = len(open(src).readlines())
    shutil.copy(src, inside("inference/methods/" + name + ".txt"))
    shutil.copy(src, RESULTS_DIR + "/" + name + ".txt")

    run("git checkout -- " + GENERATED)
    print("   %s.txt  %d dòng  ->  %s/  (đã khôi phục tệp gốc của tác giả)"
          % (name, n_lines, RESULTS_DIR))


def collect_scores(name):
    """Chép bảng điểm evaluate.py vừa ghi ra runs/tn0/, rồi dọn tệp tạm."""
    shutil.copy(inside("inference/methods/scores_" + name + ".csv"),
                RESULTS_DIR + "/scores_" + name + ".csv")

    tmp = inside("inference/methods/" + name + ".txt")
    if os.path.exists(tmp):
        os.remove(tmp)

    print("   scores_%s.csv  ->  %s/" % (name, RESULTS_DIR))


def check_clean():
    """Repo tác giả chỉ được mang đúng vá đã biết."""
    ok, mo_ta = check_repo.check_patched_only(MOBIVITAL_DIR)
    print()
    if not ok:
        sys.exit("DỪNG — repo MobiVital " + mo_ta)
    print("repo MobiVital:", mo_ta)


# ---------------------------------------------------------------

def case_prep():
    """Sinh data_final/*.npy — lệnh đầu tiên trong README của tác giả."""
    if all(os.path.exists(inside(f)) for f in NPY_FILES):
        print("data_final đã có đủ hai tệp, bỏ qua")
    else:
        run("python dataset_preparation/prep_breath_final.py")

    for f in NPY_FILES:
        if not os.path.exists(inside(f)):
            sys.exit("thiếu " + inside(f))
    print()
    subprocess.run("ls -la data_final/", shell=True, cwd=MOBIVITAL_DIR)


def case_a():
    """TN0a — tính điểm từ tệp lựa chọn kênh tác giả cung cấp, chưa đụng model."""
    shutil.copy(inside(GENERATED), RESULTS_DIR + "/TN0a.txt")

    run("python -m inference.evaluate"
        " -m " + os.path.basename(GENERATED) +
        " -d " + OLD_NAMES +
        " --save_file scores_TN0a.csv")

    shutil.copy(inside("inference/methods/scores_TN0a.csv"),
                RESULTS_DIR + "/scores_TN0a.csv")
    print("   TN0a.txt và scores_TN0a.csv  ->  %s/" % RESULTS_DIR)


def case_b():
    """TN0b — chọn kênh bằng tệp trọng số tác giả phát hành."""
    run("python -m inference.mobivital_gen")
    save_txt("TN0b")
    run("python -m inference.evaluate -m TN0b.txt --save_file scores_TN0b.csv")
    collect_scores("TN0b")
    check_clean()


def case_c():
    """TN0c — train lại LSTM từ đầu bằng chính vòng train của tác giả."""
    run("python -m training.autoreg_training --model_name lstm_retrained")
    run("python -m inference.mobivital_gen --model_name lstm_retrained")
    save_txt("TN0c")
    run("python -m inference.evaluate -m TN0c.txt --save_file scores_TN0c.csv")
    collect_scores("TN0c")
    check_clean()


# ---------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True, choices=["prep", "a", "b", "c"])
args = parser.parse_args()

if not os.path.isdir(MOBIVITAL_DIR + "/.git"):
    sys.exit("không thấy " + MOBIVITAL_DIR + " — chạy scripts/setup_colab.py trước")

os.makedirs(RESULTS_DIR, exist_ok=True)

{"prep": case_prep, "a": case_a, "b": case_b, "c": case_c}[args.case]()
