# Tóm tắt thesis — mô hình dự báo gọn cho pipeline MobiVital

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar

**Nhánh nộp:** `final_submission`

**Năm bước, nối nhau:**

| | làm gì | kết quả |
|---|---|---|
| **TN0** | tái lập MobiVital | micro 0,8195 trên 537 phiên G H I J, khớp bài báo |
| **TN1** | chọn kiến trúc giữa bốn ứng viên | DS-TCN 64 k3n4 — **37.081 tham số** |
| **TN2** | chọn tầm nhìn: kernel 3, 5, 7, 9 | tầm nhìn **61** — càng rộng càng tệ |
| **TN3** | chọn hàm loss: quét 10 mức alpha | đưa Pearson vào loss hơn MSE thuần **0,019** |
| **TN4** | chấm trên G H I J, chưa dùng để chọn gì | **0,803590 ± 0,015350** |

**Mô hình cuối: DS-TCN 64, kernel 3, 4 khối, loss lai alpha 0,6 — 37.081 tham số.**

Trên tập kiểm tra độc lập nó **ngang** mốc LSTM-352 của MobiVital (0,810302) với
**ít hơn 40,5 lần tham số**; chênh 0,0067 nhỏ hơn dao động seed của chính nó
(0,0154). Không phát biểu là "tốt hơn". Xem [BANG_TCN.md](BANG_TCN.md) mục 5.

**Cách đọc:** mục **0** là hướng dẫn setup → dữ liệu/Drive → chạy → lưu kết quả;
mục **1–2** là bài toán và giao thức; mục **3–4** là notebook và kết quả TN0–TN4;
mục **5–7** giải thích lựa chọn cuối và nguồn chi tiết. Các lệnh là hướng dẫn
tái chạy, không phải thông báo vừa train lại.

## 0. Hướng dẫn thực hiện từ setup đến test cuối

Phần này mô tả **cách chạy và vị trí tệp theo mã hiện tại**. Các mục 1–7 bên dưới giải thích bài toán, kết quả đã có và quyết định nghiên cứu. Không cần train lại chỉ để đọc kết quả: notebook trong repo đã lưu output. Các đường dẫn Drive dưới đây là nơi script đọc/ghi; tài liệu không xác nhận mọi ZIP hiện vẫn tồn tại trong tài khoản Drive của người dùng.

### 0.1. Code, dữ liệu và kết quả nằm ở đâu?

| Nơi lưu | Vai trò | Có giữ được khi Colab xóa runtime? |
|---|---|---|
| GitHub, nhánh `submission` | Code, 17 notebook, tài liệu, checksum và một số CSV/TXT bằng chứng | Có, với những thay đổi đã commit/push. |
| `/content/UWB_RADAR/` | Bản code và các file làm việc trong Colab | Không nên xem là lưu bền vững. |
| `/content/UWB_RADAR/external/mobivital/` | Mã tác giả, CSV radar/đai và dữ liệu NPY dựng từ CSV | Phải tải/dựng lại nếu runtime mất. |
| `/content/UWB_RADAR/data/processed/` | NPZ theo người và cửa sổ để train | Khôi phục được từ hai gói trên Drive. |
| `/content/UWB_RADAR/runs/` | Checkpoint, curve và scores đang chạy | Chỉ giữ được qua runtime mới nếu đã sao lưu. Đây là thư mục local, không phải symlink Drive. |
| `/content/drive/MyDrive/mobivital/` | Dữ liệu xử lý đóng gói và ZIP kết quả | Có, sau khi sao chép hoàn tất vào Drive. |

```text
GitHub submission ──clone──> /content/UWB_RADAR
Zenodo tripod.zip ──giải nén──> external/mobivital/dataset/mobivital/tripod/
                                      ↓ xử lý dữ liệu lần đầu
                              data/processed/
                                      ↓ đóng gói
Google Drive / MyDrive / mobivital / by_user.tar + windows.tar.gz
                                      ↓ khôi phục ở runtime mới
                              data/processed/
                                      ↓ train + chấm điểm
                              runs/<experiment>/
                                      ↓ save_results.py
Google Drive / MyDrive / mobivital / <tên kết quả>.zip
```

### 0.2. Setup Colab — thực hiện ở mỗi runtime mới

Chọn runtime Python có GPU để train; CPU phù hợp cho một số bước kiểm tra nhưng không dùng thời gian CPU để so với lượt GPU. Dựng dữ liệu đầy đủ cần khoảng **25 GB ổ local trống** theo notebook chuẩn bị. Hai gói dữ liệu xử lý khoảng **2,7 GB trên Drive**, chưa gồm ZIP checkpoint/kết quả; đây là kích thước quan sát, không phải giới hạn bộ nhớ train.

**Ô 1 — mount Drive và tạo nơi lưu:**

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/mobivital')
DRIVE.mkdir(parents=True, exist_ok=True)
```

**Ô 2 — tải mã và setup:** chỉ clone khi runtime mới chưa có thư mục repo.

```python
!git clone -q --branch submission --single-branch https://github.com/quangminhho004-blip/UWB_RADAR.git /content/UWB_RADAR
%cd /content/UWB_RADAR
!python scripts/setup_colab.py
```

`setup_colab.py` tải mã MobiVital và ghim commit **`4319731d2769d4134c92088dd846666e262f18e9`**, cài `einops`, tạo `runs/`, in commit project và GPU. Script không khóa toàn bộ phiên bản môi trường Colab. Nếu cần tái lập một lượt lịch sử, dùng `git_commit` của lượt đó; nhánh `submission` có thể tiếp tục thay đổi.

**Ô 3 — lưu môi trường của lượt chạy mới** (bổ sung để đóng gói bằng chứng; không phải tệp đã có sẵn của mọi lượt cũ):

```python
import json, platform, subprocess
import numpy as np
import torch

metadata = {
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'python': platform.python_version(),
    'numpy': np.__version__,
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
}
Path('runs').mkdir(exist_ok=True)
Path('runs/environment.json').write_text(json.dumps(metadata, indent=2))
Path('runs/requirements_runtime.txt').write_text(
    subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
print(metadata)
```

Nguồn thao tác: [setup_colab.py](../scripts/setup_colab.py). Lệnh `%cd` phải chạy trong notebook, để các ô sau đứng đúng thư mục gốc project.

### 0.3. Chuẩn bị dữ liệu lần đầu — `DATA_PREPARE.ipynb`

Cách ưu tiên: chạy [DATA_PREPARE.ipynb](../notebooks/DATA_PREPARE.ipynb) từ đầu đến cuối. Luồng tương đương bằng các lệnh dưới đây, sau bước setup:

```python
!python scripts/download_dataset.py
!python scripts/mobivital/setup_dataset.py
!python scripts/mobivital/run_tn0.py --case prep
!python scripts/make_npz.py
!python scripts/check_data.py
!python scripts/make_windows.py
!python scripts/checksums.py
!git --no-pager diff -- data/checksums.txt
```

Chạy từng bước và kiểm tra thành công trước khi sang bước sau.

| Bước | Đọc gì? | Sinh ra gì / kiểm tra gì? |
|---|---|---|
| `download_dataset.py` | Zenodo record `15022885`, tệp `tripod.zip` | Tải `/content/tripod.zip` (~5,7 GB); giải nén CSV vào `external/mobivital/dataset/mobivital/tripod/`. Script kiểm 1.874 CSV thô. |
| `mobivital/setup_dataset.py` | CSV vừa giải nén | Chuẩn bị ánh xạ tên cũ cho evaluator tác giả và loại dữ liệu khỏi Git của repo tác giả; không tạo cửa sổ train. |
| `mobivital/run_tn0.py --case prep` | CSV, chạy `prep_breath_final.py` của tác giả | `external/mobivital/data_final/training_breath_tripod_data.npy` và `testing_breath_tripod_data.npy`. Mỗi tệp được ghi lần lượt mảng radar và GT. |
| `make_npz.py` | CSV cùng nguồn | 12 tệp `data/processed/by_user/A.npz` đến `L.npz`; tổng 1.826 phiên đủ 1.500 mẫu, gồm 1.289 dev và 537 test. Số phiên hợp lệ khác số CSV tải về. |
| `check_data.py` | NPY tác giả và NPZ project | Đối chiếu nội dung radar/GT; kiểm đủ 1.289/1.289 dev và 537/537 test. Đây là bằng chứng dữ liệu phục vụ TN0, ngoài so điểm a/b/c. |
| `make_windows.py` | NPZ dev theo người và NPY train gộp | Cửa sổ train `X`, `y`; dùng `generate_dataset` của tác giả. |
| `checksums.py` | Nội dung mảng trong `by_user/*.npz` | Ghi lại `data/checksums.txt`; dùng `git diff` so với mốc đã commit. Script tự ghi đè, không tự đánh giá pass/fail. |

**Định dạng dữ liệu sau xử lý:**

```text
/content/UWB_RADAR/data/processed/
├── by_user/
│   ├── A.npz ... F.npz, K.npz, L.npz      dev: 8 người
│   └── G.npz, H.npz, I.npz, J.npz        test: 4 người
└── windows/
    ├── dev_cv/
    │   └── <người>_corr0.9_h200_f25.npz   8 tệp A–F, K, L
    └── final_train/
        └── train_corr0.9_h200_f25.npz    train gộp ABCDEFKL
```

| Tệp | Các mảng / ý nghĩa |
|---|---|
| `by_user/<người>.npz` | `uwb`: `(số phiên, 1500, 120)`, complex64; `gt`: `(số phiên, 1500)`, float32; `files`: tên CSV để truy vết. GT đã min–max về `[-1,1]`; radar phức I/Q chưa chuẩn hóa. |
| `windows/...npz` | `X`: `(số cửa sổ, 200)`; `y`: `(số cửa sổ, 25)`. `y` là đoạn tiếp theo của chính chuỗi đang cắt. Không phải mọi `y` đều là sóng đai. |

**Chuẩn hóa và lọc diễn ra thế nào?**

```text
CSV → GT chuẩn hóa riêng [-1,1]; radar I/Q giữ nguyên → by_user NPZ
Radar của người train → bin 20–28 → abs/real/imag/phase → 36 ứng viên
→ min–max RIÊNG mỗi ứng viên 1.500 mẫu về [-1,1]
→ giữ ứng viên có Pearson với GT > 0,9
→ thêm chuỗi GT chuẩn hóa của phiên
→ cắt history 200, future 25, stride 25 → 52 cặp/chuỗi
→ X, y → shuffle và batch khi train
```

Min/max lấy trên từng chuỗi của một phiên trước khi cắt cửa sổ, không gộp toàn bộ người/dataset. Chuỗi hằng số được đưa về toàn 0. Min–max này khác chuẩn hoá theo từng cửa sổ, và khác BatchNorm bên trong model. Xem [PIPELINE_2.md](PIPELINE_2.md) để trình bày bằng sơ đồ.

### 0.4. Cất dữ liệu đã xử lý lên Drive

Sau khi kiểm tra dữ liệu thành công, chạy trong notebook:

```python
!tar -cf /content/drive/MyDrive/mobivital/by_user.tar -C data/processed by_user
!tar -czf /content/drive/MyDrive/mobivital/windows.tar.gz -C data/processed windows
!ls -lh /content/drive/MyDrive/mobivital/by_user.tar /content/drive/MyDrive/mobivital/windows.tar.gz
```

Hai archive chứa thư mục `by_user/` và `windows/`, vì vậy lúc restore phải giải nén vào **`data/processed/`**. Không giải nén vào `data/processed/by_user/` lần nữa, tránh lồng thêm tầng.

CSV thô và `data_final/*.npy` không nằm trong hai archive này. Giữ dữ liệu raw ở Zenodo và dựng lại khi cần đường chạy của tác giả; không đưa CSV/checkpoint dung lượng lớn vào GitHub.

### 0.5. Runtime sau — khôi phục dữ liệu và kết quả

Sau mount Drive, clone và setup:

```python
!python scripts/restore_processed_data_on_drive.py
```

Script lấy **đúng hai tên** `by_user.tar`, `windows.tar.gz` trong `/content/drive/MyDrive/mobivital/`. Nếu không mount hoặc thiếu archive, script in thông báo và thoát; **không tự tải CSV hay tự chạy preprocessing**. Khi thiếu dữ liệu phải quay lại mục 0.3.

- **TN1 qua runner của đồ án:** dữ liệu xử lý đầy đủ đủ để train/chấm; `scoring.py` đọc radar/GT từ `by_user`. Không cần tải lại CSV chỉ để chạy các runner này.
- **TN0 chạy mã tác giả và kiểm tra dữ liệu:** cần thêm CSV và `data_final`; làm đúng các ô chuẩn bị của TN0.
- Kiểm tra dữ liệu đã khôi phục trước khi train:

```python
assert len(list(Path('data/processed/by_user').glob('*.npz'))) == 12
assert len(list(Path('data/processed/windows/dev_cv').glob('*.npz'))) == 8
assert Path('data/processed/windows/final_train/train_corr0.9_h200_f25.npz').exists()
```

Kiểm số tệp không thay thế kiểm nội dung/checksum. Bước restore hiện bỏ qua khi số tệp đáp ứng điều kiện, không xác minh mọi giá trị mảng.

**Khôi phục kết quả cũ:** dùng ô “Khôi phục kết quả” trong notebook tương ứng. Ô này giải từng ZIP vào thư mục tạm, gộp metric theo `(experiment, run_id)` vào `runs/summary.csv`, rồi chép checkpoint/curve/scores về `runs/<experiment>/`. Nếu có các lượt khác nhau nhưng trùng `run_id`, không gộp chúng thành một lượt; dùng archive riêng và đối chiếu commit.

Chỉ `unzip ... -d runs/` chưa đủ cho resume: runner tra cứu **`runs/summary.csv`**, còn ZIP chứa **`runs/<experiment>/summary.csv`**. Không giải nhiều ZIP rồi để bản summary cuối đè mất các dòng trước. ZIP là bản chụp cả thư mục thực nghiệm ở runtime đó; tên chứa C64 không đảm bảo bên trong chỉ có C64 nếu đã chạy nhiều cấu hình chung runtime.

### 0.6. TN0 — chạy và lưu bằng chứng trước khi khảo sát model

Mở [TN0.ipynb](../notebooks/TN0.ipynb), thực hiện setup/CSV/NPY/NPZ/check_data/windows. Trên bản clone mới, **áp dụng bản vá inference đã công khai** trước khi chạy phía tác giả:

```python
!python scripts/mobivital/apply_patched_files.py
```

Bản vá thêm `model.eval()`; setup clone không tự áp dụng bản vá này. Ô lệnh riêng trên là bước cần làm rõ khi tái chạy TN0 với repo tác giả mới tải.

Sau đó chạy theo thứ tự:

```python
!python scripts/mobivital/run_tn0.py --case a
!python scripts/mobivital/run_tn0.py --case b
!python scripts/mobivital/run_tn0.py --case c
!python scripts/run_tn0.py --case a
!python scripts/run_tn0.py --case b
!python scripts/run_tn0.py --case c
!python scripts/run_tn0.py --compare
!python scripts/save_results.py tn0
```

Checkpoint công bố dùng cho TN0b: `external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth`. TN0c phía project đặt seed **1234**, train LSTM-352 hai layer bằng MSE, 20 epoch; đây không phải nhóm seed 0/1/2 của TN1. Không coi TN0c là phép so cùng seed hoàn toàn giữa hai vòng train nếu chưa đối chiếu cách đặt RNG phía tác giả.

Kỳ vọng đối chiếu lịch sử: TN0a micro **0,819481**, TN0b micro **0,822175**, trùng **537/537 cặp `(bin, method)`**; TN0c ghi tham khảo. Bảng giải thích đầy đủ và giới hạn nằm ở mục 4.

Gói tạo ra: `runs/tn0.zip` → Drive `mobivital/tn0.zip`. Phải giữ TXT lựa chọn, scores từng phiên, `compare.csv`, log/output và checkpoint train lại nếu cần tái lập inference. **Repo hiện chưa chứa đầy đủ các artifact này**, dù notebook có output; kiểm trong ZIP của lần chạy thực tế trước khi bàn giao. `summary.csv` ít hoặc không có dòng không tự có nghĩa TN0 thất bại, vì bảng so TN0 dùng `compare.csv` và `scores_*.csv`.

### 0.7. TN0 → TN1 → TN2 → TN3 → TN4: chạy gì, thay gì?

Mũi tên đi tiếp nghĩa là giữ **cấu hình**, không phải huấn luyện tiếp từ
checkpoint của bước trước. Mỗi lượt dựng model và train lại từ đầu theo seed.

| Bước | Phạm vi chạy | Đầu vào train → đầu vào chấm | Đầu ra quyết định |
|---|---|---|---|
| TN0 | Chạy lại pipeline MobiVital | Checkpoint của tác giả → 537 phiên G H I J | Micro theo phiên, so với con số bài báo |
| TN1 | 4 kiến trúc × seed 0/1/2 × 4 fold | Cửa sổ 6 người → phiên radar 2 người validation | CV macro; chọn DS-TCN 64 k3n4 |
| TN2 | kernel 5, 7, 9 × seed 0 × 4 fold; k3 dùng lại TN1 | như TN1 | Giữ RF61; mang thêm RF121 sang TN3 |
| TN3 | 2 kiến trúc × alpha 0,0…0,9 × seed 0 × 4 fold | như TN1 | Chọn alpha riêng cho từng kiến trúc |
| TN4 | 3 tổ hợp × seed 0/1/2 | `windows/final_train/` ABCDEFKL → `by_user/` G H I J | Macro từng seed, trung bình ± độ lệch chuẩn mẫu |

Bốn fold: `val_AB` train CDEFKL; `val_CE` train ABDFKL; `val_DF` train ABCEKL;
`val_KL` train ABCDEF. Cửa sổ validation đã lọc bằng GT chỉ dùng cho đường cong
`val_mse`/`val_pearson`; **chọn cấu hình bằng điểm chọn ứng viên trên toàn bộ
phiên validation**, không chọn bằng loss trên cửa sổ đã lọc.

**TN1** — bốn lệnh, mỗi lệnh chạy tiếp seed 1 và 2 trong notebook:

```python
!python scripts/run_cv.py --experiment tn1 --model ds_tcn --channels 64 --kernel_size 3 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --seed 0
!python scripts/run_cv.py --experiment tn1 --model lstm --seed 0
!python scripts/run_cv.py --experiment tn1 --model lstm --hidden 67 --seed 0
!python scripts/run_cv.py --experiment tn1 --model cnn_lstm --hidden 58 --seed 0
```

**TN2** — cùng cấu trúc bốn khối, chỉ đổi `--kernel_size` và `--experiment`:

```python
!python scripts/run_cv.py --experiment tn2_rf --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --seed 0
```

Chạy tiếp với `--kernel_size 7` và `9`. Không thêm `--folds` khi đang chạy CV
bốn fold; cờ đó chỉ dành cho vòng sàng lọc một fold, và vòng sàng lọc không được
đưa vào bảng kết luận.

**TN3** — thêm `--loss mse_pearson --alpha`, quét 0,0 tới 0,9:

```python
!python scripts/run_cv.py --experiment tn3 --model ds_tcn --channels 64 --kernel_size 3 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.6 --seed 0
```

`--alpha` là trọng số của **MSE**, không phải ngưỡng chọn kênh. `--alpha 0` là
Pearson thuần. Mốc `alpha = 1` (MSE thuần) lấy lại từ TN1 và TN2, không train lại
chỉ để đổi tên thực nghiệm.

**TN4** — đổi sang `run_final_test.py`, chạy đủ ba seed:

```python
!python scripts/run_final_test.py --experiment tn4 --model ds_tcn --channels 64 --kernel_size 3 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.6 --seed 0
```

| tổ hợp | đổi gì trong lệnh trên |
|---|---|
| **DS-TCN 64/RF61, alpha 0,6** — mô hình cuối | (đúng lệnh trên) |
| DS-TCN 64/RF121, Pearson thuần | `--kernel_size 5 --alpha 0.0` |
| DS-TCN 64/RF121, MSE thuần — đối chứng loss | `--kernel_size 5 --loss mse`, bỏ `--alpha` |
| Mốc LSTM 352 | `--experiment tn1_ghij --model lstm` |

**Chấm một phiên validation hoặc test:** radar 120 bin → 240 ứng viên abs/phase
→ chuẩn hoá mỗi chuỗi → inversion detector → cửa sổ 200→25 → model → Pearson dự
báo với tương lai của chính ứng viên → cộng điểm 52 cửa sổ → chọn ứng viên →
Pearson sóng đã chọn với GT của phiên. `invert_detector` trả 0/1; phép so `< 0.8`
trong mã giữ 0 và loại 1, không phải lọc Pearson với đai ở ngưỡng 0,8.

### 0.8. Kết quả lưu file gì? Tên trên Drive là gì?

```text
runs/
├── summary.csv                       metric chung, runner tra để bỏ qua lượt đã xong
└── <experiment>/                     tn0, tn1, tn1_ghij, tn2_rf, tn3, tn4
    ├── summary.csv                   bản lọc theo experiment khi save_results
    ├── README.txt                    lúc đóng gói, commit đóng gói
    ├── scores_<run_id>.csv           điểm + bin/method từng phiên
    └── <run_id>/
        ├── final.pth                 state_dict sau epoch cuối
        ├── curve.csv                 loss/MSE/Pearson theo epoch
        └── last.pth                  trạng thái resume, chỉ khi đang train dở
```

`run_id` của CV có hậu tố fold. Ví dụ checkpoint cấu hình được chọn, seed 0,
fold AB:

```text
runs/tn1/ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed0_val_AB/final.pth
```

Trong repo, `runs/tn1/` đã được xếp lại thành
`<tên cấu hình>/seed<N>/<fold>/` cho dễ đọc — cùng nội dung, chỉ khác cách lồng
thư mục. Xem [runs/tn1/README.md](../runs/tn1/README.md).

`final.pth` chứa trọng số, **không** chứa định nghĩa kiến trúc; phải giữ
`run_id` và lệnh tương ứng để nạp đúng. `last.pth` lưu trạng thái train để
resume và bị xoá khi hoàn tất. Runner sao lưu sang Drive sau mỗi fold; các lệnh
sao lưu gọi với `check=False`, nên phải kiểm thông báo **"đã chép sang …"** và
tệp trên Drive, không chỉ thấy model train xong.

**Tên gói trên Drive**, tất cả dưới `/content/drive/MyDrive/mobivital/`:

| Thực nghiệm | ZIP |
|---|---|
| Dữ liệu | `by_user.tar`, `windows.tar.gz` |
| TN0 | `tn0.zip` |
| TN1 DS-TCN 64 | `tn1_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed{0,1,2}.zip` |
| TN1 LSTM 352 | `tn1_lstm_mse_corr0.9_seed{1,2}.zip`, `tn1_lstm.zip` |
| TN1 LSTM 67 | `tn1_lstm_h67_mse_corr0.9_seed{0,1,2}.zip` |
| TN1 CNN-LSTM 58 | `tn1_cnn_lstm_h58_c32_k5_mse_corr0.9_seed{0,1,2}.zip`, `tn1_cnn_lstm_h58.zip` |
| Mốc LSTM 352 trên GHIJ | `tn1_lstm.zip` — thư mục `tn1_ghij/` bên trong |
| TN2 tầm nhìn | `tn2_rf_c64_4fold.zip` |
| TN3 hàm loss | `tn3_ds_tcn_c64.zip` (RF61), `tn3_ds_tcn_c64_k5.zip` (RF121) |
| TN4 | `tn4_<run_id>.zip`, runner tự tạo sau mỗi seed |

Các ZIP **tích luỹ**: `save_results.py` chạy lại sau mỗi seed, nên bản của seed
cuối chứa cả những seed trước. Danh mục đầy đủ: [DANH_MUC_ZIP.md](DANH_MUC_ZIP.md).

### 0.9. Tổng hợp điểm và bàn giao kết quả

```python
!python scripts/compare_cv.py --experiment tn1
!python scripts/compare_cv.py --experiment tn2_rf
!python scripts/compare_cv.py --experiment tn3
!python scripts/compare_cv.py --experiment tn4 --final
!python scripts/compare_cv.py --experiment tn1_ghij --final
```

Chế độ `--final` gộp nhiều seed thành trung bình ± độ lệch chuẩn mẫu, dùng cho
bảng G H I J; nó không có cột `cv_score` vì test cuối không chia fold.

- **CV:** dòng `fold=TONG` ghi điểm gộp của một cấu hình ở một seed; kiểm đủ bốn
  fold trước khi gọi là CV đầy đủ. `score_std` của dòng đó là dao động giữa
  **bốn fold**, không phải giữa các seed.
- **Giữa các seed:** tính điểm của từng seed trước, rồi mới lấy trung bình và
  độ lệch chuẩn mẫu (`ddof=1`). Không chọn seed cao nhất làm đại diện.
- **`scores_*.csv`:** giữ `user`, `session_file`, `bin`, `method`,
  `n_candidates_kept`, `pearson`. Đây là dữ liệu để tính lại macro/micro và so
  lựa chọn từng phiên.
- **`curve.csv`:** dùng kiểm tra hội tụ. `train_loss` khác thang khi đổi loss,
  không dùng nó thay điểm chọn sóng cuối.
- **`git_commit`, `device`:** xem trong dòng summary của từng lượt. Commit trong
  `README.txt` là lúc đóng gói, có thể khác commit lúc train. Cột `device` chỉ
  ghi `cuda` hay `cpu`; thời gian chạy ở các phiên Colab khác nhau **không** dùng
  làm bằng chứng model nhanh hơn.

Bộ bàn giao gồm: code kèm commit, notebook có output, checksum dữ liệu, ZIP kết
quả chứa checkpoint + scores + curve + summary, và đường dẫn tải ZIP có quyền
truy cập. Repo ghi đường dẫn Drive theo quy ước nhưng **chưa có liên kết chia sẻ
công khai đã được xác minh cho toàn bộ artifact**. Không ghi là đã bàn giao đủ
checkpoint/CSV nếu mới chỉ có output notebook.

Nguồn cho hướng dẫn vận hành: [make_npz.py](../scripts/make_npz.py),
[make_windows.py](../scripts/make_windows.py),
[restore_processed_data_on_drive.py](../scripts/restore_processed_data_on_drive.py),
[run_cv.py](../scripts/run_cv.py), [run_final_test.py](../scripts/run_final_test.py),
[save_results.py](../scripts/save_results.py),
[training.py](../src/training.py), [results.py](../src/results.py).

## 1. Mục tiêu và bài toán

Nghiên cứu mô hình dự báo dùng để chọn tín hiệu hô hấp từ các ứng viên radar UWB
theo pipeline MobiVital. Model nhận 200 mẫu lịch sử và dự báo 25 mẫu tiếp theo
của chính ứng viên. Dự báo được đối chiếu với tương lai radar đã quan sát để chấm
khả năng tự dự báo, rồi chọn ứng viên.

Sóng được đánh giá cuối là **sóng radar của ứng viên được chọn**, không phải ghép
các dự báo thành sóng đai. GT đai dùng để chọn dữ liệu train và để chấm kết quả;
GT **không** tham gia lựa chọn ứng viên lúc inference.

Câu hỏi của nhánh này: **kiến trúc nào chọn tín hiệu tốt nhất, ở ngân sách tham
số nào?**

## 2. Giao thức và cách đọc điểm

- Tập phát triển: **ABCDEFKL**; bốn fold validation **AB, CE, DF, KL**, mỗi fold
  train trên sáu người còn lại.
- Tập kiểm tra: **G H I J**, 537 phiên. Mô hình cuối train đủ tám người phát
  triển rồi chấm một lần trên đó.
- **G H I J không dùng để chọn bất cứ thứ gì** — không chọn kiến trúc, không
  chọn tầm nhìn, không chọn alpha. Mọi lựa chọn nằm ở TN1, TN2, TN3, đều trên
  validation.
- TN1 và TN4: **3 seed**. TN2 và TN3: **seed 0** cho mỗi cấu hình.
- Điểm chính: **Pearson macro theo người**. Riêng bảng tái lập TN0 dùng **micro
  theo phiên**, giải thích ngay dưới.
- Nền train giống nhau ở mọi cấu hình: Adam, LR 1e-4, weight decay 0, batch 64,
  20 epoch, checkpoint sau epoch cuối. MSE ở TN1 và TN2; TN3 và TN4 đổi loss,
  không đổi gì khác.

Các phép sàng lọc một fold được ghi riêng và **không** đưa vào bảng kết luận.
Không lấy điểm của riêng fold KL thay cho CV macro.

Độ lệch chuẩn giữa các seed **không phải kiểm định thống kê**. Nó chỉ nói: chênh
lệch nhỏ hơn con số đó thì chưa xếp hạng được.

Tập kiểm tra tách theo người. **G H I J đã được chấm ở TN0**, phần tái lập
MobiVital — không mô tả nó là chưa từng được nhìn, cũng không nói "chỉ mở đúng
một lần".

### 2.1. Macro theo người và micro theo phiên là gì?

Trước tiên, **mỗi phiên đo có một điểm Pearson** giữa sóng radar được chọn và sóng đai tham chiếu của phiên đó. Macro và micro khác nhau ở cách gộp các điểm phiên này.

| Cách gộp | Cách tính | Ai có trọng số bằng nhau? |
|---|---|---|
| **Macro theo người** | Tính điểm trung bình các phiên của từng người → lấy trung bình các người. | Mỗi người có trọng số như nhau, dù có nhiều hay ít phiên. |
| **Micro theo phiên** | Cộng điểm của tất cả phiên → chia tổng số phiên. | Mỗi phiên có trọng số như nhau; người có nhiều phiên ảnh hưởng nhiều hơn. |

Với người `u` có `n_u` phiên và điểm phiên là `r_u,j`:

```text
Điểm người u = tổng điểm các phiên của u / n_u
Macro        = tổng điểm trung bình của từng người / số người
Micro        = tổng điểm của tất cả phiên / tổng số phiên
```

**Ví dụ minh họa, không phải kết quả thực nghiệm:** người A có 100 phiên, trung bình 0,50; người B có 10 phiên, trung bình 0,90.

```text
Macro = (0,50 + 0,90) / 2                  = 0,7000
Micro = (100 × 0,50 + 10 × 0,90) / 110    = 0,5364
```

Micro **không mặc định cao hơn hoặc tốt hơn** macro. Nếu người có nhiều phiên đạt điểm thấp, micro bị kéo xuống; nếu họ đạt điểm cao, micro được kéo lên. Chọn macro làm điểm chính để mỗi người được tính ngang nhau trong đánh giá khả năng áp dụng sang người khác.

Trong tài liệu này, “micro” là **trung bình điểm Pearson theo phiên**, không phải nối mọi mẫu của mọi phiên thành một chuỗi rồi tính một Pearson duy nhất.

### 2.2. Thực nghiệm nào dùng điểm nào?

| Thực nghiệm | Dữ liệu đánh giá | Điểm dùng trong bảng chính |
|---|---|---|
| **TN0** | 537 phiên G H I J | **Micro theo phiên**, khớp cách bài báo MobiVital báo cáo. |
| **TN1, TN2, TN3** | Validation AB, CE, DF, KL | **Macro theo người**. Mỗi người xuất hiện đúng một lần ở validation; tám người có trọng số bằng nhau. |
| **TN4** | G H I J, sau khi train đủ tám người | **Macro theo người** là điểm chính; micro có ghi kèm, dán nhãn riêng. |

Với nhiều seed: tính điểm của **từng seed** trước, rồi báo cáo trung bình ± độ
lệch chuẩn mẫu (`ddof=1`). "3 seed" không phải ba người hay ba fold.

**Không đặt micro của TN0 cạnh macro của TN4 rồi lấy hiệu để kết luận cải
thiện.** Hai con số đó khác cả thước đo lẫn mục đích.

## 3. Mười bốn notebook

Khi chạy Colab, ô setup tải mã của nhánh `final_submission`.

| Bước | Notebook | Nhiệm vụ |
|---|---|---|
| Chuẩn bị | [DATA_PREPARE](../notebooks/DATA_PREPARE.ipynb) | Dựng và lưu dữ liệu đã xử lý để dùng chung. |
| **TN0** | [TN0](../notebooks/TN0.ipynb) | Tái lập điểm và đối chiếu bộ chọn kênh với MobiVital. |
| **TN1** | [DS-TCN 64 k3n4](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb) | Cấu hình được chọn, 3 seed × 4 fold. |
| **TN1** | [LSTM 352](../notebooks/TN1_LSTM.ipynb) | Kiến trúc MobiVital, làm mốc. CV và cả nhánh G H I J. |
| **TN1** | [LSTM 67](../notebooks/TN1_LSTM_small.ipynb) | Mốc cùng ngân sách tham số. CV và cả nhánh G H I J. |
| **TN1** | [CNN-LSTM 58](../notebooks/TN1_CNN_LSTM.ipynb) | Mốc lai tích chập + hồi quy, cùng ngân sách. |
| **TN2** | [Tầm nhìn, 4 fold](../notebooks/TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb) | kernel 5, 7, 9 — vòng kết luận. |
| **TN2** | [Tầm nhìn, 1 fold](../notebooks/TN2_ReceptiveField_DS_TCN_c64.ipynb) | Vòng **sàng lọc**, không đưa vào bảng kết luận. |
| **TN3** | [Loss, RF61](../notebooks/TN3_HybridLoss_DS_TCN_c64.ipynb) | Quét alpha, giữ 0,6. |
| **TN3** | [Loss, RF121](../notebooks/TN3_HybridLoss_DS_TCN_c64_rf121.ipynb) | Quét alpha, giữ 0. |
| **TN4** | [Test RF61](../notebooks/TN4_final_test_ds_tcn_c64.ipynb) | Mô hình cuối trên G H I J, 3 seed. |
| **TN4** | [Test RF121](../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb) | Pearson thuần và đối chứng MSE, mỗi loss 3 seed. |

Hai notebook công cụ không train gì:
[NAP_KET_QUA_TN1](../notebooks/NAP_KET_QUA_TN1.ipynb) và
[NAP_MOC_GHIJ](../notebooks/NAP_MOC_GHIJ.ipynb) lấy kết quả đã chạy từ tệp nén
trên Drive về; [TAI_ZIP_TN123](../notebooks/TAI_ZIP_TN123.ipynb) và
[TAI_ZIP_TN4](../notebooks/TAI_ZIP_TN4.ipynb) gom tệp nén.

**Ba notebook thiếu log ở vài ô**, ghi rõ để không ai hiểu nhầm là chưa chạy:
`TN2_..._c64_4fold` thiếu 1 trong 4 ô train, `TN2_..._c64` (sàng lọc) thiếu 1
trong 6. Kết quả của những ô đó vẫn nằm trong `runs/`, chỉ là log không được lưu
lại trong notebook.

## 4. Quá trình lựa chọn

### TN0 — kiểm tra pipeline

**Mục tiêu:** kiểm tra việc triển khai pipeline MobiVital trước khi thay LSTM bằng TCN/DS-TCN. TN0 có ba phép riêng, không phải ba seed hay ba fold. Tất cả điểm dưới đây là **micro trên 537 phiên GHIJ**.

| Phép | Đầu vào và việc đã làm | Kiểm tra điều gì? |
|---|---|---|
| **TN0a — chấm lại lựa chọn có sẵn** | TXT lựa chọn ứng viên do tác giả cung cấp → lấy sóng theo `(bin, method)` → so với đai → lấy trung bình điểm phiên. Chạy evaluator phía tác giả và phía project. | Hai evaluator có chấm nhất quán khi nhận cùng lựa chọn không? Không train, không chạy model để chọn lại. |
| **TN0b — chạy lại inference** | Checkpoint LSTM công bố → chạy chọn ứng viên bằng hai pipeline → ghi TXT/CSV → so lựa chọn và điểm từng phiên. | Pipeline inference của project có cho cùng lựa chọn với phía tác giả khi dùng cùng checkpoint không? Không train lại. |
| **TN0c — train lại từ đầu** | Dữ liệu train → train LSTM với MSE bằng từng pipeline → checkpoint mới → chọn ứng viên trên GHIJ → chấm điểm. | Kết quả khi tái chạy cả quá trình huấn luyện và inference. Đây là kết quả tham khảo, không yêu cầu hai lượt train ngẫu nhiên ra đúng cùng điểm. |

**Output đã lưu trong notebook TN0 hiện tại:**

| Phép | Phía MobiVital | Phía project | Kết quả đối chiếu |
|---|---:|---:|---|
| TN0a | 0,819481 | 0,819481 | Sai lệch điểm từng phiên lớn nhất bằng 0 trong output; gần mốc 0,819 dùng để đối chiếu bài báo. |
| TN0b | 0,822175 | 0,822175 | Trùng **537/537 cặp `(bin, method)`**; sai lệch điểm từng phiên lớn nhất bằng 0. |
| TN0c | 0,812839 | 0,822642 | Hai kết quả train lại được ghi để tham khảo; không phải điểm trung bình ba seed. |

“Trùng kênh” ở TN0b được kiểm tra bằng **cả bin và cách biểu diễn**, ví dụ `(24, phase)`; cùng bin nhưng khác `abs/phase` không tính là cùng ứng viên. TN0a và TN0b có thể khác điểm vì TN0a dùng TXT được cung cấp, còn TN0b sinh lại lựa chọn từ checkpoint.

Kết quả a/b hỗ trợ tính nhất quán của chấm điểm và inference trong phép thử này. Không diễn giải thành chứng minh mọi bước xử lý dữ liệu, mọi checkpoint hoặc mọi mô hình đều tương đương.

**Mã thực hiện và dấu vết kết quả:**

- [scripts/mobivital/run_tn0.py](../scripts/mobivital/run_tn0.py): chạy các bước phía mã MobiVital.
- [scripts/run_tn0.py](../scripts/run_tn0.py): chạy phía project và đối chiếu. Hàm `report`/`compare` lấy trung bình điểm phiên, vì vậy bảng TN0 là micro.
- [TN0.ipynb](../notebooks/TN0.ipynb): lệnh và output đối chiếu hiện đang lưu trong bản nộp.
- Phía mã tác giả có bổ sung **`model.eval()`** khi inference bằng [apply_patched_files.py](../scripts/mobivital/apply_patched_files.py); không mô tả là chạy mã tác giả hoàn toàn không chỉnh sửa.

Trong `runs/tn0/` hiện có `TN0a.txt`, `TN0b.txt`, `scores_TN0a.csv`, `scores_TN0b.csv`, `scores_ours_a.csv` và `README.txt`. Output notebook còn ghi việc sinh `compare.csv` cùng các kết quả b/c phía project, nhưng các tệp đó **chưa có trong thư mục được lưu ở repo hiện tại**. Do đó, bảng trên dẫn output notebook; không tuyên bố đã đóng gói đầy đủ mọi CSV của cả ba phép.

#### TN0 và các tệp outdated: đọc cái nào?

| Tệp / nhóm tệp | Trạng thái và cách dùng |
|---|---|
| `notebooks/TN0.ipynb` | Notebook tái lập được giữ trong bản nộp; dùng output hiện có cho bảng TN0 ở trên. |
| `notebooks/TN0.md` | Ghi chú lịch sử, có phần từ MacBook/CPU ngày 02/09/2026, cách dựng `work` và kết quả TN0c cũ **0,798748**. Không thay số này vào bảng Colab hiện tại. Đã đánh dấu outdated ở đầu tệp. |
| `runs/tn0/` | Các TXT/CSV đã lưu để truy vết; xem từng tệp, không suy ra thư mục đã có đủ a/b/c chỉ từ tên thư mục. |
| Ghi chú giao thức trước đó (đã loại khỏi bản nộp) | Mô tả giai đoạn trước, gồm cách chia TN0 khác hiện tại. Dùng notebook/runner hiện tại và tài liệu này để mô tả bản nộp. |
| `old_expreiment_outdated_donotuse/THUC_NGHIEM_1.ipynb`, `THUC_NGHIEM_2.ipynb`, `THUC_NGHIEM_3.ipynb`, `THUC_NGHIEM_4_alpha07.ipynb` | Notebook khảo sát cũ ở máy local, ngoài danh mục push. **Không phải TN0a/b**, cũng không tự động tương ứng TN1 hiện tại. Không trộn điểm khi khác split, metric hoặc cấu hình. |

Outdated nghĩa là không còn dùng làm hướng dẫn chạy/kết luận chính cho bản nộp hiện tại; không có nghĩa mọi số liệu cũ đều sai. Khi trích lại phải ghi rõ lần chạy và giao thức.

### TN1 — bốn kiến trúc, cùng một giao thức

**Cấu hình được so:** tất cả nhận 200 mẫu, xuất 25 mẫu, train MSE, seed 0/1/2 ×
4 fold. Không cấu hình nào được ưu ái thêm epoch, thêm seed hay đổi learning rate.

| Cấu hình | Họ | Cấu tạo | Tham số | Seed × fold |
|---|---|---|---:|---|
| **DS-TCN 64, k3 n4** | tích chập | 4 khối, dilation 1-2-4-8, không norm, `nn.Dropout(0.2)` theo phần tử | **37.081** | 3 × 4 |
| LSTM 352 | hồi quy | 2 tầng, một chiều — kiến trúc MobiVital, giữ nguyên | 1.502.713 | 3 × 4 |
| LSTM 67 | hồi quy | như trên, thu nhỏ để cùng ngân sách tham số | 56.908 | 3 × 4 |
| CNN-LSTM 58 | lai | 2 tầng Conv1d stride 2 (32 kênh, k5) rồi LSTM 2 tầng | 55.667 | 3 × 4 |

**Cấu trúc thực sự của DS-TCN trong code:** đầu vào qua Conv1d kernel 1 để tăng
từ một kênh lên C kênh. Mỗi khối có hai tầng; mỗi tầng làm đệm trái →
convolution → chuẩn hoá (nếu có) → ReLU → dropout. Convolution của DS-TCN gồm
depthwise rồi pointwise, **không chèn activation giữa hai phép này**. Sau hai
tầng, cộng nhánh tắt, **không có ReLU sau phép cộng**. Cuối mạng lấy vị trí thời
gian cuối rồi Linear(C, 25). Các khối giữ nguyên C nên nhánh tắt là identity.

Tầm nhìn tính bằng `1 + 2 × (kernel − 1) × Σ dilation`, ra **61**. Đây là tầm
nhìn lý thuyết; với đầu vào 200 mẫu có đệm trái, model chỉ thấy 61 mẫu gần nhất.

**Khối tích chập này là thiết kế của đồ án**, không phải bản tái lập một kiến
trúc đã công bố. Tài liệu không gọi nó là bản cài đặt của bài báo nào, và các
lựa chọn cụ thể — 4 khối, bỏ chuẩn hoá, dropout 0,2 theo phần tử — là thiết lập
được đánh giá **trong cả cấu hình**, chưa có thí nghiệm riêng chứng minh từng
cái tối ưu hay là nguyên nhân tăng điểm.

| Phép so trong TN1 | Yếu tố thay đổi | Kết luận được phép rút ra |
|---|---|---|
| DS-TCN 64 ↔ LSTM 67 | Họ kiến trúc; ngân sách tham số gần nhau (37k so với 57k) | So tích chập với hồi quy ở cùng cỡ model. |
| DS-TCN 64 ↔ CNN-LSTM 58 | Họ kiến trúc; ngân sách gần nhau (37k so với 56k) | So tích chập thuần với bản lai tích chập + hồi quy. |
| LSTM 67 ↔ CNN-LSTM 58 | Thêm tầng tích chập phía trước, chuỗi vào LSTM ngắn đi 4 lần | Hai thứ đổi cùng lúc; nếu hơn thì chưa biết nhờ cái nào. |
| DS-TCN 64 ↔ LSTM 352 | Cả kiến trúc lẫn ngân sách (37k so với 1,5 triệu) | Chỉ đọc được là "ngang điểm với ít hơn 40 lần tham số". |

Nguồn cấu hình: [models.py](../src/models.py) và bốn notebook ở mục 3.

### Kết quả TN1

| Cấu hình | Tham số | seed 0 | seed 1 | seed 2 | CV macro ± std |
|---|---:|---:|---:|---:|---:|
| **DS-TCN 64, k3 n4** | **37.081** | 0,758245 | 0,760101 | 0,764287 | **0,760878 ± 0,003095** |
| LSTM 352 | 1.502.713 | 0,760698 | 0,752525 | 0,757769 | 0,756998 ± 0,004141 |
| LSTM 67 | 56.908 | 0,753584 | 0,751081 | 0,754958 | 0,753208 ± 0,001966 |
| CNN-LSTM 58 | 55.667 | 0,749769 | 0,756900 | 0,751329 | 0,752666 ± 0,003749 |

**Đọc được gì:**

DS-TCN 64 hơn LSTM 67 **0,0077** và hơn CNN-LSTM 58 **0,0082**, với **ít hơn
1,5 lần tham số** so với cả hai. Hai chênh lệch này lớn hơn dao động seed của
mọi cấu hình trong bảng, nên xếp hạng đọc được.

So với LSTM 352, DS-TCN 64 hơn **0,0039** với **ít hơn 40,5 lần tham số**. Con
số 0,0039 **nhỏ hơn** dao động seed của LSTM 352 (0,0041), nên phát biểu an toàn
là **ngang điểm ở ít hơn 40 lần tham số**, không phải "vượt".

**Đọc KHÔNG được:**

Bảng này là bảng chọn cấu hình. Cả bốn dòng đều chấm trên cùng tám người
ABCDEFKL, nên chênh lệch nhỏ có thể là do chọn trúng chứ không phải kiến trúc
tốt hơn. Nhánh này không có tập test độc lập để kiểm chứng điều đó.

DS-TCN 64 khác các mốc ở nhiều thứ cùng lúc — họ kiến trúc, số khối, chuẩn hoá,
dropout, tầm nhìn — nên không quy kết quả cho riêng phép tích chập depthwise.

Tầm nhìn 61 nghĩa là model chỉ thấy **1,2 giây** gần nhất trong khi một nhịp thở
khoảng 4 giây. Nó vẫn cho điểm cao nhất bảng. Đây là quan sát đo được, chưa có
giải thích nào được kiểm chứng.

Bảng đầy đủ kèm cách đọc: [BANG_TCN.md](BANG_TCN.md).

### TN2 — tầm nhìn

Giữ nguyên DS-TCN 64, 4 khối, không chuẩn hoá, dropout 0,2 theo phần tử. Đổi
**đúng một thứ**: kernel. Tầm nhìn tính bằng `1 + 2 × (kernel − 1) × Σ dilation`.

| kernel | tầm nhìn | tham số | CV macro | seed × fold |
|---:|---:|---:|---:|---|
| **3** | **61** | **37.081** | **0,760878** ± 0,003095 | 3 × 4 *(lấy lại từ TN1)* |
| 5 | 121 | 38.105 | 0,757855 | 1 × 4 |
| 7 | 181 | 39.129 | 0,743657 | 1 × 4 |
| 9 | 241 | 40.153 | 0,736970 | 1 × 4 |

**Tầm nhìn càng rộng điểm càng thấp, đơn điệu.** Cửa sổ vào chỉ có 200 mẫu nên
k9 đã phủ dư (241 > 200) mà vẫn không giúp gì.

Chênh RF61 với RF121 là **0,0030**, xấp xỉ dao động seed của RF61 — chưa tách
được hai cái này. Vì vậy **cả hai** được mang xuống TN3, để xem hàm loss có đổi
thứ hạng không. Chênh với RF181 và RF241 thì đủ lớn để loại.

Một chi tiết phải ghi: cấu hình được chọn chỉ nhìn **61 mẫu gần nhất**, tức
**1,2 giây** ở tần số 50 Hz, trong khi một nhịp thở khoảng 4 giây. Nó không nhìn
đủ một chu kỳ thở mà vẫn cho điểm cao nhất. Đây là quan sát đo được, chưa có
giải thích nào được kiểm chứng.

### TN3 — hàm loss lai

`loss = alpha × MSE + (1 − alpha) × (1 − Pearson)`. `alpha` là trọng số của
**MSE**; `alpha = 0` là Pearson thuần, `alpha = 1` là MSE thuần.

**Vì sao thử:** mọi sóng ứng viên đã được kéo về `[-1, 1]` trước khi vào model,
nên biên độ không còn mang thông tin phân biệt. MSE thì phạt sai biên độ. Model
dành sức khớp một thứ đã bị chuẩn hoá mất là phí.

| alpha | DS-TCN 64/RF61 | DS-TCN 64/RF121 |
|---:|---:|---:|
| 0 | 0,776667 | **0,780306** |
| 0,1 | 0,769390 | 0,763183 |
| 0,2 | 0,775264 | 0,771996 |
| 0,3 | 0,771931 | 0,776213 |
| 0,4 | 0,779266 | 0,771848 |
| 0,5 | 0,779419 | 0,763457 |
| 0,6 | **0,780028** | 0,752386 |
| 0,7 | 0,771665 | 0,761892 |
| 0,8 | 0,776611 | 0,760013 |
| 0,9 | 0,764941 | 0,758236 |
| **1 — MSE thuần** | 0,760878 *(TN1)* | 0,757855 *(TN2)* |

**19 trong 20 mức alpha hơn MSE thuần.** Mức tốt nhất hơn **0,0192** (RF61) và
**0,0225** (RF121). Ngoại lệ duy nhất là RF121 ở alpha 0,6 — 0,752386, thấp hơn
mốc MSE của chính nó 0,0055. Một seed nên đọc là nhiễu, nhưng ghi ra chứ không
làm tròn thành "mọi mức đều hơn".

**Đừng đọc từng mức alpha như một xếp hạng.** Dải 0,776–0,780 của RF61 trải
trong khoảng 0,004 — đúng cỡ dao động seed — mà bảng này chỉ có **một** seed.
Kết luận đọc được là *"đưa Pearson vào loss thì tốt hơn MSE thuần"*, không phải
*"0,6 là mức tối ưu"*.

Giữ `RF61 + alpha 0,6` và `RF121 + alpha 0` mang xuống TN4.

### TN4 — kiểm tra trên G H I J

Train đủ tám người ABCDEFKL rồi chấm 537 phiên của bốn người chưa dùng để chọn
bất cứ thứ gì. Ba seed mỗi tổ hợp.

| cấu hình | tham số | macro | micro | thực nghiệm |
|---|---:|---:|---:|---|
| **LSTM 352** *(mốc MobiVital)* | 1.502.713 | **0,810302** ± 0,015402 | 0,805309 | TN1 GHIJ |
| **DS-TCN 64/RF61, alpha 0,6** | **37.081** | **0,803590** ± 0,015350 | — | TN4 |
| DS-TCN 64/RF121, Pearson thuần | 38.105 | 0,801739 ± 0,009968 | — | TN4 |
| DS-TCN 64/RF121, MSE thuần *(đối chứng)* | 38.105 | 0,762191 ± 0,021433 | — | TN4 |

**Hàm loss là thứ có tác dụng rõ nhất.** Cùng kiến trúc 64/RF121, chỉ đổi loss:
Pearson thuần **0,801739** so với MSE thuần **0,762191** — chênh **0,0395**, gần
bốn lần dao động seed. Đây là phép so sạch nhất trong cả đồ án: cùng kiến trúc,
cùng ba seed, cùng dữ liệu, khác đúng một thứ. Và nó khớp chiều với TN3.

**Nhưng thứ hạng của TN1 không giữ nguyên sang TN4.**

| | CV (ABCDEFKL) | G H I J |
|---|---:|---:|
| DS-TCN 64 | **0,760878** | 0,803590 |
| LSTM 352 | 0,756998 | **0,810302** |

Trên tập dùng để chọn, DS-TCN đứng đầu. Trên tập không dùng để chọn, LSTM 352
đứng đầu. Đó là dấu hiệu cấu hình được chọn hợp với tám người ABCDEFKL hơn là
hợp với bài toán nói chung. **Phải ghi rõ chuyện này khi báo cáo, đừng chỉ trình
bảng CV.**

Chênh lệch trên G H I J: DS-TCN thấp hơn LSTM 352 **0,0067** — nhỏ hơn dao động
seed của cả hai bên (0,0154), nên không xếp hạng được.

**Phát biểu đúng:** DS-TCN **37.081** tham số cho kết quả **ngang** LSTM
**1.502.713** tham số trên tập kiểm tra độc lập — ít hơn **40,5 lần**. Không nói
"tốt hơn", ở cả hai chiều.

Bảng đầy đủ kèm cách đọc: [BANG_TCN.md](BANG_TCN.md).

## 5. Vì sao mô hình cuối có các tham số này?

**Mô hình cuối: DS-TCN 64, kernel 3, 4 khối, không chuẩn hoá, dropout 0,2 theo
phần tử, loss lai alpha 0,6 — 37.081 tham số.**

| Thành phần | Thiết lập | Lý do và mức bằng chứng |
|---|---|---|
| Tích chập tách rời | Depthwise theo thời gian + pointwise trộn kênh | Ở C=64, k=3: 12.352 tham số xuống 4.416. Giảm trọng số là mục tiêu của đồ án. |
| Số kênh | 64 | Nhánh gọn có điểm CV tốt ở ngân sách nhỏ. **Không** khảo sát riêng số kênh trên nhánh này. |
| Số khối | 4 | Giữ độ sâu cố định để TN2 khảo sát tầm nhìn qua kernel. **Không** khẳng định 4 khối tối ưu. |
| Kernel / tầm nhìn | 3 / 61 | **Có bằng chứng: TN2.** Tầm nhìn rộng hơn cho điểm thấp hơn, đơn điệu qua 61 → 121 → 181 → 241. |
| Chuẩn hoá | Không có trong khối | Giữ cấu trúc đơn giản. Lý do để đưa vào khảo sát, **chưa** chứng minh bỏ chuẩn hoá tự làm tăng điểm. |
| Dropout | Theo phần tử, p = 0,2 | Cửa sổ 200 mẫu trượt 25 mẫu chồng lấn rất nhiều, nên số cửa sổ không phải số quan sát độc lập. Đó là động cơ dùng regularization, **không** chứng minh riêng p = 0,2 tối ưu. |
| Hàm loss | Lai, alpha 0,6 | **Có bằng chứng mạnh nhất: TN3 và TN4.** Hơn MSE thuần 0,0192 trên CV và 0,0395 trên G H I J với đối chứng cùng kiến trúc. Riêng **mức** 0,6 thì chỉ dựa trên một seed. |
| Vào / ra | 200 → 25 | Giữ giao diện và cách chấm của pipeline MobiVital. |
| LR / batch / epoch | 1e-4 / 64 / 20 | Thiết lập chung cho mọi cấu hình; không tối ưu riêng cho cái nào. |

Phải phân biệt **lý do đưa một thiết lập vào khảo sát** và **bằng chứng cho hiệu
quả của nó**. Trong bảng trên chỉ có hai dòng — kernel và hàm loss — được một
thực nghiệm cô lập riêng. Các dòng còn lại là lựa chọn thiết kế đi kèm cả cấu
hình, không quy được chênh lệch cho riêng chúng.

## 6. Kiểm mã trước khi train

```bash
python3 scripts/check_model.py --model ds_tcn --channels 64 --kernel_size 3 \
    --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element
python3 scripts/check_model.py --model cnn_lstm --hidden 58 --compare-with lstm --compare-hidden 67
```

Một lượt CV mất một tới ba giờ. Bản cài sai một chi tiết vẫn chạy, vẫn ra số,
rồi cho kết luận "kiến trúc này thua" trong khi thật ra là mã hỏng.
`check_model.py` kiểm shape vào ra, giá trị hữu hạn, gradient lan ngược được, số
tham số, lưu và nạp lại `state_dict`; riêng họ tích chập kiểm thêm loại chuẩn
hoá, loại dropout và tầm nhìn; riêng `cnn_lstm` kiểm chuỗi vào LSTM đúng 50 bước
và LSTM là một chiều.

## 7. Tài liệu đọc tiếp

- [BANG_TCN.md](BANG_TCN.md) — bảng kết quả TN1 đến TN4 và những gì đọc được
- [CHIA_DU_LIEU.md](CHIA_DU_LIEU.md) — vì sao chia theo người, bốn fold cố định
- [PIPELINE_2.md](PIPELINE_2.md) — sơ đồ train và inference từng khối
- [DANH_MUC_ZIP.md](DANH_MUC_ZIP.md) — tệp nén kết quả trên Drive
- [SO_DO_DU_LIEU.md](SO_DO_DU_LIEU.md) — dữ liệu đi từ CSV thô tới cửa sổ train
- [CAU_TRUC_MA_NGUON.md](CAU_TRUC_MA_NGUON.md) — vai trò từng phần mã nguồn
- [ARTIFACTS.md](../ARTIFACTS.md) — danh mục bàn giao
- [notebooks/TN0.md](../notebooks/TN0.md) — ghi chú TN0, có phần đã lỗi thời

**Nhánh này không trích dẫn kiến trúc từ bài báo nào.** Khối tích chập trong
`src/models.py` là thiết kế của đồ án; không có câu nào nói "cài theo bài X".
Tham chiếu ngoài duy nhất còn giữ là [MobiVital](https://arxiv.org/abs/2503.11064)
— bài gốc mà đồ án cải tiến, là nguồn của pipeline, của mốc LSTM 352, và là đích
của TN0.
