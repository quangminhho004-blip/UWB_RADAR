# Tóm tắt thesis — thực nghiệm TCN và DS-TCN

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar

**Nhánh nộp:** `submission`

**Hai mô hình cuối được chọn, theo thứ tự ưu tiên:**

1. **DS-TCN 64/RF121 — phương án chính:** Pearson loss thuần (α = 0), **38.105 tham số**.
2. **DS-TCN 64/RF61 — phương án thứ hai:** hybrid loss (α = 0,6), **37.081 tham số**.

Thứ tự này thể hiện phương án được nhóm ưu tiên sử dụng và trình bày; không phải thứ hạng điểm test. Nhánh C192 được giữ làm đối chứng dung lượng.

**Cách đọc:** mục **0** là hướng dẫn setup → dữ liệu/Drive → chạy → lưu kết quả; mục **1–2** là bài toán và giao thức; mục **3–4** là notebook, thiết kế và kết quả TN0–TN4; mục **5–7** giải thích lựa chọn cuối và nguồn chi tiết. Các lệnh là hướng dẫn tái chạy, không phải thông báo vừa train lại.

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

Min/max lấy trên từng chuỗi của một phiên trước khi cắt cửa sổ, không gộp toàn bộ người/dataset. Chuỗi hằng số được đưa về toàn 0. Min–max này khác RevIN theo cửa sổ và khác BatchNorm trong model. Xem [PIPELINE_2.md](PIPELINE_2.md) để trình bày bằng sơ đồ.

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

- **TN1–TN4 qua runner project:** dữ liệu xử lý đầy đủ đủ để train/chấm; `scoring.py` đọc radar/GT từ `by_user`. Không cần tải lại CSV chỉ để chạy các runner này.
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

Checkpoint công bố dùng cho TN0b: `external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth`. TN0c phía project đặt seed **1234**, train LSTM-352 hai layer bằng MSE, 20 epoch; đây không phải nhóm seed 0/1/2 của TN1–TN4. Không coi TN0c là phép so cùng seed hoàn toàn giữa hai vòng train nếu chưa đối chiếu cách đặt RNG phía tác giả.

Kỳ vọng đối chiếu lịch sử: TN0a micro **0,819481**, TN0b micro **0,822175**, trùng **537/537 cặp `(bin, method)`**; TN0c ghi tham khảo. Bảng giải thích đầy đủ và giới hạn nằm ở mục 4.

Gói tạo ra: `runs/tn0.zip` → Drive `mobivital/tn0.zip`. Phải giữ TXT lựa chọn, scores từng phiên, `compare.csv`, log/output và checkpoint train lại nếu cần tái lập inference. **Repo hiện chưa chứa đầy đủ các artifact này**, dù notebook có output; kiểm trong ZIP của lần chạy thực tế trước khi bàn giao. `summary.csv` ít hoặc không có dòng không tự có nghĩa TN0 thất bại, vì bảng so TN0 dùng `compare.csv` và `scores_*.csv`.

### 0.7. TN1 → TN2 → TN3 → TN4: chạy gì, thay gì?

Không cần Optuna. Dùng các cấu hình đã định trong notebook; mỗi lượt dựng model và train từ đầu theo seed. Mũi tên đi tiếp nghĩa là giữ **cấu hình**, không phải fine-tune checkpoint TN trước.

| Bước | Thay đổi / phạm vi chạy | Đầu vào train → đầu vào chấm | Đầu ra quyết định |
|---|---|---|---|
| TN1 | 5 cấu hình nền, mỗi cấu hình seed 0/1/2 × 4 fold | Windows của 6 người train → phiên radar của 2 người validation | CV macro; giữ C64/RF61 và C192/RF61 cho khảo sát tiếp. |
| TN2 RF | C64/C192, k5/k7/k9, mỗi cấu hình seed 0 × 4 fold; k3 dùng lại TN1 | Như TN1 | Giữ 64/RF61, 64/RF121, 192/RF121. |
| TN2 RevIN | DS-TCN nền RF253 + RevIN, seed 0/1/2 × 4 fold | Như TN1 | Ghi kết quả giảm điểm; không đi tiếp nhánh này. |
| TN3 | Ba cấu hình giữ lại; α = 0,0 đến 0,9 bước 0,1; seed 0 × 4 fold | Như TN1 | Chọn alpha riêng; MSE làm mốc từ TN1/TN2, không cần train lại chỉ để đổi tên TN. |
| TN4 | Bốn tổ hợp chính: RF121 Pearson, RF61 hybrid, C192 hybrid, RF121 MSE; mỗi tổ hợp 3 seed | `windows/final_train/` ABCDEFKL → `by_user/` GHIJ | Test macro từng seed, trung bình ± sample std. |

Bốn fold: `val_AB` train CDEFKL; `val_CE` train ABDFKL; `val_DF` train ABCEKL; `val_KL` train ABCDEF. Windows validation đã lọc bằng GT chỉ dùng cho đường cong `val_mse/val_pearson`; **chọn cấu hình bằng điểm chọn ứng viên trên toàn bộ phiên validation**, không chọn bằng loss trên windows đã lọc.

**Lệnh mẫu TN1 C64/RF61, seed 0**; chạy tiếp seed 1, 2 trong notebook:

```python
!python scripts/check_model.py --model ds_tcn --channels 64 --kernel_size 3 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element
!python scripts/run_cv.py --experiment tn1 --model ds_tcn --channels 64 --kernel_size 3 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse --seed 0
```

TN2 RF: cùng cấu trúc bốn block, đổi `--experiment tn2_rf` và `--kernel_size 5`, `7`, `9`; mỗi mức chạy seed 0. Đổi channels sang 192 cho nhánh C192. Không thêm `--folds val_KL` nếu đang chạy CV bốn fold; flag này chỉ dành cho notebook sàng lọc một fold.

**Lệnh mẫu TN3 C64/RF121 với Pearson thuần:**

```python
!python scripts/run_cv.py --experiment tn3 --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.0 --seed 0
```

Các ô sau thay alpha theo dải đã định. `--loss mse_pearson --alpha 0` chính là Pearson thuần; `--alpha` là trọng số MSE, không phải ngưỡng chọn kênh.

**TN4 — phương án chính RF121/Pearson:**

```python
!python scripts/run_final_test.py --experiment tn4 --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.0 --seed 0
!python scripts/run_final_test.py --experiment tn4 --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.0 --seed 1
!python scripts/run_final_test.py --experiment tn4 --model ds_tcn --channels 64 --kernel_size 5 --n_blocks 4 --dropout 0.2 --norm none --dropout_kind element --loss mse_pearson --alpha 0.0 --seed 2
```

Các tổ hợp còn lại vẫn chạy đủ seed 0/1/2:

| Tổ hợp | Thay gì trong lệnh TN4 trên? |
|---|---|
| Ưu tiên 2: C64/RF61 hybrid | `--kernel_size 3 --alpha 0.6` |
| Đối chứng C192/RF121 hybrid | `--channels 192 --alpha 0.2` |
| Đối chứng C64/RF121 MSE | Dùng `--loss mse`, bỏ `--alpha` |
| Mốc DS-TCN nền | Chạy notebook `TN1_final_evaluation.ipynb`, experiment `tn1_ghij`; model DS-TCN C64 mặc định 6 block, BN, dropout 0. |

**Chấm một phiên validation/test:** radar 120 bin → 240 ứng viên abs/phase → chuẩn hóa mỗi chuỗi → inversion detector → cửa sổ 200→25 → model → Pearson dự báo với tương lai của chính ứng viên → cộng điểm 52 cửa sổ → chọn ứng viên → Pearson sóng đã chọn với GT của phiên. `invert_detector` trả 0/1; phép so `< 0.8` trong mã giữ 0, loại 1, không phải lọc Pearson với đai ở ngưỡng 0,8.

### 0.8. Kết quả lưu file gì? Tên trên Drive là gì?

```text
runs/
├── summary.csv                             metric chung, runner tra để bỏ qua lượt đã xong
└── <experiment>/                           tn0, tn1, tn2_rf, tn2, tn3, tn4...
    ├── summary.csv                         bản lọc theo experiment khi save_results
    ├── README.txt                          lúc đóng gói, commit đóng gói
    ├── scores_<run_id>.csv                  điểm + bin/method từng phiên
    ├── <run_id>.txt                         lựa chọn ứng viên (runner test cuối)
    └── <run_id>/
        ├── final.pth                       state_dict sau epoch cuối
        ├── curve.csv                       loss/MSE/Pearson theo epoch
        └── last.pth                        trạng thái resume, chỉ khi đang train dở
```

CV có hậu tố fold trong `run_id`; test cuối có seed nhưng không có fold validation. Ví dụ checkpoint phương án chính, test seed 0:

```text
runs/tn4/ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed0/final.pth
```

`final.pth` chứa trọng số, không chứa đủ định nghĩa kiến trúc; phải giữ run_id/lệnh/cấu hình tương ứng để load đúng. `last.pth` lưu trạng thái train để resume và được xóa khi hoàn tất. Checkpoint local giữa epoch **chưa tự có trên Drive**: runner CV sao lưu sau mỗi fold, runner test cuối sau mỗi lượt train+test. Nếu runtime mất trước lần sao lưu tiếp theo, phần mới làm có thể mất. Các lệnh sao lưu gọi với `check=False`, nên phải kiểm thông báo **“đã chép sang …”** và tệp trên Drive, không chỉ thấy model train xong.

**Tên gói tổng hợp theo notebook:** tất cả nằm dưới `/content/drive/MyDrive/mobivital/` khi Drive đã mount.

| Thực nghiệm | ZIP / quy tắc đặt tên |
|---|---|
| Dữ liệu | `by_user.tar`, `windows.tar.gz` |
| TN0 | `tn0.zip` |
| TN1 nền TCN/DS-TCN | `tn1_tcn_weightnorm.zip` (tên ô lưu cuối; nội dung là cả `runs/tn1/` của runtime đó) |
| TN1 C64/C192 RF61 | `tn1_ds_tcn_rf61_c64.zip`, `tn1_ds_tcn_rf61_c192.zip` |
| TN2 RF bốn fold | `tn2_rf_c64_4fold.zip`, `tn2_rf_c192_4fold.zip` |
| TN2 C64 sàng lọc KL | `tn2_rf_ds_tcn_c64.zip` |
| TN2 RevIN | `tn2_ds_tcn_revin.zip` |
| Phép thử C192 BN/dropout kênh | `tn_test_ds_tcn_c192.zip` |
| TN3 | `tn3_ds_tcn_c64.zip` (RF61), `tn3_ds_tcn_c64_k5.zip` (RF121), `tn3_ds_tcn_c192.zip` |
| TN4 | Runner tự tạo `tn4_<run_id>.zip` sau mỗi seed/tổ hợp. |
| Test DS-TCN nền | `tn1_ghij.zip` qua ô lưu cuối, ngoài các ZIP tự động. |

Runner CV còn tự tạo `<experiment>_<config_id>.zip` sau mỗi fold. ZIP đó chứa checkpoint theo fold trong cùng experiment; không nhầm tên ZIP với tên thư mục phải giải nén.

**Sao lưu thủ công và thêm môi trường vào gói**, ví dụ TN4:

```python
import shutil
for filename in ['environment.json', 'requirements_runtime.txt']:
    if Path('runs', filename).exists():
        shutil.copy2(Path('runs', filename), Path('runs/tn4', filename))
!python scripts/save_results.py tn4 --out tn4_tong_hop
!ls -lh /content/drive/MyDrive/mobivital/tn4_tong_hop.zip
```

`--out` chỉ đổi **tên ZIP**, không lọc nội dung theo model. Cùng tên ZIP sẽ bị thay thế ở lần lưu sau. Nếu chạy lại một cấu hình ở commit/thiết lập khác, đặt `--experiment` riêng và sao lưu riêng để không nhầm với kết quả cũ. Các tên experiment mới sẽ cần điều chỉnh mẫu ZIP trong ô khôi phục.

### 0.9. Tổng hợp điểm và bàn giao kết quả

Sau khi chạy hoặc khôi phục cả metric lẫn scores:

```python
!python scripts/compare_cv.py --experiment tn1
!python scripts/compare_cv.py --experiment tn2_rf
!python scripts/compare_cv.py --experiment tn3
!python scripts/compare_cv.py --experiment tn4
```

- **CV:** `fold=TONG` ghi điểm gộp của cấu hình/seed; kiểm đủ bốn fold trước khi gọi là CV đầy đủ. `score_std` của dòng CV không phải độ lệch chuẩn giữa seed.
- **Test:** một dòng cho mỗi tổ hợp/seed; gộp ba seed thành mean và sample std (`ddof=1`). Không chọn seed có điểm GHIJ cao nhất làm đại diện duy nhất rồi gọi đó là điểm mô hình.
- **`scores_*.csv`:** giữ `user`, `session_file`, `bin`, `method`, `n_candidates_kept`, `pearson`. Đây là dữ liệu để tính lại macro/micro và so lựa chọn từng phiên.
- **`curve.csv`:** dùng kiểm tra hội tụ. `train_loss` khác thang khi đổi loss, không dùng nó thay điểm chọn sóng cuối.
- **`git_commit`, `device`:** xem trong dòng summary của từng lượt. Commit trong `README.txt` là lúc đóng gói, có thể khác commit train của các dòng được khôi phục. Thời gian chạy khác GPU/phiên không dùng làm bằng chứng model nhanh hơn.

Bộ bàn giao nên có code/commit, notebook có output, checksum dữ liệu, ZIP kết quả chứa checkpoint+scores+curve+summary và đường dẫn tải ZIP có quyền truy cập. Hiện repo ghi đường dẫn Drive theo quy ước nhưng **chưa có một liên kết chia sẻ công khai đã được xác minh cho toàn bộ artifact**. Không ghi là đã bàn giao đủ checkpoint/CSV nếu mới có output notebook. Các thiếu hụt TN0 được ghi ở mục 4; bảng từng seed cũng ghi C192/RF241 có điểm từ notebook nhưng thiếu trong ZIP đã đối chiếu.

Nguồn cho hướng dẫn vận hành: [make_npz.py](../scripts/make_npz.py), [make_windows.py](../scripts/make_windows.py), [restore_processed_data_on_drive.py](../scripts/restore_processed_data_on_drive.py), [run_cv.py](../scripts/run_cv.py), [run_final_test.py](../scripts/run_final_test.py), [save_results.py](../scripts/save_results.py), [training.py](../src/training.py), [results.py](../src/results.py).

## 1. Mục tiêu và bài toán

Nghiên cứu mô hình dự báo dùng để chọn tín hiệu hô hấp từ các ứng viên radar UWB theo pipeline MobiVital. Model nhận 200 mẫu lịch sử và dự báo 25 mẫu tiếp theo của chính ứng viên. Dự báo được đối chiếu với tương lai radar đã quan sát để chấm khả năng tự dự báo và chọn ứng viên.

Sóng được đánh giá cuối là **sóng radar của ứng viên được chọn**, không phải ghép các forecast thành sóng đai. GT đai dùng để chọn dữ liệu train và đánh giá kết quả; GT không tham gia lựa chọn ứng viên lúc inference.

Câu hỏi chính: **cấu hình TCN/DS-TCN nào, với RF và loss nào, phù hợp cho nhiệm vụ chọn tín hiệu?**

## 2. Giao thức và cách đọc điểm

- Dev: **ABCDEFKL**; bốn fold validation **AB, CE, DF, KL**, mỗi fold train trên sáu người còn lại.
- Test cuối: **GHIJ**; model cuối train đủ tám người dev.
- TN1: **3 seed × 4 fold**.
- TN2 RF mới và TN3: **seed 0 × 4 fold** cho mỗi cấu hình/alpha.
- TN4: **3 seed 0, 1, 2**, không chạy CV bốn fold.
- Điểm chính TN1–TN4: **Pearson macro theo người**. Riêng bảng tái lập TN0 dùng **micro theo phiên**, giải thích dưới đây.
- Nền train: Adam, LR 1e-4, weight decay 0, batch 64, 20 epoch, checkpoint epoch cuối; MSE cho TN1/TN2.

Các phép sàng lọc một fold được ghi riêng. Không lấy điểm KL thay cho CV macro. Độ lệch chuẩn giữa seed không phải kiểm định thống kê về tương đương.

Tập test được tách theo người. Tài liệu lịch sử có ghi nhận GHIJ từng được xem trong project; không mô tả nó là chưa từng được nhìn trong toàn bộ quá trình. Khảo sát/chọn RF và loss ở các bảng chính dùng validation.

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
| **TN0a/b/c** | 537 phiên của GHIJ | **Micro theo phiên**, khớp cách báo cáo tái lập của runner TN0. |
| **TN1, TN2, TN3 — CV** | Validation AB, CE, DF, KL | **Macro theo người**. Mỗi người dev xuất hiện đúng một lần ở validation; tám người có trọng số bằng nhau. |
| Sàng lọc một fold | Ví dụ validation KL | Macro của riêng K và L; không phải CV tám người. |
| **TN4** | GHIJ, sau khi train đủ tám người dev | **Macro theo người** là điểm chính; micro nếu báo cáo thêm phải ghi nhãn riêng. |

Với nhiều seed: tính điểm của **từng seed** trước, sau đó báo cáo trung bình ± độ lệch chuẩn giữa seed. “3 seed” không phải ba người hay ba fold. Không đặt micro TN0 cạnh macro TN4 rồi lấy hiệu để kết luận cải thiện.

## 3. Thứ tự notebook chính

Các đường dẫn dưới mở notebook trong repo. Khi chạy Colab, ô setup tải mã của **nhánh submission**.

| Bước | Notebook | Nhiệm vụ |
|---|---|---|
| Chuẩn bị | [DATA_PREPARE](../notebooks/DATA_PREPARE.ipynb) | Dựng và lưu dữ liệu xử lý để dùng chung. |
| **TN0** | [TN0](../notebooks/TN0.ipynb) | Tái lập điểm và đối chiếu bộ chọn kênh với MobiVital; giữ LSTM ở đây vì là mốc kiểm chứng. |
| **TN1** | [TCN và DS-TCN nền](../notebooks/TN1_TCN_DSTCN_model_selection.ipynb) | Khảo sát các cấu hình TCN/DS-TCN ban đầu. |
| **TN1** | [DS-TCN 64/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb) | Nền gọn, 3 seed × 4 fold. |
| **TN1** | [DS-TCN 192/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c192.ipynb) | Đối chứng dung lượng lớn, 3 seed × 4 fold. |
| **TN2** | [RF nhóm 64](../notebooks/TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb) | Khảo sát RF121/181/241, so với RF61 nền. |
| **TN2** | [RF nhóm 192](../notebooks/TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb) | Khảo sát RF ở mức dung lượng lớn hơn. |
| **TN3** | [Loss 64/RF61](../notebooks/TN3_HybridLoss_DS_TCN_c64.ipynb) | Quét alpha, giữ α = 0,6. |
| **TN3** | [Loss 64/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c64_rf121.ipynb) | Quét alpha, giữ α = 0. |
| **TN3** | [Loss 192/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c192.ipynb) | Quét alpha, giữ α = 0,2. |
| **TN4** | [64/RF121 — phương án chính](../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb) | Test Pearson thuần và MSE đối chứng, mỗi loss 3 seed. |
| **TN4** | [64/RF61 — phương án thứ hai](../notebooks/TN4_final_test_ds_tcn_c64.ipynb) | Test hybrid α = 0,6, 3 seed. |
| **TN4** | [192/RF121](../notebooks/TN4_final_test_ds_tcn_c192.ipynb) | Test hybrid α = 0,2, 3 seed. |

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
| Ghi chú giao thức/kế hoạch cũ (đã loại khỏi bản nộp) | Có mô tả giai đoạn trước, gồm cách chia TN0 và kế hoạch Optuna khác hiện tại. Dùng notebook/runner hiện tại và tài liệu này để mô tả bản nộp. |
| `old_expreiment_outdated_donotuse/THUC_NGHIEM_1.ipynb`, `THUC_NGHIEM_2.ipynb`, `THUC_NGHIEM_3.ipynb`, `THUC_NGHIEM_4_alpha07.ipynb` | Notebook khảo sát cũ ở máy local, ngoài danh mục push. **Không phải TN0a/b/c**, cũng không tự động tương ứng TN1–TN4 hiện tại. Không trộn điểm khi khác split, metric hoặc cấu hình. |

Outdated nghĩa là không còn dùng làm hướng dẫn chạy/kết luận chính cho bản nộp hiện tại; không có nghĩa mọi số liệu cũ đều sai. Khi trích lại phải ghi rõ lần chạy và giao thức.

### TN1 — giữ cấu hình gọn và đối chứng dung lượng

**Cấu hình được so:** tất cả nhận 200 mẫu, xuất 25 mẫu, train MSE, RevIN tắt, seed 0/1/2 × 4 fold. Mỗi block có hai phép tích chập; DS-TCN thay conv thường bằng depthwise + pointwise.

| Cấu hình | Channels | Block | Kernel | Dilation các block | Norm | Dropout | Tham số |
|---|---:|---:|---:|---|---|---|---:|
| TCN BatchNorm | 64 | 6 | 3 | 1, 2, 4, 8, 16, 32 | BatchNorm | 0 | 151.513 |
| TCN WeightNorm | 64 | 6 | 3 | 1, 2, 4, 8, 16, 32 | WeightNorm | 0 | 150.745 |
| DS-TCN nền | 64 | 6 | 3 | 1, 2, 4, 8, 16, 32 | BatchNorm | 0 | 56.281 |
| DS-TCN 64/RF61 | 64 | 4 | 3 | 1, 2, 4, 8 | Không | `nn.Dropout(0.2)` theo phần tử | 37.081 |
| DS-TCN 192/RF61 | 192 | 4 | 3 | 1, 2, 4, 8 | Không | `nn.Dropout(0.2)` theo phần tử | 307.801 |

Ba bản sáu block có RF253; hai bản bốn block k3 có RF61. So giữa các tổ hợp không cô lập riêng tác dụng bỏ norm, giảm block hoặc thêm dropout. Đối chiếu bảng dưới với [điểm từng seed](BANG_TCN_TUNG_SEED.md).

| Cấu hình | Tham số | CV macro, 3 seed |
|---|---:|---:|
| TCN-64 BatchNorm, RF253 | 151.513 | 0,7423 ± 0,0044 |
| TCN-64 WeightNorm, RF253 | 150.745 | 0,7463 ± 0,0030 |
| DS-TCN nền C64, RF253 | 56.281 | 0,7421 ± 0,0007 |
| **DS-TCN 64/RF61** | **37.081** | **0,7609 ± 0,0031** |
| DS-TCN 192/RF61 | 307.801 | 0,7480 ± 0,0122 |

64/RF61 có điểm trung bình cao nhất và ít tham số nhất trong bảng nên được giữ làm nền gọn. Bản 192 được giữ để khảo sát ảnh hưởng của tăng dung lượng, không gọi là thắng TN1.

### TN2 — giữ nhiều RF để nghiên cứu tiếp

TN2 có **nhánh khảo sát RF** dưới đây và **nhánh RevIN** trình bày ngay sau bảng RF. Trong lệnh chạy, nhánh RF dùng experiment `tn2_rf`, nhánh RevIN dùng `tn2`; cách gọi này không thay đổi file kết quả đã có.

So cùng **seed 0, bốn fold, MSE**:

| RF | Kernel | C64 | C192 |
|---|---:|---:|---:|
| 61 | 3 | 0,758244 | 0,757493 |
| 121 | 5 | 0,757855 | 0,764428 |
| 181 | 7 | 0,743657 | 0,732562 |
| 241 | 9 | 0,736970 | 0,738416 |

Giữ **64/RF61, 64/RF121, 192/RF121**. RF121 không thắng RF61 ở C64 với MSE, nhưng có điểm gần nhau trong lượt khảo sát; nhóm tiếp tục thử loss trên cả hai. Đổi kernel cũng làm thay số trọng số depthwise, nên đây là khảo sát **kernel/RF khi giữ độ sâu và số kênh**.

#### TN2 — nhánh DS-TCN + RevIN

**Câu hỏi khảo sát:** chuẩn hóa riêng mỗi cửa sổ đầu vào rồi khôi phục thang đo đầu ra có cải thiện việc chọn ứng viên không?

RevIN ở code này tính mean và độ lệch chuẩn từ **200 mẫu history của từng cửa sổ**, chuẩn hóa trước model và khôi phục dự báo 25 mẫu bằng chính thống kê đó:

```text
mean = trung bình 200 mẫu history
std  = sqrt(variance(history, unbiased=False) + 1e-5)
x_norm = (history - mean) / std
y_pred = model(x_norm) × std + mean
```

Không dùng GT đai hoặc 25 mẫu tương lai để tính mean/std. Bản RevIN này không có tham số affine học được nên không làm tăng số tham số. RevIN cũng khác chuẩn hóa dữ liệu ban đầu và khác BatchNorm bên trong block.

**Cấu hình nền của phép thử DS-TCN + RevIN:** C64, kernel 3, **6 block**, RF253, **BatchNorm**, dropout 0, MSE; Adam/LR/batch/epoch theo nền chung. Đây không phải nhánh 4 block, không norm, dropout 0,2 của 64/RF61 hoặc 64/RF121.

| Mô hình | Tham số | Seed 0 | Seed 1 | Seed 2 | CV macro trung bình ± std | Trạng thái |
|---|---:|---:|---:|---:|---:|---|
| DS-TCN nền, không RevIN | 56.281 | 0,741375 | 0,742256 | 0,742697 | **0,742109 ± 0,000673** | 3 seed × 4 fold, dùng làm đối chứng. |
| **DS-TCN + RevIN** | **56.281** | **0,723776** | **0,737823** | **0,727547** | **0,729715 ± 0,007271** | **Đã chạy đủ 3 seed × 4 fold.** |

Các điểm seed được chép từ output/bảng hiện có và làm tròn sáu chữ số; có thể lệch một đơn vị ở chữ số cuối giữa các bản tổng hợp.

**Kết luận được phép rút ra:** thêm RevIN vào **DS-TCN nền RF253 có BatchNorm** giảm CV trung bình khoảng **0,012394**, và thấp hơn nền ở cả ba seed. Vì thế nhánh này không được giữ trong cấu hình cuối. Kết quả không chứng minh RevIN luôn có hại với mọi kiến trúc, cũng không chứng minh bỏ BatchNorm sẽ tốt hơn.

**Bằng chứng đã lưu:** [notebook DS-TCN + RevIN](../notebooks/TN2_DS_TCN_RevIN.ipynb) có output kiểm tra model, log train/validation cho ba seed trên các fold AB, CE, DF, KL và output lưu kết quả. Ô `runtime.unassign()` trống không có nghĩa thực nghiệm chưa chạy; đây là lệnh ngắt phiên Colab. Bảng tổng hợp: [bảng từng seed](BANG_TCN_TUNG_SEED.md); công thức RevIN: [src/models.py](../src/models.py).

### TN3 — chọn loss riêng cho từng cấu hình

`Loss = α × MSE + (1 − α) × (1 − Pearson)`.

| Cấu hình | Alpha được giữ | CV macro, seed 0 × 4 fold |
|---|---:|---:|
| 64/RF61 | 0,6 | 0,780028 |
| **64/RF121** | **0 — Pearson thuần** | **0,780306** |
| 192/RF121 | 0,2 | 0,776011 |

Đây là các alpha có điểm cao nhất quan sát được trong dải đã thử, không phải giá trị tối ưu phổ quát.

**Toàn bộ dải khảo sát — CV macro, seed 0 × 4 fold:**

| Alpha | 64/RF121 — phương án chính | 64/RF61 — phương án thứ hai | 192/RF121 — đối chứng |
|---|---:|---:|---:|
| 0,0 — Pearson thuần | **0,780306** | 0,776667 | 0,772640 |
| 0,1 | 0,763183 | 0,769390 | 0,771848 |
| 0,2 | 0,771996 | 0,775264 | **0,776011** |
| 0,3 | 0,776213 | 0,771931 | 0,774584 |
| 0,4 | 0,771848 | 0,779266 | 0,768337 |
| 0,5 | 0,763457 | 0,779419 | 0,769950 |
| 0,6 | 0,752386 | **0,780028** | 0,765333 |
| 0,7 | 0,761892 | 0,771665 | 0,769263 |
| 0,8 | 0,760013 | 0,776611 | 0,770282 |
| 0,9 | 0,758236 | 0,764941 | 0,771893 |
| 1,0 — MSE, dùng lại nền seed 0 | 0,757855 | 0,758244 | 0,764428 |

Nguồn các alpha 0–0,9: bảng output trong ba notebook TN3 được dẫn ở mục 3. Hàng MSE lấy lại seed 0 của TN1/TN2. **Riêng notebook TN3 C64/RF61 đang in mốc MSE 0,760878**, là mốc trung bình ba seed làm tròn trong bảng đó, không phải seed 0. Bảng trên dùng **0,758244** để so cùng seed; không sửa output lịch sử thành một kết quả chạy mới. Các bảng có thể lệch chữ số thập phân cuối do làm tròn; khi xuất số chính thức phải dùng nhất quán nguồn summary/scores.

### TN4 — hai mô hình cuối và các đối chứng

| Vai trò | Cấu hình/loss | Test macro, 3 seed |
|---|---|---:|
| **Ưu tiên 1 — phương án chính** | **64/RF121, Pearson thuần** | **0,8017 ± 0,0100** |
| **Ưu tiên 2 — phương án thứ hai** | **64/RF61, hybrid α = 0,6** | **0,8036 ± 0,0154** |
| Đối chứng loss | 64/RF121, MSE | 0,7622 ± 0,0214 |
| Đối chứng dung lượng | 192/RF121, hybrid α = 0,2 | 0,8007 ± 0,0107 |
| Mốc DS-TCN nền | C64/RF253, BatchNorm, MSE | 0,7958 ± 0,0154 |

**Cách trình bày lựa chọn cuối trong thesis:** Nhóm chọn hai mô hình DS-TCN C64, ưu tiên **RF121 với Pearson loss thuần** làm phương án chính, sau đó là **RF61 với hybrid loss α = 0,6**. RF121 có phạm vi quan sát lý thuyết lớn hơn với mức tăng 1.024 tham số so với RF61; Pearson thuần cũng không cần hệ số phối trộn hai thành phần loss. Đây là những đặc điểm thiết kế của phương án chính, không tự chứng minh mô hình chính xác hơn RF61.

Về bằng chứng thực nghiệm, 64/RF121 với Pearson tăng trung bình **0,039548** so với MSE trên cùng cấu hình và tăng ở cả ba seed đã chạy. RF61 hybrid vẫn được chọn là mô hình cuối thứ hai và có điểm test trung bình nhỉnh hơn RF121 khoảng **0,001851**. Chênh lệch này được báo cáo nguyên vẹn; không dùng thứ tự ưu tiên để tuyên bố RF121 thắng về độ chính xác hoặc hai mô hình đã được chứng minh tương đương.

## 5. Vì sao hai mô hình cuối có các tham số này?

Hai mô hình cùng dùng C64, bốn block, dilation 1–2–4–8, không norm và dropout theo phần tử p = 0,2. Bảng dưới giải thích **phương án chính RF121**. Phương án thứ hai RF61 dùng kernel 3 thay cho 5 và hybrid α = 0,6 thay cho Pearson thuần; các lựa chọn này được truy vết qua TN2 và TN3.

| Thành phần | Thiết lập | Lý do thiết kế và mức bằng chứng |
|---|---|---|
| DS convolution | Depthwise theo thời gian + pointwise trộn kênh | Giảm trọng số so với conv thường cùng kích thước. |
| Số kênh | 64 | Nhánh gọn có điểm CV tốt; bản 192 cung cấp đối chứng dung lượng. |
| Block / dilation | 4 block; 1, 2, 4, 8 | Giữ độ sâu cố định để khảo sát RF qua kernel. Không khẳng định 4 block tối ưu độc lập. |
| Kernel / RF | 5 / 121 | Được giữ sau TN2 và khảo sát loss ở TN3. |
| Norm | Không có trong block | Giữ cấu trúc đơn giản, không dùng thống kê BatchNorm trong block. Đây là lý do để khảo sát, chưa chứng minh bỏ norm tự làm tăng điểm. |
| Dropout | Theo phần tử, p = 0,2 | Regularization khi train. Cửa sổ 200 mẫu trượt 25 mẫu có mức chồng lấn lớn; số cửa sổ không phải số quan sát độc lập. Điều này tạo động cơ regularization, không chứng minh riêng p = 0,2 tối ưu. |
| Loss | Pearson thuần | Mức α = 0 có CV cao nhất quan sát được ở 64/RF121; có đối chứng cùng kiến trúc tại TN4. |
| Input/output | 200 → 25 | Giữ giao diện và cách chấm của pipeline. |
| LR / batch / epoch | 1e-4 / 64 / 20 | Thiết lập chung cho các phép so; không tối ưu riêng từng mô hình. |

Phải phân biệt **lý do đưa một thiết lập vào khảo sát** và **bằng chứng cho hiệu quả của cả cấu hình**. TN1 đổi nhiều thành phần giữa một số ứng viên; không quy toàn bộ chênh lệch cho dropout hoặc norm.

Bảng giải thích đầy đủ, số tham số và câu trả lời hội đồng: [BAO_CAO_QUA_TRINH_THUC_NGHIEM.md](BAO_CAO_QUA_TRINH_THUC_NGHIEM.md).

## 6. Notebook bổ sung được giữ

| Notebook | Vai trò / trạng thái |
|---|---|
| [Mốc DS-TCN trên GHIJ](../notebooks/TN1_final_evaluation.ipynb) | Kết quả test của DS-TCN nền; giữ lệnh/output DS-TCN. |
| [RF C64, một fold](../notebooks/TN2_ReceptiveField_DS_TCN_c64.ipynb) | Sàng lọc sơ bộ, tách khỏi kết quả CV bốn fold. |
| [DS-TCN + RevIN](../notebooks/TN2_DS_TCN_RevIN.ipynb) | Đã chạy 3 seed × 4 fold: **0,729715 ± 0,007271**, thấp hơn nền 0,012394. Xem phân tích nhánh RevIN tại mục 4. |
| [DS-TCN-192 thử BatchNorm](../notebooks/TN_test_ds_tcn_192.ipynb) | Phép thử bổ sung; khác cả norm và loại dropout với nhánh chính. |

Phép thử C192/RF121 với BatchNorm và dropout theo kênh đạt **0,756618** (seed 0 × 4 fold), so với khoảng **0,764428** của tổ hợp không norm và dropout phần tử. Đây là so hai tổ hợp, không phải ablation chỉ thay BatchNorm. Bảng cấu hình/số tham số: [BANG_TCN.md](BANG_TCN.md); các kết quả một fold C64 và RF301/RF361 tra output notebook sàng lọc, không nhập vào bảng CV bốn fold.

Các notebook ngoài phạm vi không nằm trong danh mục bản nộp. Danh mục máy đọc được: [SUBMISSION_NOTEBOOKS.json](SUBMISSION_NOTEBOOKS.json). Mã hỗ trợ nhiều model vẫn được giữ để không phá runner và phần tái lập TN0; đó không phải danh mục thực nghiệm của bản nộp.

**Lưu ý tái lập:** output train/test cũ được giữ làm bằng chứng các lần chạy đã ghi. Ô clone đã được đổi sang nhánh submission cho lần chạy mới; output setup cũ có thể hiển thị commit cũ. Không coi việc chỉnh notebook trình bày là một lượt train mới.

## 7. Tài liệu đọc tiếp

- [Danh mục artifact và hướng dẫn bàn giao](../ARTIFACTS.md).
- [Sơ đồ nhánh TN1–TN4](SO_DO_NHANH.md).
- [Báo cáo chi tiết và lý do từng tham số](BAO_CAO_QUA_TRINH_THUC_NGHIEM.md).
- [Bảng TCN từng seed](BANG_TCN_TUNG_SEED.md).
- [Pipeline train/inference](PIPELINE_2.md).
- [TN0 — ghi chú lịch sử, có phần outdated](../notebooks/TN0.md); bảng hiện tại nằm ở mục 4 của tài liệu này.

Các bảng/tài liệu lịch sử trong repo có thể còn mô hình ngoài phạm vi hoặc trạng thái cũ. Dùng danh mục và quá trình trong tài liệu này để xác định nội dung bản nộp.

Cơ sở kiến trúc: [Bai et al., 2018](https://arxiv.org/abs/1803.01271), [Howard et al., 2017](https://arxiv.org/abs/1704.04861). Cơ sở regularization: [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html). Các bài này không quy định chính xác tổ hợp tham số của đồ án.
