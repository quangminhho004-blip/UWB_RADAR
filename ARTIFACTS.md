# Bàn giao artifact — nhánh `final_submission`

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar.

**Phạm vi:** hai thực nghiệm. **TN0** tái lập MobiVital; **TN1** so bốn kiến
trúc. Không có thực nghiệm nào khác trên nhánh này.

**Cấu hình được chọn:** DS-TCN 64, kernel 3, 4 khối — **37.081 tham số**, CV
macro **0,760878 ± 0,003095**.

| Cấu hình | Tham số | CV macro (4 fold × 3 seed) | Artifact trong git |
|---|---:|---:|---|
| **DS-TCN 64, k3 n4** | **37.081** | **0,760878 ± 0,003095** | **đủ** — 12 checkpoint, curve, scores |
| LSTM 352 | 1.502.713 | 0,756992 ± 0,004156 | chưa — nạp từ Drive |
| LSTM 67 | 56.908 | 0,753208 ± 0,001967 | chưa — nạp từ Drive |
| CNN-LSTM 58 | 55.667 | 0,752658 ± 0,003757 | chưa — nạp từ Drive |


## 1. Trạng thái bàn giao

**Đủ trong git:** mã nguồn, sáu notebook có output, `runs/tn0/`, và toàn bộ
artifact của cấu hình được chọn ở `runs/tn1/DS-TCN-C64-RF61/`.

**Còn thiếu:** artifact của ba cấu hình mốc. Chúng **đã train xong** — log đầy đủ
nằm trong `TN1_LSTM.ipynb`, `TN1_LSTM_small.ipynb`, `TN1_CNN_LSTM.ipynb` — nhưng
tệp kết quả chỉ được nén lên Drive, chưa bao giờ vào git.

Lấy về bằng [`notebooks/NAP_KET_QUA_TN1.ipynb`](notebooks/NAP_KET_QUA_TN1.ipynb).
Notebook đó tự tính lại điểm macro từ `scores.csv` rồi so với bảng trên; lệch
quá 0,0001 là báo.

| Mục người đóng gói cần điền | Trạng thái |
|---|---|
| Link Drive chia sẻ được cho `by_user.tar`, `windows.tar.gz` | chưa có |
| Link Drive chia sẻ được cho các ZIP kết quả TN1 | chưa có |
| Ba thư mục baseline trong `runs/tn1/` | chưa nạp |

**Không đánh dấu hoàn tất chỉ vì notebook có output.** Output chứng minh đã
chạy; nó không phải là checkpoint và không phải là điểm từng phiên.


## 2. Bộ code nộp gồm gì

```text
src/         models.py, training.py, scoring.py, losses.py, results.py,
             mobivital_reference.py
scripts/     chuẩn bị dữ liệu · run_cv.py · compare_cv.py · check_model.py
             · save_results.py · gop_summary.py · run_tn0.py
notebooks/   6 notebook có output + NAP_KET_QUA_TN1
docs/        THESIS.md là điểm vào
runs/        tn0 và tn1
data/        chỉ giữ checksums.txt; dữ liệu tải riêng
external/    mã MobiVital tải riêng, ghim commit
```

Mã của các kiến trúc từng thử mà không công bố **không** có trên nhánh này.
`git log` vẫn giữ nguyên lịch sử.


## 3. Dữ liệu — lấy ở đâu, giải nén vào đâu

| tệp trên Drive | giải vào | dùng cho |
|---|---|---|
| `by_user.tar` | `data/processed/by_user/` | chấm điểm, đối chiếu TN0 |
| `windows.tar.gz` | `data/processed/windows/` | train |

Hoặc dựng lại từ đầu theo [README.md](README.md) mục "Chuẩn bị dữ liệu". Dựng
lại xong chạy `python scripts/checksums.py` rồi so với `data/checksums.txt` —
script **ghi đè** tệp đó và không tự báo đạt/trượt, nên phải giữ bản mốc trước
khi chạy rồi `git diff`.

Dữ liệu thô 13 GB **không** đưa lên GitHub.


## 4. TN0 — kiểm trước khi đọc TN1

TN0 chứng minh pipeline của đồ án cho ra đúng số của pipeline MobiVital. Nếu TN0
không đạt thì mọi so sánh ở TN1 đều vô nghĩa.

| bằng chứng | ở đâu | phải thấy gì |
|---|---|---|
| điểm tái lập | `runs/tn0/scores_TN0a.csv` | micro **0,8195** trên 537 phiên, khớp con số bài báo công bố trong dung sai 0,001 |
| bảng lựa chọn kênh | `runs/tn0/TN0a.txt`, `TN0b.txt` | 537 dòng mỗi tệp |
| đối chiếu byte dữ liệu | output `notebooks/TN0.ipynb` | ABCDEFKL 1289/1289 · G H I J 537/537 |

Kiểm nhanh:

```bash
python3 -c "
import csv, statistics
# TN0a/TN0b là đầu ra của pipeline tác giả: hai cột, không có tiêu đề chuẩn
for f in ('TN0a', 'TN0b'):
    r = [row for row in csv.reader(open('runs/tn0/scores_%s.csv' % f))][1:]
    print(f, len(r), 'phiên · micro %.4f' % statistics.mean(float(x[1]) for x in r))
"
```

Ra `TN0a 537 phiên · micro 0.8195` và `TN0b 537 phiên · micro 0.8222`.


## 5. ZIP kết quả cần tìm trên Drive

Tất cả dưới `MyDrive/mobivital/`. Các ZIP **tích luỹ**: bản của seed cuối chứa
cả những seed trước.

| cấu hình | ZIP | dung lượng |
|---|---|---:|
| DS-TCN 64 | `tn1_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed2.zip` | 1,7 MB |
| LSTM 352 | `tn1_lstm_mse_corr0.9_seed2.zip` | 42,6 MB |
| LSTM 67 | `tn1_lstm_h67_mse_corr0.9_seed2.zip` | 2,5 MB |
| CNN-LSTM 58 | `tn1_cnn_lstm_h58.zip` | 2,5 MB |
| TN0 | `tn0.zip` | 5,4 MB |

Danh mục đầy đủ: [docs/DANH_MUC_ZIP.md](docs/DANH_MUC_ZIP.md).


## 6. Checkpoint

`runs/tn1/DS-TCN-C64-RF61/seed{0,1,2}/val_{AB,CE,DF,KL}/final.pth` — **12 tệp,
đã commit**. Mỗi tệp là `state_dict` sau epoch cuối của một fold, không kèm
trạng thái Adam.

Kiểm cả 12 nạp được:

```bash
python3 -c "
import glob, torch, sys; sys.path.insert(0, '.')
from src.models import build_model
kw = dict(channels=64, kernel_size=3, n_blocks=4, dropout=0.2,
          norm='none', dropout_kind='element')
n = 0
for p in sorted(glob.glob('runs/tn1/DS-TCN-C64-RF61/seed*/val_*/final.pth')):
    build_model('ds_tcn', **kw).load_state_dict(
        torch.load(p, map_location='cpu', weights_only=True), strict=True)
    n += 1
print(n, '/ 12 nạp được')
"
```

`final.pth` chứa trọng số, **không** chứa định nghĩa kiến trúc. Phải giữ
`config_id` và lệnh tương ứng mới nạp đúng — `config_id` nằm trong
`runs/tn1/summary.csv`.


## 7. Kiểm trước khi đánh dấu đủ

| kiểm | lệnh | phải thấy |
|---|---|---|
| bốn cấu hình dựng đúng | `python3 scripts/check_model.py --model <tên> ...` | `TẤT CẢ ĐẠT` |
| số tham số | như trên | 37.081 · 1.502.713 · 56.908 · 55.667 |
| bảng kết quả dựng lại được | `python3 scripts/compare_cv.py --experiment tn1` | bốn dòng khớp bảng ở mục đầu |
| notebook có output | mở trên GitHub | mọi ô mã đều có kết quả in ra |

Nếu `compare_cv.py` chỉ ra một dòng thì ba cấu hình mốc chưa được nạp — xem
mục 1.


## 8. Khôi phục để đọc hoặc chạy lại

```bash
git clone --branch final_submission --single-branch \
    https://github.com/quangminhho004-blip/UWB_RADAR.git
cd UWB_RADAR
python scripts/setup_colab.py          # tải mã MobiVital, ghim commit
# giải by_user.tar và windows.tar.gz vào data/processed/
python scripts/compare_cv.py --experiment tn1
```

Chạy lại một cấu hình: xem lệnh trong [docs/THESIS.md](docs/THESIS.md) mục 0.7.
Mỗi lượt CV mất một tới ba giờ.


## 9. Điều kiện chốt phần thực nghiệm

Đủ khi tất cả các dòng dưới đây đúng:

- `runs/tn1/` có bốn thư mục, mỗi thư mục 3 seed × 4 fold × 3 tệp
- `runs/tn1/summary.csv` có 60 dòng (4 cấu hình × 3 seed × 5 dòng)
- `compare_cv.py --experiment tn1` in ra bốn dòng khớp bảng ở mục đầu
- TN0 đạt theo mục 4
- Link Drive chia sẻ được đã điền cho dữ liệu và cho từng ZIP kết quả

Hiện tại **chưa đủ**: thiếu ba thư mục baseline và thiếu link Drive.
