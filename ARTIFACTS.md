# Bàn giao artifact — nhánh `final_submission`

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar.

**Phạm vi:** năm thực nghiệm. **TN0** tái lập MobiVital; **TN1** chọn kiến trúc;
**TN2** chọn tầm nhìn; **TN3** chọn hàm loss; **TN4** chấm trên G H I J.

**Mô hình cuối:** DS-TCN 64, kernel 3, 4 khối, loss lai alpha 0,6 — **37.081
tham số**, macro G H I J **0,803590 ± 0,015350**.

| Thực nghiệm | Cấu hình | Artifact trong git |
|---|---:|---|
| TN0 | tái lập MobiVital | đủ — bảng lựa chọn kênh, điểm 537 phiên |
| TN1 | 4 kiến trúc × 3 seed × 4 fold | đủ — 48 checkpoint |
| TN2 | 3 mức kernel × 4 fold | đủ — 12 checkpoint |
| TN3 | 20 mức alpha × 4 fold | đủ — 80 checkpoint |
| TN4 | 3 tổ hợp × 3 seed | đủ — 9 checkpoint |
| mốc G H I J | LSTM 352 × 3 seed | đủ — 3 checkpoint |

**152 checkpoint, tất cả nạp được `strict=True`.**

## 1. Trạng thái bàn giao

**Đủ trong git:** mã nguồn, 14 notebook có output, và toàn bộ `runs/` của năm
thực nghiệm.

**Còn thiếu:** **link Drive chia sẻ được** cho dữ liệu và cho các tệp nén kết
quả. Đó là mục duy nhất chưa xong.

| Mục người đóng gói cần điền | Trạng thái |
|---|---|
| Link Drive chia sẻ được cho `by_user.tar`, `windows.tar.gz` | chưa có |
| Link Drive chia sẻ được cho các ZIP kết quả | chưa có |

**Không đánh dấu hoàn tất chỉ vì notebook có output.** Output chứng minh đã
chạy; nó không phải là checkpoint và không phải là điểm từng phiên.

**Ba notebook thiếu log ở vài ô** — ghi ra chứ không giấu:
`TN2_..._c64_4fold` thiếu 1 trong 4 ô train, `TN2_..._c64` (vòng sàng lọc)
thiếu 1 trong 6. Kết quả của những ô đó vẫn nằm trong `runs/`.

## 2. Bộ code nộp gồm gì

```text
src/         models.py, training.py, scoring.py, losses.py, results.py,
             mobivital_reference.py
scripts/     chuẩn bị dữ liệu · run_cv.py · run_final_test.py · compare_cv.py
             · check_model.py · save_results.py · gop_summary.py · run_tn0.py
notebooks/   12 notebook có output + 2 notebook nạp kết quả
docs/        THESIS.md là điểm vào
runs/        tn0, tn1, tn1_ghij, tn2_rf, tn3, tn4
data/        chỉ giữ checksums.txt; dữ liệu tải riêng
external/    mã MobiVital tải riêng, ghim commit
```

Mã của các kiến trúc từng thử mà không công bố, và nhánh C192, **không** có trên
nhánh này. Bản gốc còn ở nhánh `submission`; `git log` giữ nguyên lịch sử.


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
| TN0 | `tn0.zip` | 5,4 MB |
| TN1 DS-TCN 64 | `tn1_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed2.zip` | 1,7 MB |
| TN1 LSTM 352 | `tn1_lstm_mse_corr0.9_seed2.zip` | 42,6 MB |
| TN1 LSTM 67 | `tn1_lstm_h67_mse_corr0.9_seed2.zip` | 2,5 MB |
| TN1 CNN-LSTM 58 | `tn1_cnn_lstm_h58.zip` | 2,5 MB |
| Mốc G H I J | `tn1_lstm.zip` | 37,3 MB |
| TN2 | `tn2_rf_c64_4fold.zip` | |
| TN3 | `tn3_ds_tcn_c64.zip`, `tn3_ds_tcn_c64_k5.zip` | |
| TN4 | `tn4_<run_id>.zip`, mỗi seed một tệp | |

Danh mục đầy đủ: [docs/DANH_MUC_ZIP.md](docs/DANH_MUC_ZIP.md).


## 6. Checkpoint

**152 tệp `final.pth`, đã commit**, trải khắp năm thực nghiệm. CV có hậu tố
fold trong đường dẫn; test cuối chỉ có seed. Mỗi tệp là `state_dict` sau epoch cuối của một fold, không kèm
trạng thái Adam.

Kiểm cả 12 nạp được:

```bash
python3 -c "
import glob, torch, sys; sys.path.insert(0, '.')
from src.models import build_model
DS = dict(channels=64, n_blocks=4, dropout=0.2, norm='none', dropout_kind='element')
K  = {61: 3, 121: 5, 181: 7, 241: 9}
import re
def doan(p):
    if 'LSTM-352' in p:    return 'lstm', {}
    if 'LSTM-67'  in p:    return 'lstm', dict(hidden=67)
    if 'CNN-LSTM-58' in p: return 'cnn_lstm', dict(hidden=58)
    m = re.search(r'rf(\d+)|64-(\d+)__', p, re.IGNORECASE)
    return 'ds_tcn', dict(kernel_size=K[int(m.group(1) or m.group(2))], **DS)
n = 0
for p in sorted(glob.glob('runs/*/**/final.pth', recursive=True)):
    name, kw = doan(p)
    build_model(name, **kw).load_state_dict(
        torch.load(p, map_location='cpu', weights_only=True), strict=True)
    n += 1
print(n, '/ 152 nạp được')
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
| mọi checkpoint nạp được | đoạn mã ở mục 6 | 152 / 152 |
| bảng kết quả dựng lại được | `compare_cv.py` cho cả năm thực nghiệm | khớp bảng ở [docs/BANG_TCN.md](docs/BANG_TCN.md) |
| notebook có output | mở trên GitHub | mọi ô mã đều có kết quả in ra |

Bảng G H I J đọc bằng `--final`; bảng CV đọc không cờ.


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

- `runs/` có đủ sáu thư mục: `tn0` `tn1` `tn1_ghij` `tn2_rf` `tn3` `tn4`
- 152 tệp `final.pth` nạp được `strict=True`
- mọi cấu hình trong `runs/` truy được về một notebook có output
- `compare_cv.py` dựng lại đúng các bảng trong [docs/BANG_TCN.md](docs/BANG_TCN.md)
- TN0 đạt theo mục 4
- Link Drive chia sẻ được đã điền cho dữ liệu và cho từng ZIP kết quả

Hiện tại năm dòng đầu **đã đủ**. Còn thiếu **link Drive chia sẻ được** — mục
duy nhất chưa xong.
