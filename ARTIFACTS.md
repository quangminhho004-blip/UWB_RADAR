# Bàn giao artifact — nhánh `final_submission`

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar.

**Phạm vi:** năm thực nghiệm. **TN0** tái lập MobiVital; **TN1** chọn kiến trúc;
**TN2** chọn tầm nhìn; **TN3** chọn hàm loss; **TN4** chấm trên G H I J.

**Mô hình cuối:** DS-TCN 64, kernel 5, 4 khối, Pearson thuần (alpha 0) —
**38.105 tham số**, macro G H I J **0,801739 ± 0,009968**.

| Thực nghiệm | Cấu hình | Artifact trong git |
|---|---:|---|
| TN0 | tái lập MobiVital | đủ — bảng lựa chọn kênh, điểm 537 phiên |
| TN1 | 4 kiến trúc × 3 seed × 4 fold | đủ — 48 checkpoint |
| TN2 | 3 mức kernel × 4 fold | đủ — 12 checkpoint |
| TN3 | 20 mức alpha × 4 fold | đủ — 80 checkpoint |
| TN4 | 2 tổ hợp × 3 seed | đủ — 6 checkpoint |
| mốc G H I J | LSTM 352 × 3 seed, nằm trong TN4 | đủ — 3 checkpoint |

**149 checkpoint, tất cả nạp được `strict=True`.**

## 1. Trạng thái bàn giao

**Đủ trong git:** mã nguồn, 11 notebook có output, và toàn bộ `runs/` của năm
thực nghiệm.

**Tùy chọn tải nhanh qua Drive:** chưa có liên kết chia sẻ dữ liệu đã xử lý
và ZIP kết quả. Link Drive không bắt buộc để đọc kết quả và checkpoint trong
repo. Người chạy lại có thể dựng dữ liệu từ nguồn theo mục 3; muốn dùng các ô
khôi phục từ Drive thì cần tự cung cấp các tệp và đường dẫn tương ứng.

| Mục người đóng gói cần điền | Trạng thái |
|---|---|
| Link Drive chia sẻ được cho `by_user.tar`, `windows.tar.gz` | chưa có |
| Link Drive chia sẻ được cho các ZIP kết quả | chưa có |

**Không đánh dấu hoàn tất chỉ vì notebook có output.** Output chứng minh đã
chạy; nó không phải là checkpoint và không phải là điểm từng phiên.

**Mười một ô mã không có output:** bảy ô là
`runtime.unassign()` (ngắt phiên Colab, không in gì theo thiết kế), hai ô khôi
phục tệp nén chỉ in khi thật sự khôi phục, một ô đối chiếu checksum, một ô
`compare_cv.py` chạy lại bảng đã có sẵn ở `runs/`.

Trong 11 notebook được Git theo dõi, **44/44 ô gọi trực tiếp**
`run_cv.py` hoặc `run_final_test.py` có output. Sự hiện diện của output không
tự chứng minh mỗi lượt chạy đã hoàn tất thành công. Hai ô checksum và so bảng
không có output cần chạy lại khi xác minh bản bàn giao.

## 2. Bộ code nộp gồm gì

```text
src/         models.py, training.py, scoring.py, losses.py, results.py,
             mobivital_reference.py
scripts/     chuẩn bị dữ liệu · run_cv.py · run_final_test.py · compare_cv.py
             · check_model.py · save_results.py · run_tn0.py
notebooks/   11 notebook chạy thực nghiệm
runs/        tn0, tn1, tn2_rf, tn3, tn4
data/        chỉ giữ checksums.txt; dữ liệu tải riêng
external/    mã MobiVital tải riêng, ghim commit
```

Mã của các kiến trúc từng thử mà không công bố, và nhánh C192, **không** có trên
nhánh này. Bản gốc còn ở nhánh `submission`; `git log` giữ nguyên lịch sử.


## 3. Dữ liệu — lấy ở đâu, giải nén vào đâu

**Đường chuẩn bị dữ liệu không cần liên kết Drive: dựng lại từ đầu** theo [README.md](README.md)
mục "Chuẩn bị dữ liệu". Bộ script tải dataset từ Zenodo rồi dựng `by_user` và
`windows`. Chưa chạy lại toàn bộ bước tải và xử lý dữ liệu trong lượt rà soát này.

Đường thứ hai — tải tệp đã xử lý từ Drive — **chưa bàn giao được** vì chưa có
liên kết chia sẻ. Đường dẫn Drive trong notebook không thay thế liên kết tải
dành cho người nhận.

| tệp trên Drive | giải vào | dùng cho | liên kết |
|---|---|---|---|
| `by_user.tar` | `data/processed/by_user/` | chấm điểm, đối chiếu TN0 | **chưa có** |
| `windows.tar.gz` | `data/processed/windows/` | train | **chưa có** |

Dựng lại xong chạy `python scripts/checksums.py` rồi so với `data/checksums.txt` —
script **ghi đè** tệp đó và không tự báo đạt/trượt, nên phải giữ bản mốc trước
khi chạy rồi `git diff`.

Dữ liệu thô 13 GB **không** đưa lên GitHub.


## 4. TN0 — kiểm trước khi đọc TN1

TN0 chứng minh pipeline của đồ án cho ra đúng số của pipeline MobiVital. Nếu TN0
không đạt, cần xác định sai khác trước khi khẳng định đã tái lập pipeline gốc.

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

Tất cả dưới `MyDrive/mobivital/` — **Drive riêng của nhóm, chưa có liên kết chia
sẻ**. Bảng dưới là danh mục để người trong nhóm tìm, không phải đường bàn giao.

Mọi kết quả cần để đọc bản nộp **đã nằm trong `runs/`** của repo; các ZIP này chỉ
cần khi muốn dựng lại từ nguồn gốc. Các ZIP **tích luỹ**: bản của seed cuối chứa
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



## 6. Checkpoint

**149 tệp `final.pth`, đã commit**, thuộc TN1–TN4. CV có hậu tố
fold trong đường dẫn; test cuối chỉ có seed. Mỗi tệp là `state_dict` sau epoch cuối của một lượt train, không kèm
trạng thái Adam.

Kiểm 149 checkpoint được Git theo dõi:

```bash
python3 -c "
import subprocess, torch, sys; sys.path.insert(0, '.')
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
paths = subprocess.check_output(['git', 'ls-files', 'runs'], text=True).splitlines()
for p in sorted(p for p in paths if p.endswith('/final.pth')):
    name, kw = doan(p)
    build_model(name, **kw).load_state_dict(
        torch.load(p, map_location='cpu', weights_only=True), strict=True)
    n += 1
assert n == 149, n
print(n, '/ 149 nạp được')
"
```

`final.pth` chứa trọng số, **không** chứa định nghĩa kiến trúc. Phải giữ
cấu hình và lệnh tương ứng mới nạp đúng. Tra `run_id` cùng các cột cấu hình
trong `runs/<thực nghiệm>/summary.csv` và đối chiếu lệnh trong notebook.


## 7. Kiểm trước khi đánh dấu đủ

| kiểm | lệnh | phải thấy |
|---|---|---|
| bốn cấu hình dựng đúng | `python3 scripts/check_model.py --model <tên> ...` | `TẤT CẢ ĐẠT` |
| số tham số | như trên | 37.081 · 1.502.713 · 56.908 · 55.667 |
| mọi checkpoint nạp được | đoạn mã ở mục 6 | 149 / 149 |
| bảng kết quả dựng lại được | bốn lệnh `compare_cv.py` trong README | bảng TN1–TN4 từ `runs/`; TN0 kiểm riêng ở mục 4 |
| ô runner có output | mở notebook trên GitHub | 44/44 ô gọi trực tiếp `run_cv.py` hoặc `run_final_test.py` có output |
| truy nguồn kết quả | đối chiếu cấu hình, seed, fold với lệnh và log notebook | không dùng số ô có output thay cho số cấu hình đã xác minh |

Bảng G H I J đọc bằng `--final`; bảng CV đọc không cờ.


## 8. Khôi phục để đọc hoặc chạy lại

```bash
git clone --branch final_submission --single-branch \
    https://github.com/quangminhho004-blip/UWB_RADAR.git
cd UWB_RADAR
python scripts/setup_colab.py          # tải mã MobiVital, ghim commit
# Đọc bảng đã lưu không cần tải dữ liệu hoặc train lại.
python scripts/compare_cv.py --experiment tn1
```

Chạy lại một cấu hình: lấy lệnh từ chính notebook của cấu hình đó trong
`notebooks/` — ô gọi `run_cv.py` hoặc `run_final_test.py` ghi đủ mọi cờ.
Trước khi train lại, chuẩn bị dữ liệu theo mục 3. Thời gian chạy phụ thuộc cấu hình và thiết bị.


## 9. Điều kiện chốt phần thực nghiệm

Đủ khi tất cả các dòng dưới đây đúng:

- `runs/` có đủ năm thư mục: `tn0` `tn1` `tn2_rf` `tn3` `tn4`
- 149 tệp `final.pth` nạp được `strict=True`
- mọi cấu hình trong `runs/` truy được về một notebook có output
- Bốn lệnh `compare_cv.py` trong README dựng được bảng TN1–TN4 từ `runs/`
- TN0 đạt theo mục 4
- Người nhận có hướng dẫn chuẩn bị dữ liệu từ nguồn ở README; nếu chọn cung cấp thêm bản xử lý sẵn qua Drive, phải điền và thử các liên kết tải

Các liên kết Drive chưa được điền; đường tải nhanh này chưa được cung cấp.
Không cần tải lại ZIP kết quả từ Drive để đọc các artifact đã có trong `runs/`.
Quy trình tải và xử lý dữ liệu từ đầu chưa được chạy lại trong lượt rà soát này.
