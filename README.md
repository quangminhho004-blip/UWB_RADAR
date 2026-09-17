# Contactless and Robust Respiration Monitoring based on UWB Radar

Bản nộp trên nhánh **final_submission**, năm thực nghiệm nối nhau.

| | làm gì | kết quả |
|---|---|---|
| **TN0** | tái lập MobiVital | micro 0,8195 trên 537 phiên G H I J |
| **TN1** | chọn kiến trúc giữa bốn ứng viên | DS-TCN 64 k3n4 — 37.081 tham số |
| **TN2** | chọn tầm nhìn: kernel 3, 5, 7, 9 | tầm nhìn rộng hơn thì điểm thấp hơn, đơn điệu |
| **TN3** | chọn hàm loss: quét 10 mức alpha | Pearson thuần hơn MSE thuần **0,022**; chốt RF121 |
| **TN4** | chấm trên G H I J | **0,801739 ± 0,009968** |

**Mô hình cuối: DS-TCN 64, kernel 5, 4 khối, Pearson thuần (alpha 0) — 38.105 tham số.**

Trên tập kiểm tra độc lập nó đạt **0,801739 ± 0,009968**, so với mốc LSTM-352 của
MobiVital **0,810302 ± 0,015402** — điểm trung bình **thấp hơn 0,0086** với
**ít hơn 39,4 lần tham số**. Đồ án **chưa thực hiện kiểm định thống kê** nào,
nên chưa kết luận hai mô hình tương đương hay khác biệt có ý nghĩa thống kê.
Dấu ± là độ lệch chuẩn giữa ba seed, không phải khoảng tin cậy.

**Bàn giao cho thành viên đóng gói bản nộp:** [ARTIFACTS.md](ARTIFACTS.md).

## Kết quả đầy đủ

| | |
|---|---|
| bảng TN1–TN4, cách đọc, giới hạn | [runs/](runs/) — đọc `summary.csv` của từng thực nghiệm |
| điểm từng buổi ghi | `runs/<thực nghiệm>/<cấu hình>/**/scores.csv` |
| trọng số model | `runs/<thực nghiệm>/<cấu hình>/**/final.pth` — 149 tệp |
| lệnh đã chạy và log | 11 notebook trong [notebooks/](notebooks/), output còn nguyên |

Dựng lại bảng của một thực nghiệm:

```bash
python3 scripts/compare_cv.py --experiment tn1
python3 scripts/compare_cv.py --experiment tn2_rf
python3 scripts/compare_cv.py --experiment tn3
python3 scripts/compare_cv.py --experiment tn4 --final
```

Nhánh này giữ mã của bốn kiến trúc đã công bố. Các kiến trúc từng thử mà không công bố, và nhánh C192 (đối chứng dung lượng lớn), không nằm ở đây — bản gốc còn ở nhánh `submission`. Các commit lịch sử không bị viết lại.

Tài liệu dài (tóm tắt thesis, bảng kết quả kèm cách đọc, lý do chia dữ liệu, sơ đồ pipeline) nằm ngoài nhánh này theo chủ ý; phần cần để đọc kết quả đã gói trong `README.md` và `ARTIFACTS.md`.

## Bài toán

Mô hình dự báo 25 mẫu tiếp theo từ 200 mẫu lịch sử của từng ứng viên radar. Pipeline dùng độ khớp dự báo để chọn ứng viên, rồi đánh giá **sóng radar được chọn** với đai tham chiếu. Đây không phải mô hình trực tiếp xuất nhịp thở/phút hoặc tái tạo sóng đai.

## Chạy trên Colab

Mở notebook từ nhánh `final_submission`, ví dụ:

[DATA_PREPARE](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/final_submission/notebooks/DATA_PREPARE.ipynb) · [TN0](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/final_submission/notebooks/TN0.ipynb) · [TN1 cấu hình được chọn](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/final_submission/notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb)

Ô setup trong notebook lấy đúng nhánh:

```bash
git clone --branch final_submission --single-branch https://github.com/quangminhho004-blip/UWB_RADAR.git
cd UWB_RADAR
python scripts/setup_colab.py
```

Script setup tải mã MobiVital riêng vào external/mobivital và ghim commit `4319731d2769d4134c92088dd846666e262f18e9`. Dữ liệu tải riêng; 149 checkpoint thực nghiệm nằm trong `runs/`.

## Chuẩn bị dữ liệu

Ưu tiên chạy [DATA_PREPARE.ipynb](notebooks/DATA_PREPARE.ipynb). Nếu dựng dữ liệu bằng lệnh, thứ tự chính là:

```bash
python scripts/download_dataset.py
python scripts/mobivital/setup_dataset.py
(cd external/mobivital && python dataset_preparation/prep_breath_final.py)
python scripts/make_npz.py
python scripts/check_data.py
python scripts/make_windows.py
python scripts/checksums.py
```

Các notebook thực nghiệm khôi phục dữ liệu đã xử lý bằng `scripts/restore_processed_data_on_drive.py`. Kiểm tra đường dẫn Drive trong notebook trước khi chạy.

## Trình tự thực nghiệm

| Bước | Việc thực hiện | Seed/fold |
|---|---|---|
| TN0 | Đối chiếu pipeline với MobiVital | Theo từng phép kiểm chứng trong notebook |
| TN1 | So bốn kiến trúc: DS-TCN 64, LSTM 352, LSTM 67, CNN-LSTM 58 | 3 seed × 4 fold, đủ cho cả bốn |
| TN2 | Khảo sát tầm nhìn qua kernel 5, 7, 9 | seed 0 × 4 fold; kernel 3 dùng lại TN1 |
| TN3 | Quét 10 mức alpha của hàm loss lai | seed 0 × 4 fold mỗi alpha |
| TN4 | Train đủ ABCDEFKL, chấm G H I J | 3 seed mỗi tổ hợp |

**G H I J không dùng để chọn bất cứ thứ gì** — kiến trúc, tầm nhìn và alpha đều chọn trên validation. Vòng sàng lọc một fold ghi riêng, không trộn vào bảng kết luận.

## Mã và kết quả

```text
notebooks/   Notebook theo danh mục bản nộp
src/         Model, huấn luyện, loss, chọn ứng viên, kết quả
scripts/     Chuẩn bị dữ liệu, runner và lưu/so sánh kết quả
data/        Dữ liệu tải riêng; repo giữ checksums.txt
runs/        Artifact TN0–TN4; checkpoint có commit
external/    Mã MobiVital tải riêng
```

`scripts/run_cv.py` đánh giá trên bốn fold thuộc ABCDEFKL. `scripts/run_final_test.py` train đủ ABCDEFKL rồi chấm một lần trên G H I J. Số chính của đồ án là Pearson macro theo người. Thư mục kết quả được đặt bằng `--experiment`; cấu hình và seed tạo tên run riêng.

Output trong notebook là bằng chứng lần chạy đã lưu, không phải kết quả chạy lại sau khi chỉnh bản nộp. **Chính tệp README.md này là điểm vào của bản nộp**; tài liệu dài hơn được bàn giao riêng, không nằm trong repo.

### `data/checksums.txt` dùng để làm gì?

Tệp này lưu mã băm MD5 làm mốc đối chiếu nội dung dữ liệu đã xử lý trong `data/processed/by_user/*.npz`. Script `scripts/checksums.py` băm tên, shape, dtype và giá trị các mảng theo thứ tự cố định; không băm trực tiếp vỏ ZIP của NPZ. Nhờ đó có thể kiểm tra dữ liệu dựng lại trên máy khác có khớp mốc đã lưu không. Đây là kiểm tra tính nhất quán dữ liệu, không phải điểm model hay bằng chứng pipeline đúng về mặt khoa học.

Script hiện **ghi đè** `data/checksums.txt`, không tự báo pass/fail. Muốn đối chiếu, giữ bản mốc trước khi chạy, rồi so hai tệp hoặc xem `git diff -- data/checksums.txt`. Script chỉ băm `by_user`, không băm checkpoint hoặc tập windows; không dùng nó để kết luận mọi artifact đều giống nhau.

## Nguồn dữ liệu và phương pháp nền

- [Bài báo MobiVital](https://arxiv.org/abs/2503.11064).
- [Mã nguồn MobiVital](https://github.com/nesl/mobivital-public).
- [Dữ liệu Zenodo](https://doi.org/10.5281/zenodo.15022885).
