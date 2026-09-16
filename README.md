# Contactless and Robust Respiration Monitoring based on UWB Radar

Bản nộp trên nhánh **final_submission** có hai thực nghiệm: **TN0** tái lập
MobiVital, **TN1** so bốn kiến trúc trên cùng dữ liệu và cùng giao thức.

**Cấu hình được chọn: DS-TCN 64, kernel 3, 4 khối — 37.081 tham số, CV macro 0,760878 ± 0,003095.**

| Cấu hình | Tham số | CV macro (4 fold × 3 seed) |
|---|---:|---:|
| **DS-TCN 64, k3 n4** | **37.081** | **0,760878 ± 0,003095** |
| LSTM 352 — kiến trúc MobiVital | 1.502.713 | 0,756992 ± 0,004156 |
| LSTM 67 | 56.908 | 0,753208 ± 0,001967 |
| CNN-LSTM 58 | 55.667 | 0,752658 ± 0,003757 |

**Bắt đầu đọc:** [Tóm tắt thesis và danh mục thực nghiệm](docs/THESIS.md).

Danh mục tài liệu và vai trò từng file: [docs/README.md](docs/README.md).

**Bàn giao cho thành viên đóng gói bản nộp:** [ARTIFACTS.md](ARTIFACTS.md) — gói dữ liệu/kết quả cần lấy, checkpoint, kiểm TN0 và các link Drive cần điền.

- [Bảng kết quả TN1](docs/BANG_TCN.md).
- [Vì sao chia dữ liệu như vậy](docs/CHIA_DU_LIEU.md).
- [Pipeline train và inference](docs/PIPELINE_2.md).

Nhánh này chỉ giữ mã của bốn kiến trúc đã công bố. Các kiến trúc từng thử mà không công bố, và các thực nghiệm sau TN1, không nằm ở đây. Các commit lịch sử không bị viết lại.

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

Script setup tải mã MobiVital riêng vào external/mobivital và ghim commit `4319731d2769d4134c92088dd846666e262f18e9`. Dữ liệu và trọng số không nằm trong repo.

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

Không cấu hình nào chạy ít seed hay ít fold hơn cấu hình khác. G H I J không dùng để chọn cấu hình, và nhánh này không có bước test cuối trên chúng. Danh mục notebook nằm trong [THESIS.md](docs/THESIS.md).

## Mã và kết quả

```text
notebooks/   Notebook theo danh mục bản nộp
src/         Model, huấn luyện, loss, chọn ứng viên, kết quả
scripts/     Chuẩn bị dữ liệu, runner và lưu/so sánh kết quả
docs/        Tóm tắt thesis, báo cáo và sơ đồ
data/        Dữ liệu tải riêng; repo giữ checksums.txt
runs/        Artifact thực nghiệm; checkpoint của tn1 có commit
external/    Mã MobiVital tải riêng
```

`scripts/run_cv.py` đánh giá trên bốn fold thuộc ABCDEFKL. Số chính của đồ án là Pearson macro theo người. Thư mục kết quả được đặt bằng `--experiment`; cấu hình và seed tạo tên run riêng.

Output cũ trong notebook là bằng chứng lần chạy đã lưu, không phải kết quả chạy lại sau khi chỉnh bản nộp. Các tài liệu cũ được giữ để tra cứu; **THESIS.md là điểm vào của bản nộp hiện tại**.

### `data/checksums.txt` dùng để làm gì?

Tệp này lưu mã băm MD5 làm mốc đối chiếu nội dung dữ liệu đã xử lý trong `data/processed/by_user/*.npz`. Script `scripts/checksums.py` băm tên, shape, dtype và giá trị các mảng theo thứ tự cố định; không băm trực tiếp vỏ ZIP của NPZ. Nhờ đó có thể kiểm tra dữ liệu dựng lại trên máy khác có khớp mốc đã lưu không. Đây là kiểm tra tính nhất quán dữ liệu, không phải điểm model hay bằng chứng pipeline đúng về mặt khoa học.

Script hiện **ghi đè** `data/checksums.txt`, không tự báo pass/fail. Muốn đối chiếu, giữ bản mốc trước khi chạy, rồi so hai tệp hoặc xem `git diff -- data/checksums.txt`. Script chỉ băm `by_user`, không băm checkpoint hoặc tập windows; không dùng nó để kết luận mọi artifact đều giống nhau.

## Nguồn dữ liệu và phương pháp nền

- [Bài báo MobiVital](https://arxiv.org/abs/2503.11064).
- [Mã nguồn MobiVital](https://github.com/nesl/mobivital-public).
- [Dữ liệu Zenodo](https://doi.org/10.5281/zenodo.15022885).
