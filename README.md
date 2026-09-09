# Contactless and Robust Respiration Monitoring based on UWB Radar

Bản nộp trên nhánh **submission** tập trung vào **TCN/DS-TCN**, theo thứ tự **TN0 → TN1 → TN2 → TN3 → TN4**. Hai mô hình cuối: ưu tiên **DS-TCN 64/RF121 với Pearson loss thuần**, sau đó **DS-TCN 64/RF61 với hybrid α = 0,6**.

**Bắt đầu đọc:** [Tóm tắt thesis và danh mục thực nghiệm](docs/THESIS.md).

Danh mục tài liệu và vai trò từng file: [docs/README.md](docs/README.md).

- [Báo cáo chi tiết: vì sao chọn từng tham số](docs/BAO_CAO_QUA_TRINH_THUC_NGHIEM.md).
- [Sơ đồ các nhánh](docs/SO_DO_NHANH.md).
- [Pipeline train và inference](docs/PIPELINE_2.md).
- [Danh mục 17 notebook của bản nộp](docs/SUBMISSION_NOTEBOOKS.json).

Các notebook mô hình ngoài phạm vi không có trong cây tệp hiện tại của bản nộp. TN0 giữ mô hình tham chiếu MobiVital để kiểm tra pipeline; mã dùng chung trong src/ và scripts/ được giữ để các notebook chạy được. Các commit lịch sử không bị viết lại.

## Bài toán

Mô hình dự báo 25 mẫu tiếp theo từ 200 mẫu lịch sử của từng ứng viên radar. Pipeline dùng độ khớp dự báo để chọn ứng viên, rồi đánh giá **sóng radar được chọn** với đai tham chiếu. Đây không phải mô hình trực tiếp xuất nhịp thở/phút hoặc tái tạo sóng đai.

## Chạy trên Colab

Mở notebook từ nhánh submission, ví dụ:

[DATA_PREPARE trên Colab](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/submission/notebooks/DATA_PREPARE.ipynb) · [TN0 trên Colab](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/submission/notebooks/TN0.ipynb) · [TN4 cấu hình cuối trên Colab](https://colab.research.google.com/github/quangminhho004-blip/UWB_RADAR/blob/submission/notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb)

Ô setup trong notebook lấy đúng nhánh:

```bash
git clone --branch submission --single-branch https://github.com/quangminhho004-blip/UWB_RADAR.git
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
| TN1 | Khảo sát TCN/DS-TCN; giữ hai mức dung lượng | 3 seed × 4 fold |
| TN2 | Khảo sát kernel/RF | RF mới: seed 0 × 4 fold; RF61 dùng lại nền |
| TN3 | Khảo sát alpha của loss | Seed 0 × 4 fold mỗi alpha |
| TN4 | Train đủ ABCDEFKL, test GHIJ | 3 seed cho mỗi tổ hợp |

Notebook phụ một fold là sàng lọc sơ bộ, không trộn điểm với CV bốn fold. Danh mục từng notebook và trạng thái nằm trong [THESIS.md](docs/THESIS.md).

## Mã và kết quả

```text
notebooks/   Notebook theo danh mục bản nộp
src/         Model, huấn luyện, loss, chọn ứng viên, kết quả
scripts/     Chuẩn bị dữ liệu, runner và lưu/so sánh kết quả
docs/        Tóm tắt thesis, báo cáo và sơ đồ
data/        Dữ liệu tải riêng; repo giữ checksums.txt
runs/        Artifact thực nghiệm; checkpoint không commit
external/    Mã MobiVital tải riêng
```

`scripts/run_cv.py` đánh giá trên bốn fold thuộc ABCDEFKL. `scripts/run_final_test.py` train đủ ABCDEFKL rồi đánh giá GHIJ. Số chính của đồ án là Pearson macro theo người. Thư mục kết quả được đặt bằng `--experiment`; cấu hình và seed tạo tên run riêng.

Output cũ trong notebook là bằng chứng lần chạy đã lưu, không phải kết quả chạy lại sau khi chỉnh bản nộp. Các tài liệu cũ được giữ để tra cứu; **THESIS.md là điểm vào của bản nộp hiện tại**.

### `data/checksums.txt` dùng để làm gì?

Tệp này lưu mã băm MD5 làm mốc đối chiếu nội dung dữ liệu đã xử lý trong `data/processed/by_user/*.npz`. Script `scripts/checksums.py` băm tên, shape, dtype và giá trị các mảng theo thứ tự cố định; không băm trực tiếp vỏ ZIP của NPZ. Nhờ đó có thể kiểm tra dữ liệu dựng lại trên máy khác có khớp mốc đã lưu không. Đây là kiểm tra tính nhất quán dữ liệu, không phải điểm model hay bằng chứng pipeline đúng về mặt khoa học.

Script hiện **ghi đè** `data/checksums.txt`, không tự báo pass/fail. Muốn đối chiếu, giữ bản mốc trước khi chạy, rồi so hai tệp hoặc xem `git diff -- data/checksums.txt`. Script chỉ băm `by_user`, không băm checkpoint hoặc tập windows; không dùng nó để kết luận mọi artifact đều giống nhau.

## Nguồn dữ liệu và phương pháp nền

- [Bài báo MobiVital](https://arxiv.org/abs/2503.11064).
- [Mã nguồn MobiVital](https://github.com/nesl/mobivital-public).
- [Dữ liệu Zenodo](https://doi.org/10.5281/zenodo.15022885).
