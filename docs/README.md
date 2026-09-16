# Tài liệu của nhánh `final_submission`

Nhánh này chỉ có **TN0** (tái lập MobiVital) và **TN1** (chọn kiến trúc, bốn cấu
hình). Các thực nghiệm sau — tầm nhìn, hàm loss lai, kiểm tra trên GHIJ — và mã
của những kiến trúc từng thử mà không công bố đều không nằm ở đây.

Đọc từ [THESIS.md](THESIS.md).

| File | Giữ để làm gì? |
|---|---|
| [THESIS.md](THESIS.md) | Giao thức, cách chấm điểm macro/micro, bốn cấu hình và kết quả. |
| [BANG_TCN.md](BANG_TCN.md) | Bảng kết quả TN1 và những gì đọc được từ nó. |
| [CHIA_DU_LIEU.md](CHIA_DU_LIEU.md) | Vì sao chia theo người, bốn fold cố định, G H I J để riêng. |
| [PIPELINE_2.md](PIPELINE_2.md) | Sơ đồ train và inference, giải thích từng khối. |
| [SO_DO_DU_LIEU.md](SO_DO_DU_LIEU.md) | Dữ liệu đi từ CSV thô tới cửa sổ train. |
| [CAU_TRUC_MA_NGUON.md](CAU_TRUC_MA_NGUON.md) | Vai trò từng phần mã nguồn. |
| [DANH_MUC_ZIP.md](DANH_MUC_ZIP.md) | Tệp nén kết quả trên Drive, cái nào đã vào git. |

Đóng gói để nộp: [ARTIFACTS.md](../ARTIFACTS.md).

## Bốn cấu hình

| cấu hình | tham số | CV macro |
|---|---:|---:|
| **DS-TCN 64, k3 n4** | **37.081** | **0,760878 ± 0,003095** |
| LSTM 352 — kiến trúc MobiVital | 1.502.713 | 0,756992 ± 0,004156 |
| LSTM 67 | 56.908 | 0,753208 ± 0,001967 |
| CNN-LSTM 58 | 55.667 | 0,752658 ± 0,003757 |

## Bằng chứng gốc

- [notebooks/](../notebooks/) — sáu notebook, mỗi ô đều còn nguyên output lúc chạy
- [runs/tn0/](../runs/tn0/) — bảng lựa chọn kênh và điểm từng phiên của phần tái lập
- [runs/tn1/](../runs/tn1/) — checkpoint, đường cong loss, điểm từng phiên của bốn cấu hình
- [data/checksums.txt](../data/checksums.txt) — mốc đối chiếu nội dung mảng `by_user` giữa các máy. Không phải điểm model; `scripts/checksums.py` tạo lại tệp mốc chứ không tự báo đạt/trượt.

## Không trích dẫn kiến trúc từ bài báo nào

Khối tích chập trong `src/models.py` là thiết kế của đồ án. Nó không phải bản
tái lập của một kiến trúc đã công bố, nên tài liệu ở đây không nói "cài theo
bài X" ở bất kỳ chỗ nào. Tham chiếu duy nhất còn giữ là **MobiVital** — bài gốc
mà đồ án cải tiến, và là đích của TN0.
