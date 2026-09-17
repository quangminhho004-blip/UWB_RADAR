# Tài liệu của nhánh `final_submission`

Năm thực nghiệm nối nhau. Đọc từ [THESIS.md](THESIS.md).

| | làm gì | kết quả |
|---|---|---|
| **TN0** | tái lập MobiVital | micro 0,8195 trên 537 phiên G H I J |
| **TN1** | chọn kiến trúc giữa bốn ứng viên | DS-TCN 64 k3n4 — **37.081 tham số** |
| **TN2** | chọn tầm nhìn: kernel 3, 5, 7, 9 | tầm nhìn **61**; càng rộng càng tệ |
| **TN3** | chọn hàm loss: quét 10 mức alpha | Pearson trong loss hơn MSE thuần **0,019** |
| **TN4** | chấm trên G H I J | **0,803590 ± 0,015350** |

**Mô hình cuối: DS-TCN 64, kernel 3, 4 khối, loss lai alpha 0,6 — 37.081 tham số.**

Trên tập kiểm tra độc lập nó **ngang** mốc LSTM-352 của MobiVital (0,810302) với
**ít hơn 40,5 lần tham số**. Chênh 0,0067 nhỏ hơn dao động seed của chính nó
(0,0154) nên không phát biểu là "tốt hơn".

| File | Giữ để làm gì? |
|---|---|
| [THESIS.md](THESIS.md) | Giao thức, cách chấm điểm, quá trình chọn qua TN0–TN4. |
| [BANG_TCN.md](BANG_TCN.md) | Bảng kết quả bốn thực nghiệm và những gì đọc được. |
| [CHIA_DU_LIEU.md](CHIA_DU_LIEU.md) | Vì sao chia theo người, bốn fold cố định, G H I J để riêng. |
| [PIPELINE_2.md](PIPELINE_2.md) | Sơ đồ train và inference, giải thích từng khối. |
| [SO_DO_DU_LIEU.md](SO_DO_DU_LIEU.md) | Dữ liệu đi từ CSV thô tới cửa sổ train. |
| [CAU_TRUC_MA_NGUON.md](CAU_TRUC_MA_NGUON.md) | Vai trò từng phần mã nguồn. |
| [DANH_MUC_ZIP.md](DANH_MUC_ZIP.md) | Tệp nén kết quả trên Drive, cái nào đã vào git. |

Đóng gói để nộp: [ARTIFACTS.md](../ARTIFACTS.md).

## Không giữ gì so với các bản trước

Nhánh C192 (đối chứng dung lượng lớn) và các kiến trúc từng thử mà không công bố
đều **không** nằm ở nhánh này. Bản gốc còn nguyên ở nhánh `submission` và trong
lịch sử Git.

## Bằng chứng gốc

- [notebooks/](../notebooks/) — 14 notebook, output lúc chạy còn nguyên
- [runs/](../runs/) — checkpoint, đường cong loss, điểm từng phiên của mọi cấu hình
- [data/checksums.txt](../data/checksums.txt) — mốc đối chiếu nội dung mảng `by_user`. Không phải điểm model; `scripts/checksums.py` tạo lại tệp mốc chứ không tự báo đạt/trượt.

## Không trích dẫn kiến trúc từ bài báo nào

Khối tích chập trong `src/models.py` là thiết kế của đồ án. Nó không phải bản
tái lập của một kiến trúc đã công bố, nên tài liệu ở đây không nói "cài theo
bài X" ở bất kỳ chỗ nào. Tham chiếu duy nhất còn giữ là **MobiVital** — bài gốc
mà đồ án cải tiến, và là đích của TN0.
