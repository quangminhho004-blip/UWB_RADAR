# Tài liệu của nhánh submission

Đọc từ `THESIS.md`; các file còn lại cung cấp bằng chứng và giải thích chi tiết. Hai mô hình cuối được trình bày theo thứ tự **64/RF121 + Pearson**, rồi **64/RF61 + hybrid α = 0,6**.

Đóng gói để nộp hội đồng: [ARTIFACTS.md](../ARTIFACTS.md) liệt kê dữ liệu, ZIP kết quả, checkpoint, trạng thái thiếu và cách kiểm trước khi bàn giao.

| File | Giữ để làm gì? |
|---|---|
| [THESIS.md](THESIS.md) | Tổng quan TN0–TN4, macro/micro, cấu hình, kết quả và lựa chọn cuối. |
| [BAO_CAO_QUA_TRINH_THUC_NGHIEM.md](BAO_CAO_QUA_TRINH_THUC_NGHIEM.md) | Giải thích thiết kế, lý do từng tham số và giới hạn kết luận. |
| [BANG_TCN.md](BANG_TCN.md) | Bảng kết quả tổng hợp TCN/DS-TCN. |
| [BANG_TCN_TUNG_SEED.md](BANG_TCN_TUNG_SEED.md) | Điểm từng seed để kiểm tra dao động và đối chiếu bảng tổng hợp. |
| [SO_DO_NHANH.md](SO_DO_NHANH.md) | Sơ đồ cấu hình đi qua TN1–TN4. |
| [PIPELINE_2.md](PIPELINE_2.md) | Sơ đồ train/inference và giải thích từng block cho slide. |
| [CHIA_DU_LIEU.md](CHIA_DU_LIEU.md) | Bằng chứng về chia theo người và các fold. Giao thức đang dùng được tóm tắt trong THESIS.md. |
| [CAU_TRUC_MA_NGUON.md](CAU_TRUC_MA_NGUON.md) | Tra cứu vai trò các phần mã nguồn; danh mục notebook hiện hành nằm trong manifest. |
| [TOC_DO.md](TOC_DO.md) | Ghi chú hiệu năng lịch sử khi tái lập TN0, không phải benchmark mô hình cuối. |
| [SUBMISSION_NOTEBOOKS.json](SUBMISSION_NOTEBOOKS.json) | Danh mục chính xác 17 notebook được giữ trong bản nộp. |

## TN0: các bằng chứng cần giữ

- [TN0.ipynb](../notebooks/TN0.ipynb): lệnh/output của TN0a, TN0b, TN0c.
- [Ghi chú TN0 lịch sử](../notebooks/TN0.md): đã gắn nhãn outdated để tránh trộn các lần chạy.
- [runs/tn0](../runs/tn0/): TXT và CSV hiện có; trạng thái những artifact chưa có được ghi trong THESIS.md.
- [data/checksums.txt](../data/checksums.txt) và [scripts/checksums.py](../scripts/checksums.py): mốc đối chiếu nội dung mảng dữ liệu `by_user` giữa các máy. Không phải điểm model; script tạo lại tệp mốc, không tự kiểm tra pass/fail.

## Tài liệu đã loại khỏi bản nộp

Các bản pipeline cũ, bảng tổng hợp ngoài phạm vi, kế hoạch chưa sử dụng và thiết kế MixLinear đã được loại khỏi `docs/`. Nội dung cần cho báo cáo hiện tại được dẫn ở bảng trên. Bản local lưu tại `.submission_archive/docs/`, được Git bỏ qua; tài liệu từng commit vẫn có thể truy vết trong lịch sử Git. Việc dọn tài liệu không thay đổi số liệu hoặc output notebook.
