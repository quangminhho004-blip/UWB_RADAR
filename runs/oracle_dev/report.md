# Phân tích oracle và lựa chọn kênh trên dev

Sinh lúc: 2026-09-07T04:07:36.411545+00:00; commit đồ án: `2b3d1932c3e1fb14327c6d5c9c66087be04050ed`.
SHA256 script (kể cả khi chưa commit): `5cac6c219e1f32244b31dddeeeb0627a124cd3e44f49c6641f1f67c0604b1806`.
Commit MobiVital đang cài: `4319731d2769d4134c92088dd846666e262f18e9`.
Nguồn oracle: `runs/oracle_dev/oracle_sessions.csv`.

## 1. Cách hiểu và phạm vi

Chỉ ABCDEFKL, 1289 buổi; không dùng GHIJ. Không train, không chạy model.
Oracle nhìn GT để chọn sóng có Pearson cao nhất; không dự báo hay tái tạo.
All: cả 240 ứng viên. Kept: chỉ ứng viên có invert_detector < 0.8.
Pearson có dấu, không lấy abs(Pearson), không lật sóng.
Điểm chính = trung bình mỗi người, rồi trung bình 8 người (macro), đúng cv_score.
Oracle không có seed. Model báo từng seed; không tự xếp hạng các nhóm thiếu seed.
Số 0.943 của bài báo thuộc phép đánh giá khác, không dùng so trực tiếp ở đây.

## 2. Oracle

| Người | Buổi | Oracle all | Oracle sau lọc | Mất do lọc |
|---|---:|---:|---:|---:|
| A | 224 | 0.881919 | 0.881654 | 0.000265 |
| B | 156 | 0.930920 | 0.929080 | 0.001840 |
| C | 211 | 0.869794 | 0.868997 | 0.000798 |
| D | 206 | 0.933515 | 0.932735 | 0.000780 |
| E | 126 | 0.924011 | 0.923845 | 0.000166 |
| F | 102 | 0.880540 | 0.876682 | 0.003858 |
| K | 148 | 0.957078 | 0.957078 | 0.000000 |
| L | 116 | 0.928261 | 0.925621 | 0.002640 |
| **Macro 8 người** | 1289 | **0.913255** | **0.911961** | **0.001293** |

Bộ lọc làm giảm điểm oracle ở 34/1289 buổi (chênh > 1e-9).
Số ứng viên còn lại: min 86, trung bình 125.66, max 167.

Trung vị oracle sau lọc trên các buổi (không phải macro): 0.942845.
Buổi có oracle sau lọc < 0.3: 0/1289 (0.00%, micro).
Buổi có oracle sau lọc < 0.5: 8/1289 (0.62%, micro).

| Fold | Oracle all (macro 2 người) | Oracle kept (macro 2 người) |
|---|---:|---:|
| val_AB | 0.906420 | 0.905367 |
| val_CE | 0.896903 | 0.896421 |
| val_DF | 0.907027 | 0.904708 |
| val_KL | 0.942669 | 0.941349 |

## 3. Điểm model so với oracle

Chỉ tính khi đủ đúng mọi session dev; cv_macro được tính lại từ scores CSV.

| Cấu hình | Seed | CV macro | Oracle kept − model |
|---|---:|---:|---:|

## 4. Chọn trùng hay chỉ bằng điểm?

Ghép cùng seed và cùng (user, session_file). Cùng bin khác method vẫn là khác sóng.
Tỷ lệ macro: trung bình tỷ lệ từng người. Micro: tỷ lệ trên toàn bộ buổi.
CSV ghi cả số buổi và hai tỷ lệ; bảng dưới dùng macro.

| A | B | Seed | Trùng bin | Trùng (bin, method) | Bằng điểm (<1e-6) |
|---|---|---:|---:|---:|---:|

## 5. Giới hạn và dữ liệu còn thiếu

- Chưa có bộ scores CV đầy đủ: chưa kết luận được các model chọn trùng nhau.
- Điểm oracle là trung bình, không có nghĩa mọi buổi đều có sóng tốt.
- Khoảng cách tới oracle không bảo đảm đổi kiến trúc sẽ thu hồi được.
- Trùng lựa chọn chỉ mô tả hành vi; chưa chứng minh nguyên nhân hay ý nghĩa thống kê.
- Chưa đánh giá sai số số lần thở/phút trong script này.
- Nếu dùng --oracle-csv, phải là kết quả cùng dữ liệu và cùng phiên bản bộ lọc.

## 6. Nguồn CSV/ZIP đã đọc

Chưa tìm thấy scores CV phù hợp. Đưa ZIP vào --scores để bổ sung.
