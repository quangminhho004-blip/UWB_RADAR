# Bảng kết quả TN1 — bốn cấu hình

Một thực nghiệm duy nhất: chọn kiến trúc trên tập phát triển ABCDEFKL, bốn fold
cố định, ba seed. Mọi dòng dưới đây cùng dữ liệu, cùng giao thức train, cùng
cách chấm điểm — nên so trực tiếp được với nhau.

Số tham số dựng lại bằng `models.build_model` rồi đếm, không chép tay.


## Đọc bảng này thế nào

**macro** — trung bình điểm từng người, rồi trung bình các người. Số chính thức
của đồ án, theo [THESIS.md](THESIS.md) mục 2.

**micro** — trung bình toàn bộ phiên đo. Bài báo MobiVital công bố theo thước này.

**Chênh lệch nhỏ hơn `seed_std` thì không xếp hạng được.** Dao động giữa ba seed
của chính các cấu hình này trải từ 0,0020 đến 0,0042.


## Bảng chính — 4 fold trên ABCDEFKL, 3 seed

| cấu hình | họ | tham số | tầm nhìn | CV macro | seed_std |
|---|---|---:|---:|---:|---:|
| **DS-TCN 64, k3 n4** | tích chập | **37.081** | 61 | **0,760878** | 0,003095 |
| LSTM 352 | hồi quy | 1.502.713 | cả 200 mẫu | 0,756992 | 0,004156 |
| LSTM 67 | hồi quy | 56.908 | cả 200 mẫu | 0,753208 | 0,001967 |
| CNN-LSTM 58 | lai | 55.667 | cả 200 mẫu | 0,752658 | 0,003757 |

`LSTM 352` là kiến trúc của MobiVital, giữ nguyên, làm mốc.


## Hai phép so đọc được

### Cùng ngân sách tham số — ba cấu hình quanh 56 nghìn

| | tham số | CV macro | chênh so với DS-TCN |
|---|---:|---:|---:|
| **DS-TCN 64** | **37.081** | **0,760878** | — |
| LSTM 67 | 56.908 | 0,753208 | −0,0077 |
| CNN-LSTM 58 | 55.667 | 0,752658 | −0,0082 |

DS-TCN dùng **ít hơn 1,5 lần tham số** mà điểm cao hơn cả hai. Chênh lệch
0,0077 và 0,0082 đều lớn hơn `seed_std` của mọi cấu hình trong bảng, nên xếp
hạng này đọc được.

### So với mốc MobiVital

| | tham số | CV macro |
|---|---:|---:|
| **DS-TCN 64** | **37.081** | **0,760878** |
| LSTM 352 | 1.502.713 | 0,756992 |

**Ít hơn 40,5 lần tham số**, điểm cao hơn 0,0039. Con số 0,0039 nằm trong
khoảng dao động seed của LSTM 352 (0,0042), nên phát biểu an toàn là **ngang
điểm với ít hơn 40 lần tham số**, không phải "vượt".


## Giới hạn phải ghi khi báo cáo

**Bảng này chọn cấu hình, không kết luận về khả năng tổng quát hoá.** Cả bốn
dòng đều chấm trên cùng tám người ABCDEFKL. Chênh lệch nhỏ trên tập dùng để
chọn thì có thể là do chọn trúng, không phải do kiến trúc tốt hơn.

**DS-TCN 64 khác các mốc ở nhiều thứ cùng lúc**, không chỉ ở phép tích chập:
4 khối, không chuẩn hoá, dropout 0,2 theo phần tử, tầm nhìn 61. Không quy được
kết quả cho riêng thứ nào trong số đó.

**Tầm nhìn 61 trên cửa sổ 200 mẫu** nghĩa là model chỉ thấy 1,2 giây gần nhất,
trong khi một nhịp thở khoảng 4 giây. Nó vẫn cho điểm cao nhất bảng. Đây là
quan sát đo được, chưa có giải thích được kiểm chứng.

**Cột micro của bảng này còn trống.** `run_cv.py` ghi `score_micro` vào dòng
từng fold nhưng không ghi vào dòng `TONG`, mà bảng chỉ đọc dòng `TONG`. Chữa
được mà không phải chạy lại: micro bốn fold bằng trung bình micro từng fold có
trọng số theo số phiên đo, cả hai số đều đã nằm trong dòng từng fold. Bốn fold
chia hết tám người nên mỗi phiên đo đếm đúng một lần.


## Dựng lại bảng này

```bash
python3 scripts/compare_cv.py --experiment tn1
```

Đọc `runs/tn1/summary.csv`. Muốn tính lại từ gốc thì đọc thẳng `scores.csv`
từng fold — điểm Pearson của từng phiên đo nằm ở đó.
