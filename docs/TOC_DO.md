# Tốc độ và bộ nhớ — đo trên Colab L4, 2026-09-04

Đo trong lúc chạy TN0. Dùng để biết chỗ nào đáng tối ưu trước khi làm TN1–TN6.


## 1. Rào cản bộ nhớ của code MobiVital

`inference/mobivital_gen.py` nạp trọng số xong **không gọi `model.eval()`**.
LSTM ở chế độ train nên cuDNN cấp thêm vùng nhớ dự trữ cho backward, dù cả hàm
nằm trong `torch.no_grad()`.

Bước chọn kênh đưa cả một buổi ghi vào LSTM một lần:

```
129 ứng viên sống sót (trên 240) × 52 cửa sổ = 6708 chuỗi × 200 mẫu
gấp 105 lần batch lúc train (64)

dự trữ backward = 2 lớp × 4 cổng × 6708 × 200 × 352 × 4 byte = 15.1 GB
```

Đo thật, cùng buổi ghi đầu tiên:

| cấu hình | kết quả |
|---|---|
| T4 15 GB | tràn, xin 16.95 GB |
| L4 22 GB | tràn, xin 21.00 GB |
| L4 + `expandable_segments:True` | vẫn tràn — cần thật ~33 GB, không phải phân mảnh |
| L4 + vá `model.eval()` | qua được buổi 27/1874, xin 11.35 GB (9.36 GB bị giữ mà bỏ không) |
| L4 + vá + `expandable_segments:True` | **chạy trọn 1874 buổi, 17 phút** |

Bài báo ghi tác giả dùng **GTX 1080 Ti, 11 GB** — ít hơn L4 — mà vẫn chạy được.
`requirements.txt` của họ ghim `torch==2.3.0`; bản đó bỏ qua vùng dự trữ khi đang
trong `no_grad()`. PyTorch hiện hành cấp theo cờ `model.training`. Không cài lại
`torch==2.3.0` được vì bản đó không có wheel cho Python 3.13.

Vá: `scripts/mobivital/patch_eval.py`, thêm **đúng một dòng** `model.eval()`.
Model chỉ gồm `nn.LSTM(dropout=0)` và `nn.Linear` nên train và eval cho forward
giống hệt — không đổi kết quả, chỉ đổi cách xin bộ nhớ.

**Thời gian bước chọn kênh: CPU 1 giờ 40 → GPU đã vá 17 phút.**


## 2. Thời gian đi đâu — một buổi ghi

Đo trên L4, buổi ghi đầu của người G, 129 ứng viên sống sót → 6708 cửa sổ:

| bước | thiết bị | thời gian | phần trăm |
|---|---|---|---|
| dựng 240 ứng viên (`np.abs`, `np.angle`, `np.unwrap`) | CPU | 0.015 s | 1.3% |
| lọc `invert_detector` | CPU | 0.281 s | **24.7%** |
| cắt cửa sổ | CPU | 0.009 s | 0.8% |
| LSTM forward | GPU | 0.395 s | 34.8% |
| chấm Pearson từng cửa sổ | CPU | 0.435 s | **38.3%** |
| **tổng** | | **1.134 s** | |

**GPU chỉ chiếm 34.8% thời gian.** 65% còn lại là numpy chạy CPU, GPU ngồi chờ.
537 buổi ghi × 1.13 s ≈ 10 phút.

VRAM dùng 14.8/22.5 GB, nhưng phần lớn là bể chứa của bộ cấp phát chứ không phải
đang tính. Nút thắt không nằm ở VRAM.


## 3. Chỗ đáng tối ưu, và cái giá phải trả

Hai chỗ chiếm 63% thời gian, đều nằm trong `src/scoring.py` nên sửa được:

**Chấm Pearson — 38.3%.** Hiện gọi `np.corrcoef` 6708 lần trong vòng lặp Python.
Gộp được thành một phép ma trận: trừ trung bình, chia độ lệch chuẩn, nhân vô
hướng theo hàng. Gần như biến mất.

**Lọc `invert_detector` — 24.7%.** 240 lần gọi hàm của tác giả. Gộp theo lô được,
nhưng hàm đó nhập từ repo họ.

**Cái giá:** cả hai đều đổi thứ tự cộng số thực, tức đổi chữ số cuối. Bước chọn
kênh kết thúc bằng `argmax`; hai ứng viên gần bằng điểm thì đảo thứ hạng. TN0b
đang khớp **537/537** sẽ hỏng.

Nên nếu tối ưu thì phải chạy lại TN0 để đo lại mức khớp, và báo cáo con số mới.
Chưa làm.


## 4. Vòng train

292.708 cửa sổ, 20 epoch, batch 64:

| | thời gian |
|---|---|
| `training/autoreg_training.py` của tác giả | 24 phút |
| `src/training.py` của đồ án | 25 phút |

Batch 64 là cấu hình MobiVital công bố trong `checkpoints/optimal_params.json`,
giữ nguyên để so được. Batch nhỏ như vậy không lấp đầy L4 — nhưng đổi batch là
đổi cấu hình, phải báo cáo riêng.
