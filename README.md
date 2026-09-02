# MobiVital — đồ án tốt nghiệp

Cải tiến mô hình dự báo sóng nhịp thở cho MobiVital (radar UWB không tiếp xúc):
thay LSTM baseline bằng TCN nhân quả có RevIN.

Bài báo gốc: [arXiv 2503.11064](https://arxiv.org/abs/2503.11064) ·
Code: [nesl/mobivital-public](https://github.com/nesl/mobivital-public) ·
Dữ liệu: [Zenodo 10.5281/zenodo.15022885](https://doi.org/10.5281/zenodo.15022885)

---

## Bài toán

Radar UWB đo phản hồi ở **120 khoảng cách** khác nhau. Người ngồi ở đâu đó trong 120
khoảng cách đó, không biết trước. Với mỗi buổi ghi 30 giây phải tìm ra kênh nào bắt
được nhịp thở rõ nhất — **mà không được nhìn nhịp thở thật**, vì ngoài đời không có
cảm biến đo.

Cách MobiVital giải: cho một model dự báo trước 25 mẫu tiếp theo của từng kênh, kênh
nào model đoán chuẩn nhất thì chọn. Sóng thở đều đặn nên dễ đoán, sóng nhiễu lộn xộn
nên đoán trật.

Đồ án này thay model dự báo đó từ LSTM sang TCN.

## Kết quả đã có

Bảng 4 bài báo gốc:

| Method | điểm |
|---|---|
| **MobiVital** | **0.819** |
| SNR | 0.745 |
| CFAR | 0.516 |
| Variance | 0.514 |
| **Oracle** (được nhìn nhịp thở thật) | **0.943** |

Dựng lại bằng chính code của họ, không sửa dòng nào — xem
[notebooks/TN0.md](notebooks/TN0.md):

```
TN0a  chấm file kết quả họ commit sẵn      0.819481   khớp bài báo
TN0b  checkpoint của họ, mình tự chạy      0.822175
TN0c  tự train lại từ đầu                  0.798748
```

`Oracle 0.943` là trần trên. MobiVital đạt `0.819`. **Dư địa là 0.124** — mọi cải tiến
chỉ có thể ăn trong khoảng này.

## Chuẩn bị dữ liệu

Chạy lần lượt, một lần duy nhất:

| # | lệnh | ra cái gì |
|---|---|---|
| 1 | `python scripts/1_organize_raw.py` | gom CSV thô thành `data/raw/A/` … `L/` |
| 2 | `python scripts/2_make_npz.py` | `data/processed/by_user/*.npz` — **pipeline dev** |
| 3 | `python scripts/3_run_mobivital_prep.py` | `data/processed/mobivital_original/*.npy` — **pipeline gốc** |
| 4 | `python scripts/4_check_data.py` | đối chiếu hai pipeline, sai lệch phải bằng 0 |
| 5 | `python scripts/5_make_windows.py` | `data/processed/windows/` — cửa sổ cắt sẵn |

Bước 3 chạy `prep_breath_final.py` của MobiVital **nguyên bản, 0 dòng sửa**, bằng cách
dựng một thư mục tạm có đúng cấu trúc mà code họ đòi rồi `cd` vào đó.

Bước 4 in ra `sai lệch gt lớn nhất = 0.00e+00` — bằng chứng dữ liệu của mình giống hệt
dữ liệu MobiVital dùng.

## Cấu trúc

```
scripts/     chuẩn bị dữ liệu, chạy một lần
src/         pipeline của đồ án
notebooks/   thí nghiệm, chạy trên Colab
docs/        luật thí nghiệm và lý do thiết kế

data/        KHÔNG commit, để trên Google Drive
runs/        KHÔNG commit, checkpoint và kết quả
external/mobivital/   KHÔNG commit, clone riêng
```

Repo MobiVital **không có LICENSE** nên tuyệt đối không chép vào đây. Clone riêng:

```bash
git clone https://github.com/nesl/mobivital-public.git external/mobivital
```

Bản đang dùng: commit `4319731d2769d4134c92088dd846666e262f18e9`.

## Ranh giới — cái gì của ai

| Mượn lại từ MobiVital | Tự viết ở đồ án này |
|---|---|
| `generate_dataset` cắt cửa sổ train | TCN, DS-TCN, RevIN |
| `sequence_transforms`, `transform` | loss MSE + Pearson |
| `self_normalize` | vòng train, checkpoint, resume |
| `invert_detector` | 4-fold CV trên ABCDEFKL |
| `LSTMMultiStep` làm baseline | bộ chọn kênh dùng chung cho LSTM và TCN |

[`src/mobivital_reference.py`](src/mobivital_reference.py) là chỗ duy nhất chạm vào code
MobiVital — chỉ `import` sáu hàm thuần tính toán, không nạp file script nào của họ.

## Đọc tiếp

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — luật thí nghiệm. Đọc trước khi chạy bất cứ gì.
- [`docs/WHY_SPLIT_BY_USER.md`](docs/WHY_SPLIT_BY_USER.md) — vì sao chia dữ liệu theo người.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — chạy toàn bộ trên Colab.
- [`notebooks/TN0.md`](notebooks/TN0.md) — dựng lại kết quả MobiVital.

## Nguyên tắc

- **Không sửa code MobiVital.** `git diff external/mobivital` phải trống.
- **`GHIJ` là tập test**, không dùng để chọn cấu hình. Pool phát triển là `ABCDEFKL`.
- Mọi lựa chọn cấu hình quyết định bằng `cv_score` trên 4 fold, `test_GHIJ` chỉ để nhìn.
