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

Giải nén `tripod.zip` (Zenodo) thẳng vào thư mục MobiVital. **Chỉ giữ một bản CSV**:

```
external/mobivital/dataset/mobivital/tripod/   1874 CSV, 13 GB
        |
        +--> prep_breath_final.py cua HO   -> data_final/*.npy
        |
        +--> scripts/make_npz.py cua MINH  -> data/processed/by_user/*.npz
```

| # | lệnh | ra cái gì |
|---|---|---|
| 1 | `unzip tripod.zip -d external/mobivital/dataset/mobivital/` | 1874 CSV |
| 2 | `python scripts/mobivital/setup_dataset.py` | vá 52 tên file lỗi thời, giấu dữ liệu khỏi git của họ |
| 3 | `cd external/mobivital && python dataset_preparation/prep_breath_final.py` | `data_final/*.npy` — **pipeline gốc**, script của họ |
| 4 | `python scripts/make_npz.py` | `by_user/*.npz` — **pipeline của mình** |
| 5 | `python scripts/check_data.py` | đối chiếu hai bên, phải khớp từng byte |
| 6 | `python scripts/make_windows.py` | `data/processed/windows/` — cửa sổ cắt sẵn |

`data/` chỉ chứa thứ pipeline của mình sinh ra. `scripts/mobivital/` chỉ **dọn
chỗ** — việc đọc CSV do chính `prep_breath_final.py` của MobiVital làm, nguyên
bản, 0 dòng sửa.

Bước 5 in ra:

```
ABCDEFKL  1289/1289 buổi ghi khớp TỪNG BYTE   = training_breath_tripod_data.npy
GHIJ       537/537  buổi ghi khớp TỪNG BYTE   = testing_breath_tripod_data.npy
```

Từ đó mọi thí nghiệm sau chỉ đọc `by_user/*.npz`, bỏ được CSV thô 13 GB.

## Cấu trúc

```
src/         pipeline của đồ án — models.py, training.py, scoring.py, results.py
scripts/     gọi src/ theo đúng thứ tự; notebook chỉ chạy một dòng
  make_npz.py        đọc CSV trong thư mục MobiVital -> by_user/*.npz
  make_windows.py    -> data/processed/windows/
  check_data.py      đối chiếu dữ liệu mình với dữ liệu MobiVital
  checksums.py       -> results/checksums.txt
  run_cv.py          4 fold trên ABCDEFKL, chọn cấu hình
  run_final_test.py  train đủ ABCDEFKL, test GHIJ một lần duy nhất
  mobivital/         để riêng: dọn chỗ cho pipeline gốc chạy nguyên bản
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

## Chạy thí nghiệm

```bash
python scripts/run_cv.py --model ds_tcn --revin true          # chọn cấu hình
python scripts/run_final_test.py --model ds_tcn --revin true  # số công bố
```

Cả hai nhận chung một bộ cờ: `--model lstm|tcn|ds_tcn`, `--revin true|false`,
`--loss mse|mse_pearson`, `--alpha`, `--corr`, `--seed`, `--epochs`.

```
run_cv.py           ABCDEFKL -> 4 fold (train 6, chấm 2) -> cv_score
run_final_test.py   ABCDEFKL -> final.pth -> test GHIJ   -> .txt + scores.csv
```

Mỗi lần chạy thêm một dòng vào `runs/summary.csv`. **G H I J không bao giờ được
nhìn lúc chọn cấu hình** — xem [docs/PROTOCOL.md](docs/PROTOCOL.md).

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


## Chuỗi bằng chứng

```
[Colab] DATA_PREPARE   dữ liệu từ Zenodo -> by_user, windows
                       -> sai lệch hai pipeline = 0.0
                       -> results/checksums.txt

[Colab] TN0    a  chấm bảng MobiVital commit sẵn   -> khớp Table 4 bài báo (0.819)
               b  checkpoint có sẵn, tự chọn kênh
               c  train lại từ đầu
               chạy hai lần: code MobiVital bản gốc (0 dòng sửa), rồi code mình
               -> a và b khớp tới chữ số cuối, lựa chọn kênh trùng 537/537

[Colab] TN1..TN6       TCN, 4 fold                       <- từ đây trở đi
```

Cả hai pipeline chạy **trong cùng một phiên Colab, cùng một GPU**. Bước chọn kênh
là `argmax`, GPU và CPU cộng số theo thứ tự khác nên hai ứng viên gần bằng điểm
có thể đảo thứ hạng — đã đo: cùng checkpoint, GPU và CPU chọn khác kênh ở 251/537
buổi ghi. Đối chiếu từng dòng chỉ có nghĩa khi cùng thiết bị.

Chi tiết: [`notebooks/TN0.md`](notebooks/TN0.md) mục "TN0 nối với TN0.1".

## Dữ liệu

Dữ liệu thô 13 GB nằm trên Zenodo, **không đưa lên GitHub** (GitHub chặn file quá
100 MB). Thay vào đó repo giữ:

- [`notebooks/DATA_PREPARE.ipynb`](notebooks/DATA_PREPARE.ipynb) — đi từ DOI Zenodo
  tới dữ liệu đã xử lý, có đủ output
- [`results/checksums.txt`](results/checksums.txt) — mã băm **nội dung mảng** của 21
  file, để ai chạy lại cũng đối chiếu được

Đo được khi chạy ở hai máy khác nhau:

```
by_user/*.npz     12/12 giong TUNG SO
windows/*.npz     so cua so giong het, gia tri lech ~2e-8
```

`by_user` chỉ dùng `+ - x :` nên chính xác tuyệt đối. `windows` lệch vì phép
`phase` dùng `np.unwrap`: cộng `2pi` 22 lần trong `float32` làm `max-min` lệch
`1e-6`, rồi `self_normalize` **chia** cho số đó nên khuếch đại ra toàn mảng.
Ba phép `abs`, `real`, `imag` khớp tới chữ số 13.

## Đọc tiếp

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — luật thí nghiệm. Đọc trước khi chạy bất cứ gì.
- [`docs/WHY_SPLIT_BY_USER.md`](docs/WHY_SPLIT_BY_USER.md) — vì sao chia dữ liệu theo người.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — chạy toàn bộ trên Colab.
- [`notebooks/TN0.md`](notebooks/TN0.md) — dựng lại kết quả MobiVital.

## Nguyên tắc

- **Không sửa code MobiVital.** `git diff external/mobivital` phải trống.
- **`GHIJ` là tập test**, không dùng để chọn cấu hình. Pool phát triển là `ABCDEFKL`.
- Mọi lựa chọn cấu hình quyết định bằng `cv_score` trên 4 fold, `test_GHIJ` chỉ để nhìn.
