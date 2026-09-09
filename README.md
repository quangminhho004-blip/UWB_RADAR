# MobiVital — đồ án tốt nghiệp

Khảo sát mô hình dự báo dùng trong bộ chọn kênh của MobiVital (radar UWB không
tiếp xúc): kiến trúc, tầm nhìn, và hàm mục tiêu.

**Mọi số liệu nằm ở [`docs/BANG_DIEM.md`](docs/BANG_DIEM.md)**, không ghi trong
tệp này — để chỉ có một chỗ duy nhất phải cập nhật khi có kết quả mới.

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

## Cài đặt

Repo này **không chứa mã MobiVital và không chứa dữ liệu**. Cả hai phải lấy về,
đúng chỗ mà mã MobiVital đòi. Ba bước dưới làm đúng việc đó.

### Bước 1 — lấy repo đồ án

```bash
git clone https://github.com/quangminhho004-blip/UWB_RADAR.git
cd UWB_RADAR
```

### Bước 2 — lấy mã MobiVital vào `external/mobivital`

Repo MobiVital **không có LICENSE** nên không được chép vào repo này. Phải clone
riêng, và **ghim đúng commit** đã dùng cho mọi số liệu trong luận văn:

```bash
git clone https://github.com/nesl/mobivital-public.git external/mobivital
git -C external/mobivital checkout 4319731d2769d4134c92088dd846666e262f18e9
```

Trên Colab thì một lệnh làm cả hai việc, kèm cài `einops`:

```bash
python scripts/setup_colab.py
```

Đường dẫn `external/mobivital` là **bắt buộc**, không đổi tên được — mã
MobiVital dùng đường dẫn tương đối, và `src/mobivital_reference.py` trỏ vào đó.

### Bước 3 — lấy dữ liệu vào thư mục `tripod`

Dữ liệu thô 13 GB trên Zenodo, giải nén thẳng vào **bên trong** thư mục
MobiVital vừa clone:

```bash
python scripts/download_dataset.py
```

Ra đúng chỗ này:

```
external/mobivital/
    dataset/
        mobivital/
            tripod/        <- 1874 tệp CSV, 13 GB
```

Vì sao đúng chỗ đó: `prep_breath_final.py` dòng 18 đọc theo đường dẫn tương đối
`./dataset/mobivital/tripod/`. Đặt sai chỗ là mã của tác giả không chạy được.

Script tải bằng `aria2c` 16 luồng — đo thật trên Colab: `wget` một luồng mất
**2,3 giờ**, `aria2c -x16` mất **2–4 phút**, vì Zenodo bóp băng thông mỗi kết
nối. Chạy lại được: đủ 1874 tệp thì bỏ qua, không tải lại.

Muốn tải tay thì lấy `tripod.zip` từ
[Zenodo](https://doi.org/10.5281/zenodo.15022885) rồi:

```bash
unzip tripod.zip -d external/mobivital/dataset/mobivital/
```

Giải nén vào `.../mobivital/`, **không** vào `.../mobivital/tripod/` — trong tệp
nén đã có sẵn thư mục `tripod/`, vào sâu một tầng nữa là lồng hai lớp.

**Chỉ giữ một bản CSV duy nhất.** Hai pipeline đọc chung bản đó, nên khi
`check_data.py` báo sai lệch bằng 0 thì không ai cãi được là do hai bản dữ liệu
khác nhau. `data/` chỉ chứa thứ pipeline của đồ án sinh ra.


## Dựng dữ liệu đã xử lý

Chạy lần lượt, một lần duy nhất. Cả hai pipeline đọc chung bộ CSV ở bước 3:

```
external/mobivital/dataset/mobivital/tripod/   1874 CSV, 13 GB
        |
        +--> prep_breath_final.py CỦA TÁC GIẢ  -> data_final/*.npy
        |
        +--> scripts/make_npz.py CỦA ĐỒ ÁN     -> data/processed/by_user/*.npz
```

| # | lệnh | ra cái gì |
|---|---|---|
| 1 | `python scripts/mobivital/setup_dataset.py` | vá 52 tên tệp lỗi thời, giấu dữ liệu khỏi git của MobiVital |
| 2 | `cd external/mobivital && python dataset_preparation/prep_breath_final.py` | `data_final/*.npy` — **pipeline gốc**, script của tác giả, 0 dòng sửa |
| 3 | `python scripts/make_npz.py` | `by_user/*.npz` — **pipeline của đồ án** |
| 4 | `python scripts/check_data.py` | đối chiếu hai bên, phải khớp từng byte |
| 5 | `python scripts/make_windows.py` | `data/processed/windows/` — cửa sổ cắt sẵn |

Bước 4 phải in ra:

```
ABCDEFKL  1289/1289 buổi ghi khớp TỪNG BYTE   = training_breath_tripod_data.npy
GHIJ       537/537  buổi ghi khớp TỪNG BYTE   = testing_breath_tripod_data.npy
```

Không ra thế thì dừng, mọi so sánh về sau vô nghĩa.

Từ đây mọi thí nghiệm chỉ đọc `by_user/*.npz` và `windows/`, bỏ được CSV thô
13 GB.

Chạy `notebooks/DATA_PREPARE.ipynb` một lần là hai tệp `by_user.tar` và
`windows.tar.gz` nằm sẵn trên Drive. Từ đó mọi notebook sau gọi
`python scripts/restore_processed_data_on_drive.py` để lấy về (2 phút) thay vì
chạy lại bước 3 và 5 (~16 phút). CSV thô không cất lên Drive vì pipeline
MobiVital đọc thẳng CSV, mà tải lại từ Zenodo chỉ mất vài phút.

Muốn kiểm dữ liệu của mình có trùng với bản dùng trong luận văn không:

```bash
python scripts/checksums.py        # so với data/checksums.txt
```

## Cấu trúc

```
src/         pipeline của đồ án — models.py, training.py, scoring.py, results.py
scripts/     gọi src/ theo đúng thứ tự; notebook chỉ chạy một dòng !python
  setup_colab.py       clone mã nguồn, ghim commit MobiVital, nối runs/ vào Drive
  download_dataset.py  tải Zenodo -> giải nén vào thư mục MobiVital
  make_npz.py          đọc CSV trong thư mục MobiVital -> by_user/*.npz
  check_data.py        so dữ liệu đồ án với dữ liệu MobiVital, từng byte
  make_windows.py      -> data/processed/windows/
  checksums.py         -> data/checksums.txt
  run_tn0.py           TN0 pipeline ĐỒ ÁN: --case a|b|c, --compare
  run_cv.py            4 fold trên ABCDEFKL, chọn cấu hình
  run_final_test.py    train đủ ABCDEFKL, test GHIJ một lần duy nhất
  save_results.py      nén runs/<thực nghiệm>/ thành runs/<thực nghiệm>.zip
  mobivital/           chạy code tác giả nguyên bản
    setup_dataset.py   vá 52 tên tệp lỗi thời, giấu dữ liệu khỏi git MobiVital
    run_tn0.py         TN0 pipeline MOBIVITAL: --case prep|a|b|c
notebooks/   thí nghiệm, chạy trên Colab
docs/        luật thí nghiệm và lý do thiết kế
```

**Hai tệp trùng tên `run_tn0.py`** — phân biệt bằng thư mục:

```
scripts/run_tn0.py             chạy code CỦA ĐỒ ÁN trong src/, và in bảng --compare
scripts/mobivital/run_tn0.py   chạy code CỦA TÁC GIẢ (autoreg_training, mobivital_gen,
                               evaluate, prep_breath_final)

Nằm trong mobivital/  =>  chạy code của tác giả.

data/        KHÔNG commit, để trên Google Drive
runs/        KHÔNG commit, checkpoint và kết quả
external/mobivital/   KHÔNG commit, clone riêng
```

Mã MobiVital và dữ liệu lấy riêng — xem mục **Cài đặt** ở trên.

## Chạy thí nghiệm

```bash
python scripts/run_cv.py         --experiment tn1 --model ds_tcn --revin true  # chọn cấu hình
python scripts/run_final_test.py --experiment tn7 --model ds_tcn --revin true  # số công bố
```

`--experiment` **bắt buộc** — nó quyết định thư mục kết quả, mỗi thực nghiệm
một thư mục riêng, không dùng chung:

```
--experiment tn1  ->  runs/tn1/
--experiment tn2  ->  runs/tn2/
```

Trong `runs/<tên>/` có đủ checkpoint, đường cong loss, bảng lựa chọn kênh, điểm
từng buổi ghi và metric. Xong thì nén lại mang đi:

```bash
python scripts/save_results.py tn1                    # -> runs/tn1.zip
python scripts/save_results.py tn1 --out tn1_phien2   # -> runs/tn1_phien2.zip
unzip tn1.zip -d runs/                                # bung lại đúng chỗ cũ
```

Trên Colab, `save_results.py` chép luôn tệp nén sang Drive. Và hai script train
**tự gọi nó** — `run_cv.py` sau mỗi fold, `run_final_test.py` sau mỗi lần chạy —
nên ngắt phiên giữa chừng cũng không mất phần đã xong. Tên tệp nén chứa cấu hình
nên hai phiên chạy song song không đè lên nhau.

Chạy lại một lệnh đã có kết quả thì script bỏ qua, không train lại.

Hai script nhận chung một bộ cờ:

```
--model lstm|tcn|ds_tcn      --revin true|false
--loss mse|mse_pearson       --alpha        trọng số MSE khi dùng mse_pearson
--corr                       ngưỡng lọc cửa sổ train, mặc định 0.9
--seed                       --epochs

--hidden                     chiều ẩn LSTM, mặc định 352
--channels                   số kênh TCN, mặc định 64
--kernel_size --n_blocks --dropout        riêng cho TCN
```

Không gõ cờ nào thì nó lấy mặc định — cấu hình MobiVital công bố trong
`checkpoints/optimal_params.json`. Cờ đi vào tên lần chạy, và hậu tố chỉ xuất
hiện khi giá trị **khác mặc định**, nên tên của các lần chạy cũ không đổi khi
thêm cờ mới.

```
run_cv.py           ABCDEFKL -> 4 fold (train 6, chấm 2) -> cv_score
run_final_test.py   ABCDEFKL -> final.pth -> test GHIJ   -> .txt + scores.csv
```

Mỗi lần chạy thêm một dòng vào `runs/summary.csv`. **G H I J không bao giờ được
nhìn lúc chọn cấu hình** — xem [docs/PROTOCOL.md](docs/PROTOCOL.md).

Vì sao chia bốn fold như vậy, vì sao ghép cặp người như thế, mỗi cấu hình train
bao nhiêu lần, và bằng chứng của từng con số nằm ở đâu:
[docs/CHIA_DU_LIEU.md](docs/CHIA_DU_LIEU.md).

Cấu hình đầy đủ của từng model trong TN1, kể cả tham số lấy mặc định:
[docs/CAU_HINH_TN1.txt](docs/CAU_HINH_TN1.txt).

Các model nhận `--model lstm | bilstm | cnn_lstm | tcn | ds_tcn`.

## Kết quả và metric

Mỗi thực nghiệm một thư mục `runs/<tên>/`, chứa đủ checkpoint, đường cong loss,
bảng lựa chọn kênh, điểm từng buổi ghi và metric. Ý nghĩa từng cột, từng tệp:
[runs/README.md](runs/README.md).

Số quyết định là **`score_macro`** — Pearson trung bình theo người, đo trên buổi
ghi thô qua bộ chọn kênh. Không phải `train_mse`: cửa sổ lúc train đã lọc bằng
`corr(sóng, nhịp thở thật) > 0.9`, tức đã nhìn đáp án.

## Ranh giới — cái gì của ai

| Mượn lại từ MobiVital | Tự viết ở đồ án này |
|---|---|
| `generate_dataset` cắt cửa sổ train | TCN, DS-TCN, RevIN |
| `sequence_transforms`, `transform` | loss MSE + Pearson |
| `self_normalize` | vòng train, checkpoint, resume |
| `invert_detector` | 4-fold CV trên ABCDEFKL |
| `LSTMMultiStep` làm baseline | bộ chọn kênh dùng chung cho LSTM và TCN |

[`src/mobivital_reference.py`](src/mobivital_reference.py) là chỗ duy nhất chạm vào code
MobiVital — chỉ `import` sáu hàm thuần tính toán, không nạp file script nào của tác giả.


## Chuỗi bằng chứng

```
[Colab] DATA_PREPARE   dữ liệu từ Zenodo -> by_user, windows
                       -> đối chiếu hai pipeline, phải khớp từng byte
                       -> data/checksums.txt

[Colab] TN0    a  chấm bảng lựa chọn kênh tác giả commit sẵn
               b  checkpoint tác giả, đồ án tự chạy inference
               c  train lại từ đầu
               chạy hai lần: mã MobiVital bản gốc (0 dòng sửa), rồi mã đồ án
               -> đối chiếu từng dòng, và đối chiếu với bài báo

[Colab] TN1..TN4       so kiến trúc, tầm nhìn, hàm loss, rồi test cuối
```

Cả hai pipeline chạy **trong cùng một phiên Colab, cùng một thiết bị**. Bước
chọn kênh là `argmax`, thiết bị khác nhau cộng số theo thứ tự khác nên hai ứng
viên gần bằng điểm có thể đảo thứ hạng. Đối chiếu từng dòng chỉ có nghĩa khi
cùng thiết bị.

Chi tiết và số liệu: [`notebooks/TN0.md`](notebooks/TN0.md).

## Dữ liệu

Dữ liệu thô 13 GB nằm trên Zenodo, **không đưa lên GitHub** (GitHub chặn file quá
100 MB). Thay vào đó repo giữ:

- [`notebooks/DATA_PREPARE.ipynb`](notebooks/DATA_PREPARE.ipynb) — đi từ DOI Zenodo
  tới dữ liệu đã xử lý, có đủ output
- [`data/checksums.txt`](data/checksums.txt) — mã băm **nội dung mảng** của 12
  file, để ai chạy lại cũng đối chiếu được

Chạy lại ở máy khác thì `by_user` khớp từng số, còn `windows` lệch rất nhỏ ở
phép `phase`: `np.unwrap` cộng `2pi` nhiều lần trong `float32` làm `max − min`
lệch, rồi `self_normalize` **chia** cho số đó nên khuếch đại ra toàn mảng. Ba
phép `abs`, `real`, `imag` thì khớp gần như tuyệt đối.

## Đọc tiếp

- [`docs/BANG_DIEM.md`](docs/BANG_DIEM.md) — **mọi số liệu của đồ án**, một chỗ duy nhất.
- [`docs/NHANH_DS_TCN.md`](docs/NHANH_DS_TCN.md) — cây nhánh DS-TCN: thực nghiệm nào kế thừa cấu hình nào, nhánh nào cụt.
- [`docs/BANG_NHANH_TCN.md`](docs/BANG_NHANH_TCN.md) — bảng cột TN1–TN4, hàng kiến trúc: cái nào đi tiếp, cái nào dừng.
- [`docs/BANG_TCN_TUNG_SEED.md`](docs/BANG_TCN_TUNG_SEED.md) — họ TCN, điểm từng seed, cả micro lẫn macro.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — luật thí nghiệm. Đọc trước khi chạy bất cứ gì.
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — sơ đồ khối hai giai đoạn, dùng cho slide.
- [`docs/CAU_TRUC_MA_NGUON.md`](docs/CAU_TRUC_MA_NGUON.md) — mỗi tệp trong `src/` và `scripts/` làm gì.
- [`docs/RESEARCH_GAP.md`](docs/RESEARCH_GAP.md) — khoảng trống nghiên cứu và cách phát biểu kết quả.
- [`docs/CHIA_DU_LIEU.md`](docs/CHIA_DU_LIEU.md) — vì sao chia dữ liệu theo người, vì sao bốn fold.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — chạy toàn bộ trên Colab.
- [`notebooks/TN0.md`](notebooks/TN0.md) — dựng lại kết quả MobiVital.

## Nguyên tắc

- **Không sửa code MobiVital.** `git diff external/mobivital` phải trống.
- **`GHIJ` là tập test**, không dùng để chọn cấu hình. Pool phát triển là `ABCDEFKL`.
- Mọi lựa chọn cấu hình quyết định bằng `cv_score` trên 4 fold, `test_GHIJ` chỉ để nhìn.
