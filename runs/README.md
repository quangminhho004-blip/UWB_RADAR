# runs/ — kết quả thực nghiệm

Năm thực nghiệm nối nhau. Bảng đầy đủ kèm cách đọc:
[docs/BANG_TCN.md](../docs/BANG_TCN.md).

---

## `tn0/` — tái lập MobiVital

Chạy lại pipeline của tác giả rồi đối chiếu với pipeline của đồ án trên cùng dữ
liệu. Nếu bước này không đạt thì mọi so sánh sau đều vô nghĩa.

| bằng chứng | phải thấy gì |
|---|---|
| `scores_TN0a.csv` | micro **0,8195** trên 537 phiên, khớp con số bài báo trong dung sai 0,001 |
| `scores_TN0b.csv` | micro 0,8222 |
| `TN0a.txt` `TN0b.txt` | 537 dòng mỗi tệp — bảng lựa chọn kênh |

---

## `tn1/` — chọn kiến trúc (4 fold · 3 seed)

Layout `<cấu hình>/seed<N>/<fold>/{curve.csv, scores.csv, final.pth}`.

| thư mục | notebook | config_id | CV macro |
|---|---|---|---:|
| `DS-TCN-C64-RF61/` | `TN1_DS_TCN_RF61_no_norm_do02_c64` | `ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9_seed{0,1,2}` | **0,760878** ± 0,003095 |
| `LSTM-352/` | `TN1_LSTM` | `lstm_mse_corr0.9_seed{0,1,2}` | 0,756998 ± 0,004141 |
| `LSTM-67/` | `TN1_LSTM_small` | `lstm_h67_mse_corr0.9_seed{0,1,2}` | 0,753208 ± 0,001966 |
| `CNN-LSTM-58/` | `TN1_CNN_LSTM` | `cnn_lstm_h58_c32_k5_mse_corr0.9_seed{0,1,2}` | 0,752666 ± 0,003749 |

`DS-TCN-C64-RF61` là cấu hình được chọn, mang xuống TN2 → TN3 → TN4.

**Hai dòng `TONG` dựng lại.** Seed 2 của `LSTM-352` và `LSTM-67` không có dòng
`TONG` trong tệp nén gốc: `run_cv.py` ghi dòng đó sau khi xong cả bốn fold, nên
ô lưu kết quả chạy trước thì chưa kịp. Chúng được tính lại từ `scores.csv`
per-fold, đúng công thức `run_cv.py` dùng — macro là trung bình bốn fold,
`score_std` là độ lệch chuẩn tổng thể. Công thức đã đối chiếu với bảy dòng
`TONG` có sẵn, khớp từng chữ số.

---

## `tn2_rf/` — tầm nhìn (4 fold · 1 seed)

Giữ nguyên DS-TCN 64, 4 khối, không chuẩn hoá, dropout 0,2 theo phần tử. Đổi
**đúng một thứ**: kernel. Tầm nhìn = `1 + 2 × (kernel − 1) × Σ dilation`.

| thư mục | kernel | tầm nhìn | tham số | cv_score |
|---|---:|---:|---:|---:|
| *(ở `tn1/`)* | **3** | **61** | **37.081** | **0,760878** ± 0,003095 *(3 seed)* |
| `c64-rf121/` | 5 | 121 | 38.105 | 0,757855 |
| `c64-rf181/` | 7 | 181 | 39.129 | 0,743657 |
| `c64-rf241/` | 9 | 241 | 40.153 | 0,736970 |

Tầm nhìn càng rộng điểm càng thấp, đơn điệu. Cửa sổ vào chỉ 200 mẫu nên kernel 9
đã phủ dư (241 > 200) mà vẫn không giúp gì.

Chênh RF61 với RF121 là 0,0030 — xấp xỉ dao động seed của RF61, chưa tách được.
Vì vậy **cả hai** mang xuống TN3.

Notebook `TN2_ReceptiveField_DS_TCN_c64` còn chạy kernel 11 và 13 **chỉ trên một
fold** để xem đẩy tầm nhìn vượt hẳn cửa sổ có đảo chiều xu hướng không. Không
đảo, nên hai mức đó dừng ở vòng sàng lọc và không có trong thư mục này.

---

## `tn3/` — hàm loss lai (10 mức alpha · 4 fold · 1 seed)

Layout `<cấu hình>/a<alpha>/seed0/<fold>/...`. `a0` = Pearson thuần.
`loss = alpha × MSE + (1 − alpha) × (1 − Pearson)` — `alpha` là trọng số của
**MSE**, không phải ngưỡng chọn kênh.

| thư mục | tầm nhìn | đỉnh | mốc MSE thuần |
|---|---:|---|---|
| `c64-rf61/` | 61 | alpha 0,6 → **0,780028** | 0,760878 *(tn1)* |
| `c64-rf121/` | 121 | alpha 0 → **0,780306** | 0,757855 *(tn2_rf)* |

**19 trong 20 mức alpha hơn MSE thuần.** Đỉnh hơn mốc 0,0192 và 0,0225. Ngoại lệ
duy nhất: RF121 ở alpha 0,6 được 0,752386, thấp hơn mốc của chính nó 0,0055 —
một seed nên đọc là nhiễu.

Bảng này **một seed**. Dải 0,776–0,780 của RF61 trải trong khoảng 0,004, đúng cỡ
dao động seed đo ở TN1. Kết luận đọc được là *"đưa Pearson vào loss thì tốt hơn
MSE thuần"*, không phải *"0,6 là mức tối ưu"*.

---

## `tn4/` — kiểm tra trên G H I J (3 seed)

Train đủ tám người ABCDEFKL rồi chấm một lần trên 537 phiên của G H I J.
G H I J không dùng để chọn bất cứ thứ gì.

| thư mục | | tham số | macro | micro |
|---|---|---:|---:|---:|
| `LSTM-352/` | mốc MobiVital, giữ nguyên | 1.502.713 | **0,810302** ± 0,015402 | 0,805309 |
| `64-61__hybrid-a0.6/` | DS-TCN 64/RF61 · alpha 0,6 ★ **mô hình cuối** | **37.081** | **0,803590** ± 0,015350 | 0,798315 |
| `64-121__pearson/` | DS-TCN 64/RF121 · Pearson thuần | 38.105 | 0,801739 ± 0,009968 | 0,796845 |
| `64-121__mse/` | DS-TCN 64/RF121 · MSE thuần — đối chứng | 38.105 | 0,762191 ± 0,021433 | 0,755965 |

**Hàm loss có tác dụng rõ nhất.** Cùng kiến trúc 64/RF121, chỉ đổi loss:
0,801739 so với 0,762191 — chênh **0,0395**, thắng cả ba seed. Phép so sạch nhất
trong đồ án.

**LSTM 352 đứng đầu, không phải DS-TCN.** Mô hình cuối thấp hơn 0,0067 — nhỏ hơn
dao động seed của cả hai bên (0,0154), nên không xếp hạng được. Phát biểu đúng
là **ngang điểm với ít hơn 40,5 lần tham số**. Ngược thứ hạng ở TN1; xem
[docs/BANG_TCN.md](../docs/BANG_TCN.md) mục 5.

Ba lượt của `LSTM-352/` đặt `--experiment tn1_ghij` lúc chạy nên log trong
`TN1_LSTM.ipynb` in ra `runs/tn1_ghij/`. Kết quả cùng giao thức nên xếp chung
vào đây; cột `experiment` trong `summary.csv` đã đổi thành `tn4`, `run_id` giữ
nguyên. `SOURCE_MANIFEST.json` chỉ phủ ba thư mục DS-TCN — nó lập trước khi mốc
LSTM chuyển vào.

---

## Dựng lại mọi bảng trên

```bash
python3 scripts/compare_cv.py --experiment tn1
python3 scripts/compare_cv.py --experiment tn2_rf
python3 scripts/compare_cv.py --experiment tn3
python3 scripts/compare_cv.py --experiment tn4 --final
```

---

Mỗi thực nghiệm một thư mục. Mọi thứ của nó nằm chung một chỗ: checkpoint, đường
cong loss, bảng lựa chọn kênh, điểm từng buổi ghi, metric.

```
runs/
├── summary.csv          bảng metric chung cả đồ án, mỗi lần chạy một dòng
│
├── tn0/                 <- notebooks/TN0.ipynb
│   ├── TN0a.txt  TN0b.txt              lựa chọn kênh, pipeline MobiVital
│   ├── ours_b.txt  ours_c.txt            lựa chọn kênh, pipeline đồ án
│   ├── scores_*.csv                      điểm từng buổi ghi, 537 dòng
│   ├── compare.csv                       bảng ĐẠT / KHÔNG ĐẠT
│   ├── summary.csv  README.txt           do save_results.py sinh
│   └── ours_c/final.pth  curve.csv       trọng số, loss từng epoch
├── tn0.zip
│
├── tn1/                 <- scripts/run_cv.py --experiment tn1
│   ├── <cấu hình>_val_AB/final.pth  curve.csv
│   ├── scores_<cấu hình>_val_AB.csv
│   └── summary.csv  README.txt
├── tn1.zip
│
└── tn4/                 <- scripts/run_final_test.py --experiment tn4
```

`--experiment` **bắt buộc** ở `run_cv.py` và `run_final_test.py`; nó quyết định
tên thư mục. Không có thùng dùng chung, không thực nghiệm nào ghi đè thực nghiệm
khác.


## Từng loại tệp chứa gì

### `summary.csv` — bảng metric, mỗi lần chạy một dòng

Đây là bảng để dựng bảng kết quả trong luận văn. 28 cột, chia bốn nhóm.

**Nhận dạng** — tự điền, không phải truyền vào

| cột | nghĩa |
|---|---|
| `run_id` | tên lần chạy, gộp từ cấu hình |
| `experiment` | `tn0`, `tn1`, … — trùng tên thư mục |
| `timestamp` `git_commit` `device` | chạy lúc nào, bản code nào, `cuda` hay `cpu`. Cố ý không ghi tên đời GPU |

**Cấu hình đang thử** — đây là thứ TN1 thay đổi giữa bốn cấu hình

| cột | nghĩa |
|---|---|
| `model` | `lstm` · `cnn_lstm` · `tcn` · `ds_tcn` |
| `loss` `alpha` | `mse` hoặc `mse_pearson`; `alpha` là trọng số phần MSE |
| `corr_threshold` | ngưỡng lọc sóng đáng học lúc cắt cửa sổ, mặc định 0.9 |
| `seed` | hạt giống ngẫu nhiên |
| `fold` `val_users` | `val_AB` và `AB` — chỉ có ở CV. `fold = TONG` là dòng tổng của cả 4 fold |

**Lúc train** — đo trên **cửa sổ cắt sẵn**

| cột | nghĩa |
|---|---|
| `n_params` | số tham số học được. Trả lời câu "tốt hơn vì kiến trúc hay vì model to hơn" |
| `n_train_windows` `epochs` | lượng dữ liệu và số vòng |
| `train_mse` `train_pearson` `train_loss` | ghi cả MSE lẫn Pearson mọi lúc, bất kể đang tối ưu cái nào, để so được giữa các thí nghiệm dùng loss khác nhau |
| `minutes_train` | chỉ để tính giờ Colab, **không dùng làm bằng chứng tốc độ** — phần cứng Colab đổi giữa các phiên |
| `resumed` | 1 nếu lần chạy này nối tiếp một phiên bị ngắt |

**Lúc chấm điểm** — đo trên **buổi ghi thô**, model tự chọn kênh

| cột | nghĩa |
|---|---|
| **`score_macro`** | **số quyết định**. Trung bình theo người, không theo buổi ghi — mỗi người có số buổi khác nhau, tính gộp thì người ghi nhiều buổi bị tính nặng ký vô lý |
| `score_micro` | trung bình theo buổi ghi, để tham khảo |
| `score_std` | độ lệch chuẩn giữa 4 fold, chỉ có ở dòng `fold = TONG` |
| `n_sessions` | số buổi ghi đã chấm ở fold đó |
| `n_negative` | số buổi Pearson âm — bắt lỗi thầm, sóng chọn ra ngược pha |
| `minutes_score` | thời gian chấm |

`train_mse` và `score_macro` là **hai thước đo khác nhau, không quy đổi cho nhau**.
`train_mse` đo trên cửa sổ đã lọc bằng `corr(sóng, nhịp thở thật) > 0.9` — tức đã
nhìn đáp án. `score_macro` đo trên buổi ghi thô, model tự chọn kênh, không nhìn
đáp án. Chọn cấu hình phải nhìn `score_macro`. `train_mse` thấp không đảm bảo
`score_macro` cao.

### `scores_*.csv` — điểm từng buổi ghi

Một dòng một buổi ghi: hai người của fold đó khi CV, hoặc đủ 537 phiên
G H I J khi test cuối.

| cột | nghĩa |
|---|---|
| `user` `session_file` | người nào, tệp CSV nào |
| `bin` `method` | kênh khoảng cách và phép biến đổi mà model đã chọn |
| `n_candidates_kept` | còn bao nhiêu ứng viên sau khi loại sóng đảo chiều |
| `pearson` | điểm của buổi ghi đó |

Dùng để so hai cấu hình theo kiểu **thắng / hoà / thua trên từng buổi ghi**, chứ
không chỉ so hai con số trung bình. Hai bảng điểm khác hẳn nhau vẫn có thể cho
cùng một trung bình.

### `*.txt` — bảng lựa chọn kênh

Định dạng của MobiVital, để đối chiếu trực tiếp với tệp họ commit sẵn:

```
tên_tệp_csv , kênh khoảng cách , phép biến đổi , cờ lật
240416_userH_tripod_04_2.csv,28,phase,0
```

Cờ lật MobiVital ghi cứng 0, không bao giờ bằng 1.

### `<run_id>/curve.csv` — loss từng epoch

`epoch`, `train_mse`, `train_pearson`, `train_loss`, `minutes`. Dùng vẽ đường hội
tụ, và để thấy model có overfit không.

### `<run_id>/final.pth` — trọng số

Chỉ trọng số, không kèm trạng thái Adam. Trong lúc train còn có `last.pth` nặng
gấp ba (kèm Adam và trạng thái sinh số ngẫu nhiên) để chạy tiếp khi Colab ngắt
phiên; train xong thì xoá.

### `compare.csv` — riêng TN0

Bảng ĐẠT / KHÔNG ĐẠT: `kiem_tra`, `mobivital`, `pipeline_do_an`, `ket_luan`,
`chi_tiet`.


## Nén và mang đi

```bash
python scripts/save_results.py tn0     # -> runs/tn0.zip
unzip tn0.zip -d runs/                 # bung lại đúng chỗ cũ
```

Tệp nén **tự chứa đủ**: `summary.csv` bên trong đã lọc sẵn các dòng metric của
riêng thực nghiệm đó, tải về đọc được ngay mà không cần bảng chung.


## Cái gì lên GitHub, cái gì không

| | |
|---|---|
| lên GitHub | `.txt`, `.csv` — nhỏ, là bằng chứng cho hội đồng xem |
| không lên | `*.pth` trọng số, `*.zip` tệp nén — xem `.gitignore` |

Mã băm dữ liệu nằm riêng ở `data/checksums.txt` vì nó mô tả dữ liệu, không thuộc
thực nghiệm nào.
