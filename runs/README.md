# runs/ — kết quả thực nghiệm

Mỗi thực nghiệm một thư mục. Mọi thứ của nó nằm chung một chỗ: checkpoint, đường
cong loss, bảng lựa chọn kênh, điểm từng buổi ghi, metric.

```
runs/
├── summary.csv          bảng metric chung cả đồ án, mỗi lần chạy một dòng
│
├── tn0/                 <- notebooks/TN0.ipynb
│   ├── TN0a.txt  TN0b.txt  TN0c.txt      lựa chọn kênh, pipeline MobiVital
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
└── tn7/                 <- scripts/run_final_test.py --experiment tn7
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
| `timestamp` `git_commit` `device` | chạy lúc nào, bản code nào, phần cứng gì |

**Cấu hình đang thử** — đây là thứ TN1–TN6 thay đổi

| cột | nghĩa |
|---|---|
| `model` | `lstm` · `tcn` · `ds_tcn` |
| `revin` | 0 hoặc 1 |
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
| `n_sessions` | số buổi ghi đã chấm. Test GHIJ phải đúng **537** |
| `n_negative` | số buổi Pearson âm — bắt lỗi thầm, sóng chọn ra ngược pha |
| `minutes_score` | thời gian chấm |
| `test_ghij_macro` | chỉ `run_final_test.py` điền. **Số công bố trong luận văn** |

`train_mse` và `score_macro` là **hai thước đo khác nhau, không quy đổi cho nhau**.
`train_mse` đo trên cửa sổ đã lọc bằng `corr(sóng, nhịp thở thật) > 0.9` — tức đã
nhìn đáp án. `score_macro` đo trên buổi ghi thô, model tự chọn kênh, không nhìn
đáp án. Chọn cấu hình phải nhìn `score_macro`. `train_mse` thấp không đảm bảo
`score_macro` cao — đó chính là lý do có TN3.

### `scores_*.csv` — điểm từng buổi ghi

Một dòng một buổi ghi, 537 dòng khi test GHIJ.

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
