# Kết quả thí nghiệm

Mỗi thực nghiệm một thư mục riêng, nén thành một tệp riêng khi cất lên Drive:

```
results/tn0/   <- notebooks/TN0.ipynb                        -> Drive/mobivital/tn0.tar.gz
results/tn1/   <- run_cv.py --experiment tn1                 -> Drive/mobivital/tn1.tar.gz
results/tn2/   <- run_cv.py --experiment tn2                 -> Drive/mobivital/tn2.tar.gz
...                                                             ...
results/tn7/   <- run_final_test.py --experiment tn7         -> Drive/mobivital/tn7.tar.gz

results/checksums.txt   băm dữ liệu, không thuộc thực nghiệm nào
```

`--experiment` là **bắt buộc** ở cả `run_cv.py` và `run_final_test.py`: nó quyết
định `results/<tên>/` và `runs/<tên>/`. Không có thùng dùng chung, không thực
nghiệm nào ghi đè kết quả của thực nghiệm khác.

Nén lên Drive bằng `python scripts/save_results.py <tên thực nghiệm>`.

### Trong mỗi tệp nén có gì

`.tar.gz` giống `.zip`: `tar` gộp cả thư mục thành một tệp, `gz` nén lại.

```
tn1/
  summary.csv       metric — các dòng của tn1 trong runs/summary.csv
  compare.csv       bảng ĐẠT / KHÔNG ĐẠT (chỉ TN0)
  README.txt        sinh lúc nào, commit nào
  *.txt             tệp lựa chọn kênh
  scores_*.csv      điểm từng buổi ghi
  runs/             checkpoint .pth và đường cong loss từng epoch
```

Tệp nén **tự chứa đủ**: tải riêng về vẫn đọc được metric mà không cần bảng chung.
Bung ra xem, đúng chỗ cũ:

```bash
tar -xzf tn1.tar.gz -C results/     # -> results/tn1/...
```

### Metric lưu ở đâu

| chỗ | nội dung |
|---|---|
| `runs/summary.csv` | bảng chính của cả đồ án, mỗi lần chạy một dòng, 27 cột |
| `results/<tn>/scores_*.csv` | Pearson từng buổi ghi |
| `runs/<tn>/<run_id>/curve.csv` | loss từng epoch |
| `runs/<tn>/<run_id>/final.pth` | trọng số |


## results/tn0/

Ba kiểm tra, chạy hai lần — một lần bằng code MobiVital, một lần bằng code đồ án.
Mỗi lần ghi hai tệp: tệp lựa chọn kênh `.txt` và điểm từng buổi ghi `.csv`.

| kiểm tra | pipeline MobiVital | pipeline đồ án |
|---|---|---|
| TN0a  tính điểm từ tệp lựa chọn kênh có sẵn | `TN0a.txt` `scores_TN0a.csv` | `scores_ours_a.csv` |
| TN0b  chọn kênh bằng tệp trọng số tác giả | `TN0b.txt` `scores_TN0b.csv` | `ours_b.txt` `scores_ours_b.csv` |
| TN0c  train lại LSTM | `TN0c.txt` `scores_TN0c.csv` | `ours_c.txt` `scores_ours_c.csv` |

TN0a phía đồ án không có tệp `.txt` riêng: nó tính điểm ngay trên `TN0a.txt`.

Định dạng tệp lựa chọn kênh: `tên_tệp_csv , kênh khoảng cách , phép biến đổi , cờ lật`

```
240416_userH_tripod_04_2.csv,28,phase,0
```

TN0a và TN0b bắt buộc hai pipeline khớp tới chữ số cuối. TN0c chỉ tham khảo vì
đây là hai lần train độc lập.

`TN0a.txt` là bản sao tệp MobiVital commit sẵn trong repo họ, giữ ở đây để chạy
lại được. Nguồn: `inference/methods/` của
[mobivital-public](https://github.com/nesl/mobivital-public).


## checksums.txt

Mã băm **nội dung mảng** của 21 file dữ liệu đã xử lý, sinh bởi
`scripts/6_checksums.py`.

Dữ liệu thô 13 GB nằm trên Zenodo (DOI 10.5281/zenodo.15022885), không đưa lên
GitHub được — GitHub chặn file quá 100 MB. Ai muốn kiểm chứng thì chạy
`notebooks/DATA_PREPARE.ipynb` rồi đối chiếu với file này.

Không băm thẳng file `.npz` vì `.npz` là ZIP, mà ZIP nhúng thời điểm ghi vào
từng mục — hai file nội dung y hệt vẫn khác md5.

Đo được khi chạy ở hai máy khác nhau (MacBook và Colab):

```
by_user/*.npz     12/12 giong TUNG SO
windows/*.npz     so cua so giong het, gia tri lech ~2e-8
```

`by_user` chỉ dùng `+ - x :` nên chính xác tuyệt đối. `windows` dùng `np.angle`,
`np.unwrap`, `np.corrcoef` — hàm siêu việt, chữ số cuối phụ thuộc thư viện toán
từng máy. Lệch `2e-8` nhỏ hơn cả sai số biểu diễn của `float32` (`1.2e-7`).
