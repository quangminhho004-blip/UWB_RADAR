# Kết quả thí nghiệm — file đối chứng

Sinh ra từ `notebooks/TN0.ipynb`, chạy một mạch trên Colab. Xem output trong đó.

Notebook làm ba việc hai lần: một lần bằng code MobiVital bản gốc, một lần bằng
code của mình. Mỗi lần ghi ra hai file — bảng lựa chọn kênh và điểm từng buổi
ghi — nên so được cả hai.

| bậc | việc | MobiVital | của mình |
|---|---|---|---|
| a | chấm bảng MobiVital commit sẵn | `scores_TN0a.csv` | `scores_ours_a.csv` |
| b | checkpoint có sẵn, tự chọn kênh | `TN0b.txt` `scores_TN0b.csv` | `ours_b.txt` `scores_ours_b.csv` |
| c | train lại từ đầu | `TN0c.txt` `scores_TN0c.csv` | `ours_c.txt` `scores_ours_c.csv` |

Bậc a của mình không có file `.txt` riêng: nó chấm đúng bảng `TN0a.txt`.

Định dạng `.txt`: `tên_file_csv , bin , phép , cờ_lật`

```
240416_userH_tripod_04_2.csv,28,phase,0
```

Bậc a và b phải khớp tuyệt đối — cùng dữ liệu, cùng thuật toán, không có gì ngẫu
nhiên. Bậc c thì không, vì thứ tự xáo trộn dữ liệu khác nhau.

`TN0a.txt` là bản sao bảng MobiVital commit sẵn trong repo họ, giữ ở đây để mục 6
của notebook chấm lại được. Nguồn: `inference/methods/` của
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
