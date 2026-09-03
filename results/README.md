# Kết quả thí nghiệm — file đối chứng

Sinh ra từ `notebooks/tn0.ipynb`, xem output trong đó.

| file | do ai sinh | nội dung |
|---|---|---|
| `TN0b.txt` | `mobivital_gen.py` của MobiVital, checkpoint của họ | lựa chọn kênh, 537 dòng |
| `TN0c.txt` | như trên, checkpoint mình train lại | |
| `TN0_1.txt` | `src/scoring.py` của mình, checkpoint MobiVital | |
| `scores_TN0b.csv` | `evaluate.py` của MobiVital | điểm từng buổi ghi, 537 dòng |
| `scores_TN0c.csv` | như trên | |
| `scores_TN0_1.csv` | `src/scoring.py` của mình | |

Pipeline gốc ghi ra hai file mỗi lần chạy: `mobivital_gen.py` ghi lựa chọn kênh,
`evaluate.py` ghi điểm. Bên mình ghi đúng hai file tương ứng để so được cả hai.

Định dạng `.txt`: `tên_file_csv , bin , phép , cờ_lật`

```
240416_userH_tripod_04_2.csv,28,phase,0
```

Điểm tương ứng, chạy bằng `evaluate.py` bản gốc:

```
TN0b   0.8221751511496864     checkpoint MobiVital
TN0c   0.7987479281268859     checkpoint mình train lại
```

Không commit `TN0a.txt` — đó là file gốc trong repo MobiVital, không redistribute.


## TN0.1 — đối chiếu

`notebooks/TN0_1.ipynb`, nạp cùng checkpoint LSTM của MobiVital:

```
lua chon kenh          TRUNG 537 / 537,  khac 0
diem tung buoi ghi     lech lon nhat 1.11e-16  = 1 don vi lam tron cuoi float64
                       so buoi ghi lech > 1e-12:  0
diem trung binh        0.8221751511496862  vs  0.8221751511496864
                       lech 2.22e-16
```

Không chỉ trùng lựa chọn kênh, mà **cả 537 điểm từng buổi ghi đều giống hệt**.
