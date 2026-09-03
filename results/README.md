# Kết quả thí nghiệm — file đối chứng

Sinh ra từ `notebooks/tn0.ipynb`, xem output trong đó.

| file | do ai sinh | dùng để |
|---|---|---|
| `TN0b.txt` | `mobivital_gen.py` của MobiVital + checkpoint của họ | mốc đối chiếu cho TN0.1 |
| `TN0c.txt` | như trên, checkpoint mình train lại | |
| `scores_TN0b.csv` | `evaluate.py` của MobiVital | điểm từng buổi ghi |
| `scores_TN0c.csv` | như trên | |

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
