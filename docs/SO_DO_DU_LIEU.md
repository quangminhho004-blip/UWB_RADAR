# Sơ đồ tư duy — dữ liệu đi từ đâu tới đâu

Từ 5,7 GB trên Zenodo tới điểm số cuối cùng. Mỗi hộp ghi **file gì**, mũi tên
ghi **script nào biến nó thành file kế tiếp**.


## Toàn cảnh

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ZENODO   tripod.zip   5,7 GB                                                 │
│  DOI 10.5281/zenodo.15022885                                                  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  download_dataset.py   (aria2c 16 luồng, 2–4 phút)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  external/mobivital/dataset/mobivital/tripod/*.csv     1874 file · 13 GB      │
│  mỗi CSV = 1 buổi ghi 30 giây · 1500 dòng                                     │
│    cột 12..131   phần thực UWB  (120 kênh cự ly)                              │
│    cột 132..251  phần ảo  UWB                                                 │
│    cột áp chót   nhịp thở tham chiếu (đai ngực)                               │
└───────────┬──────────────────────────────────────────┬───────────────────────┘
            │                                          │
            │  make_npz.py                             │  prep_breath_final.py
            │  (gom theo người, CỦA ĐỒ ÁN)             │  (CỦA TÁC GIẢ, 0 dòng sửa)
            ▼                                          ▼
┌───────────────────────────────────┐   ┌──────────────────────────────────────┐
│  data/processed/by_user/          │   │  external/mobivital/data_final/      │
│  A.npz … L.npz   ·  12 file · 2,6 GB│   │  training_breath_tripod_data.npy     │
│                                   │   │  testing_breath_tripod_data.npy      │
│  mỗi .npz:                        │   │                                      │
│    uwb    (n, 1500, 120) số phức  │   │  chỉ để ĐỐI CHIẾU, không train      │
│           CHƯA chuẩn hoá          │   └──────────────┬───────────────────────┘
│    gt     (n, 1500)  ĐÃ [-1,1]    │                  │
│    files  tên CSV gốc             │                  │
└──────────┬────────────────────────┘                  │
           │                                           │
           │        check_data.py  ◄───────────────────┘
           │        so TỪNG BYTE:  ABCDEFKL 1289/1289 · GHIJ 537/537
           │        khớp thì mọi so sánh về sau mới có nghĩa
           │
           │  make_windows.py   (gọi generate_dataset của MobiVital)
           │  LỌC sóng có corr > 0,9 với nhịp thở thật  →  CÓ NHÌN NHÃN
           │  CHUẨN HOÁ về [-1,1]  ·  cắt cửa sổ 200 vào / 25 ra
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  data/processed/windows/     502 MB     ·  CHỈ ĐỂ TRAIN                       │
│                                                                              │
│    dev_cv/                            final_train/                            │
│    A_corr0.9_h200_f25.npz … L         train_corr0.9_h200_f25.npz              │
│    8 người, cắt RIÊNG từng người      8 người GỘP, đúng thứ tự MobiVital      │
│    → run_cv.py ghép 4 fold tuỳ ý      → run_final_test.py                     │
│                                                                              │
│    KHÔNG có GHIJ trong đây                                                    │
└──────────────────────────────────────────────────────────────────────────────┘


           ═══════════  hai file trên cất lên Drive  ═══════════

           by_user.tar     2,6 GB   ← dữ liệu THÔ, cho bước chấm điểm
           windows.tar.gz  106 MB   ← cửa sổ ĐÃ CẮT, cho bước train

           mỗi phiên Colab:  restore_processed_data_on_drive.py  bung về (~2 phút)
           thay vì dựng lại từ CSV (~20 phút)
```


## Lúc chạy một thực nghiệm

```
                         restore_processed_data_on_drive.py
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
   data/processed/windows/                        data/processed/by_user/
   (đã lọc, đã chuẩn hoá)                          (thô)
              │                                                 │
              │  training.load_windows                          │  scoring.score_all
              │  6 người train / fold                           │  đọc buổi ghi thô
              ▼                                                 ▼
        ┌───────────┐                                  ┌──────────────────┐
        │  TRAIN    │  20 epoch · Adam · MSE           │  CHẤM ĐIỂM        │
        │  model    │  hoặc MSE + Pearson              │  240 ứng viên     │
        └─────┬─────┘                                  │  lọc còn ~126     │
              │                                        │  model tự chọn    │
              │  final.pth                             │  KHÔNG nhìn nhãn  │
              └──────────────────────────────────────► │  argmax → 1 sóng  │
                                                       └────────┬─────────┘
                                                                │
                                                     Pearson(sóng chọn, nhịp thở thật)
                                                                │
                                                                ▼
                                              điểm mỗi buổi ghi → mỗi người → macro
                                              ghi vào runs/summary.csv
                                              nén runs/<TN>/ → <TN>.zip → Drive
```


## Ai đọc file nào

| bước | đọc từ | vì sao |
|---|---|---|
| **train** (run_cv, run_final_test) | `windows/` | cửa sổ đã cắt sẵn, đã lọc, đã chuẩn hoá — nhanh |
| **chấm 2 người validate** (trong CV) | `by_user/*.npz` thô | cửa sổ `windows/` có nhìn nhãn → dùng để chấm là rò rỉ |
| **chấm GHIJ** (test cuối) | `by_user/*.npz` thô | y như trên, và GHIJ vốn không có trong `windows/` |
| **đối chiếu dữ liệu** | `by_user/` so `data_final/*.npy` | chứng minh hai pipeline đọc ra cùng byte |


## Ba chỗ chuẩn hoá — đừng lẫn

```
1.  gt (nhịp thở tham chiếu)     make_npz.py       min-max [-1,1], MỘT LẦN cả 1500 mẫu
2.  mỗi sóng ứng viên            transform()       min-max [-1,1], MỘT LẦN cả 1500 mẫu
                                (trong make_windows và trong scoring)
3.  RevIN  (chỉ thí nghiệm TN2)  bên trong model   z-score TỪNG cửa sổ 200 mẫu, đảo được
```

`uwb` thô trong `by_user/*.npz` **chưa** qua bước nào — chuẩn hoá xảy ra ở
`transform()` khi dựng ứng viên, không phải lúc lưu file.
