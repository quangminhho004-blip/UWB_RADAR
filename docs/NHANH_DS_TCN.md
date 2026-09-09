# Nhánh DS-TCN — thực nghiệm nào kế thừa cấu hình nào

Bốn bậc, mỗi bậc chốt một thứ rồi truyền xuống bậc sau:

```
TN1  chọn kiến trúc nền   ->   TN2  chọn tầm nhìn   ->   TN3  chọn hàm loss   ->   TN4  test cuối
```

Nhánh nào không được chọn thì **cụt** ở đó. Dưới đây ghi rõ cụt vì lý do gì.

Mọi điểm trong tài liệu này là **macro trên tập phát triển**, trừ chỗ ghi GHIJ.


## Cây đầy đủ

```
TN1 — chọn kiến trúc nền
│    notebook: TN1_TCN_DSTCN_model_selection
│
├─ TCN-64 BatchNorm      151.513 ts   tầm nhìn 253   0,742337 ± 0,004369   3 seed
│     └─ CỤT: gấp 2,7 lần tham số của DS-TCN mà điểm ngang
│
├─ TCN-64 WeightNorm     150.745 ts   tầm nhìn 253   0,746250 ± 0,002997   3 seed
│     └─ CỤT: hơn DS-TCN 0,0041, nhỏ hơn dao động seed, không đủ để trả 2,7 lần tham số
│
└─ DS-TCN-64             56.281 ts    tầm nhìn 253   0,742109 ± 0,000673   3 seed   ★ CHỌN
   │  GHIJ 0,795782 ± 0,015413 (TN1_final_evaluation)
   │
   ├─ + RevIN            56.281 ts    tầm nhìn 253   0,729715 ± 0,007271   3 seed
   │     notebook: TN2_DS_TCN_RevIN
   │     └─ CỤT: tệ hơn 0,0124, gấp 18 lần dao động seed.
   │        RevIN chia cho độ lệch chuẩn, tức xoá biên độ — mà biên độ là
   │        manh mối thật, tương quan 0,53 với chất lượng ứng viên.
   │
   └─ bỏ chuẩn hoá · dropout 0,2 theo phần tử · 4 khối
      │  tầm nhìn 253 -> 61, tham số 56.281 -> 37.081
      │
      ├─ c64  k3    37.081 ts   tầm nhìn 61   0,760877 ± 0,003095   3 seed   ★ CHỌN
      │     notebook: TN1_DS_TCN_RF61_no_norm_do02_c64
      │
      └─ c192 k3   307.801 ts   tầm nhìn 61   0,747955 ± 0,012189   3 seed
            notebook: TN1_DS_TCN_RF61_no_norm_do02_c192
            └─ CỤT ở nhánh này: 8 lần tham số, điểm thấp hơn 0,0129,
               dao động seed lớn gấp bốn lần. Nhưng số kênh 192 vẫn được
               mang xuống TN2 để kiểm xem quy luật tầm nhìn có lặp lại không.
```

```
TN2 — chọn tầm nhìn        kế thừa: bỏ chuẩn hoá · dropout 0,2 · 4 khối · loss MSE
│    notebook: TN2_ReceptiveField_DS_TCN_c64_4fold  và  _c192_4fold
│
├─ VÒNG SÀNG LỌC 1 fold, chỉ để loại nhanh
│     notebook: TN2_ReceptiveField_DS_TCN_c64  và  _c192
│     kernel 11 tầm nhìn 301   val_KL 0,7940
│     kernel 13 tầm nhìn 361   val_KL 0,7759
│     └─ CỤT: tệ hơn hẳn ba mức còn lại, không đưa vào vòng 4 fold
│
├─ 64 kênh
│  ├─ k3  tầm nhìn  61   37.081 ts   0,760877 ± 0,003095  3 seed   ← NỀN, kế thừa từ TN1
│  ├─ k5  tầm nhìn 121   38.105 ts   0,757855             1 seed   ★ mang xuống TN3
│  ├─ k7  tầm nhìn 181   39.129 ts   0,743657             1 seed   └─ CỤT: thua nền 0,0172
│  └─ k9  tầm nhìn 241   40.153 ts   0,736970             1 seed   └─ CỤT: thua nền 0,0239
│
└─ 192 kênh
   ├─ k3  tầm nhìn  61  307.801 ts   0,747955 ± 0,012189  3 seed   ← NỀN, kế thừa từ TN1
   ├─ k5  tầm nhìn 121  310.873 ts   0,764428             1 seed   ★ mang xuống TN3
   ├─ k7  tầm nhìn 181  313.945 ts   0,732562             1 seed   └─ CỤT: thua k5 0,0319
   └─ k9  tầm nhìn 241  317.017 ts   0,736623             1 seed   └─ CỤT: thua k5 0,0278

   Nền là cấu hình TN1 đã chốt. TN2 giữ nguyên mọi thứ, CHỈ đổi kernel — nên
   không chạy lại nền, đó là điểm của giao thức bậc thang.
```

```
TN3 — chọn hàm loss        kế thừa: kiến trúc và tầm nhìn đã chốt ở TN1 và TN2
│    loss = alpha·MSE + (1 − alpha)·(1 − Pearson), quét mười mức
│
├─ c64  k3 RF61    notebook: TN3_HybridLoss_DS_TCN_c64
│     đỉnh alpha 0,6  ->  0,780028      10/10 mức hơn MSE thuần
│
├─ c64  k5 RF121   notebook: TN3_HybridLoss_DS_TCN_c64_rf121
│     đỉnh alpha 0,0  ->  0,780306      9/10 mức hơn MSE thuần
│     alpha 0,6 = 0,752386, mức DUY NHẤT trong cả đồ án THUA MSE thuần
│
└─ c192 k5 RF121   notebook: TN3_HybridLoss_DS_TCN_c192
      đỉnh alpha 0,2  ->  0,776011      10/10 mức hơn MSE thuần
```

```
TN4 — test cuối trên GHIJ   kế thừa: cả ba lựa chọn trên
│    train đủ ABCDEFKL, chấm một lần trên GHIJ, 3 seed
│
├─ c64  k3 RF61  alpha 0,6    notebook: TN4_final_test_ds_tcn_c64
│     micro 0,798314 ± 0,016034   macro 0,803590 ± 0,015350   ĐÃ CHẠY
│
├─ c192 k5 RF121 alpha 0,2    notebook: TN4_final_test_ds_tcn_c192
│     micro 0,795856 ± 0,011006   macro 0,800721 ± 0,010705   ĐÃ CHẠY
│
└─ c64  k5 RF121 alpha 0,6    notebook: TN4_final_test_ds_tcn_c64_rf121
      CHƯA CHẠY
```


## Hai chỗ nhánh bị đứt mạch

### 1. TN4 thiếu nền MSE thuần

Ba dòng TN4 đều dùng hàm loss lai. Không có dòng nào cùng kiến trúc mà dùng MSE
thuần, nên **không trừ ra được đóng góp của hàm loss trên tập test**. Hiện chỉ
chứng minh được nó giúp trên tập phát triển.

Vá được: notebook `TN4_final_test_ds_tcn_c64_rf121` đã có sẵn mục 4 làm việc đó,
ba seed, khoảng 1 giờ.

### 2. Alpha mang xuống TN4 chọn theo đỉnh của từng cấu hình

`c64 k3` lấy alpha 0,6 và `c192 k5` lấy alpha 0,2 — mỗi cái theo đỉnh TN3 của
chính nó. Nhưng TN3 vừa cho thấy thứ hạng alpha **không chuyển được**: alpha 0,6
là đỉnh ở `c64 k3` lại là mức duy nhất thua MSE thuần ở `c64 k5`.

Nên hai giá trị alpha đó là **mức tốt nhất đo được trên tập phát triển**, không
phải mức tối ưu đã chứng minh. Phải ghi câu đó khi báo cáo.


## Nhánh không thuộc mạch này

**Khảo sát MixLinear** — `TN1_MixLinear`, `TN_MixLinear_C0/C2/C3`. Câu hỏi khác
hẳn: model cực nhỏ 63 đến 992 tham số. Thang điểm 0,65–0,71, không so được với
mạch DS-TCN. Xuống phụ lục.

**ModernTCN, BiLSTM, CNN-LSTM, LSTM** — cùng bậc TN1 nhưng không phải họ TCN
nhân quả. Chúng ở lại TN1 làm mốc so, không đi tiếp xuống TN2.

**`TN_test_ds_tcn_192`** — bản DS-TCN-192 tầm nhìn 121 dùng BatchNorm và dropout
theo kênh, chạy ở `--experiment tn_test`. Là bản thử trước khi chốt cấu hình,
không thuộc bậc nào.
