# Mạch thực nghiệm DS-TCN — bốn bậc, cái gì đã có, cái gì còn thiếu

Bốn bậc, mỗi bậc chốt một thứ rồi truyền xuống bậc sau:

```
TN1 chọn kiến trúc  ->  TN2 chọn tầm nhìn  ->  TN3 chọn hàm loss  ->  TN4 test cuối
```

Điểm trong tài liệu này là **macro trên tập phát triển**, trừ chỗ ghi GHIJ.


## Bảng tên — tên đọc, tên trong mã, tệp thực nghiệm

Tên đọc chỉ dùng trong tài liệu và slide. **`config_id` trong mã và trên Drive
giữ nguyên**, không đổi.

| tên đọc | `config_id` | tham số | notebook |
|---|---|---:|---|
| **DS-TCN-nền** *(ứng viên TN1)* | `ds_tcn_c64` | 56.281 | `TN1_TCN_DSTCN_model_selection` |
| **DS-TCN-nền + RevIN** | `ds_tcn_c64_revin` | 56.281 | `TN2_DS_TCN_RevIN` |
| **Ours-64/61** | `ds_tcn_c64_k3_n4_none_do0.2_dpel` | **37.081** | `TN1_DS_TCN_RF61_no_norm_do02_c64` |
| **Ours-192/61** | `ds_tcn_c192_k3_n4_none_do0.2_dpel` | 307.801 | `TN1_DS_TCN_RF61_no_norm_do02_c192` |
| **Ours-64/121** | `ds_tcn_c64_k5_n4_none_do0.2_dpel` | 38.105 | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| **Ours-192/121** | `ds_tcn_c192_k5_n4_none_do0.2_dpel` | 310.873 | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| Ours-64/181 · Ours-64/241 | `..._c64_k7_...` · `..._c64_k9_...` | 39.129 · 40.153 | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| Ours-192/181 · Ours-192/241 | `..._c192_k7_...` · `..._c192_k9_...` | 313.945 · 317.017 | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| *bản thử 192/121-BN* | `ds_tcn_c192_k5_n4_do0.2` | 313.945 | `TN_test_ds_tcn_192` |

`Ours-⟨kênh⟩/⟨tầm nhìn⟩` — đọc là biết ngay bao nhiêu kênh, nhìn xa bao nhiêu mẫu.

Cùng nhóm còn hai kiến trúc đối chứng, **không phải DS-TCN**:
`tcn_c64` và `tcn_c64_weight`, cùng notebook `TN1_TCN_DSTCN_model_selection`.


## DS-TCN-nền bám tài liệu nào

**DS-TCN-nền** là ứng viên dựng bám khuyến nghị của hai bài, ghi rõ mục.
Nó là một trong năm ứng viên TN1, không phải điểm xuất phát của các ứng viên kia:

| thành phần | theo |
|---|---|
| tích chập nhân quả, đệm trái `(k−1)·d` | **Bai 2018, mục 3.2** |
| độ giãn tăng luỹ thừa `d = 2ⁱ` | **Bai 2018, mục 3.3** |
| hai tầng conv mỗi khối, ReLU, nối tắt | **Bai 2018, mục 3.4 và Hình 1(b)** |
| dropout theo kênh sau mỗi conv | **Bai 2018, mục 3.4** |
| **số khối 6 để tầm nhìn 253 ≥ 200** | **Bai 2018, mục A.1** |
| tách depthwise và pointwise | **Howard 2017, mục 3.1** |
| BatchNorm | **Howard 2017, mục 3.1** — lệch Bai 2018 mục 3.4 vốn dùng WeightNorm |

> Bai, S., Kolter, J. Z., Koltun, V. *An Empirical Evaluation of Generic
> Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271
>
> Howard, A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for
> Mobile Vision Applications*. arXiv:1704.04861

**Ba chỗ nền cố ý lệch Bai**, có lý do, ghi trong `docs/THAM_CHIEU.md`: dùng
BatchNorm thay WeightNorm, số kênh 64 không theo mục A.1, không có nhánh 1×1.
Nên gọi là **DS-TCN-nền** chứ đừng gọi là "bản Bai".

**Ours cũng là DS-TCN**, cùng phép tích chập depthwise theo Howard 2017 mục 3.1.
Hai cái khác nhau ở chuẩn hoá, dropout, số khối và tầm nhìn — bốn lựa chọn cấu
hình, không phải hai kiến trúc khác nhau.

**Chỗ Ours đi ngược Bai — và là đóng góp chính.** Mục A.1 viết:

> *"The most important factor for picking parameters is to make sure that the TCN
> has a sufficiently large receptive field by choosing k and d that can cover the
> amount of context needed for the task."*

TN2 đo ngược lại: tầm nhìn **61 và 121 hơn hẳn 181 và 241**, dù cả bốn đều nhỏ
hơn hoặc bằng cửa sổ vào 200 mẫu. Đây là phát hiện, không phải "bỏ BatchNorm".


## TN1 — so các ứng viên NGANG HÀNG để chọn một

TN1 không phải "dựng nền rồi sửa dần". Nó đặt **năm cấu hình ngang hàng** lên
cùng một giao thức — cùng 4 fold, cùng 3 seed, cùng dữ liệu — rồi chọn cái tốt
nhất mang xuống TN2.

```
                      ┌──────────────── THỰC NGHIỆM 1 ────────────────┐
                      │      năm ứng viên, không cái nào là gốc       │
                      └───────────────────────────────────────────────┘
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ TCN-64      │ │ TCN-64      │ │ DS-TCN-nền  │ │ Ours-64/61  │ │ Ours-192/61 │
   │ BatchNorm   │ │ WeightNorm  │ │             │ │             │ │             │
   ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
   │ 151.513 ts  │ │ 150.745 ts  │ │  56.281 ts  │ │  37.081 ts  │ │ 307.801 ts  │
   │ RF253       │ │ RF253       │ │ RF253       │ │ RF61        │ │ RF61        │
   │ k3 · 6 khối │ │ k3 · 6 khối │ │ k3 · 6 khối │ │ k3 · 4 khối │ │ k3 · 4 khối │
   │ BatchNorm   │ │ WeightNorm  │ │ BatchNorm   │ │ không norm  │ │ không norm  │
   │ dropout 0   │ │ dropout 0   │ │ dropout 0   │ │ do 0,2 p.tử │ │ do 0,2 p.tử │
   │ conv thường │ │ conv thường │ │ depthwise   │ │ depthwise   │ │ depthwise   │
   ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
   │  0,742337   │ │  0,746250   │ │  0,742109   │ │ **0,760877**│ │  0,747955   │
   │  ± 0,004369 │ │  ± 0,002997 │ │  ± 0,000673 │ │  ± 0,003095 │ │  ± 0,012189 │
   │  3 seed     │ │  3 seed     │ │  3 seed     │ │  3 seed     │ │  3 seed     │
   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └──────┬──────┘
                                                          │               │
                                                       ★ CHỌN      mang xuống TN2
                                                          │        làm đối chứng
                                                          ▼        kích cỡ khác
                                                        TN2
```

Xếp hạng: **Ours-64/61 hơn cả bốn ứng viên còn lại.**

| | so với Ours-64/61 | có đọc được không |
|---|---:|---|
| Ours-192/61 | −0,012922 | được — lớn hơn dao động seed của Ours-64/61 |
| TCN-64 WeightNorm | −0,014627 | được |
| DS-TCN-nền | −0,018768 | được — gấp sáu lần dao động seed |
| TCN-64 BatchNorm | −0,018540 | được |

Bốn khoảng cách đều lớn hơn `seed_std` của Ours-64/61 (0,003095), nên **thứ hạng
này đọc được**, không phải nhiễu.

### Chỗ phải ghi rõ khi báo cáo

Ours-64/61 khác DS-TCN-nền **bốn thứ cùng lúc** — chuẩn hoá, dropout, số khối,
tầm nhìn. Vì hai cái là **hai ứng viên ngang hàng**, không phải một cái sửa từ
cái kia, nên chuyện đó **không sai giao thức**: TN1 làm nhiệm vụ *chọn*, không
làm nhiệm vụ *tách đóng góp*.

Nhưng hệ quả vẫn phải nêu: **không nói được bỏ BatchNorm giúp bao nhiêu, hay
dropout giúp bao nhiêu.** Chỉ nói được *cấu hình này hơn cấu hình kia 0,0188*.
Muốn tách thì cần một khảo sát riêng, đổi một biến mỗi lần.

### Nhánh RevIN — thuộc TN2, không thuộc TN1

```
   DS-TCN-nền  ──►  + RevIN   56.281 ts · RF253   0,729715 ± 0,007271   3 seed
                              notebook: TN2_DS_TCN_RevIN   thực nghiệm: tn2
                              CỤT: tệ hơn nền 0,0124, gấp 18 lần dao động seed
                              RevIN chia cho độ lệch chuẩn, tức xoá biên độ,
                              mà biên độ có tương quan 0,53 với chất lượng ứng viên
```

Nó gắn vào **DS-TCN-nền** chứ không gắn vào Ours, nên kết quả này không nói được
RevIN ảnh hưởng thế nào tới cấu hình đã chọn.

## TN2 — chọn tầm nhìn

Kế thừa cấu hình Ours đã chốt, **chỉ đổi kernel**. Nền là chính điểm TN1.

```
                    ┌─────────── 64 kênh ───────────┐   ┌────────── 192 kênh ──────────┐

   RF 61   nền      Ours-64/61    0,760877 ± 0,0031     Ours-192/61   0,747955 ± 0,0122
                        3 seed                              3 seed
                          │                                    │
   RF121   ★         Ours-64/121   0,757855  1 seed       Ours-192/121  0,764428  1 seed
                          │                                    │
   RF181   CỤT       Ours-64/181   0,743657  thua nền 0,0172   Ours-192/181  0,732562
   RF241   CỤT       Ours-64/241   0,736970  thua nền 0,0239   Ours-192/241  0,736623

   notebook: TN2_ReceptiveField_DS_TCN_c64_4fold   và   _c192_4fold
```

Vòng sàng lọc 1 fold trước đó đã loại RF301 và RF361 *(0,7940 và 0,7759)* —
notebook `TN2_ReceptiveField_DS_TCN_c64` và `_c192`.

**Hai nhóm tách nhau: 61 và 121 tốt, 181 trở lên tệ rõ.** Trong nhóm tốt không
xếp hạng được — ở c64 nền nhỉnh hơn 0,0030, ở c192 mức 121 nhỉnh hơn 0,0165, hai
chiều ngược nhau.


## TN3 — chọn hàm loss

Kế thừa kiến trúc và tầm nhìn. `loss = alpha·MSE + (1 − alpha)·(1 − Pearson)`,
quét mười mức, mỗi mức đủ 4 fold, 1 seed.

```
   Ours-64/61     đỉnh alpha 0,6  ->  0,780028     10/10 mức hơn MSE thuần
                  notebook: TN3_HybridLoss_DS_TCN_c64

   Ours-64/121    đỉnh alpha 0,0  ->  0,780306      9/10 mức hơn MSE thuần
                  alpha 0,6 = 0,752386  <- mức DUY NHẤT cả đồ án THUA MSE thuần
                  notebook: TN3_HybridLoss_DS_TCN_c64_rf121

   Ours-192/121   đỉnh alpha 0,2  ->  0,776011     10/10 mức hơn MSE thuần
                  notebook: TN3_HybridLoss_DS_TCN_c192
```

**29/30 mức hơn MSE thuần** — lặp ở ba cấu hình. Nhưng **ba đỉnh nằm ba chỗ khác
nhau**, và alpha 0,6 là đỉnh của cấu hình thứ nhất lại là mức duy nhất thua ở
cấu hình thứ hai, mà hai cấu hình đó chỉ khác nhau tầm nhìn.


## TN4 — test cuối trên GHIJ

Train đủ ABCDEFKL, chấm một lần trên GHIJ, 3 seed.

```
   Ours-64/61  alpha 0,6     micro 0,798314 ± 0,016034   macro 0,803590 ± 0,015350   ĐÃ CHẠY
                             notebook: TN4_final_test_ds_tcn_c64

   Ours-192/121 alpha 0,2    micro 0,795856 ± 0,011006   macro 0,800721 ± 0,010705   ĐÃ CHẠY
                             notebook: TN4_final_test_ds_tcn_c192

   Ours-64/121 alpha 0,6     CHƯA CHẠY
                             notebook: TN4_final_test_ds_tcn_c64_rf121

   mốc: LSTM-352 (kiến trúc MobiVital, 1.502.713 ts)
        micro 0,805308 ± 0,016308   macro 0,810302 ± 0,015403
```


## CÒN THIẾU GÌ

| # | việc | thời gian | được gì |
|---:|---|---|---|
| 1 | **Ours-64/121 với MSE thuần trên GHIJ**, 3 seed | ~1 giờ | Trừ ra được **đóng góp của hàm loss trên tập test**. Hiện chỉ chứng minh được trên tập phát triển. Notebook `TN4_final_test_ds_tcn_c64_rf121` mục 4 đã có sẵn. |
| 2 | **Ours-64/121 với loss lai trên GHIJ**, 3 seed | ~1 giờ | Cấu hình đem demo có số công bố. Cùng notebook, mục 3. |
| 3 | Ours-64/121 và Ours-192/121 thêm seed 1, 2 ở TN2 | ~4 giờ | Có `seed_std` cho hai mức tầm nhìn, hiện mới 1 seed |
| 4 | Khảo sát tách bốn biến của bước nền → Ours | ~8 giờ | Trả lời được "bỏ BatchNorm giúp bao nhiêu" |
| 5 | Bản thử 192/121-BN thêm seed 1, 2 | ~2 giờ | Một phép so hai biến sạch: chuẩn hoá và loại dropout |

**Việc 1 và 2 đáng làm nhất** — cùng một notebook, tổng 2 giờ, và trả lời đúng câu hội đồng sẽ hỏi về đóng góp chính.

Việc 4 là chỗ yếu thật của đồ án nhưng tốn 8 giờ; nếu không làm thì phải ghi rõ
giới hạn đó khi báo cáo, đừng nói "bỏ BatchNorm giúp tăng điểm".


## Nhánh không thuộc mạch này

**MixLinear** — `TN1_MixLinear`, `TN_MixLinear_C0/C2/C3`. Câu hỏi khác: model cực
nhỏ 63 đến 992 tham số. Thang điểm 0,65–0,71, không so được với mạch DS-TCN.

**ModernTCN, BiLSTM, CNN-LSTM, LSTM** — cùng bậc TN1 làm mốc so, không đi tiếp
xuống TN2.
