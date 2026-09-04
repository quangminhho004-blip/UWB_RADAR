# Protocol thí nghiệm

Luật chạy thí nghiệm cho đồ án. Chốt trước khi chạy, không đổi giữa chừng.

---

## 1. Hai pipeline

Đồ án dùng **hai pipeline riêng biệt**, mỗi cái một vai trò:

| | **Pipeline TRAIN / VALIDATE** | **Pipeline TEST** |
|---|---|---|
| Code | của đồ án (`scripts/make_npz.py`, `scripts/make_windows.py` → `.npz` theo từng người) | của MobiVital (`prep_breath_final.py` nguyên bản) |
| Dùng để | train model, **chọn cấu hình** qua 4-fold CV | ra **số công bố** trong luận văn |
| Dữ liệu | pool `ABCDEFKL`, chia 4 fold | train full pool `ABCDEFKL`, test `GHIJ` |
| Chạy khi nào | mọi thí nghiệm TN1–TN6 | TN0 (reproduce) + bảng so sánh cuối |
| Cho ra | `cv_score` (để chọn) và `test_GHIJ` (để nhìn) | số báo cáo |

### MỌI thí nghiệm đều test GHIJ

Hai pipeline **không phải** "sớm / muộn". Cả hai đều test `GHIJ`, chỉ khác mục
đích: một để dò đường, một để công bố.

```
MỖI THÍ NGHIỆM (TN1..TN6), với mỗi cấu hình đem thử:

   PIPELINE TRAIN / VALIDATE
        |
        +--> train 4 fold (6 người/fold) --> cv_score   --> DÙNG ĐỂ CHỌN
        |
        +--> chính 4 model đó chạy GHIJ  --> test_GHIJ  --> CHỈ ĐỂ NHÌN
        |
        +--> ra CẤU HÌNH TỐI ƯU (chỉ là mấy con số lựa chọn)
                     |
                     v
KẾT QUẢ CÔNG BỐ  (chạy MỘT lần, sau khi đã khóa hết lựa chọn):

   PIPELINE TEST
        |
        +--> LSTM train full pool --> test GHIJ --> số LSTM
        +--> TCN  train full pool --> test GHIJ --> số TCN
                                          |
                                   BẢNG SO SÁNH CUỐI
```

### Vì sao hai pipeline khác nhau vẫn hợp lệ

Thứ đi từ pipeline trên xuống pipeline dưới **chỉ là cấu hình** — mấy con số lựa
chọn (kiến trúc nào, có RevIN không, alpha bao nhiêu...). Không phải model, không
phải dữ liệu, không phải trọng số.

Bảng so sánh cuối chạy **trọn vẹn trong pipeline test**, cho cả LSTM lẫn TCN →
công bằng và tái lập được.

TN0b là **reproduce**: phải dùng pipeline test từ đầu tới cuối, không thay
preprocessing của đồ án vào — thay vào thì không còn là reproduce nữa.

Phép bảo hiểm cho việc chuyển cấu hình giữa hai pipeline là TN0a (mục 5).

### Đọc `test_GHIJ` lúc phát triển thế nào

Số `test_GHIJ` trong pipeline train/validate **thấp hơn** số công bố — bình
thường, vì model fold chỉ train trên 6 người còn model công bố train trên 8. Nó
dùng để xem **xu hướng lên/xuống**, không phải đọc giá trị tuyệt đối.

---

## 2. Chia dữ liệu

Dataset có 12 người. Chia hai nhóm, không trộn:

```
Pool phát triển  =  A B C D E F K L    (8 người)
Test             =  G H I J            (4 người)
```

Chia theo **người**, không chia ngẫu nhiên theo session. Một người có hàng trăm
session; chia ngẫu nhiên sẽ để cùng một người vừa ở train vừa ở validation, model
học "chữ ký" của người đó thay vì quy luật chung.

### Bốn fold

Chia 8 người của pool thành 4 cặp. Lần lượt lấy từng cặp ra làm validation, 6
người còn lại làm train:

| fold | train (6 người) | chấm điểm trên (2 người) |
|---|---|---|
| `val_AB` | C D E F K L | A, B |
| `val_CE` | A B D F K L | C, E |
| `val_DF` | A B C E K L | D, F |
| `val_KL` | A B C D E F | K, L |

Đánh giá **một** cấu hình tốn 4 lần train (nhân thêm số seed).

Cách ghép cặp: mỗi cặp gồm một người nhiều dữ liệu và một người ít dữ liệu, để 4
fold có lượng dữ liệu train xấp xỉ nhau (lệch ~9%, so với ~35% nếu ghép theo bảng
chữ cái A B / C D / E F / K L).

**Bộ 4 fold này cố định, dùng y nguyên cho MỌI thí nghiệm.** Nhờ vậy mọi so sánh
đều diễn ra trên đúng cùng cách chia dữ liệu.

---

## 3. Luật quyết định

**Người thắng của mỗi thí nghiệm = cấu hình có `cv_score` cao nhất. Luôn luôn.**

Mỗi thí nghiệm ghi lại **hai** cột:

| TN | cấu hình | `cv_score` (pool) | `test_GHIJ` |
|---|---|---|---|
| 1 | TCN thường | 0.712 | 0.698 |
| 1 | TCN-DS | **0.741** | 0.725 |
| 2 | DS + RevIN | **0.768** | 0.751 |

- `cv_score` — **dùng để quyết định**.
- `test_GHIJ` — **chỉ để nhìn**, biết đang đi lên hay đi xuống. Không bao giờ
  tham gia vào việc chọn.

Kể cả khi `test_GHIJ` nói ngược lại `cv_score`, vẫn chọn theo `cv_score`.

### Giữ kỷ luật bằng code, không bằng ý chí

Hàm chọn người thắng **chỉ nhận `cv_score` làm đầu vào**. `test_GHIJ` chỉ được
ghi vào log, không truyền vào hàm chọn.

### Ghi trong luận văn

> Mỗi thí nghiệm chúng tôi ghi cả điểm CV và điểm test GHIJ. Mọi quyết định chọn
> cấu hình chỉ dựa trên điểm CV; điểm test được ghi lại để theo dõi và trình bày
> minh bạch, không tham gia vào bất kỳ lựa chọn nào.

Nói thẳng ra là an toàn. Giấu mới nguy hiểm.

### Phần thưởng: một hình cho luận văn

Cuối cùng vẽ scatter `cv_score` (trục X) vs `test_GHIJ` (trục Y), mỗi điểm là một
cấu hình đã thử. Hai cái tương quan cao = bằng chứng protocol CV dự đoán đúng
test. Đây là điểm cộng, không phải điểm trừ.

---

## 4. Cách tính điểm

```
điểm mỗi người  =  trung bình official score các session của CHÍNH người đó
điểm mỗi fold   =  trung bình điểm của 2 người validation
cv_score        =  trung bình 4 điểm fold      <- số dùng để chọn cấu hình
cv_std          =  độ lệch chuẩn 4 điểm fold   <- chỉ để báo cáo
```

Vì mỗi fold có đúng 2 người validation, `cv_score` bằng đúng trung bình 8
điểm-mỗi-người. **Mỗi người một phiếu bằng nhau**, bất kể ghi 102 hay 224 session.

Báo cáo kèm bảng điểm từng người (8 số) để thấy độ phân tán giữa người.

### Seed

Cùng cấu hình, cùng fold, đổi seed thì kết quả vẫn lệch chút vì model xuất phát
từ điểm ngẫu nhiên khác. Chạy nhiều seed rồi lấy trung bình để khử may rủi khởi
tạo — so cấu hình chứ không so cú tung đồng xu.

Số seed mỗi fold: **chưa chốt** (1, 2 hay 3). Chi phí = `4 fold x số seed` lần
train cho mỗi cấu hình.

---

## 5. Chuỗi thí nghiệm

Mỗi thí nghiệm đổi **đúng một biến**, kế thừa toàn bộ người thắng của các thí
nghiệm trước. TN1–TN6 đều cho ra **cả hai** số: `cv_score` và `test_GHIJ`.

| # | tên | so cái gì | pipeline |
|---|---|---|---|
| **0a** | Kiểm eval | checkpoint LSTM của tác giả → eval trên GHIJ, so số trong paper | test |
| **0b** | Reproduce | train lại LSTM bằng code và tham số của tác giả, 3–4 seed → test GHIJ | **test, nguyên bản** |
| **1** | Kiến trúc | TCN thường vs TCN depthwise-separable | train/validate |
| **2** | RevIN | người thắng TN1, có vs không RevIN | train/validate |
| **3** | Loss | MSE thuần vs MSE + Pearson (alpha 0.5 / 0.7 / 0.9) | train/validate |
| **4** | corr-threshold | quét 0.70 / 0.80 / 0.85 / 0.90 / 0.95 | train/validate |
| **5** | Hyperparameter | Optuna: channels, kernel, blocks, dropout, lr, weight decay | train/validate |
| **6** | Xác nhận | chạy lại 2 so sánh then chốt của TN1–3 tại threshold + HP cuối | train/validate |
| **công bố** | So sánh | LSTM và TCN, cùng train full pool → test GHIJ | **test** |

TN2 tái dùng số của nhánh "không RevIN" từ TN1, không train lại.

TN6 cần thiết vì TN1–3 chạy ở threshold 0.9 (giá trị công bố của upstream); sau
khi TN4 đổi threshold thì phải kiểm lại kết luận kiến trúc còn đúng không.

---

## 6. Kết quả công bố

Trong lúc phát triển, `test_GHIJ` được đo bằng chính 4 model fold (mỗi model chỉ
train trên 6 người). Số đó dùng để theo dõi xu hướng, **không phải** số đem báo
cáo.

Số báo cáo lấy từ một model khác, train lại từ đầu trong pipeline test:

```
4 fold  ->  cv_score  ->  chọn cấu hình tốt nhất
                                 |
              train lại cấu hình đó trên ĐỦ 8 người ABCDEFKL
                     (PIPELINE TEST, prep_breath_final.py)
                                 |
                        test GHIJ  ->  số công bố
```

Model fold chỉ train trên 6 người và chỉ dùng để so cấu hình; xong việc thì bỏ,
không đem làm sản phẩm cuối.

LSTM ở bảng cuối cũng train trên đủ 8 người, cùng pipeline test, để công bằng.

**Chọn model đơn hay ensemble nhiều seed: phải chốt TRƯỚC**, không được nhìn điểm
test rồi mới chọn.

---

## 7. Checklist một thí nghiệm hợp lệ

- [ ] Dùng đúng bộ 4 fold cố định ở mục 2
- [ ] Chỉ đổi **một** biến so với thí nghiệm trước
- [ ] Mọi cấu hình chạy cùng số seed
- [ ] Người thắng chọn bằng `cv_score`, không bằng `test_GHIJ`
- [ ] Ghi lại: cấu hình đầy đủ, seed, commit git, `cv_score`, `cv_std`, bảng điểm
      8 người, `test_GHIJ`
- [ ] Kết quả lưu ra file, không chỉ nằm trong output của notebook
