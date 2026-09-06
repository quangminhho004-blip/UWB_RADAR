# Vì sao chia dữ liệu như vậy

Tài liệu trả lời bốn câu: ai vào tập test, vì sao bốn fold, vì sao ghép cặp
người như thế, và vì sao tính điểm theo người.

Mọi con số dưới đây đo được từ dữ liệu, không phải ước lượng. Lệnh dựng lại
từng bảng ghi ở cuối tài liệu.


## 1. Dữ liệu có gì

Bộ `tripod` của MobiVital có **12 người**, ký hiệu A đến L, tổng **1826 buổi
ghi**. Mỗi buổi ghi dài 1500 mẫu, gồm tín hiệu radar UWB ở 120 khoảng cách và
một đường nhịp thở thật đo bằng đai ngực.

Số buổi ghi mỗi người **không đều**, chênh nhau tới hơn hai lần:

| người | buổi ghi | | người | buổi ghi |
|---|---:|---|---|---:|
| A | 224 | | G | 134 |
| B | 156 | | H | 138 |
| C | 211 | | I | 145 |
| D | 206 | | J | 120 |
| E | 126 | | K | 148 |
| F | **102** | | L | 116 |

Đây là dữ liệu tác giả thu, đồ án không can thiệp. Người A có 224 buổi, người F
chỉ có 102 — gấp 2,2 lần. Sự chênh lệch này quyết định hai lựa chọn ở mục 3 và
mục 4.


## 2. Ai vào tập test — không phải đồ án chọn

Tám người `A B C D E F K L` dùng để phát triển, bốn người `G H I J` để test.

**Cách chia này là của tác giả**, ghi cứng trong code họ:

```
external/mobivital/dataset_preparation/prep_breath_final.py

    dòng 29    training_user = "ABCDEFKL"
    dòng 61    testing_user  = "GHIJ"
```

Đồ án giữ nguyên, không xáo lại. Lý do:

**Để so được với số công bố.** Bài báo báo cáo 0.819 trên đúng bốn người này.
Xáo lại là mọi so sánh với bài báo mất nghĩa.

**Để không chọn tập test có lợi cho mình.** Nếu đồ án tự chia, luôn có nghi vấn
đã thử vài cách rồi giữ cách cho điểm đẹp. Dùng cách chia có sẵn thì không còn
chỗ cho nghi vấn đó.

Kết quả: 1289 buổi ghi để phát triển, 537 buổi ghi để test.

**`G H I J` không được nhìn khi chọn cấu hình.** Mọi quyết định — kiến trúc nào,
tham số nào — chỉ dựa trên tám người phát triển. Xem [PROTOCOL.md](PROTOCOL.md).


## 3. Vì sao bốn fold, không phải một lần chia

Nếu chỉ chia tám người thành một tập train và một tập validation, kết quả phụ
thuộc nặng vào việc **ai rơi vào tập validation**.

Số đo được ở TN1 cho thấy điều đó rất rõ. Cùng một model, cùng một hạt giống,
chỉ đổi nhóm người đem chấm:

```
val_AB   0.7987          val_DF   0.6265
val_CE   0.7919          val_KL   0.8257
```

Chênh giữa fold cao nhất và thấp nhất là **0.20**. Trong khi đổi hạt giống ngẫu
nhiên chỉ làm điểm dao động **0.004**. Tức là **chọn ai đem chấm ảnh hưởng gấp
50 lần chọn hạt giống nào**.

Một lần chia thì con số thu được là số của riêng nhóm người đó, không đại diện.
Bốn fold thì mỗi người đều được đem chấm đúng một lần, và điểm cuối là trung
bình của bốn lần.

### Vì sao bốn chứ không phải tám

Tám người thì chia được tám fold, mỗi fold chấm một người. Không làm vì hai lẽ:

**Gấp đôi thời gian máy.** Tám fold là tám lần train thay vì bốn. Toàn bộ kế
hoạch có sáu thí nghiệm, mỗi thí nghiệm vài cấu hình, mỗi cấu hình ba hạt
giống — nhân đôi là thêm khoảng năm mươi giờ GPU.

**Chấm một người thì nhiễu hơn.** Điểm của một fold là trung bình trên những
người đem chấm. Chấm hai người thì trung bình của hai số, ổn định hơn chấm một
người. Người F chỉ có 102 buổi ghi, chấm riêng người đó cho một con số rất
thiếu chắc chắn.

Bốn fold là chỗ cân bằng: mỗi người vẫn được chấm đúng một lần, mà chi phí chỉ
bằng một nửa.


## 4. Vì sao ghép cặp `AB · CE · DF · KL`

Bốn fold cố định:

```
val_AB    train C D E F K L    chấm A B
val_CE    train A B D F K L    chấm C E
val_DF    train A B C E K L    chấm D F
val_KL    train A B C D E F    chấm K L
```

Vấn đề: số cửa sổ train mỗi người rất khác nhau, vì số buổi ghi khác nhau **và**
tỉ lệ cửa sổ vượt ngưỡng lọc cũng khác nhau:

| người | buổi ghi | cửa sổ train |
|---|---:|---:|
| A | 224 | 42.640 |
| B | 156 | 39.104 |
| C | 211 | 47.996 |
| D | 206 | 53.300 |
| E | 126 | 25.792 |
| F | 102 | **9.256** |
| K | 148 | 50.804 |
| L | 116 | 23.816 |
| **tổng** | **1289** | **292.708** |

Người F chỉ đóng góp 9.256 cửa sổ, người D đóng góp 53.300 — gấp **5,8 lần**.

Nếu ghép cặp tuỳ tiện thì bốn fold có lượng dữ liệu train lệch nhau nhiều, và
fold nào train ít hơn sẽ cho điểm thấp hơn **vì thiếu dữ liệu chứ không phải vì
model kém**. Điểm trung bình bốn fold khi đó pha lẫn hai nguyên nhân.

### Nguyên tắc: ghép người nhiều dữ liệu với người ít dữ liệu

Kết quả của cách đang dùng, so với cách ghép theo bảng chữ cái:

| | val_AB | val_CE | val_DF | val_KL | chênh lệch |
|---|---:|---:|---:|---:|---:|
| **AB · CE · DF · KL** | 210.964 | 218.920 | 230.152 | 218.088 | **9,1%** |
| AB · CD · EF · KL | 210.964 | 191.412 | 257.660 | 218.088 | 34,6% |

Ghép theo bảng chữ cái thì fold `val_EF` được train nhiều hơn fold `val_CD` tới
**34,6%** — vì E và F đều ít dữ liệu nên bị giữ lại nhiều cho tập train.

### Đây là cách đều nhất có thể

Tám người chia thành bốn cặp có **105 cách**. Đã thử hết cả 105:

```
đều nhất trong 105 cách : 9,1%   ->  AB · CE · DF · KL
cách đang dùng          : 9,1%
```

Không cách nào đều hơn. Cách đang dùng là tối ưu, không phải chọn cảm tính.


## 5. Vì sao tính điểm trung bình theo NGƯỜI

Điểm của một fold là trung bình Pearson **theo người**, không phải theo buổi ghi:

```
điểm fold val_AB = ( điểm trung bình của người A + điểm trung bình của người B ) / 2
```

Không tính gộp tất cả buổi ghi rồi chia tổng số. Lý do: người A có 224 buổi,
người F có 102 buổi. Tính gộp thì người ghi nhiều buổi có tiếng nói nặng gấp đôi
người ghi ít, mà không có lý do gì để một người quan trọng hơn người khác.

Hệ thống này dùng cho người dùng cuối, mỗi người một lần. Nên chỉ số đúng là
"trung bình model làm tốt đến đâu **trên một người**", không phải "trên một buổi
ghi".

Trong `runs/summary.csv`, cột này tên `score_macro`. Cột `score_micro` là cách
tính gộp, chỉ để đối chiếu, không dùng để quyết định gì.


## 6. Mỗi cấu hình train bao nhiêu lần

### Cấu hình huấn luyện — giữ nguyên của MobiVital

Mọi lần train trong đồ án dùng đúng bộ tham số tác giả công bố ở
`external/mobivital/checkpoints/optimal_params.json`:

```json
{ "batch_size": 64, "epochs": 20, "future_length": 25,
  "hidden_size": 352, "history_length": 200, "lr": 0.0001, "num_layers": 2 }
```

Cộng thêm: Adam, hàm loss MSE, ngưỡng lọc cửa sổ `corr > 0.9`. Chép sang
`src/mobivital_reference.py` dòng 47–58 làm giá trị mặc định.

Giữ nguyên để mỗi thí nghiệm chỉ đổi **đúng một biến**. Đổi thêm số epoch hay
learning rate thì không biết chênh lệch đến từ kiến trúc hay từ cái vừa đổi.

### Ba hạt giống, không phải một

Mỗi cấu hình train lại **ba lần** với `--seed 0`, `1`, `2`. Ba hạt giống đổi
trọng số khởi tạo và thứ tự xáo trộn dữ liệu, mọi thứ khác giữ nguyên.

Vì một con số đơn không phân biệt được "hơn thật" với "hơn may". Đo được trên
tập test: cùng model cùng dữ liệu, đổi hạt giống làm điểm dao động tới **0,028**
giữa lần cao nhất và thấp nhất.

Báo cáo `mean ± std` của ba lần, kèm cả ba số riêng lẻ.

### Tổng số lần train của TN1

Bốn cấu hình, mỗi cấu hình bốn fold, mỗi fold ba hạt giống:

```
chọn cấu hình   4 cấu hình x 4 fold x 3 hạt giống = 48 lần train
kiểm chứng      3 cấu hình x 1 lần  x 3 hạt giống =  9 lần train
                                                    -----
                                                    57 lần train
```

Bước kiểm chứng train trên **đủ tám người** rồi chấm 537 buổi ghi của `G H I J`,
nên chỉ một lần mỗi hạt giống, không chia fold.

TCN-64 chưa chạy bước kiểm chứng, vì `cv_score` cho thấy nó và DS-TCN-64 hoà
nhau (chênh 0,0002) mà DS-TCN đã có số.

### Vì sao ba, không phải năm hay mười

Ba là số nhỏ nhất còn tính được độ lệch chuẩn. Nhân lên năm hạt giống là thêm
khoảng bốn mươi giờ GPU cho riêng TN1, và còn năm thí nghiệm nữa phía sau.

Đây là hạn chế phải nêu: độ lệch chuẩn tính từ ba mẫu bản thân nó cũng nhiễu.
Vì vậy tài liệu luôn in **cả ba số riêng lẻ** bên cạnh `mean ± std`, để người
đọc tự nhìn dải giá trị thay vì chỉ tin vào một con số tóm tắt.


## 7. Bốn fold cố định cho mọi thí nghiệm

Bốn fold trên **không đổi** từ TN1 đến TN6. Không xáo lại, không chọn ngẫu nhiên
theo từng lần chạy.

Vì mục đích của các thí nghiệm là so cấu hình với nhau. Nếu mỗi thí nghiệm chia
fold khác nhau, chênh lệch quan sát được có thể do cách chia chứ không do cấu
hình. Cố định fold thì mọi cấu hình bị chấm bằng đúng một cái thước.

Định nghĩa nằm ở `scripts/run_cv.py`, phần `FOLDS` đầu tệp.


## 8. Những câu có thể bị hỏi

**"Sao không chia ngẫu nhiên theo buổi ghi cho đều?"**

Chia theo buổi ghi thì buổi ghi của cùng một người rơi vào cả tập train lẫn tập
chấm. Model học được đặc điểm riêng của người đó — nhịp thở, dáng ngồi, khoảng
cách tới radar — rồi được chấm trên chính người đó. Điểm sẽ cao giả tạo và
không nói được gì về người mới.

Chia theo người thì người đem chấm là người model **chưa từng thấy**, đúng tình
huống dùng thật.

**"Bốn người test có đại diện không?"**

Không hoàn toàn, và đây là hạn chế phải nêu. Bốn người là ít, và điểm giữa họ
chênh nhau lớn — trong TN1, người G và J đạt trên 0,91 còn người H chỉ 0,63–0,69
ở mọi kiến trúc và mọi hạt giống. Con số công bố là trung bình của bốn người
này, không phải của dân số nói chung.

Đây là giới hạn của bộ dữ liệu gốc, không phải của cách chia.

**"Sao không dùng cả 12 người rồi kiểm chéo hết?"**

Làm thế thì không còn tập nào chưa từng đụng tới để báo cáo. Bốn người `G H I J`
được giữ riêng và chỉ chấm sau khi đã chốt xong cấu hình, đúng một lần cho mỗi
cấu hình cuối cùng.

**"Chênh lệch 9,1% giữa các fold có ảnh hưởng kết quả không?"**

Có, nhưng ảnh hưởng như nhau lên mọi cấu hình. Vì fold cố định nên khi so hai
cấu hình, cả hai đều chịu đúng cùng một mức chênh. Chênh lệch này làm `cv_std`
lớn (0,065–0,092), nhưng không làm lệch việc **so sánh**.


## 9. Bằng chứng — số nào do đâu ra

Không con số nào trong tài liệu này là ước lượng hay chép tay. Bảng dưới chỉ rõ
script sinh ra nó và tệp lưu lại nó.

| con số trong tài liệu | script sinh ra | bằng chứng lưu ở |
|---|---|---|
| số buổi ghi mỗi người (mục 1) | `scripts/make_npz.py` | `data/processed/by_user/*.npz` |
| dữ liệu khớp từng byte với pipeline tác giả | `scripts/check_data.py` | output ô 18 của `notebooks/TN0.ipynb` |
| chia `ABCDEFKL` / `GHIJ` (mục 2) | của tác giả | `external/mobivital/dataset_preparation/prep_breath_final.py` dòng 29 và 61 |
| số cửa sổ train mỗi người (mục 4) | `scripts/make_windows.py` | `data/processed/windows/dev_cv/*.npz` |
| điểm từng fold, `cv_score` (mục 3, 6) | `scripts/run_cv.py` | `runs/summary.csv` và `runs/tn1/scores_*.csv` |
| điểm trên `GHIJ` (mục 6) | `scripts/run_final_test.py` | `runs/tn1_ghij/*.txt` và `scores_*.csv` |
| bảng so bốn cấu hình | `scripts/compare_cv.py` | output notebook, đọc từ hai nguồn trên |
| pipeline đồ án tương đương pipeline gốc | `scripts/run_tn0.py --compare` | `runs/tn0/` và output `notebooks/TN0.ipynb` |

### Notebook nào chạy gì

| notebook | sinh ra bằng chứng gì |
|---|---|
| `DATA_PREPARE.ipynb` | dựng `by_user/*.npz` và `windows/`, đối chiếu với dữ liệu tác giả |
| `TN0.ipynb` | chứng minh pipeline đồ án cho ra đúng số của pipeline MobiVital: 537/537 buổi ghi chọn trùng kênh, chênh lệch 0 |
| `TN1_LSTM.ipynb` | LSTM-352: 4 fold x 3 hạt giống, và `GHIJ` 3 hạt giống |
| `TN1_TCN_DSTCN_model_selection.ipynb` | TCN-64 và DS-TCN-64: 4 fold x 3 hạt giống |
| `TN1_LSTM_small.ipynb` | LSTM-67: 4 fold x 3 hạt giống, và `GHIJ` 3 hạt giống |
| `TN1_final_evaluation.ipynb` | gộp kết quả các phiên, bảng so sánh, `GHIJ` cho DS-TCN-64 |

Mọi notebook đều lưu kèm output. Mở trên GitHub là thấy đúng con số đã chạy,
không cần chạy lại.

### Mỗi lần train lưu lại những gì

`runs/<tên thực nghiệm>/<tên cấu hình>/` chứa:

```
final.pth      trọng số sau khi train xong
curve.csv      mse, pearson, số phút của từng epoch
```

Và cùng cấp thư mục thực nghiệm:

```
scores_<tên cấu hình>.csv   điểm Pearson của TỪNG buổi ghi, kèm kênh đã chọn
<tên cấu hình>.txt          bảng chọn kênh, đúng định dạng của tác giả
summary.csv                 một dòng mỗi lần train, 28 cột
```

Nhờ `scores_*.csv` giữ điểm từng buổi ghi mà so được hai cấu hình ở mức chi
tiết hơn trung bình — đếm được bao nhiêu buổi hơn, bao nhiêu buổi hoà, bao
nhiêu buổi kém.

Cột `git_commit` trong `summary.csv` ghi mã commit lúc chạy, nên tra ngược được
dòng số đó sinh ra từ bản code nào.


## Dựng lại các bảng trên

Số buổi ghi và số cửa sổ mỗi người:

```python
import numpy as np, glob, os
for f in sorted(glob.glob("data/processed/by_user/*.npz")):
    print(os.path.basename(f)[0], len(np.load(f)["gt"]))
for f in sorted(glob.glob("data/processed/windows/dev_cv/*.npz")):
    print(os.path.basename(f)[0], np.load(f)["X"].shape[0])
```

Kiểm cách ghép cặp có tối ưu không — duyệt hết 105 cách:

```python
import itertools
DEV = list("ABCDEFKL")
tong = sum(CUA_SO[u] for u in DEV)
ket = []
for c1 in itertools.combinations(DEV, 2):
    con1 = [u for u in DEV if u not in c1]
    for c2 in itertools.combinations(con1, 2):
        con2 = [u for u in con1 if u not in c2]
        for c3 in itertools.combinations(con2, 2):
            c4 = tuple(u for u in con2 if u not in c3)
            cach = sorted([c1, c2, c3, c4])
            tr = [tong - CUA_SO[a] - CUA_SO[b] for a, b in cach]
            ket.append((100 * (max(tr) - min(tr)) / min(tr), cach))
print(min(ket))
```


## Đọc thêm

- [PROTOCOL.md](PROTOCOL.md) — giao thức thí nghiệm, quy tắc không nhìn `G H I J`
- `scripts/run_cv.py` — định nghĩa bốn fold, phần `FOLDS` đầu tệp
- `runs/README.md` — ý nghĩa từng cột trong bảng kết quả


## 10. Visualize data và xuất phần chia validation / inference

Mở `notebooks/VISUALIZE_DATA.ipynb` trong VS Code, chọn **Select Kernel →
Python Environments → .venv**, rồi **Run All**. Biểu đồ hiển thị trực tiếp dưới
các ô code, không cần mở HTML. Notebook mặc định dùng số tham chiếu; đổi
`SOURCE = "actual"` để đọc NPZ thực tế. Phần tín hiệu tự bỏ qua ở chế độ tham chiếu.
Đặt `SAVE_PNG = True` nếu muốn lưu thêm ảnh.

Nếu tạo môi trường trên máy khác, cài `numpy matplotlib ipykernel` vào Python
được chọn làm kernel. Script HTML cũ vẫn có thể chạy riêng, từ thư mục gốc dự án:

```bash
python scripts/visualize_data.py --source documented
python scripts/visualize_data.py --source actual
```

- `documented`: không cần thư viện ngoài; sinh báo cáo HTML bằng số tham chiếu ở
  tài liệu này, ghi rõ chưa đo trên máy.
- `actual`: cần NumPy và các NPZ do DATA_PREPARE sinh ra; đếm session/cửa sổ thực,
  kiểm tra tên người và kích thước GT/X/y. Chế độ này không chạy model hay tính điểm test.
- Kết quả ở `reports/data_visualization/<source>/index.html` và `split_plan.json`.
  Chế độ actual còn xuất `session_splits.csv`: stage, role, user, đường dẫn NPZ,
  chỉ số session (bắt đầu từ 0), tên CSV gốc. Manifest tham chiếu dữ liệu, không sao chép.
- Notebook có thêm heatmap radar, sóng ở một bin và GT; xuất hình PNG khi có dữ liệu.
  Script đọc cửa sổ actual có thể cần hàng trăm MB RAM cho một người; notebook
  chỉ đọc header khi đếm, và nạp radar khi xem tín hiệu.

### Validation

Bốn fold dùng định nghĩa chung tại `src/data_splits.py`.
Train chỉ lấy cửa sổ của sáu người trong fold; validation lấy toàn session của
hai người còn lại. Không dùng cửa sổ đã lọc corr với GT để chấm validation.
Lệnh chạy hiện có:

```bash
python scripts/run_cv.py --experiment tn_visualize_cv --model lstm --seed 0
```

Lặp lại với seed 1 và 2 khi chạy đầy đủ thí nghiệm. Chọn cấu hình bằng CV, rồi
train lại đủ ABCDEFKL và đánh giá GHIJ bằng `scripts/run_final_test.py`.

### Inference

Inference là thao tác chạy model, không phải tập dữ liệu thứ tư ngoài
train/validation/test. Khi kiểm chứng cuối, dữ liệu inference là radar GHIJ;
khi triển khai, đó là radar của người mới.

`scoring.pick_channel(uwb, model)` nhận radar một session `(1500, 120)` dạng số
phức và model đã nạp checkpoint, đặt `eval()`. Nó trả sóng được chọn, chỉ số
ứng viên và số ứng viên còn lại. Không truyền GT vào bộ chọn kênh. GT chỉ được
dùng sau đó nếu cần tính Pearson. Kết quả chưa phải số nhịp thở BPM và pipeline
hiện tại chưa xử lý streaming.

**Điểm chưa thống nhất của tài liệu cũ:** PROTOCOL.md còn mô tả xem GHIJ trong
phát triển và chưa chốt số seed. Phần mới tuân theo CHIA_DU_LIEU.md và run_cv.py:
không chấm GHIJ trong CV; dùng seed 0, 1, 2 khi đánh giá đầy đủ.
