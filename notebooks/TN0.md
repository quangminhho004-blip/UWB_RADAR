# TN0 — dựng lại kết quả MobiVital

Giải thích cho [tn0.ipynb](tn0.ipynb). Chạy bằng code MobiVital bản gốc, **không sửa dòng nào**.

| | Chuỗi | Bậc này kiểm tra |
|---|---|---|
| TN0a | TXT sẵn trong repo MobiVital → `evaluate.py` | CSV + evaluator, đối chiếu số trong báo |
| TN0b | ckpt MobiVital → `mobivital_gen.py` → TXT mới → `evaluate.py` | inference chọn bin/method |
| TN0c | `.npy` → `autoreg_training.py` → ckpt mới → `mobivital_gen.py` → TXT → `evaluate.py` | công thức train |

Mỗi bậc thêm đúng một việc do mình tự làm. Bậc nào lệch đầu tiên thì lỗi nằm ở đúng cái
vừa thêm.

---

## Mục 1 — vì sao phải dựng thư mục `work`

Code MobiVital dùng đường dẫn tương đối:

```python
data_folder      = './dataset/mobivital/tripod/'    # mobivital_gen.py:113
args.data_folder = './data_final'                   # autoreg_training.py:34
args.model_folder= 'checkpoints'
```

`./` = "thư mục đang đứng". Repo này để dữ liệu ở `data/raw/`, `data/processed/` — tên khác hẳn.

Nên dựng một thư mục rỗng có đúng những cái tên đó, bên trong toàn symlink trỏ về file thật
(gần như 0 byte). `cd` vào đấy rồi chạy. Code MobiVital tưởng nó đang nằm trong repo của nó.

```
runs/tn0/work/
├── dataset/mobivital/tripod/   1874 symlink -> data/raw/*/*.csv
├── data_final/                 2 symlink -> data/processed/mobivital_original/*.npy
├── checkpoints/                BẢN SAO .pth + optimal_params.json
└── inference/methods/          nơi code MobiVital ghi TXT ra
```

### Vì sao checkpoint phải copy chứ không symlink

TN0c sẽ ghi file `.pth` mới. Nếu là symlink thì ghi xuyên qua và đè mất bản gốc trong
`external/mobivital/checkpoints/`.

### `optimal_params.json`

Đọc **sau** `parse_args()` và ghi đè lên mọi tham số dòng lệnh (`autoreg_training.py:42-46`).
Nên cấu hình thật là các số trong file này, không phải số mặc định ghi ở `add_argument`:

```
batch_size 64 | epochs 20 | lr 1e-4 | hidden_size 352 | num_layers 2
history_length 200 | future_length 25
```

---

## Mục 2 — TN0b

`get_best_sequence` (`mobivital_gen.py:56`), chạy cho từng session:

1. **240 ứng viên** — 120 kênh phức × 2 phép (`abs`, `phase`)
2. **lọc** `invert_detector(seq) < 0.8` — savgol làm mượt rồi so bề rộng đỉnh, trả về 0/1.
   Không dùng nhịp thở thật.
3. **cắt 52 cửa sổ** — 200 mẫu vào → 25 mẫu đáp án, trượt 25. `(1500-200)//25 = 52`
4. **LSTM dự báo**, rồi `corrcoef(dự_báo, 25_mẫu_thật_của_chính_kênh_đó)`
5. `argmax(sum(corr))` → chọn kênh. **Vẫn không dùng nhịp thở thật** — đây là điểm chính
   của bài báo: chọn kênh tốt bằng cách xem model dự báo kênh nào dễ nhất
6. ghi ra `f"{file},{bin},{method},{invert_bit}"`

Rồi `evaluate.py` đọc TXT đó, mở lại CSV, lấy đúng kênh đã ghi, tính Pearson với cột 252
(nhịp thở thật), trung bình 537 session.

### Khoá chống ghi đè

`mobivital_gen.py:121` **luôn** ghi ra đúng một tên `tripod_mobivital_pre_invert_0.9.txt`.
Chạy hai lần mà quên đổi tên là mất kết quả lần trước.

Nên trước mỗi lần chạy có một ô in `True/False` — phải là `False` mới chạy tiếp. Xong thì
`shutil.move` sang `TN0b.txt` / `TN0c.txt`.

Checkpoint cũng vậy: TN0c dùng `--model_name lstm_retrained` nên file mới là
`lstm_retrained_tripod_0.9.pth`, bản gốc `lstm_pred_tripod_0.9.pth` giữ nguyên.

---

## Mục 3 — TN0c

### Dữ liệu train được cắt thế nào

`generate_sequences` (`training/utils/model_utils.py:61`) — chọn sóng nào đáng học:

```python
for i in range(len(y_breath)):                     # 1289 session ABCDEFKL
    for bins in range(20, 29):                     # 9 bin, KHÔNG phải 120
        for seq in sequence_transforms(X_uwb[i,:,bins]):   # 4 phep: abs real imag phase
            if corrcoef(seq, self_normalize(y_breath[i])) > 0.9:
                above_thresh_seqs.append(seq)
    above_thresh_seqs.append(self_normalize(y_breath[i]))        # LUÔN thêm sóng GT
```

Ba chỗ dễ bỏ sót:

- chỉ quét **9 kênh** (bin 20–28), không phải cả 120
- **4 phép biến đổi** (`abs`, `real`, `imag`, `phase`) → 36 ứng viên mỗi session.
  Lúc inference chỉ có **2 phép** (`abs`, `phase`). Có **hai file `model_utils.py`
  trùng tên nhưng nội dung khác nhau**:
  `training/utils/model_utils.py:29` dùng 4 phép, `utils/model_utils.py:30` dùng 2.
- dòng cuối: **sóng nhịp thở thật của chính session đó luôn được thêm vào tập train**,
  không cần qua ngưỡng nào
- `autoreg_training.py:99` viết `training_seqs, _ = generate_sequences(...)` — list
  `below_thresh` (tương quan âm mạnh) bị **vứt đi**

Đo thật ở ngưỡng 0.9 (`scripts/5_make_windows.py`): trong **36 ứng viên** mỗi session chỉ
giữ **3.4**, cộng sóng GT → **4.37 sóng/session**, tổng **292.708 cửa sổ train**. Tức gần
**1/4 dữ liệu train chính là sóng ground truth** chứ không phải tín hiệu radar. Đây là chỗ
TN4 sẽ đánh vào.

`generate_dataset` (`:80`) — cắt cửa sổ:

```python
start_idx = 0
end_idx   = start_idx + 200 + 25
while end_idx <= 1500:
    X.append(seq[start_idx : start_idx+200])       # 200 mẫu -> đầu vào
    y.append(seq[start_idx+200 : end_idx])         #  25 mẫu -> đáp án
    start_idx += 25
    end_idx    = start_idx + 200 + 25
```

→ **52 cửa sổ mỗi sóng**, y hệt lúc inference. Rồi `DataLoader(batch_size=64, shuffle=True)`.

`autoreg_training.py:64` — `Adam(lr=1e-4)`, `MSELoss()`, 20 epoch, lưu epoch cuối.

### Hai chỗ lọc, đừng nhầm

| | lúc train | lúc inference |
|---|---|---|
| quét bao nhiêu kênh | 9 (bin 20–28) | 120 (tất cả) |
| phép biến đổi | **4** — abs, real, imag, phase | **2** — abs, phase |
| số ứng viên / session | 36 | 240 |
| lọc bằng | `corr(sóng, GT) > 0.9` | `invert_detector < 0.8` |
| còn lại / session | 3.4 (+1 sóng GT) | ~105 |
| có dùng nhịp thở thật | **có** | **không** |
| cửa sổ | 200 → 25, 52 cái | 200 → 25, 52 cái |

**Chỗ này không nhất quán trong thiết kế của MobiVital.** Model được học trên cả `real` và
`imag`, nhưng lúc chấm điểm lại chỉ đem ra chọn giữa `abs` và `phase`. Một nửa dữ liệu train
là loại tín hiệu không bao giờ xuất hiện lúc dùng thật. Đáng làm một thí nghiệm: train chỉ
với `abs` + `phase` cho khớp với inference.

Ngưỡng `0.9` lọc **dữ liệu train**. Ngưỡng `0.8` lọc **ứng viên lúc chấm điểm**.

### Ba điểm đáng ghi vào báo cáo

- Train bằng **MSE**, chấm bằng **Pearson**. Hai thước đo khác nhau.
- **Không có validation.** `test_dataloader` được truyền vào `train()` nhưng bên trong không
  dùng lần nào; hàm `get_loss()` viết ra rồi bỏ đó. Train cứng 20 epoch, không early-stop,
  không chọn epoch tốt nhất. ⇒ GHIJ có nạp vào bộ nhớ nhưng **không đụng tới lúc train**,
  không rò rỉ.
- `seed = 1234` ghi cứng ở `autoreg_training.py:14` ⇒ chạy script gốc chỉ train lại được
  **một lần**. Muốn nhiều seed để tính `± ` thì phải gọi hàm `train()` qua `mv_run.py`,
  lúc đó mình tự khởi tạo model nên tự đặt seed được.

---

## Mục 4 — kết quả (chạy 2026-09-02, MacBook, CPU)

```
bai bao (Table 4)   0.819
TN0a                0.819481      <- khop chinh xac
TN0b                0.822175      <- cao hon cong bo
TN0c                0.798748
```

| user | session | TN0a | TN0b | TN0c |
|---|---|---|---|---|
| G | 134 | 0.9226 | 0.9173 | 0.9176 |
| H | 138 | 0.6907 | 0.6819 | 0.6347 |
| I | 145 | 0.7658 | 0.7775 | 0.7594 |
| J | 120 | 0.9173 | 0.9313 | 0.9022 |

```
        micro    macro    std
TN0a    0.8195   0.8241   0.0995
TN0b    0.8222   0.8270   0.1031
TN0c    0.7987   0.8035   0.1153
```

**micro** = trung bình 537 session. **macro** = trung bình 4 người — đây là số chính thức
theo `docs/PROTOCOL.md` mục 4, vì I có 145 session còn J chỉ có 120, tính micro thì I bị
tính nặng ký hơn một cách vô lý.

Thời gian: TN0b 45 phút · TN0c train **7h06** + gen 25 phút. (0.426 s/batch, 3079
batch/epoch, 20 epoch, CPU 10 lõi.)

### Khoảng cách 0.023 đến từ đâu

Không phải "train lại kém đều". So từng session, cùng 537 bản ghi:

```
289  hoà            hai model chọn cùng kênh, điểm y hệt
121  TN0c thắng     thắng trung bình  +0.109
127  TN0c thua      thua  trung bình  -0.203      <- thua nặng gấp đôi

5 ca thua nặng nhất:  -1.87  -1.77  -1.64  -1.56  -1.51
```

Số lần thắng thua ngang nhau (121 vs 127). Toàn bộ khoảng cách nằm ở **mức độ**.

Pearson chỉ chạy từ −1 đến +1, nên tụt 1.87 nghĩa là từ khoảng +0.93 xuống −0.94 —
**sóng bị lộn ngược**. Khoảng 30 session chọn phải kênh có tín hiệu đảo dấu.

Mà `mobivital_gen.py:139` ghi cứng:

```python
invert_bit = 0        # khong bao gio bang 1
```

`invert_detector` đáng lẽ loại sóng lộn ngược, nhưng vài chục ca vẫn lọt, và cờ sửa dấu
luôn để 0. Đây là lỗ hổng cụ thể trong pipeline MobiVital, sửa được là ăn ~0.02.

---

## Mục 5 — hai bẫy đã vấp

### `evaluate.py:67` cắt mất dòng

```python
save_df[args.methods_file] = pd.Series(scores_dict)
```

Gán Series vào DataFrame đã có → pandas **căn theo index cũ và vứt key lạ**. Chạy nhiều lần
vào cùng một `scores.csv` thì lần đầu tạo index, những lần sau bị cắt theo index đó.

Cách tránh: mỗi lần chạy dùng `--save_file` riêng (`scores_TN0b.csv`, `scores_TN0c.csv`).
Con số `evaluate.py` **in ra màn hình** thì luôn đúng, vì nó là `total_score / count` tính
trong vòng lặp, không đi qua DataFrame.

### CPU và GPU cho kết quả khác nhau

Bước chọn kênh là `argmax`. Khi hai ứng viên gần bằng điểm nhau, sai số ở chữ số thứ 10
giữa kernel LSTM trên CUDA và trên CPU đủ để lật sang kênh khác.

Hệ quả: **mọi phép so sánh dùng để chứng minh tương đương phải chạy cùng loại thiết bị.**
Chuyển lên Colab (có GPU) thì phải ép CPU cho riêng phép đối chiếu, bằng cách gán
`gen.device = "cpu"` sau khi import — gán thuộc tính lúc chạy, không sửa file.

---

## Mục 6 — đối chiếu với bài báo

arXiv 2503.11064, Table 4 *"Average score of methods"*:

| Method | w/ Inv Det. | w/o Inv Det. |
|---|---|---|
| **MobiVital** | **0.819** | 0.816 |
| SNR | 0.745 | 0.475 |
| CFAR | 0.516 | 0.218 |
| Variance | 0.514 | 0.225 |
| **Oracle** | — | **0.943** |

```
bai bao   0.819
TN0a      0.819481     khop
TN0b      0.822175     cao hon cong bo mot chut
```

⇒ File TXT và checkpoint trong repo MobiVital **đều là thật**, không phải số bịa.

Bài báo còn ghi *"Less than 5% of the sequences selected by MobiVital have a negative
correlation"*. Đo lại: TN0a 3.0% · TN0b 3.0% · TN0c 3.5%. Khớp.

### `Oracle = 0.943` — trần trên

Điểm nếu được **nhìn nhịp thở thật** để chọn kênh tốt nhất trong 240 ứng viên. MobiVital
chọn mù nên chỉ đạt 0.819.

```
0.943  tran (Oracle)
0.819  MobiVital
       -> du dia 0.124
```

Đây là con số đóng khung toàn bộ đồ án: mọi cải tiến chỉ có thể ăn trong khoảng 0.124 này.

---

## Mục 7 — vì sao TN0a và TN0b chọn khác kênh ở 251/537 session

Điểm gần bằng nhau (+0.0027) nhưng lựa chọn bin/method chỉ trùng 286/537.

Đã đo, **loại trừ được** nguyên nhân sai số tính toán:

| thí nghiệm | kết quả |
|---|---|
| lựa chọn của MobiVital đứng hạng mấy trên model mình (trong ~120 ứng viên) | hạng **2–4**, cá biệt 8 |
| chênh điểm giữa hạng nhất và lựa chọn của họ | 0.5–1.8 trên tổng ~40 = **1–4%** |
| đổi số luồng CPU (1 vs 10) có lật lựa chọn không | **0/20** |
| thêm nhiễu 1e-6 → 1e-2 vào toàn bộ trọng số | **0/20** ở mọi mức |

Nhiễu 1% vào trọng số vẫn không lật nổi một lựa chọn nào ⇒ **47% lệch không thể do sai số
số học hay khác thiết bị**.

Nguyên nhân thật không xác định được: cả TXT, checkpoint, params và code đều nằm trong một
commit gộp duy nhất `dd6e1c6 "major overhaul"`, không có lịch sử để lần. Manh mối duy nhất
là TXT dùng tên file cũ (`231003`), tức nó ra đời trước khi dataset lên Zenodo.

Không cản trở gì: hai bên đồng thuận về việc kênh nào tốt (top 2–4 trong 120), chênh điểm
cuối chỉ 0.003, và bài báo đã được dựng lại đúng.
