# Thiết kế repo nghiên cứu MobiVital (đồ án tốt nghiệp / hướng tới paper Q3+)

## Mục tiêu

Xây lại toàn bộ code thực nghiệm hiện đang nằm trong 1 notebook Colab
(`THUC_NGHIEM_1.ipynb` ở repo cũ) thành 1 repo nghiên cứu có cấu trúc rõ ràng,
tách biệt code upstream (MobiVital) và code cải tiến của người dùng, chuẩn bị
sẵn cho: đồ án tốt nghiệp, public GitHub, và khả năng phát triển thành paper
(Q3 → Q2 → Q1).

File này là tài liệu ngữ cảnh đầy đủ — bất kỳ phiên làm việc mới nào (kể cả
không có lịch sử chat trước đó) đọc xong file này là đủ để tiếp tục, không cần
người dùng kể lại.

## Bối cảnh nguồn

- Upstream: `https://github.com/nesl/mobivital-public` — code + checkpoint của
  paper "MobiVital: Self-supervised Quality Estimation for UWB-based
  Contactless Respiration Monitoring" (arXiv 2503.11064, ACM
  10.1145/3722570.3726878).
- Bản local hiện có ở `/Users/udnb/Desktop/THUC_NGHIEM/mobivital-public/` —
  tải bằng zip, **không phải git clone**, không có commit hash, **không có
  file LICENSE**.
- Notebook thực nghiệm hiện tại: `/Users/udnb/Desktop/THUC_NGHIEM/THUC_NGHIEM_1.ipynb`
  — "MobiVital — Accuracy-first TCN + RevIN + Optuna (Validation KL Objective)",
  chạy trên Google Colab, checkpoint lưu Google Drive.
- Người dùng đang có 2 thiết bị thật để demo: radar UWB thu ở **17Hz**, đai đo
  ngực ground-truth ở **10Hz**. Dataset gốc dùng để train ghi ở **50Hz**.

## Nguyên tắc thiết kế

1. Không trộn code upstream và code của người dùng vào cùng chỗ.
2. Upstream không có LICENSE → không redistribute source của họ trong repo
   public. Local vẫn giữ để đọc/debug/reproduce, gitignore khi push.
3. Mọi khẳng định trong báo cáo/paper về "giữ nguyên setup gốc" phải đúng sự
   thật đối chiếu được với code/paper upstream — không suy diễn.
4. Ưu tiên thiết kế đúng khoa học (fair comparison, không rò rỉ test) hơn là
   tiết kiệm compute — người dùng xác nhận resource không phải ràng buộc.

## Sự thật đã xác minh về upstream (không suy đoán, đã đọc trực tiếp code + paper)

### Preprocessing

- Cột dữ liệu CSV: UWB real = cột `12:132`, UWB imag = cột `132:252`, breath
  ground truth = cột áp chót (`-2`). Mỗi session hợp lệ có 1500 sample.
- Script gốc: `dataset_preparation/prep_breath_final.py`. **Notebook của
  người dùng không gọi script này — đã tự viết lại preprocessing riêng**
  (khác biệt thật, cần ghi rõ trong báo cáo, không phải "dùng nguyên
  preprocessing gốc").

### Split dữ liệu (theo user, không theo session ngẫu nhiên — tránh rò rỉ vì 1 user có nhiều session)

- **Paper gốc** (mục 5.1.1): 12 subject. `train-dev` = 8 user
  (`A,B,C,D,E,F,K,L`), `test` = 4 user (`G,H,I,J`) — giữ nguyên để so sánh
  được với số liệu công bố.
- **Script `prep_breath_final.py` gốc**: chỉ có `training_user = "ABCDEFKL"`
  và `testing_user = "GHIJ"`. Không có validation trong script chuẩn bị dữ
  liệu.
- **Script train gốc `training/autoreg_training.py`**: nhận `test_dataloader`
  làm tham số nhưng **không hề dùng nó trong `train()`** — không early
  stopping, không chọn epoch, train đúng số epoch cố định rồi lưu thẳng
  weight cuối cùng. `get_loss()` được định nghĩa nhưng không được gọi ở đâu.
  **Kết luận: code released không có validation dưới bất kỳ hình thức nào.**
- **Paper (mục 5.1.2, "Hyperparameter Tuning")**: hyperparameter LSTM
  (`optimal_params.json`) được tune bằng thư viện **Mango** (Bayesian
  optimization). Quy trình: mỗi bộ hyperparameter train trên **6/8 subject**
  của train-dev pool, đánh giá trên **2/8 subject còn lại**, objective =
  correlation trung bình qua **3 seed**. Paper **không đề cập fold hay
  cross-validation** — chỉ là 1 split cố định (không rõ cặp user nào), không
  rotate. Script tuning này **không có trong repo public**, chỉ có kết quả
  cuối (`checkpoints/optimal_params.json`).
- **Notebook người dùng hiện tại**: tự thêm `val = "KL"` tách từ 8 user
  train-dev gốc → `train=ABCDEF (6 user) / val=KL (2 user) / test=GHIJ (4 user)`.
  Về cấu trúc giống cách paper tune (6/2), nhưng dùng ít user train hơn
  8-user gốc, và dùng Optuna (30 trial) thay Mango.

### Model baseline gốc (LSTM)

- `utils/models.py` → `LSTMMultiStep`: `hidden_size=352, num_layers=2,
  input_size=1`, head `Linear(hidden_size, future_len)`.
- `optimal_params.json`: `batch_size=64, epochs=20, lr=1e-4,
  history_length=200 (4s @ 50Hz), future_length=25 (0.5s @ 50Hz),
  hidden_size=352, num_layers=2`.
- Param count đo được: **~1.50M**. Đây là baseline đúng nghĩa để so sánh, có
  checkpoint sẵn: `checkpoints/lstm_baseline_tripod_0.9.pth` (train trên
  đúng `ABCDEFKL`/test `GHIJ`).

### Model cải tiến (TCN, trong notebook người dùng — `ForecastTCN`)

- 2 biến thể: `standard` conv và `depthwise_separable` (DS) conv, cả 2 dùng
  causal dilated Conv1d, dilation `2**block_index`.
- Tùy chọn RevIN (`use_revin`).
- Head: `Linear(channels, FUTURE_SIZE)` — **lưu ý: `FUTURE_SIZE` được đọc từ
  biến global lúc `__init__`, không truyền qua constructor** — nếu muốn
  search nhiều giá trị `future_len` trong cùng 1 study phải sửa lại để nhận
  tham số, không dùng global.
- Search space Optuna hiện tại: `conv_type∈{standard,depthwise_separable}`,
  `use_revin∈{False,True}`, `channels∈{64,96,128,192}`, `kernel_size∈{3,5,7}`,
  `num_blocks∈[3,6]`, `dropout∈[0,0.25]`, `loss_alpha∈{0.5,0.7,0.9,1.0}`,
  `learning_rate∈[1e-4,3e-3] log`, `weight_decay∈[1e-7,1e-3] log`.
- Param count đo được trên toàn search space: **~28K → ~3.1M** — phần lớn
  cấu hình DS-conv nhỏ hơn LSTM baseline nhiều lần; chỉ nhánh standard-conv
  lớn (channels=192, blocks=6) mới vượt LSTM.
- Receptive field causal TCN: `RF = 1 + 2·(kernel_size−1)·(2^num_blocks − 1)`
  — phải ràng buộc với `HISTORY_SIZE` khi search (nếu `history_len` vượt RF,
  phần lịch sử xa hơn model không thấy được).

### Loss và metric

- `CompositeLoss(alpha)`: `alpha>=1.0` → MSE thuần; `alpha<1` →
  `alpha*MSE + (1-alpha)*(1-Pearson)`.
- `pearson_per_sample`: tính bằng FP32 dù train dùng AMP FP16, để tránh sai
  số.

### "Official" evaluation ("Validation KL" / "test GHIJ")

- **"KL" và "GHIJ" là TÊN NHÓM USER** (users K,L và G,H,I,J), **không phải**
  Kullback-Leibler divergence.
- Thuật toán (`candidate_windows` → `select_best_sequence` →
  `original_mobivital_score`), tất cả trong CELL 9 của notebook:
  1. Từ radar 1 session (120 kênh phức) → mỗi kênh tách `abs` (biên độ) +
     `phase` (pha) → 240 chuỗi ứng viên.
  2. Lọc bỏ chuỗi bị đảo pha: `invert_detector(seq) < 0.8`. **Quan trọng:
     `invert_detector` (từ `utils/peak_width_inverter.py`) chỉ trả về 0 hoặc
     1 (nhị phân), không phải điểm liên tục** — nên `< 0.8` chỉ tương đương
     `== 0`, số 0.8 không mang ý nghĩa định lượng "ngưỡng 80%".
  3. Với mỗi ứng viên còn lại: cắt sliding window (`history=HISTORY_SIZE`,
     `future=FUTURE_SIZE`), cho model dự báo, tính Pearson **KHÔNG dùng
     ground truth breath** để chọn — chọn ứng viên có tổng Pearson dự báo
     cao nhất (`select_best_sequence`) → mô phỏng đúng điều kiện triển khai
     thật (không biết trước kênh nào "sạch").
  4. Điểm cuối cùng của session = `corrcoef(breath_GT_chuẩn_hóa, ứng_viên_đã_chọn)`.
  5. Điểm "Validation KL" = trung bình điểm này trên toàn bộ session của
     user K,L. "test GHIJ" = tương tự trên G,H,I,J.
- Tách biệt với `CORR_THRESHOLD=0.9` — threshold **khác hoàn toàn**, nằm
  trong `generate_sequences()` (từ `training/utils/model_utils.py`, dùng lúc
  **tạo dữ liệu train**, CÓ dùng ground truth: giữ kênh có
  `|corrcoef(kênh, breath_GT)| > 0.9`). Hai threshold 0.9 và 0.8 phục vụ 2
  mục đích khác nhau, không được nhầm lẫn.

## Vấn đề cần giải quyết trong thiết kế mới

### 1. Fair comparison bị lệch do split

Notebook hiện tại train TCN trên `ABCDEF` (6 user) trong khi baseline LSTM
gốc train trên `ABCDEFKL` (8 user) → so sánh không công bằng (TCN thiệt dữ
liệu). **Quyết định: chuyển sang thiết kế CV 2 giai đoạn (mục dưới) để TCN
cũng train trên full 8 user, baseline dùng lại checkpoint gốc không cần train
lại.**

### 2. Thiết kế CV 2 giai đoạn (thay thế fixed-split `KL`)

Lý do chọn k-fold thay vì fixed-split-nhiều-seed: với chỉ 8 user trong pool,
giữ cố định 1 cặp (2 user, 25% pool) làm val khiến toàn bộ lựa chọn
hyperparameter phụ thuộc vào đặc điểm riêng của đúng 2 người đó — phương sai
ước lượng lớn khi N nhỏ. Multi-seed (paper gốc dùng 3 seed) chỉ khử nhiễu do
khởi tạo random, không khử nhiễu do chọn nhầm cặp val.

- **Giai đoạn 1 — chọn hyperparameter kiến trúc**: group k-fold trên pool 8
  user `ABCDEFKL`. **4-fold, 2 user/fold**, xoay vòng (fold 1: val={A,B},
  fold 2: val={C,D}, fold 3: val={E,F}, fold 4: val={K,L} — thứ tự cặp cụ thể
  có thể đổi, miễn xoay đủ 4 fold phủ hết 8 user). Objective Optuna = trung
  bình `validation_KL` qua 4 fold (đổi tên biến, tránh nhầm với “KL” là tên
  user — xem mục Ghi chú đặt tên). Giữ multi-seed nhưng giảm còn **2
  seed/fold** để kiểm soát chi phí (chi phí nhân theo `fold × seed`, không
  chỉ `seed` như paper gốc).
- **Giai đoạn 2 — train model cuối**: cố định hyperparameter tốt nhất từ giai
  đoạn 1, train multi-seed trên **full 8 user `ABCDEFKL`** (đúng lượng dữ
  liệu upstream dùng). Epoch dừng / single-vs-ensemble quyết định dựa trên
  đường cong đã thấy ở 4 fold giai đoạn 1 (không cần 1 val set sống trong lúc
  train full-pool).
- **Test = `GHIJ`, giữ nguyên, chỉ đụng đúng 1 lần ở bước cuối cùng.**
- **Hệ quả**: vì giờ setup dữ liệu (`ABCDEFKL` train / `GHIJ` test) khớp
  đúng upstream, **dùng thẳng checkpoint `lstm_baseline_tripod_0.9.pth` làm
  baseline, không cần train lại LSTM.**

Không làm leave-one-user-out (8-fold): mỗi fold chỉ 1 user thì điểm val của
fold đó tự nó đã nhiễu (ít session để trung bình) — không đáng đánh đổi thêm
compute so với 4-fold.

### 3. Sample rate: 50Hz (dataset train) vs 17Hz (radar thật) vs 10Hz (đai ngực GT thật)

- Nhịp thở người là tín hiệu tần số thấp (~0.15–0.5Hz) → kể cả 10Hz sampling
  vẫn thừa Nyquist. Resample không mất thông tin nhịp thở về lý thuyết; rủi
  ro thật là **domain shift** (texture nhiễu khác giữa tín hiệu ghi thật ở
  rate gốc và tín hiệu nội suy).
- Khi cần **ghép cặp (radar, breath GT) từ 2 thiết bị thật khác rate** để làm
  dữ liệu train/fine-tune mới: **hạ radar (17Hz) xuống rate của GT (10Hz)**,
  **không nội suy nâng GT lên** — vì nội suy GT tức là bịa nhãn giám sát,
  rủi ro hơn nhiều so với hạ rate 1 tín hiệu input thật.
- `HISTORY_SIZE`/`FUTURE_SIZE` là **số mẫu**, phải tính lại theo đúng số giây
  khi đổi rate, không giữ nguyên số 200/25:
  - 50Hz: history=200 (4s), future=25 (0.5s) — cấu hình gốc.
  - 10Hz: history≈40 (4s), future≈5 (0.5s) — quy đổi thẳng tỉ lệ, **là điểm
    khởi đầu, không bắt buộc** — 2 giá trị này cũng nên được đưa vào không
    gian search (ràng buộc cùng receptive field TCN, xem mục Model ở trên),
    và có 1 chặn dưới sinh lý học: history nên ≥ 1 chu kỳ thở đầy đủ (người
    lớn ~3–5s/chu kỳ) để có đủ thông tin tuần hoàn cho model khai thác — ở
    10Hz cân nhắc history ≥ 5s (≥50 mẫu) thay vì cứng nhắc giữ đúng 4s cũ.
- Chưa có dữ liệu thật (radar+GT ghép cặp) để retrain ở giai đoạn viết spec
  này — khi có, tạo `configs/rate10_*.yaml` riêng, không trộn kết quả với
  cấu hình 50Hz.

### 4. License / attribution upstream

- `mobivital-public/` local **không có LICENSE**. Không redistribute source
  của họ trong repo public.
- Bản local hiện tại là tải zip, **không phải git clone** → không biết
  chính xác version. Trước khi rework: clone lại đàng hoàng
  (`git clone https://github.com/nesl/mobivital-public.git`), ghi commit
  hash vào README, để sau này upstream đổi vẫn biết mình dùng bản nào.

## Cấu trúc thư mục

```text
THESIS_GRADUATE/
├── external/
│   └── mobivital/                # git clone thật, pin commit hash, gitignore khi push
│
├── src/
│   ├── data/
│   │   ├── preprocess.py         # CODE CỦA NGƯỜI DÙNG (không phải upstream) — CELL 6 cũ
│   │   ├── splits.py             # khai báo tường minh: TRAIN_DEV_USERS=ABCDEFKL, TEST_USERS=GHIJ, CV_FOLDS
│   │   └── resample.py           # 50Hz ↔ 10Hz/17Hz, hạ rate input không nội suy nâng label
│   ├── models/
│   │   ├── tcn.py                # RevIN, CausalConv, CausalDSConv, TCNBlock, ForecastTCN — future_len qua constructor, không dùng global
│   │   └── baseline_lstm_wrapper.py  # load checkpoint LSTM gốc, không train lại
│   ├── losses.py                 # CompositeLoss
│   ├── metrics.py                # pearson_per_sample và các metric forecast khác
│   ├── official_eval.py          # candidate_windows, select_best_sequence, original_mobivital_score — CELL 9 cũ, viết lại có test
│   └── upstream_wrapper.py       # wrap 4 hàm thực sự dùng từ upstream: generate_dataset, self_normalize, sequence_transforms, invert_detector
│
├── configs/
│   ├── rate50_h200_f25.yaml      # rate + window nằm trong TÊN config, không hard-code trong code
│   └── (rate10_*.yaml sau khi có dữ liệu thật)
│
├── notebooks/                    # setup Colab, gọi src/, train, visualize — không chứa toàn bộ implementation
│
├── runs/                         # RUN STATE — optuna .db, checkpoint mỗi epoch, active_trial.json (gitignore, KHÔNG phải results)
│
├── results/
│   └── {config_name}/            # khóa theo config (rate+window), không chỉ theo tên model
│       └── seed_{N}/  hoặc fold_{K}_seed_{N}/
│           ├── metrics.json
│           ├── train_log.csv
│           └── prediction.png
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── tests/
│
├── docs/
│   └── superpowers/{specs,plans}/
│
├── requirements.txt
├── .gitignore                    # external/mobivital/, runs/, data/, checkpoints thô
└── README.md                     # nêu rõ: dependency MobiVital fetch riêng lúc setup, không redistribute; nêu rõ split/CV khác upstream ở điểm nào
```

## Ghi chú đặt tên (tránh nhầm lẫn đã từng xảy ra trong thảo luận)

- Không đặt tên biến/hàm là `kl_score` hay tương tự cho điểm official eval —
  dễ nhầm với Kullback-Leibler divergence. Dùng tên như
  `official_score_val_KL` hoặc đổi hẳn sang tên nhóm user rõ ràng
  (`val_group="KL"`).
- Không dùng lại đúng tên `CORR_THRESHOLD` cho cả 2 khái niệm threshold khác
  nhau (0.9 dùng ground truth lúc tạo data train, 0.8 không dùng ground
  truth lúc chọn candidate eval) — đặt 2 tên riêng biệt, ví dụ
  `TRAIN_FILTER_CORR_THRESHOLD` và `INVERT_DETECTOR_THRESHOLD`.

## Ngoài phạm vi ở bản spec này

- Chưa có dữ liệu thật (radar 17Hz + đai ngực 10Hz) ghép cặp — phần
  `resample.py` áp dụng cho dữ liệu thật sẽ hoàn thiện khi có dữ liệu.
  Model tùy chọn.
- Chưa quyết định công cụ điều khiển Colab từ agent (`colab-mcp` hay fork) —
  đây là vấn đề tooling, không ảnh hưởng cấu trúc repo, không đưa vào spec
  này.
- Chưa viết chi tiết nội dung `scripts/benchmark.py` (đo latency/memory) —
  nêu trong cấu trúc thư mục nhưng để task riêng.
