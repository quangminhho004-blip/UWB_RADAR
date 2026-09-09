# Sơ đồ khối hệ thống — dùng cho slide

Mỗi mục là **một slide**: một sơ đồ khối, và mấy ý cần nói khi trình bày.
Thuật ngữ chuyên ngành giữ nguyên tiếng Anh, phần diễn giải bằng tiếng Việt.


## Bảng thuật ngữ — nên để ở slide đầu

| Thuật ngữ | Nghĩa trong đồ án này |
|---|---|
| **range bin** | Kênh cự ly. Radar chia không gian trước mặt thành 120 lớp theo khoảng cách; mỗi lớp là một range bin. |
| **candidate** | Ứng viên. Một chuỗi tín hiệu rút ra từ một range bin bằng một phép biến đổi. |
| **magnitude / phase** | Hai phép biến đổi từ số phức sang số thực: lấy độ lớn, và lấy góc pha. |
| **ground truth** | Nhịp thở tham chiếu, đo bằng đai ngực. Đây là đáp án đúng dùng để chấm điểm. |
| **sliding window** | Cửa sổ trượt. Cắt chuỗi dài thành nhiều đoạn ngắn chồng lấn nhau. |
| **self-supervised** | Tự giám sát. Dữ liệu tự sinh nhãn cho chính nó, không cần người gán nhãn. |
| **forecasting model** | Mô hình dự báo chuỗi. Nhìn đoạn quá khứ, đoán đoạn tương lai. |
| **receptive field** | Tầm nhìn. Số mẫu quá khứ mà một đầu ra của mô hình thực sự nhìn thấy. |
| **loss function** | Hàm mục tiêu. Con số mà quá trình huấn luyện tìm cách giảm xuống. |
| **epoch** | Một vòng duyệt hết dữ liệu huấn luyện. |
| **batch** | Lô. Số mẫu đưa vào mô hình cùng một lượt. |
| **optimizer** | Thuật toán cập nhật trọng số sau mỗi lô. |
| **checkpoint** | Bản lưu trọng số giữa chừng, để mất phiên còn chạy tiếp được. |
| **argmax** | Phép lấy phần tử có giá trị lớn nhất. |
| **training** | Huấn luyện. Giai đoạn model học từ dữ liệu, trọng số thay đổi. |
| **inference** | Suy luận. Giai đoạn dùng model đã huấn luyện, trọng số đứng yên. |
| **train / validate** | Trong một lần chia dữ liệu: nhóm để học, và nhóm để đo xem cấu hình có tốt không. |
| **cross-validation** | Kiểm định chéo. Chia dữ liệu nhiều cách khác nhau rồi lấy trung bình. |
| **held-out test set** | Tập kiểm tra độc lập. Dữ liệu cất riêng, không tham gia bất kỳ quyết định nào. |
| **seed** | Số khởi tạo ngẫu nhiên. Đổi seed thì kết quả lệch chút ít. |
| **macro average** | Trung bình theo người: tính điểm từng người trước, rồi trung bình các người. |
| **Pearson correlation** | Hệ số tương quan Pearson. Đo hai chuỗi có cùng hình dạng không, không quan tâm biên độ. |


## Slide 1 — Tổng quan hệ thống

```
        ┌──────────────────────┐         ┌──────────────────────┐
        │      TRAINING        │         │      INFERENCE       │
        │    (huấn luyện)      │         │     (suy luận)       │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │ Candidate extraction │         │ Candidate extraction │
        │    36 candidate      │         │    240 candidate     │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │ Lọc bằng             │         │ Lọc chuỗi bị đảo pha │
        │ ground truth         │         │                      │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │ Self-supervised      │         │ Chấm khả năng        │
        │ sampling  200 → 25   │         │ tự dự báo            │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │ Train                │────────▶│ Channel selection    │
        │ forecasting model    │  model  └──────────┬───────────┘
        └──────────────────────┘         ┌──────────▼───────────┐
                                         │ Đánh giá             │
                                         │ Pearson, macro       │
                                         └──────────────────────┘
```

**Ý cần nói**

- Hai giai đoạn dùng chung **một model duy nhất**, nối nhau qua trọng số đã
  huấn luyện.
- Model **không phân loại, không hồi quy ra nhịp thở**. Nó chỉ làm forecasting.
  Việc chọn kênh là hệ quả của khả năng dự báo.
- **Ground truth chỉ tham gia hai khối**: lọc dữ liệu lúc training, và đánh giá
  ở bước cuối. Toàn bộ phần lõi chạy mù.


## Slide 2 — Giai đoạn TRAINING

```
   ┌───────────────────────────┐
   │  Tín hiệu UWB             │   1500 mẫu × 120 range bin, số phức
   │  Ground truth             │   1500 mẫu
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Candidate extraction     │   9 range bin × 4 phép biến đổi
   │                           │   magnitude · real · imaginary · phase
   └─────────────┬─────────────┘
                 │  36 candidate
   ┌─────────────▼─────────────┐
   │  Lọc bằng ground truth    │   giữ candidate có Pearson với
   │  ngưỡng 0,9               │   ground truth vượt ngưỡng
   └─────────────┬─────────────┘
                 │  các candidate giống nhịp thở
   ┌─────────────▼─────────────┐
   │  Self-supervised sampling │   sliding window 200 mẫu → 25 mẫu
   │                           │   52 mẫu huấn luyện cho mỗi candidate
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Train                    │   20 epoch · batch 64
   │  forecasting model        │   loss: sai số bình phương,
   │                           │   hoặc kết hợp với Pearson
   └─────────────┬─────────────┘
                 │
                 ▼
              model
```

**Ý cần nói**

- Khối lọc là chỗ **duy nhất** dùng ground truth ở giai đoạn này, và nó chỉ
  dùng để **chọn dữ liệu học**, không dùng làm nhãn dự báo.
- Mục đích: model chỉ tập dự báo những chuỗi **giống nhịp thở**. Nhờ vậy nó
  giỏi dự báo chuỗi tuần hoàn và kém dự báo nhiễu — đúng thứ inference cần.
- Hệ quả về giao thức: dữ liệu sinh ra ở đây **mang thông tin của ground
  truth**, nên chỉ dùng cho nhóm người huấn luyện, không bao giờ cho nhóm kiểm
  định.


## Slide 3 — Khối then chốt: self-supervised sampling

```
   Một candidate — chuỗi 1500 mẫu

   ├──────────── 200 mẫu ────────────┤├─ 25 ─┤
   ┌─────────────────────────────────┬───────┬─────────────────────┐
   │             INPUT               │ NHÃN  │                     │
   └─────────────────────────────────┴───────┴─────────────────────┘
    0                              199 200  224                 1499

   sliding window — trượt 25 mẫu, lặp lại

   ┌───┬─────────────────────────────────┬───────┬─────────────────┐
   │   │             INPUT               │ NHÃN  │                 │
   └───┴─────────────────────────────────┴───────┴─────────────────┘
        25                            224 225  249

                          ...  52 lần
```

**Ý cần nói**

- **Input và nhãn cắt từ cùng một chuỗi.** Không có ground truth ở đây.
- Đây chính là nghĩa của **self-supervised**: dữ liệu tự sinh nhãn cho chính nó.
- Số mẫu huấn luyện: `(1500 − 200) ÷ 25 = 52` cho mỗi candidate.
- Cùng cách sinh mẫu này được dùng lại nguyên vẹn ở inference — đó là điều
  khiến hai giai đoạn nhất quán với nhau.


## Slide 4 — Giai đoạn INFERENCE

```
   ┌───────────────────────────┐
   │  Tín hiệu UWB             │   một phiên đo
   │  1500 × 120 range bin     │   KHÔNG có ground truth
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Candidate extraction     │   120 range bin × 2 phép biến đổi
   │                           │   magnitude · phase
   └─────────────┬─────────────┘
                 │  240 candidate
   ┌─────────────▼─────────────┐
   │  Lọc chuỗi bị đảo pha     │   loại chuỗi có dạng sóng lật ngược
   └─────────────┬─────────────┘
                 │  các candidate còn lại
   ┌─────────────▼─────────────┐
   │  Chấm khả năng            │   model dự báo trên 52 mẫu
   │  tự dự báo                │   của từng candidate
   └─────────────┬─────────────┘
                 │  một điểm số cho mỗi candidate
   ┌─────────────▼─────────────┐
   │  Channel selection        │   argmax — lấy candidate điểm cao nhất
   └─────────────┬─────────────┘
                 │  một chuỗi duy nhất
   ┌─────────────▼─────────────┐
   │  Đánh giá                 │   Pearson với ground truth
   └───────────────────────────┘
```

**Ý cần nói**

- Không gian tìm kiếm rộng hơn hẳn lúc training: **240 candidate** thay vì 36,
  quét toàn bộ 120 range bin vì không biết trước người nằm cách radar bao xa.
- Bốn khối giữa chạy **hoàn toàn không có ground truth**. Hệ thống chọn mù.
- Ground truth chỉ vào ở khối cuối, và chỉ để **chấm điểm** chuỗi đã được chọn.


## Slide 5 — Khối then chốt: channel selection

```
   Các candidate còn lại
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │  Với MỖI candidate:                             │
   │                                                 │
   │    52 mẫu ──▶ model ──▶ dự báo                  │
   │                            │                    │
   │                            ▼                    │
   │            Pearson( dự báo , nhãn tự sinh )     │
   │                            │                    │
   │                            ▼                    │
   │            cộng 52 giá trị ──▶ một điểm số      │
   └─────────────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     argmax      │
                    └────────┬────────┘
                             ▼
                một candidate được chọn
```

Minh hoạ trực giác:

```
   candidate tuần hoàn   →  dự báo được        →  ██████████  điểm cao
   candidate nhiễu       →  không dự báo được  →  █           điểm thấp
```

**Ý cần nói**

- Giả thiết nền: **range bin chứa nhịp thở thì tuần hoàn, mà tuần hoàn thì dự
  báo được.** Nên "dự báo được" đóng vai trò thay cho "chứa nhịp thở".
- Nhờ giả thiết đó, hệ thống chọn được kênh **mà không cần ground truth**.
- Hệ quả phản trực giác, và là một đóng góp của đồ án: **model dự báo càng giỏi
  thì chọn kênh càng kém**, vì nó dự báo được cả nhiễu trơn. Đo được: khi tăng
  receptive field, sai số dự báo giảm 13% nhưng điểm chọn kênh giảm 0,070.


## Slide 6 — Vai trò của ground truth

```
   TRAINING                                INFERENCE

   candidate extraction                    candidate extraction
        │                                       │
   ┌────▼──────────────┐                   lọc chuỗi đảo pha
   │ LỌC               │◀── ground truth        │
   └────┬──────────────┘                   chấm tự dự báo
        │                                       │
   self-supervised sampling                channel selection
        │                                       │
   train model                            ┌─────▼─────────────┐
                                          │ ĐÁNH GIÁ          │◀── ground truth
                                          └───────────────────┘
```

**Ý cần nói**

- Ground truth xuất hiện đúng **hai lần**: chọn dữ liệu học, và chấm điểm cuối.
- Toàn bộ phần lõi — forecasting và channel selection — chạy **không có ground
  truth**.
- Đó là lý do có khoảng cách giữa hệ thống thật và giới hạn lý thuyết:

```
   0,9120   oracle — điểm nếu luôn chọn đúng candidate tốt nhất
   0,8036   hệ thống của đồ án
   0,819    công bố của công trình gốc
```


## Slide 7 — Giao thức đánh giá

```
                        12 người tham gia
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   ┌────────────────────┐               ┌────────────────────┐
   │  DEVELOPMENT SET   │               │  HELD-OUT TEST SET │
   │      8 người       │               │       4 người      │
   └─────────┬──────────┘               └─────────┬──────────┘
             │                                    │
   ┌─────────▼──────────┐                         │
   │  6 người train     │                         │
   │  2 người validate  │                         │
   └─────────┬──────────┘                         │
             │                                    │
   cross-validation                               │
   lặp 4 lần, mỗi lần đổi                         │
   hai người validate                             │
             │                                    │
             ▼                                    ▼
      chọn cấu hình  ──────────────────▶   chạy đúng MỘT lần
                                                  │
                                                  ▼
                                          số liệu công bố
```

**Ý cần nói**

- Chia 8 người thành 6 train và 2 validate, lặp 4 lần để mỗi người đều có lượt
  làm validate. Lấy trung bình 4 lần đó để chọn cấu hình.
- **Held-out test set 4 người không tham gia bất kỳ quyết định nào** — chỉ chạy
  đúng một lần ở cuối để lấy số công bố.
- Mỗi cấu hình chạy 3 seed, báo cáo kèm độ lệch. Chênh lệch nhỏ hơn độ lệch đó
  thì không kết luận thứ hạng.
