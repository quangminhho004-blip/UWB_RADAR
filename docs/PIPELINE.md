# Hai pipeline — nhìn từ xa rồi phóng to từng khối

Đồ án có **hai** pipeline chạy trên cùng một mô hình:

- **Huấn luyện** — dạy mô hình dự báo sóng
- **Suy luận** — dùng mô hình đó để **chọn kênh**, rồi mới chấm điểm

Chỗ dễ nhầm nhất: cả hai đều làm **một việc giống hệt nhau** — *nhìn 200 mẫu
của một sóng, đoán 25 mẫu tiếp theo của **chính sóng đó***. Nhịp thở thật
không bao giờ là đáp án dự báo.


## 1. Nhìn từ xa

    HUẤN LUYỆN                             SUY LUẬN
    ==========                             ========

    H1  đọc buổi ghi thô                   S1  đọc buổi ghi thô
         |                                      |
    H2  chọn sóng đáng học                 S2  dựng 240 ứng viên
        (nhịp thở thật làm BỘ LỌC)             (không dùng nhịp thở thật)
         |                                      |
    H3  cắt cửa sổ 200 -> 25               S3  lọc sóng lộn ngược -> ~126
         |                                      |
    H4  train 20 epoch                     S4  chấm từng ứng viên, argmax
         |                                      |
    H5  lưu final.pth  ------------------> S5  so sóng THẮNG với nhịp thở thật
                                                |
                                           S6  gộp điểm: buổi ghi -> người -> macro

`final.pth` là chỗ hai pipeline gặp nhau. Ngoài nó ra, hai bên không dùng chung
gì cả — kể cả dữ liệu: huấn luyện đọc `windows/`, suy luận đọc `by_user/`.


## 2. Phóng to pipeline HUẤN LUYỆN

### H1 — Đọc buổi ghi thô

    external/mobivital/dataset/.../*.csv      1874 tệp
            |  scripts/make_npz.py
            v
    data/processed/by_user/A.npz … L.npz

    mỗi tệp .npz chứa:
        uwb    (số_buổi_ghi, 1500, 120)  số phức     tín hiệu radar
        gt     (số_buổi_ghi, 1500)       float32     nhịp thở thật, đã về [-1, 1]
        files  (số_buổi_ghi,)                        tên tệp CSV gốc

Bỏ tệp nào không đủ 1500 dòng. Còn lại **1826 buổi ghi** của 12 người.

### H2 — Chọn sóng đáng học *(chỗ duy nhất nhìn nhãn)*

Cho **mỗi buổi ghi**:

    bin 20..28  (9 bin)  ×  4 phép biến đổi          ->  36 ứng viên
                            abs, real, imag, phase
            |
            |   giữ ứng viên nào có corr với nhịp thở thật > 0,9
            v
        vài sóng sống sót   +   thêm CHÍNH sóng nhịp thở thật vào

Ý đồ: chỉ cho mô hình tập đoán những sóng **trông giống nhịp thở**. Nhờ vậy nó
giỏi đoán sóng thở và dở đoán thứ khác — đúng thứ mà tiêu chí chọn kênh cần.

**Đây là chỗ duy nhất trong cả pipeline huấn luyện có nhìn nhịp thở thật, và nó
chỉ dùng để LỌC, không dùng làm đáp án.**

Hệ quả bắt buộc nhớ: cửa sổ sinh ra ở đây **mang thông tin của nhãn**. Nên
trong một fold chỉ được dùng cửa sổ của 6 người train; hai người validate phải
chấm từ dữ liệu thô. Đụng vào là rò rỉ.

### H3 — Cắt cửa sổ

Mỗi sóng sống sót dài 1500 mẫu, cắt trượt 25:

    cửa sổ  1:  mẫu    0..199  ->  đáp án mẫu  200.. 224
    cửa sổ  2:  mẫu   25..224  ->  đáp án mẫu  225.. 249
    ...
    cửa sổ 52:  mẫu 1275..1474 ->  đáp án mẫu 1475..1499

    (1500 − 200) ÷ 25 = 52 cửa sổ mỗi sóng

**Đầu vào và đáp án cắt từ CÙNG một sóng.** Mã gốc, `generate_dataset`:

    X.append(seq[start : start+200])          # cùng seq
    y.append(seq[start+200 : start+225])      # cùng seq

Cắt sẵn một lần, lưu ra tệp, khỏi cắt lại mỗi lần train:

    data/processed/windows/dev_cv/A_corr0.9_h200_f25.npz    cắt RIÊNG từng người
    data/processed/windows/final_train/train_corr0.9_…npz   cắt GỘP cả 8 người

Ngưỡng nằm trong **tên tệp**. Lúc nạp, tham số `--corr` chỉ để **chọn tệp**,
không lọc thêm lần nào nữa.

### H4 — Train

    X (n, 200)  ->  model  ->  (n, 25)   so với  y (n, 25)

    batch 64 · 20 epoch · Adam lr 1e-4 · xáo trộn mỗi epoch
    loss:  MSE            hoặc
           alpha × MSE + (1 − alpha) × (1 − Pearson)

Lưu checkpoint mỗi epoch để Colab ngắt phiên còn chạy tiếp được.

### H5 — Ra sản phẩm

    runs/<thực nghiệm>/<cấu hình>/final.pth     trọng số
    runs/<thực nghiệm>/<cấu hình>/curve.csv     loss từng epoch


## 3. Phóng to pipeline SUY LUẬN

Chạy cho **từng buổi ghi** một, độc lập nhau.

### S1 — Đọc buổi ghi thô

    by_user/<người>.npz  ->  uwb (1500, 120) số phức của MỘT buổi ghi

Không đọc `windows/`. Đây chính là chỗ chặn rò rỉ đã nói ở H2.

### S2 — Dựng 240 ứng viên

    120 bin  ×  2 phép biến đổi  =  240 ứng viên
                 abs, phase

**Khác lúc train:** train quét 9 bin × 4 phép = 36; ở đây quét **cả 120 bin**
nhưng chỉ **2 phép**. Hai file `model_utils.py` của MobiVital trùng tên nhưng
nội dung khác nhau — rất dễ đọc nhầm chỗ này.

### S3 — Lọc sóng lộn ngược

    invert_detector(sóng) < 0,8   ->   giữ
        |
        +-- làm mượt bằng savgol
        +-- đo bề rộng đỉnh dương và đỉnh âm
        +-- đỉnh âm hẹp hơn -> sóng bị lộn ngược -> trả 1 -> LOẠI

Còn khoảng **126 ứng viên**. Bộ lọc này **không nhìn nhịp thở thật**.

### S4 — Chấm từng ứng viên rồi chọn một

Bốn tầng, làm cho mỗi ứng viên:

    tầng 1   cắt 52 cửa sổ, mỗi cửa sổ:
                 model(200 mẫu)  ->  25 số đoán
                 so với 25 mẫu THẬT CỦA CHÍNH ỨNG VIÊN ĐÓ
                 ->  1 số Pearson
             => 52 số cho mỗi ứng viên

    tầng 2   cộng 52 số lại  ->  1 số cho mỗi ứng viên
             (để so được các ứng viên với nhau)

    tầng 3   argmax trên ~126 số  ->  MỘT ứng viên thắng

Ví dụ hai ứng viên trên cùng một bộ dự báo:

    sóng thở tuần hoàn   52 số ≈ 1,0 mỗi số   ->  tổng ≈ 52
    nhiễu                52 số ≈ 0,0 mỗi số   ->  tổng ≈  0

Toàn bộ ý tưởng của MobiVital nằm ở đây: **kênh nào chứa nhịp thở thì tuần
hoàn, mà tuần hoàn thì dễ tự dự báo.** Nên "tự dự báo tốt" được dùng thay cho
"chứa nhịp thở", và không cần nhãn.

Chi phí: `~126 × 52 ≈ 6.500 lượt forward` mỗi buổi ghi.

### S5 — Giờ mới đụng tới nhịp thở thật

    tầng 4   Pearson( sóng thắng 1500 mẫu , nhịp thở thật 1500 mẫu )
             ->  điểm của buổi ghi này

Nhãn xuất hiện đúng ở đây, sau khi mọi lựa chọn đã xong. Nó **chấm**, không
**chọn**.

### S6 — Gộp điểm

    mỗi buổi ghi   ->  1 số Pearson
    mỗi người      ->  trung bình các buổi ghi của người đó
    điểm chính thức->  trung bình các người            <- MACRO

Macro chứ không phải trung bình toàn bộ buổi ghi, vì mỗi người có số buổi ghi
khác nhau; tính gộp thì người ghi nhiều bị tính nặng ký hơn một cách vô lý.

Sản phẩm:

    runs/<thực nghiệm>/scores_<cấu hình>.csv   điểm từng buổi ghi
    runs/<thực nghiệm>/<cấu hình>.txt          bảng lựa chọn kênh, CHỈ ở test cuối
    runs/summary.csv                           một dòng cho mỗi lần chạy

Tệp `.txt` đúng định dạng MobiVital, để đối chiếu **từng dòng**:

    240409_userG_tripod_02_3.csv,24,phase,0
    tên buổi ghi , bin , phép , cờ lật ngược


## 4. Hai pipeline khác nhau chỗ nào

| | HUẤN LUYỆN | SUY LUẬN |
|---|---|---|
| đọc từ | `windows/*.npz` (cắt sẵn) | `by_user/*.npz` (thô) |
| bin | 20–28 *(9 bin)* | 0–119 *(120 bin)* |
| phép biến đổi | 4 — abs, real, imag, phase | 2 — abs, phase |
| ứng viên mỗi buổi ghi | 36 | 240 |
| bộ lọc | corr với nhịp thở thật > 0,9 | `invert_detector < 0,8` |
| **có nhìn nhãn không** | **CÓ** — để lọc | **KHÔNG** |
| đầu vào / đáp án | 200 / 25, **cùng một sóng** | 200 / 25, **cùng một sóng** |
| cắt cửa sổ | sẵn, một lần | tại chỗ, mỗi lần chấm |


## 5. Nhịp thở thật xuất hiện đúng hai chỗ

    H2   lọc sóng đáng học            -> quyết định HỌC GÌ
    S5   so với sóng đã thắng         -> quyết định ĐIỂM BAO NHIÊU

Giữa hai chỗ đó, pipeline chạy mà **không biết nhịp thở thật là gì**. Đó là lý
do gọi được là "chọn mù", và cũng là lý do trần trên (oracle) cao hơn hẳn: oracle
được nhìn nhãn để chọn kênh, còn pipeline thật thì không.

    0,9120   trần trên, đo trên tập dev
    0,819    MobiVital công bố
             -> dư địa 0,093


## 6. Hai giao thức chạy, cùng dùng hai pipeline trên

    run_cv.py            CHỌN cấu hình
        4 fold trên ABCDEFKL, không bao giờ đụng G H I J
        mỗi fold: train 6 người  ->  chấm 2 người
        val_AB · val_CE · val_DF · val_KL

    run_final_test.py    SỐ CÔNG BỐ
        train đủ 8 người ABCDEFKL  ->  chấm một lần trên GHIJ (537 buổi ghi)
        chỉ chạy sau khi đã chốt cấu hình
