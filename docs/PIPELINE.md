# Sơ đồ khối hệ thống — dùng cho slide

Mỗi mục là **một slide**: một sơ đồ, và mấy ý cần nói khi đứng trình bày.
Vẽ lại bằng hình khối trong PowerPoint là ra slide hoàn chỉnh.

Thuật ngữ dùng thống nhất: **phiên đo** = một lần ghi 30 giây, **ứng viên** =
một chuỗi tín hiệu rút ra từ một kênh cự ly bằng một phép biến đổi.


## Slide 1 — Tổng quan hệ thống

```
        ┌──────────────────────┐         ┌──────────────────────┐
        │   GIAI ĐOẠN HUẤN     │         │   GIAI ĐOẠN SUY      │
        │       LUYỆN          │         │       LUẬN           │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │  Trích xuất ứng viên │         │  Trích xuất ứng viên │
        │      36 ứng viên     │         │     240 ứng viên     │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │  Sàng lọc theo       │         │  Sàng lọc đảo pha    │
        │  nhịp thở tham chiếu │         │                      │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │  Sinh mẫu tự giám    │         │  Chấm khả năng       │
        │  sát   200 → 25      │         │  tự dự báo           │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
        ┌──────────▼───────────┐         ┌──────────▼───────────┐
        │  Huấn luyện mô hình  │────────▶│  Chọn kênh           │
        │  dự báo chuỗi        │  mô hình└──────────┬───────────┘
        └──────────────────────┘         ┌──────────▼───────────┐
                                         │  Đánh giá            │
                                         │  Pearson theo người  │
                                         └──────────────────────┘
```

**Ý cần nói**

- Hai giai đoạn dùng chung **một mô hình duy nhất**, nối với nhau qua trọng số
  đã huấn luyện.
- Mô hình **không phân loại, không hồi quy ra nhịp thở**. Nó chỉ dự báo chuỗi.
  Việc chọn kênh là hệ quả của khả năng dự báo.
- Nhịp thở tham chiếu chỉ tham gia **hai khối**: sàng lọc lúc huấn luyện, và
  đánh giá ở bước cuối.


## Slide 2 — Giai đoạn huấn luyện

```
   ┌───────────────────────────┐
   │  Tín hiệu UWB             │   1500 mẫu × 120 kênh cự ly, số phức
   │  Nhịp thở tham chiếu      │   1500 mẫu
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Trích xuất ứng viên      │   9 kênh cự ly × 4 phép biến đổi
   │                           │   biên độ · phần thực · phần ảo · pha
   └─────────────┬─────────────┘
                 │  36 ứng viên
   ┌─────────────▼─────────────┐
   │  Sàng lọc                 │   giữ ứng viên có tương quan với
   │  tương quan > 0,9         │   nhịp thở tham chiếu vượt ngưỡng
   └─────────────┬─────────────┘
                 │  các ứng viên giống nhịp thở
   ┌─────────────▼─────────────┐
   │  Sinh mẫu tự giám sát     │   52 mẫu cho mỗi ứng viên
   │  200 mẫu → 25 mẫu         │
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Huấn luyện               │   20 vòng · lô 64 mẫu
   │  mô hình dự báo chuỗi     │   hàm mục tiêu: sai số bình phương,
   │                           │   hoặc sai số bình phương kết hợp Pearson
   └─────────────┬─────────────┘
                 │
                 ▼
        mô hình đã huấn luyện
```

**Ý cần nói**

- Khối sàng lọc là chỗ **duy nhất** dùng nhịp thở tham chiếu, và nó chỉ dùng để
  **chọn dữ liệu học**, không dùng làm đáp án.
- Mục đích: mô hình chỉ tập dự báo những chuỗi **giống nhịp thở**. Nhờ vậy nó
  giỏi dự báo chuỗi tuần hoàn và kém dự báo nhiễu — đúng thứ giai đoạn sau cần.
- Hệ quả về giao thức: dữ liệu sinh ra ở đây **mang thông tin nhãn**, nên chỉ
  dùng cho nhóm người huấn luyện, không bao giờ cho nhóm kiểm định.


## Slide 3 — Khối then chốt: sinh mẫu tự giám sát

```
   Một ứng viên — chuỗi 1500 mẫu

   ├──────────── 200 mẫu ────────────┤├─ 25 ─┤
   ┌─────────────────────────────────┬───────┬─────────────────────┐
   │            ĐẦU VÀO              │ NHÃN  │                     │
   └─────────────────────────────────┴───────┴─────────────────────┘
    0                              199 200  224                 1499

   trượt 25 mẫu, lặp lại

   ┌───┬─────────────────────────────────┬───────┬─────────────────┐
   │   │            ĐẦU VÀO              │ NHÃN  │                 │
   └───┴─────────────────────────────────┴───────┴─────────────────┘
        25                            224 225  249

                          ...  52 lần
```

**Ý cần nói**

- **Đầu vào và nhãn cắt từ cùng một chuỗi.** Không có nhịp thở tham chiếu ở đây.
- Đây là **học tự giám sát**: dữ liệu tự sinh ra nhãn cho chính nó.
- Số mẫu: `(1500 − 200) ÷ 25 = 52` cho mỗi ứng viên.
- Cùng một cách sinh mẫu này được dùng lại nguyên vẹn ở giai đoạn suy luận —
  đó là điều khiến hai giai đoạn nhất quán với nhau.


## Slide 4 — Giai đoạn suy luận

```
   ┌───────────────────────────┐
   │  Tín hiệu UWB             │   một phiên đo
   │  1500 × 120 kênh, số phức │   KHÔNG có nhịp thở tham chiếu
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │  Trích xuất ứng viên      │   120 kênh cự ly × 2 phép biến đổi
   │                           │   biên độ · pha
   └─────────────┬─────────────┘
                 │  240 ứng viên
   ┌─────────────▼─────────────┐
   │  Sàng lọc đảo pha         │   loại chuỗi có dạng sóng bị lật ngược
   └─────────────┬─────────────┘
                 │  các ứng viên còn lại
   ┌─────────────▼─────────────┐
   │  Chấm khả năng            │   mô hình dự báo trên 52 mẫu
   │  tự dự báo                │   của từng ứng viên
   └─────────────┬─────────────┘
                 │  một điểm số cho mỗi ứng viên
   ┌─────────────▼─────────────┐
   │  Chọn kênh                │   lấy ứng viên điểm cao nhất
   └─────────────┬─────────────┘
                 │  một chuỗi duy nhất
   ┌─────────────▼─────────────┐
   │  Đánh giá                 │   Pearson với nhịp thở tham chiếu
   └───────────────────────────┘
```

**Ý cần nói**

- Không gian tìm kiếm rộng hơn hẳn lúc huấn luyện: **240 ứng viên** thay vì 36,
  quét toàn bộ 120 kênh cự ly vì không biết trước người nằm ở khoảng cách nào.
- Bốn khối giữa chạy **hoàn toàn không có nhãn**. Hệ thống chọn mù.
- Nhịp thở tham chiếu chỉ vào ở khối cuối, và chỉ để **chấm điểm** chuỗi đã
  được chọn.


## Slide 5 — Khối then chốt: bộ chọn kênh

```
   Các ứng viên còn lại
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │  Với MỖI ứng viên:                              │
   │                                                 │
   │    52 mẫu ──▶ mô hình ──▶ dự báo                │
   │                              │                  │
   │                              ▼                  │
   │              Pearson( dự báo , nhãn tự sinh )   │
   │                              │                  │
   │                              ▼                  │
   │              cộng 52 giá trị ──▶ một điểm số    │
   └─────────────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Lấy điểm cao   │
                    │  nhất           │
                    └────────┬────────┘
                             ▼
                    một ứng viên được chọn
```

Minh hoạ trực giác:

```
   ứng viên tuần hoàn   →  dự báo được        →  ██████████  điểm cao
   ứng viên nhiễu       →  không dự báo được  →  █           điểm thấp
```

**Ý cần nói**

- Giả thiết nền: **kênh chứa nhịp thở thì tuần hoàn, mà tuần hoàn thì dự báo
  được.** Nên "dự báo được" đóng vai trò thay cho "chứa nhịp thở".
- Nhờ giả thiết đó, hệ thống chọn được kênh **mà không cần nhãn**.
- Hệ quả phản trực giác — và là một đóng góp của đồ án: **mô hình dự báo quá
  giỏi sẽ chọn kênh kém đi**, vì nó dự báo được cả nhiễu trơn. Đo được: khi
  tăng tầm nhìn, sai số dự báo giảm 13% nhưng điểm chọn kênh giảm 0,070.


## Slide 6 — Vai trò của nhãn trong toàn hệ thống

```
   HUẤN LUYỆN                              SUY LUẬN

   trích xuất ứng viên                     trích xuất ứng viên
        │                                       │
   ┌────▼──────────────┐                   sàng lọc đảo pha
   │ SÀNG LỌC          │◀── nhãn                │
   └────┬──────────────┘                   chấm tự dự báo
        │                                       │
   sinh mẫu tự giám sát                    chọn kênh
        │                                       │
   huấn luyện                             ┌─────▼─────────────┐
                                          │ ĐÁNH GIÁ          │◀── nhãn
                                          └───────────────────┘
```

**Ý cần nói**

- Nhãn xuất hiện đúng **hai lần**: chọn dữ liệu học, và chấm điểm cuối cùng.
- Toàn bộ phần lõi — dự báo và chọn kênh — chạy **không có nhãn**.
- Đó là lý do có khoảng cách giữa hệ thống thật và giới hạn lý thuyết:

```
   0,9120   giới hạn trên — nếu luôn chọn đúng kênh tốt nhất
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
   │   NHÓM PHÁT TRIỂN  │               │  NHÓM KIỂM TRA     │
   │      8 người       │               │  ĐỘC LẬP  4 người  │
   └─────────┬──────────┘               └─────────┬──────────┘
             │                                    │
   ┌─────────▼──────────┐                         │
   │  6 người huấn luyện│                         │
   │  2 người kiểm định │                         │
   └─────────┬──────────┘                         │
             │                                    │
   lặp 4 lần, mỗi lần đổi                         │
   hai người kiểm định                            │
             │                                    │
             ▼                                    ▼
      chọn cấu hình  ──────────────────▶   chạy đúng MỘT lần
                                                  │
                                                  ▼
                                          số liệu công bố
```

**Ý cần nói**

- Chia 8 người thành 6 huấn luyện và 2 kiểm định, lặp 4 lần để mỗi người đều
  có lượt làm kiểm định. Lấy trung bình 4 lần đó để chọn cấu hình.
- Nhóm 4 người còn lại **không tham gia bất kỳ quyết định nào** — chỉ chạy đúng
  một lần ở cuối để lấy số công bố.
- Mỗi cấu hình chạy 3 lần với khởi tạo khác nhau, báo cáo kèm độ lệch. Chênh
  lệch nhỏ hơn độ lệch đó thì không kết luận thứ hạng.
