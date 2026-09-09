# Pipeline training và inference — tổng quan và phóng to từng khối

**Bố cục: 1 slide tổng quan + 8 slide chi tiết.** Mỗi slide chi tiết chỉ tập trung vào một nhiệm vụ, gồm **3–4 block con**. Số **1–8** chỉ khối chính; số **1.1, 1.2…** chỉ các bước bên trong khối tương ứng.

Giữ sơ đồ bằng ký tự, khung vuông và ngoặc nhóm thẳng hàng. Trên slide dùng sơ đồ và một ý chính; bảng giải thích và lời nói bên dưới để làm speaker notes, không cần nhồi tất cả lên slide.

**Ba từ cần biết:** *sóng ứng viên* là chuỗi radar đem ra lựa chọn; *mẫu* là một giá trị tín hiệu tại một thời điểm; *GT* là sóng hô hấp tham chiếu từ đai đo.

| Slide | Khối được trình bày | Vai trò |
|---|---|---|
| 1 | Tổng quan **1–8** | Thấy toàn bộ luồng |
| 2 | **1. Biến đổi và chuẩn hóa** | Đưa dữ liệu train về dạng phù hợp |
| 3 | **2. Chọn các chuỗi để học** | Lọc theo đai, thêm chuỗi đai |
| 4 | **3. Tạo cửa sổ huấn luyện** | Giải thích 200 → 25 |
| 5 | **4. Huấn luyện mô hình** | Lô dữ liệu, dự báo, loss, cập nhật |
| 6 | **5. Chuẩn bị sóng ứng viên** | Tạo và lọc ứng viên lúc inference |
| 7 | **6. Dự báo và chấm từng đoạn** | Tạo điểm dự báo cho mỗi cửa sổ |
| 8 | **7. Chọn sóng đầu ra** | Cộng điểm, chọn và lấy waveform |
| 9 | **8. Đánh giá với đai đo** | Chấm kết quả đã chọn |

## Slide 1 — Tổng quan tám khối chính

**Ý chính:** Khối 1–4 học mô hình; khối 5–7 dùng mô hình để chọn sóng; khối 8 đánh giá khi có đai tham chiếu.


```text
            HUẤN LUYỆN (TRAINING)                                                  SUY LUẬN VÀ ĐÁNH GIÁ

┌────────────────────────────────────────────┐                        ┌────────────────────────────────────────────┐
│          1. BIẾN ĐỔI VÀ CHUẨN HÓA          │                        │         5. CHUẨN BỊ SÓNG ỨNG VIÊN          │
│        Radar và đai của nhóm train         │                        │     Biến đổi, chuẩn hóa, lọc sóng đảo      │
└──────────────────────┬─────────────────────┘                        └──────────────────────┬─────────────────────┘
                       │                                                                     │
┌──────────────────────▼─────────────────────┐                        ┌──────────────────────▼─────────────────────┐
│          2. CHỌN CÁC CHUỖI ĐỂ HỌC          │        ┌──────────────▶│        6. DỰ BÁO VÀ CHẤM TỪNG ĐOẠN         │
│       Lọc theo đai + thêm chuỗi đai        │        │               │          Mô hình dự báo → Pearson          │
└──────────────────────┬─────────────────────┘        │               └──────────────────────┬─────────────────────┘
                       │                              │                                      │
┌──────────────────────▼─────────────────────┐        │  Trọng số     ┌──────────────────────▼─────────────────────┐
│          3. TẠO CỬA SỔ HUẤN LUYỆN          │        │  đã học       │            7. CHỌN SÓNG ĐẦU RA             │
│          200 mẫu → 25 mẫu kế tiếp          │        │               │       Tổng điểm → lấy sóng cao nhất        │
└──────────────────────┬─────────────────────┘        │               └──────────────────────┬─────────────────────┘
                       │                              │                                      │
┌──────────────────────▼─────────────────────┐        │               ┌──────────────────────▼─────────────────────┐
│           4. HUẤN LUYỆN MÔ HÌNH            ├────────┘               │           8. ĐÁNH GIÁ VỚI ĐAI ĐO           │
│         Học dự báo và lưu trọng số         │                        │      Pearson từng buổi → macro người       │
└────────────────────────────────────────────┘                        └────────────────────────────────────────────┘
```


**Câu nói khi trình bày:** “Bốn khối bên trái tạo dữ liệu và huấn luyện mô hình. Bên phải, hệ thống chuẩn bị ứng viên, dự báo và chọn một sóng radar. Sau khi đã chọn, chúng em mới dùng đai để đánh giá chất lượng.”

**Đường nối quan trọng:** Trọng số từ **4.4** đi vào **6.2**, không đi thẳng vào bước lấy sóng cao điểm nhất. Khối 8 là đánh giá offline, không phải điều kiện để thực hiện chọn sóng. Không có GT vẫn xuất được waveform tại 7.3.


## Slide 2 — Phóng to khối 1: biến đổi và chuẩn hóa dữ liệu

**Ý chính:** Khối 1 biến dữ liệu radar thành những chuỗi số thực có thể dùng để học.

**Đầu vào:** CSV radar và đai của nhóm train. **Đầu ra:** 36 ứng viên radar/buổi và chuỗi đai đã chuẩn hóa.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │       1.1. Đọc radar và sóng đai đo        │    │
  │        Mỗi buổi: 30 giây, 1.500 mẫu        │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │
  ┌──────────────────────▼─────────────────────┐    │
  │         1.2. Tạo 36 sóng ứng viên          │    │  TIỀN XỬ LÝ
  │      9 ô cự ly × 4 cách lấy tín hiệu       │    │  Đọc và biến đổi tín hiệu
  └──────────────────────┬─────────────────────┘    │
                         │                          │
  ┌──────────────────────▼─────────────────────┐    │
  │         1.3. Chuẩn hóa từng chuỗi          │    │
  │     Radar và đai đưa riêng về [−1, 1]      │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **1.1** | Đọc tín hiệu radar phức I + jQ và sóng hô hấp tham chiếu từ đai. |
| **1.2** | Từ mỗi ô cự ly đã chọn, lấy độ lớn, phần thực, phần ảo và pha: bốn cách biểu diễn cùng tín hiệu phức. |
| **1.3** | Đưa mỗi chuỗi về cùng khoảng giá trị để chuẩn bị cho lọc và học. |

**Câu nói gợi ý:** “Trước khi học, tín hiệu radar phức được chuyển thành các chuỗi số thực. Chúng em chuẩn hóa riêng từng chuỗi radar và đai, rồi chuyển sang bước chọn dữ liệu.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Lấy bin index **20–28**: 9 bin × 4 biểu diễn = 36 candidate trước lọc.
- Phase được unwrap để xử lý bước nhảy biểu diễn tại ±π. Đây không phải inversion detector.
- Min–max được tính trên từng chuỗi **1.500 mẫu**, không lấy min/max chung cho mọi bin.
- Chỉ đọc dữ liệu thuộc nhóm train của fold đang chạy.

</details>

## Slide 3 — Phóng to khối 2: chọn các chuỗi dùng để học

**Ý chính:** Khối 2 dùng đai để chọn radar phù hợp và bổ sung cả chuỗi đai vào tập học.

**Đầu vào:** Các chuỗi đã chuẩn hóa từ khối 1. **Đầu ra:** Tập chuỗi dùng để tạo cửa sổ huấn luyện.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │      2.1. So từng sóng radar với đai       │    │
  │          Tính tương quan Pearson           │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  LỌC BẰNG ĐAI ĐO
  ┌──────────────────────▼─────────────────────┐    │
  │      2.2. Giữ sóng có tương quan cao       │    │
  │           Chỉ nhận Pearson > 0,9           │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │        2.3. Thêm chính sóng đai đo         │    │  BỔ SUNG CHUỖI HỌC
  │       Ghép vào tập chuỗi dùng để học       │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **2.1** | So từng ứng viên radar với sóng đai của cùng buổi ghi. |
| **2.2** | Giữ các chuỗi radar có tương quan dương vượt 0,9 để đưa vào tập train. |
| **2.3** | Thêm sóng đai như một chuỗi học riêng; không chỉ dùng đai để lọc. |

**Câu nói gợi ý:** “Chúng em muốn mô hình học từ các chuỗi phù hợp với hô hấp. Đai giúp chọn các chuỗi radar này, đồng thời bản thân chuỗi đai cũng được đưa vào tập học.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Điều kiện code là **`r > 0.9`**, không phải `r ≥ 0.9` hoặc `|r| > 0.9`.
- Tính Pearson trên toàn chuỗi 1.500 mẫu **trước khi chia window**.
- Không dùng inversion detector ở khối này. Toàn bộ việc tạo dữ liệu train có sử dụng GT.
- Các chuỗi tương quan âm mạnh không được đưa vào tập train bởi nhánh tạo dataset đang dùng.

</details>

## Slide 4 — Phóng to khối 3: tạo cửa sổ huấn luyện

**Ý chính:** Khối 3 tạo các cặp đầu vào và đáp án từ cùng một chuỗi.

**Đầu vào:** Tập chuỗi radar và đai được giữ ở khối 2. **Đầu ra:** Các cặp 200 mẫu đầu vào và 25 mẫu đáp án.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │        3.1. Lấy 200 mẫu làm đầu vào        │    │
  │         Tương ứng 4 giây tín hiệu          │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  TẠO MỘT CẶP HỌC
  ┌──────────────────────▼─────────────────────┐    │
  │    3.2. Lấy 25 mẫu ngay sau làm đáp án     │    │
  │      Của chính chuỗi đó, dài 0,5 giây      │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │        3.3. Trượt 25 mẫu và lặp lại        │    │  TRƯỢT CỬA SỔ
  │         Mỗi chuỗi tạo được 52 cặp          │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **3.1** | Đây là đoạn mô hình được nhìn thấy để dự báo. |
| **3.2** | Đây là đoạn dùng để tính lỗi của dự báo; không đưa đoạn này vào đầu vào model. |
| **3.3** | Dịch vị trí bắt đầu rồi lấy cặp mới, miễn còn đủ 200 + 25 mẫu. |

**Câu nói gợi ý:** “Mô hình nhìn bốn giây để dự báo nửa giây tiếp theo. Đáp án luôn lấy từ chính chuỗi đó: radar dự báo radar, đai dự báo đai.”

**Ví dụ để chiếu hoặc nói:**

```text
  Cặp 1:  đầu vào mẫu   0–199  →  đáp án mẫu 200–224
  Cặp 2:  đầu vào mẫu  25–224  →  đáp án mẫu 225–249
  Cặp 3:  đầu vào mẫu  50–249  →  đáp án mẫu 250–274
```


<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- `floor((1500 − 200 − 25) / 25) + 1 = 52` cặp, không phải 52 điểm tín hiệu.
- Không ghép 200 mẫu radar với 25 mẫu đai.
- Đây là tạo nhãn từ chính chuỗi, có thể gọi là *self-supervised forecasting*. Tuy nhiên bước chọn dữ liệu ở khối 2 vẫn dùng GT.
- Khi gom nhiều chuỗi, không nối chúng thành một sóng dài để cắt qua ranh giới buổi ghi.

</details>

## Slide 5 — Phóng to khối 4: huấn luyện mô hình dự báo

**Ý chính:** Khối 4 lặp dự báo, tính lỗi và cập nhật trọng số, rồi lưu mô hình.

**Đầu vào:** Các cặp cửa sổ từ khối 3. **Đầu ra:** Trọng số đã học để nạp tại bước 6.2.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │      4.1. Xáo trộn và chia lô dữ liệu      │    │  CHUẨN BỊ LÔ DỮ LIỆU
  │            Mỗi lô gồm 64 cửa sổ            │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │       4.2. Dự báo 25 mẫu từ 200 mẫu        │    │
  │     Mô hình xử lý các cửa sổ trong lô      │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  VÒNG LẶP HUẤN LUYỆN
  ┌──────────────────────▼─────────────────────┐    │
  │       4.3. Tính lỗi và sửa trọng số        │    │
  │        Lặp bước 4.2–4.3 qua các lô         │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │       4.4. Lưu mô hình khi học xong        │    │  LƯU MÔ HÌNH
  │       Dùng trọng số này tại bước 6.2       │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **4.1** | Xáo trộn các cửa sổ của nhóm train rồi chia thành các lô. |
| **4.2** | Model tạo dự báo cho mỗi cửa sổ, chưa biết đáp án của cửa sổ đó. |
| **4.3** | Loss đo lỗi giữa dự báo và đáp án; gradient và optimizer cập nhật trọng số. |
| **4.4** | Sau các vòng học, lưu trọng số để sử dụng mà không học lại. |

**Câu nói gợi ý:** “Mỗi lô được đưa qua mô hình để dự báo. Sai số giúp điều chỉnh trọng số. Quá trình lặp qua các lô và các epoch; học xong thì lưu mô hình.”

**Cấu hình nền:** Adam · learning rate `0.0001` · batch `64` · `20` epoch · MSE.

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Batch là nhóm cửa sổ xử lý cùng lượt. Epoch là một vòng duyệt tập train; lặp các epoch rồi mới đến 4.4.
- Khi trình bày thực nghiệm loss, thay MSE bằng đúng loss/alpha đã dùng. Sơ đồ không mặc định mọi thực nghiệm đều dùng MSE.
- Khi inference, trọng số cố định. Checkpoint phục vụ resume cần thêm trạng thái optimizer và ngẫu nhiên phù hợp.
- Chia train/validation theo người; cửa sổ của người validation không đi vào vòng học.

</details>

## Slide 6 — Phóng to khối 5: chuẩn bị sóng ứng viên khi suy luận

**Ý chính:** Khối 5 chuẩn bị các sóng radar để lựa chọn, không dùng đai đo.

**Đầu vào:** Đoạn radar MobiVital: 1.500 mẫu × 120 bin. **Đầu ra:** Các ứng viên còn lại sau lọc, kèm bin và biểu diễn.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │         5.1. Tạo 240 sóng ứng viên         │    │
  │        120 ô cự ly × độ lớn và pha         │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  BIẾN ĐỔI VÀ CHUẨN HÓA
  ┌──────────────────────▼─────────────────────┐    │
  │       5.2. Chuẩn hóa riêng từng sóng       │    │
  │        Đưa về thang giá trị [−1, 1]        │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │     5.3. Loại sóng bị nhận diện là đảo     │    │  LỌC SÓNG ĐẢO
  │          Dùng inversion detector           │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **5.1** | Quét 120 bin. Mỗi bin lấy magnitude và phase, nên có hai ứng viên. |
| **5.2** | Chuẩn hóa các chuỗi trước khi đưa sang model để chấm. |
| **5.3** | Dựa vào hình dạng đỉnh và đáy để nhận diện sóng đảo, rồi loại những ứng viên bị gắn nhãn. |

**Câu nói gợi ý:** “Lúc sử dụng không có đai để lọc. Hệ thống tạo ứng viên từ toàn bộ 120 bin, chuẩn hóa và dùng detector hình dạng để loại sóng bị nhận diện là đảo.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Phase được unwrap trước chuẩn hóa; min–max tính riêng trên từng chuỗi 1.500 mẫu.
- Detector dùng Savitzky–Golay để làm mượt, tìm đỉnh/đáy rồi so độ rộng trung bình. Không đo lệch pha với GT.
- Hàm trả **0/1**. Giữ `< 0.8` tương đương giữ `0`, không phải xác suất 80% hoặc ngưỡng tương quan.
- Sóng bị gắn nhãn đảo bị loại, không được lật lại rồi giữ. Detector trả 0 cũng chưa chứng minh sóng có chất lượng tốt.
- Giữ metadata `(bin, biểu diễn)` để không mất danh tính ứng viên sau lọc.

</details>

## Slide 7 — Phóng to khối 6: dự báo và chấm điểm từng đoạn

**Ý chính:** Khối 6 dùng mô hình đã học để đo khả năng dự báo của mỗi sóng.

**Đầu vào:** Các ứng viên từ khối 5 và trọng số từ khối 4. **Đầu ra:** 52 điểm dự báo cho mỗi ứng viên.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │       6.1. Cắt mỗi sóng thành 52 cặp       │    │  TẠO CỬA SỔ
  │       200 mẫu đầu + 25 mẫu đối chiếu       │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │         6.2. Mô hình dự báo 25 mẫu         │    │  DỰ BÁO
  │      Nạp trọng số đã lưu tại bước 4.4      │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │         6.3. Chấm từng đoạn dự báo         │    │  CHẤM ĐIỂM DỰ BÁO
  │       Pearson với radar thật kế tiếp       │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **6.1** | Chia cửa sổ như khi train, giữ riêng đoạn 25 mẫu để đối chiếu. |
| **6.2** | Model nhận 200 mẫu và tạo dự báo; trọng số không được cập nhật. |
| **6.3** | Đối chiếu dự báo với đoạn radar thật kế tiếp của chính ứng viên đó. |

**Câu nói gợi ý:** “Chúng em chấm xem mô hình dự báo từng sóng tốt đến đâu. Đoạn để đối chiếu là radar đã đo, hoàn toàn không phải sóng đai.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Các cửa sổ đều nằm trong đoạn 30 giây đã thu đủ. “Tương lai” ở đây là tương lai so với từng history, không phải dữ liệu chưa thu ở cuối snapshot.
- Model chỉ nhận 200 mẫu; phần chấm giữ riêng 25 mẫu radar thật.
- Nếu còn `N` ứng viên, có `52 × N` cửa sổ/model. Số này khác batch size và khác tốc độ xử lý/giây.
- 52 điểm này là Pearson nội bộ để chọn ứng viên, chưa phải điểm cuối với GT.

</details>

## Slide 8 — Phóng to khối 7: chọn sóng đầu ra

**Ý chính:** Khối 7 tổng hợp điểm và lấy một sóng radar làm kết quả.

**Đầu vào:** 52 điểm dự báo của mỗi ứng viên từ khối 6. **Đầu ra:** Một waveform radar được chọn, kèm `(bin, biểu diễn)`.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │       7.1. Cộng 52 điểm của mỗi sóng       │    │
  │       Mỗi ứng viên có một tổng điểm        │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  SO SÁNH VÀ LỰA CHỌN
  ┌──────────────────────▼─────────────────────┐    │
  │     7.2. Lấy ứng viên có điểm cao nhất     │    │
  │     Argmax: chọn chỉ số tổng lớn nhất      │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │       7.3. Xuất sóng radar tương ứng       │    │  SÓNG ĐẦU RA
  │     Giữ thông tin ô cự ly và biểu diễn     │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **7.1** | Tổng hợp điểm của các đoạn thuộc cùng một ứng viên. |
| **7.2** | So tổng điểm giữa các ứng viên, chọn chỉ số cao nhất. |
| **7.3** | Lấy toàn bộ waveform đã biến đổi/chuẩn hóa của ứng viên đó làm kết quả. |

**Câu nói gợi ý:** “Hệ thống chọn sóng có tổng điểm dự báo cao nhất. Kết quả là một sóng radar đã đo, không phải các dự báo ghép lại. Đến đây quá trình chọn sóng đã hoàn tất mà không cần đai.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Cùng một bin nhưng magnitude và phase là hai ứng viên khác nhau; phải lưu cả bin và biểu diễn.
- Khả năng dự báo là dấu hiệu được dùng để chọn. Nhiễu có cấu trúc cũng có thể dễ dự báo; không bảo đảm điểm dự báo cao nhất đồng nghĩa giống đai nhất.
- Không có GT vẫn chạy được đến hết **7.3**. Bước đánh giá với GT được tách riêng ở khối 8.

</details>

## Slide 9 — Phóng to khối 8: đánh giá bằng sóng đai có sẵn trong dataset

**Ý chính:** So sóng radar đã chọn với **sóng hô hấp từ đai đo được lưu sẵn trong dataset MobiVital**, thuộc cùng buổi ghi.

**Đầu vào:** Sóng radar được chọn ở khối 7 và sóng đai tham chiếu (GT) đi kèm radar đó trong dataset. **Đầu ra:** Điểm Pearson từng buổi và macro theo người.

Mỗi buổi ghi của dataset có cả **radar** và **sóng đai đo hô hấp** tương ứng. Khối 5–7 chỉ dùng radar để chọn sóng; đến khối 8 mới lấy sóng đai đã lưu để chấm kết quả. Không cần đo thêm bằng đai trong lúc chạy notebook.

```text
  ┌────────────────────────────────────────────┐   ─┐
  │       8.1. Lấy sóng đai từ dataset         │    │
  │      GT MobiVital cùng buổi với radar      │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  CHẤM TỪNG BUỔI
  ┌──────────────────────▼─────────────────────┐    │  Có dùng GT đai
  │       8.2. So sóng được chọn với đai       │    │
  │        Tính Pearson trên 1.500 mẫu         │    │
  └──────────────────────┬─────────────────────┘   ─┘
                         │
  ┌──────────────────────▼─────────────────────┐   ─┐
  │       8.3. Tính điểm cho từng người        │    │
  │      Trung bình các buổi của người đó      │    │
  └──────────────────────┬─────────────────────┘    │
                         │                          │  TỔNG HỢP KẾT QUẢ
  ┌──────────────────────▼─────────────────────┐    │
  │      8.4. Tính điểm chung theo người       │    │
  │      Trung bình điểm các người: macro      │    │
  └────────────────────────────────────────────┘   ─┘
```

**Giải thích để thuyết trình — không đưa cả bảng lên slide:**

| Bước | Đang làm gì? |
|---|---|
| **8.1** | Đọc sóng hô hấp từ đai đã lưu trong dataset MobiVital, đúng người và đúng buổi ghi của radar đang chấm. Đây là tín hiệu tham chiếu có sẵn, không phải đầu ra của model. |
| **8.2** | So 1.500 mẫu của sóng radar được chọn với 1.500 mẫu sóng đai tương ứng bằng Pearson, thu được một điểm cho buổi ghi. |
| **8.3** | Gộp điểm các buổi của một người thành điểm người đó. |
| **8.4** | Cho mỗi người trọng số như nhau khi tính điểm chung. |

**Câu nói gợi ý:** “Trong dataset MobiVital, mỗi buổi có cả tín hiệu radar và tín hiệu hô hấp tham chiếu từ đai đo. Sau khi mô hình tự chọn sóng radar, chúng em so sóng đó với sóng đai của cùng buổi để tính điểm. Sau đó, chúng em tính trung bình từng người rồi trung bình các người.”

<details>
<summary>Chi tiết kỹ thuật khi được hỏi</summary>

- Sóng đai ở khối 8 là GT của **dataset MobiVital**, không phải dữ liệu ApneaLink của demo. Trong CSV gốc, `scripts/make_npz.py` đọc cột áp chót (`data[:, -2]`), chuẩn hóa và lưu vào trường **`gt`** trong file NPZ theo người. Trường **`uwb`** chứa radar tương ứng.
- Khi validation AB, lấy GT của chính các buổi thuộc A và B; khi test GHIJ, lấy GT của chính các buổi thuộc GHIJ. Không lấy sóng đai của nhóm train để chấm cho người khác.
- **Pearson ở 6.3:** dự báo ↔ radar thật, dùng để chọn. **Pearson ở 8.2:** sóng đã chọn ↔ đai, dùng để đánh giá.
- Pearson không phải phần trăm chính xác hay sai số nhịp thở/phút.
- Macro theo người khác micro trung bình trực tiếp các buổi. Ghi đúng cách tổng hợp ở từng bảng kết quả.
- Oracle là phân tích riêng dùng GT để chọn ứng viên tốt nhất. Chỉ so oracle/model khi cùng tập người, cùng tập ứng viên và cùng phép lấy trung bình; không trừ oracle-dev với điểm test GHIJ.

</details>

## Phụ lục — Pipeline trên chạy với những người nào?

Giao thức chia dữ liệu áp dụng cho cả tám khối; đây không phải khối thứ 9 của pipeline xử lý.

| Giai đoạn | Khối 1–4: dữ liệu dùng để học | Khối 5–8: dữ liệu dùng để chấm |
|---|---|---|
| Fold AB | C D E F K L | A B |
| Fold CE | A B D F K L | C E |
| Fold DF | A B C E K L | D F |
| Fold KL | A B C D E F | K L |
| Test cuối sau khi chốt cấu hình | A B C D E F K L | G H I J |

**Câu nói khi trình bày:** “Trong mỗi fold, mô hình học từ sáu người và được đánh giá trên hai người khác. Chọn cấu hình xong, chúng em huấn luyện lại trên đủ tám người phát triển rồi chấm trên GHIJ.”


<details>
<summary>Chi tiết để trả lời khi được hỏi</summary>

- Bốn lượt gọi là **4-fold cross-validation**. Điểm CV một seed là trung bình macro của bốn fold, tương đương trung bình tám người dev.
- Train dùng GT để tạo tập học; validation tự chọn sóng bằng inversion detector và model. GT validation chỉ chấm kết quả.
- **Seed** điều khiển các nguồn ngẫu nhiên như khởi tạo và xáo trộn. Độ dao động giữa seed khác độ dao động giữa fold.
- Ghi đúng số seed thực tế của từng khảo sát. Không lấy fold có điểm cao nhất làm kết quả chung hoặc chọn checkpoint cuối chỉ vì fold đó cao nhất.
- GHIJ đã được đánh giá trong TN0; không nói “chỉ mở đúng một lần” như một sự kiện đã xảy ra. Nếu có dùng test để điều chỉnh cấu hình thì phải nêu giới hạn đó.
- Mean ± std không tự chứng minh ý nghĩa thống kê hay tương đương giữa hai cấu hình.

</details>

## Nguồn đối chiếu

- [Đọc CSV](../scripts/make_npz.py), [tạo cửa sổ](../scripts/make_windows.py), [hàm tạo dữ liệu train gốc](https://github.com/nesl/mobivital-public/blob/4319731d2769d4134c92088dd846666e262f18e9/training/utils/model_utils.py).
- [Biến đổi sóng](https://github.com/nesl/mobivital-public/blob/4319731d2769d4134c92088dd846666e262f18e9/utils/model_utils.py), [phát hiện đảo sóng](https://github.com/nesl/mobivital-public/blob/4319731d2769d4134c92088dd846666e262f18e9/utils/peak_width_inverter.py), [chọn sóng và chấm điểm](../src/scoring.py).
- [Huấn luyện](../src/training.py), [4-fold CV](../scripts/run_cv.py), [test cuối](../scripts/run_final_test.py).

**Khi đưa vào slide:** dùng sơ đồ và một câu ý chính. Phần giải thích chi tiết dành cho người thuyết trình và câu hỏi của hội đồng.
