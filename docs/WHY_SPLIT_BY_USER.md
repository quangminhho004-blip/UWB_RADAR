# Vì sao phải chia dữ liệu theo người dùng?

## 1. Vấn đề cần tránh

Mỗi file CSV trong bộ dữ liệu MobiVital là một phiên đo của một người. Từ một
phiên dài 1.500 mẫu, pipeline tạo ra nhiều cửa sổ nhỏ để huấn luyện mô hình. Với
cấu hình gốc, mỗi mẫu huấn luyện gồm 200 mẫu lịch sử và 25 mẫu tương lai; cửa sổ
tiếp theo dịch đi 25 mẫu.

Các cửa sổ liên tiếp vì vậy chồng lấn rất mạnh. Chẳng hạn, hai cửa sổ lịch sử
liên tiếp dùng chung 175 trên 200 mẫu. Chúng cũng được tạo từ cùng người, cùng
phiên đo, cùng vị trí radar và cùng điều kiện môi trường. Do đó, chúng không phải
là hai quan sát độc lập hoàn toàn.

Nếu tạo tất cả cửa sổ trước rồi chia ngẫu nhiên, dữ liệu có thể bị phân bố như
sau:

```text
Window 1 của user A, session 01  -> train
Window 2 của user A, session 01  -> validation
Window 3 của user A, session 01  -> train
```

Khi ấy, validation chứa những đoạn tín hiệu rất gần với dữ liệu model đã nhìn
thấy lúc train. Model còn có thể học các đặc điểm riêng của user A, thiết bị hoặc
phiên đo thay vì học quy luật tổng quát của tín hiệu hô hấp. Điểm validation có
thể cao nhưng không phản ánh đúng khả năng hoạt động trên một người hoàn toàn
mới. Đây là một dạng **data leakage do chia sai nhóm**.

## 2. Cách chia đúng

Đồ án chia user trước khi tạo tập train và validation. Trong mỗi fold, hai user
được giữ lại hoàn toàn để validation; sáu user còn lại dùng để train:

| Fold | Train | Validation |
|---|---|---|
| `val_AB` | C, D, E, F, K, L | A, B |
| `val_CE` | A, B, D, F, K, L | C, E |
| `val_DF` | A, B, C, E, K, L | D, F |
| `val_KL` | A, B, C, D, E, F | K, L |

Ví dụ ở fold `val_AB`, không một session hay window nào của A và B được phép đi
vào tập train. Điểm validation của fold này vì thế đo khả năng tổng quát hóa từ
sáu người đã thấy sang hai người chưa thấy.

Việc lưu dữ liệu thành `A.npz`, `B.npz`, ..., `L.npz` giúp pipeline thực thi
ranh giới này rõ ràng. Các file theo user không có nghĩa là model được train
riêng cho từng người; chúng chỉ giúp chọn đúng nhóm user trước khi ghép dữ liệu.

## 3. Vẫn shuffle window trong tập train

Chia theo user và shuffle window là hai thao tác khác nhau:

```text
Split theo user:
    quyết định user nào thuộc train, validation và test

Shuffle window:
    đảo thứ tự các window đã thuộc tập train trước khi tạo batch
```

Sau khi đã chọn sáu train user của một fold, toàn bộ window của sáu người này
được ghép chung và shuffle ở mỗi epoch. Việc shuffle giúp các batch không bị xếp
liên tục theo user hoặc session và là hành vi bình thường khi huấn luyện.

```python
train_users = ["C", "D", "E", "F", "K", "L"]
val_users = ["A", "B"]

# Tạo dữ liệu riêng sau khi đã khóa danh sách user.
train_x, train_y = make_windows(train_users)
val_x, val_y = make_windows(val_users)

# Chỉ đảo thứ tự các window bên trong tập train.
indices = torch.randperm(len(train_x), device=train_x.device)
train_x = train_x[indices]
train_y = train_y[indices]
```

Không shuffle validation và test vì model không học từ hai tập này. Thứ tự của
chúng không làm thay đổi điểm trung bình, đồng thời giữ kết quả dễ đối chiếu.

## 4. Quan hệ với protocol MobiVital

MobiVital gốc cũng tách người dùng thành hai nhóm:

```text
Train/dev: A, B, C, D, E, F, K, L
Test:      G, H, I, J
```

Đồ án giữ nguyên `GHIJ` làm tập test và chỉ thực hiện cross-validation bên trong
pool phát triển `ABCDEFKL`. Bốn fold lần lượt giữ lại từng cặp validation nên cả
tám người trong pool đều được đánh giá đúng một lần. Tập test không được trộn
vào các fold.

Thiết kế này trả lời hai câu hỏi khác nhau:

1. Cross-validation trên `ABCDEFKL`: cấu hình nào có khả năng tổng quát hóa tốt
   hơn để được lựa chọn?
2. Test trên `GHIJ`: cấu hình đã khóa hoạt động thế nào trên nhóm người hoàn toàn
   chưa dùng trong quá trình lựa chọn?

## 5. Điều gì xảy ra nếu chia ngẫu nhiên toàn bộ window?

Chia ngẫu nhiên toàn bộ window thường tạo ra nhiều mẫu train và validation hơn,
đồng thời làm điểm validation có vẻ ổn định. Tuy nhiên, sự ổn định này có thể
đến từ việc hai tập chứa tín hiệu của cùng user hoặc cùng session. Khi triển khai
cho người mới, model không còn lợi thế đó nên chất lượng có thể giảm đáng kể.

Vì mục tiêu của MobiVital là đánh giá tín hiệu trên các subject chưa thấy, đơn
vị chia dữ liệu phù hợp là **user**, không phải window. Window chỉ là đơn vị đưa
vào model sau khi ranh giới user đã được thiết lập.

## 6. Câu trả lời ngắn khi bảo vệ

> Em không chia ngẫu nhiên toàn bộ window vì các window liền nhau chồng lấn mạnh
> và mang đặc trưng của cùng người, cùng phiên đo. Nếu một user xuất hiện ở cả
> train và validation, điểm validation có thể bị cao giả tạo do data leakage.
> Vì vậy em chia theo user trước; sau đó em vẫn ghép và shuffle toàn bộ window
> của các train user trong từng epoch. Cách này đánh giá đúng hơn khả năng tổng
> quát hóa sang người chưa từng xuất hiện trong tập train.

## 7. Nguyên tắc kiểm tra bằng code

Trước mỗi lần train, pipeline cần xác nhận hai nhóm user không giao nhau:

```python
assert set(train_users).isdisjoint(val_users)
assert set(train_users).isdisjoint(test_users)
assert set(val_users).isdisjoint(test_users)
```

Các assertion này ngăn lỗi cấu hình làm một user vô tình xuất hiện ở nhiều tập.
Danh sách user của từng fold cũng phải được ghi vào kết quả thực nghiệm để có
thể kiểm tra và tái lập về sau.
