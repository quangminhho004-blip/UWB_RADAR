"""Hàm loss để train.

    from src import losses
    value = losses.mse_pearson(pred, target, alpha=0.7)

MobiVital train bằng MSE nhưng chấm điểm bằng Pearson. Hai thước đo khác nhau:

    MSE      phạt khi sai BIÊN ĐỘ    -- dự báo 0.5 mà thật 0.9 thì bị phạt
    Pearson  chỉ quan tâm HÌNH DẠNG  -- lên xuống cùng nhịp là đủ

Sóng nhịp thở đã kéo về [-1, 1] nên biên độ không mang thông tin gì thêm. Model
dành sức khớp biên độ là phí. Đó là lý do TN3 thử đưa Pearson vào loss.

alpha là trọng số cho MSE:

    alpha = 1.0   thuần MSE, đúng như MobiVital
    alpha = 0.7   nghiêng về MSE
    alpha = 0.5   cân bằng
"""

import torch


def pearson(pred, target):
    """Tương quan Pearson của từng hàng. Trả về mảng (số_hàng,).

    Bỏ giá trị trung bình đi, rồi lấy tích vô hướng chia cho tích hai độ dài.
    Chính là cos của góc giữa hai vector đã trừ trung bình.
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (pred * target).sum(dim=1)
    denominator = pred.norm(dim=1) * target.norm(dim=1)

    # Cộng số rất nhỏ để không chia cho 0 khi gặp đoạn sóng phẳng lì.
    return numerator / (denominator + 1e-8)


def mse(pred, target):
    """Sai số bình phương trung bình. Đúng hàm MobiVital dùng."""
    return torch.nn.functional.mse_loss(pred, target)


def mse_pearson(pred, target, alpha):
    """Trộn MSE với Pearson.

        alpha * MSE  +  (1 - alpha) * (1 - Pearson)

    Pearson càng gần 1 càng tốt, nên lấy (1 - Pearson) để nó thành "càng nhỏ
    càng tốt" giống MSE, rồi mới cộng được.
    """
    mse_part = mse(pred, target)
    pearson_part = 1 - pearson(pred, target).mean()
    return alpha * mse_part + (1 - alpha) * pearson_part


def get_loss_fn(name, alpha=1.0):
    """Trả về hàm loss theo tên, để notebook chỉ cần truyền chuỗi.

        get_loss_fn("mse")
        get_loss_fn("mse_pearson", alpha=0.7)
    """
    if name == "mse":
        return mse

    if name == "mse_pearson":
        def loss_fn(pred, target):
            return mse_pearson(pred, target, alpha)
        return loss_fn

    raise ValueError("không biết loss tên " + name)
