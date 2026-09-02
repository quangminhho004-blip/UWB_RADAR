"""Hàm loss để train.

    from src import losses
    gia_tri = losses.mse_pearson(du_bao, that, alpha=0.7)

VÌ SAO CÓ FILE NÀY

MobiVital train bằng MSE nhưng chấm điểm bằng Pearson. Hai thước đo khác nhau:

    MSE      phạt khi sai BIÊN ĐỘ    -- dự báo 0.5 mà thật 0.9 thì bị phạt
    Pearson  chỉ quan tâm HÌNH DẠNG  -- lên xuống cùng nhịp là đủ, biên độ mặc kệ

Sóng nhịp thở đã được kéo về [-1, 1] rồi, nên biên độ không mang thông tin gì
thêm. Model dành sức khớp biên độ là phí. Đó là lý do TN3 thử đưa Pearson vào
loss xem có ăn điểm không.

alpha là trọng số cho MSE:

    alpha = 1.0   thuần MSE, đúng như MobiVital
    alpha = 0.7   nghiêng về MSE
    alpha = 0.5   cân bằng
"""

import torch


def pearson(du_bao, that):
    """Tương quan Pearson của từng hàng. Trả về một mảng (số_hàng,).

    Công thức: bỏ giá trị trung bình đi, rồi lấy tích vô hướng chia cho tích
    hai độ dài. Chính là cos của góc giữa hai vector đã trừ trung bình.
    """
    du_bao = du_bao - du_bao.mean(dim=1, keepdim=True)
    that = that - that.mean(dim=1, keepdim=True)

    tu_so = (du_bao * that).sum(dim=1)
    mau_so = du_bao.norm(dim=1) * that.norm(dim=1)

    # Cộng số rất nhỏ để không chia cho 0 khi gặp đoạn sóng phẳng lì.
    return tu_so / (mau_so + 1e-8)


def mse(du_bao, that):
    """Sai số bình phương trung bình. Đúng hàm MobiVital dùng."""
    return torch.nn.functional.mse_loss(du_bao, that)


def mse_pearson(du_bao, that, alpha):
    """Trộn MSE với Pearson.

        alpha * MSE  +  (1 - alpha) * (1 - Pearson)

    Pearson càng gần 1 càng tốt, nên lấy (1 - Pearson) để nó thành "càng nhỏ
    càng tốt" giống MSE, rồi mới cộng được.
    """
    phan_mse = mse(du_bao, that)
    phan_pearson = 1 - pearson(du_bao, that).mean()
    return alpha * phan_mse + (1 - alpha) * phan_pearson


def lay_ham_loss(ten, alpha=1.0):
    """Trả về hàm loss theo tên, để notebook chỉ cần truyền chuỗi.

        lay_ham_loss("mse")
        lay_ham_loss("mse_pearson", alpha=0.7)
    """
    if ten == "mse":
        return mse

    if ten == "mse_pearson":
        def ham(du_bao, that):
            return mse_pearson(du_bao, that, alpha)
        return ham

    raise ValueError("không biết loss tên " + ten)
