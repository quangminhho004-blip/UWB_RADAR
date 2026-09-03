"""Vòng train, lưu checkpoint, chạy tiếp khi Colab ngắt phiên.

    from src import training
    training.set_seed(0)
    model = models.build_model("ds_tcn", revin=True)
    result = training.train(model, train_loader, val_loader, "runs/tn1/fold1_seed0")

CHÍNH SÁCH FILE

    đang train:  last.pth     GHI ĐÈ mỗi epoch, chỉ một file
    train xong:  final.pth    chỉ trọng số
                 last.pth     XOÁ

`last.pth` chứa cả trạng thái Adam nên nặng gấp ba `final.pth`. Kế hoạch có
khoảng 170 lần chạy — không xoá là 3 GB rác trên Drive, mà Drive chỉ còn 1.4 GB.

VÌ SAO LƯU CẢ TRẠNG THÁI NGẪU NHIÊN

`DataLoader` xáo trộn dữ liệu mỗi epoch bằng bộ sinh số ngẫu nhiên. Colab ngắt
phiên, phiên sau bắt đầu lại từ trạng thái mặc định thì thứ tự batch khác hẳn,
model ra khác với lần chạy liền mạch. Lưu trạng thái đó thì chạy đứt quãng và
chạy liền mạch cho kết quả y hệt.

VÌ SAO GHI CẢ MSE LẪN PEARSON MỖI EPOCH

TN3 sẽ so MSE thuần với MSE + Pearson. Hai hàm loss cho giá trị ở thang khác
nhau, `train_loss` của hai bên không so được. Nên ghi tách từng thành phần, luôn
luôn, bất kể đang tối ưu cái nào. `train_mse` so được xuyên suốt TN1 đến TN6.

VỀ VALIDATION

`val_mse` và `val_pearson` đo trên CỬA SỔ của người validation, chỉ để nhìn có
overfit không. KHÔNG được dùng để chọn cấu hình: cửa sổ đó lọc bằng
`corr(sóng, nhịp thở thật) > 0.9`, tức đã nhìn đáp án. Điểm quyết định phải lấy
từ `src/scoring.py` chạy trên buổi ghi thô.

MobiVital không có validation — `test_dataloader` truyền vào `train()` nhưng bên
trong không dùng lần nào.
"""

import os
import random
import time

import numpy as np
import torch

from src import losses
from src import mobivital_reference as mv


def set_seed(seed):
    """Đặt hạt giống cho mọi bộ sinh số ngẫu nhiên. Gọi TRƯỚC khi dựng model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_file_name(user, corr_threshold=None):
    """Tên file cửa sổ của một người, khớp với scripts/5_make_windows.py."""
    if corr_threshold is None:
        corr_threshold = mv.CORR_THRESHOLD
    return (user
            + "_corr" + str(corr_threshold)
            + "_h" + str(mv.HISTORY_LENGTH)
            + "_f" + str(mv.FUTURE_LENGTH)
            + ".npz")


def load_windows(users, corr_threshold=None, folder=None):
    """Đọc file cửa sổ của nhiều người rồi ghép lại. Trả về (X, y).

    Ghép được vì cửa sổ cắt riêng từng người — xem scripts/5_make_windows.py.
    """
    if folder is None:
        folder = mv.PROJECT_DIR + "/data/processed/windows/dev_cv"

    all_X = []
    all_y = []
    for user in users:
        data = np.load(folder + "/" + window_file_name(user, corr_threshold))
        all_X.append(data["X"])
        all_y.append(data["y"])

    return np.concatenate(all_X), np.concatenate(all_y)


def make_loader(X, y, batch_size=None, shuffle=True):
    """Bọc hai mảng thành DataLoader.

    Không dùng num_workers: dữ liệu đã nằm sẵn trong bộ nhớ, bật tiến trình phụ
    chỉ tốn thêm thời gian chuyển dữ liệu qua lại.
    """
    if batch_size is None:
        batch_size = mv.BATCH_SIZE

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(),
                                             torch.from_numpy(y).float())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle)


def run_one_pass(model, loader, device, loss_fn, optimizer=None):
    """Chạy hết một lượt dữ liệu. Có optimizer thì học, không thì chỉ đo.

    Trả về (mse, pearson, loss) trung bình trên các batch.
    """
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_mse = 0.0
    total_pearson = 0.0
    total_loss = 0.0
    n_batches = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if is_training:
            optimizer.zero_grad()
            pred = model(X)
            value = loss_fn(pred, y)
            value.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(X)
                value = loss_fn(pred, y)

        # Ghi cả hai thành phần dù đang tối ưu cái nào, để so được giữa các
        # thí nghiệm dùng hàm loss khác nhau.
        with torch.no_grad():
            total_mse = total_mse + losses.mse(pred, y).item()
            total_pearson = total_pearson + losses.pearson(pred, y).mean().item()

        total_loss = total_loss + value.item()
        n_batches = n_batches + 1

    return total_mse / n_batches, total_pearson / n_batches, total_loss / n_batches


def train(model, train_loader, val_loader, run_dir,
          epochs=None, lr=None, loss_name="mse", alpha=1.0, verbose=True):
    """Train model, lưu checkpoint mỗi epoch, chạy tiếp được nếu bị ngắt.

    run_dir  -- thư mục lưu last.pth và final.pth

    Trả về dict gồm đường dẫn checkpoint, đường cong loss, số phút, và cờ cho
    biết lần chạy này có bị ngắt giữa chừng hay không.
    """
    if epochs is None:
        epochs = mv.EPOCHS
    if lr is None:
        lr = mv.LEARNING_RATE

    os.makedirs(run_dir, exist_ok=True)
    last_path = run_dir + "/last.pth"
    final_path = run_dir + "/final.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = losses.get_loss_fn(loss_name, alpha)

    start_epoch = 0
    curve = []
    was_interrupted = False

    # --- Có file đang dở thì chạy tiếp từ đó ---
    if os.path.exists(last_path):
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_torch"])
        np.random.set_state(saved["rng_numpy"])
        start_epoch = saved["epoch"] + 1
        curve = saved["curve"]
        was_interrupted = True
        print("chạy tiếp từ epoch", start_epoch)

    started_at = time.time()

    for epoch in range(start_epoch, epochs):
        train_mse, train_pearson, train_loss = run_one_pass(
            model, train_loader, device, loss_fn, optimizer)

        val_mse = ""
        val_pearson = ""
        if val_loader is not None:
            val_mse, val_pearson, _ = run_one_pass(
                model, val_loader, device, loss_fn)

        minutes = (time.time() - started_at) / 60
        curve.append({"epoch": epoch,
                      "train_mse": train_mse,
                      "train_pearson": train_pearson,
                      "train_loss": train_loss,
                      "val_mse": val_mse,
                      "val_pearson": val_pearson,
                      "minutes": round(minutes, 2)})

        if verbose:
            line = "epoch %2d  mse %.5f  pearson %.4f" % (
                epoch, train_mse, train_pearson)
            if val_loader is not None:
                line = line + "  |  val mse %.5f  pearson %.4f" % (
                    val_mse, val_pearson)
            print(line + "   %.1f phút" % minutes)

        # Ghi đè một file duy nhất. Colab ngắt phiên thì phiên sau đọc file này.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_numpy": np.random.get_state(),
                    "curve": curve}, last_path)

    # --- Xong: giữ trọng số, xoá file đang dở ---
    torch.save(model.state_dict(), final_path)
    if os.path.exists(last_path):
        os.remove(last_path)

    last_epoch = curve[-1]
    return {"final_path": final_path,
            "curve": curve,
            "train_mse": last_epoch["train_mse"],
            "train_pearson": last_epoch["train_pearson"],
            "train_loss": last_epoch["train_loss"],
            "minutes_train": last_epoch["minutes"],
            "resumed": 1 if was_interrupted else 0}
