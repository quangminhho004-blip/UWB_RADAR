# Notebooks — chạy trên Google Colab (cần GPU)

Chạy `0_setup.ipynb` đầu tiên mỗi phiên Colab, rồi tới các notebook sau.

| notebook | việc |
|---|---|
| `0_setup.ipynb` | mount Google Drive + git clone repo + pip install + import mv_* |
| `1_explore.ipynb` | xem thử dữ liệu, vẽ vài session |
| `2_train_one_fold.ipynb` | train 1 fold để kiểm tra pipeline |
| `3_exp_architecture.ipynb` | thí nghiệm 1: TCN vs DS-TCN |

Dữ liệu `.npz` để sẵn trên Drive tại `mobivital/processed/`, tạo bằng
`scripts/` chạy ở máy (xem `docs/PLAN.md`).
