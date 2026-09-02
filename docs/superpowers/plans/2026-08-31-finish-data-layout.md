# Finish Data Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn tất cấu trúc `data/processed/{by_user,mobivital_original}`, xác minh hai pipeline dữ liệu, rồi khóa đầu vào trước khi viết code huấn luyện Colab.

**Architecture:** Dữ liệu do đồ án chuẩn bị nằm theo user trong `by_user`; dữ liệu do script MobiVital nguyên bản tạo nằm trong `mobivital_original`. Thư mục `_workdir` chỉ là staging tạm của script 3 và phải biến mất sau một lần chạy thành công.

**Tech Stack:** Python 3, NumPy, filesystem symlinks, MobiVital `prep_breath_final.py`.

---

### Task 1: Xác minh migration đường dẫn

**Files:**
- Verify: `scripts/2_make_npz.py`
- Verify: `scripts/3_run_mobivital_prep.py`
- Verify: `README.md`
- Verify: `docs/PLAN.md`
- Verify: `docs/RUNBOOK.md`
- Verify: `.gitignore`

- [ ] **Step 1: Tìm đường dẫn cũ trong các file đang sử dụng**

Run:

```bash
rg -n "dev_processed|test_processed" README.md scripts docs/PLAN.md docs/RUNBOOK.md docs/PROTOCOL.md .gitignore
```

Expected: không có kết quả trong tài liệu và script đang sử dụng; lịch sử thiết kế cũ trong `docs/superpowers/specs/` có thể giữ nguyên như tài liệu lịch sử.

- [ ] **Step 2: Kiểm tra 12 file theo user**

Run:

```bash
find data/processed/by_user -maxdepth 1 -name '*.npz' | sort
```

Expected: đủ `A.npz` đến `L.npz`.

### Task 2: Hoàn tất pipeline MobiVital nguyên bản

**Files:**
- Execute: `scripts/3_run_mobivital_prep.py`
- Produce: `data/processed/mobivital_original/training_breath_tripod_data.npy`
- Produce: `data/processed/mobivital_original/testing_breath_tripod_data.npy`

- [ ] **Step 1: Chạy wrapper**

Run:

```bash
python3 scripts/3_run_mobivital_prep.py
```

Expected: script báo tìm thấy 1.874 CSV, gọi file nguyên bản và tạo hai file `.npy`.

- [ ] **Step 2: Xác minh staging đã được dọn**

Run:

```bash
test ! -e data/processed/mobivital_original/_workdir
```

Expected: exit code 0.

- [ ] **Step 3: Xác minh shape và dtype của hai mảng nối tiếp**

Run:

```bash
python3 -c 'import numpy as np; from pathlib import Path; paths=[Path("data/processed/mobivital_original/training_breath_tripod_data.npy"),Path("data/processed/mobivital_original/testing_breath_tripod_data.npy")]; [(lambda f: (lambda x,y: print(p.name,x.shape,x.dtype,y.shape,y.dtype))(np.load(f),np.load(f)))(open(p,"rb")) for p in paths]'
```

Expected:

```text
training_breath_tripod_data.npy (1289, 1500, 120) complex64 (1289, 1500) float32
testing_breath_tripod_data.npy (537, 1500, 120) complex64 (537, 1500) float32
```

### Task 3: Khóa hợp đồng dữ liệu trước khi viết training

**Files:**
- Modify: `docs/PROTOCOL.md`
- Modify: `docs/RUNBOOK.md`

- [ ] **Step 1: Ghi rõ vai trò hai nguồn processed data**

Nội dung cần có:

```text
data/processed/by_user             -> chọn fold và train/validation theo user
data/processed/mobivital_original  -> reproduce baseline nguyên bản
```

- [ ] **Step 2: Chốt test discipline**

Trong protocol, quy định `GHIJ` không được chạy cho từng cấu hình đang lựa chọn. Chỉ chạy sau khi cấu hình đã khóa, ngoại trừ TN0 dùng để reproduce bài gốc.

- [ ] **Step 3: Chốt seed và loại bỏ Optuna**

Ghi rõ mọi cấu hình thủ công dùng cùng `4 fold × 3 seed`; xóa TN Optuna nếu đồ án không thực hiện Optuna.

---

Lưu ý: workspace hiện chưa phải Git repository, nên kế hoạch này chưa có bước commit. Sau khi khởi tạo hoặc clone đúng Git repository, từng task nên được commit riêng.
