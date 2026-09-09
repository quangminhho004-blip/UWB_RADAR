# Bàn giao artifact thực nghiệm — nhánh submission

**Đề tài:** Contactless and Robust Respiration Monitoring based on UWB Radar.

**Phạm vi:** code và bằng chứng thực nghiệm TCN/DS-TCN, từ TN0 đến TN4. Phần demo do thành viên khác phụ trách, không nằm trong danh mục này.

**Hai mô hình cuối, theo thứ tự ưu tiên:**

1. DS-TCN **64/RF121**, Pearson thuần (`mse_pearson`, alpha 0), **38.105 tham số**.
2. DS-TCN **64/RF61**, hybrid alpha 0,6, **37.081 tham số**.

Đây là thứ tự sử dụng/trình bày, không phải xếp hạng điểm test. Giữ kết quả cả ba seed và các đối chứng.

## 1. Trạng thái bàn giao

Tài liệu được lập từ notebook và script trong repo. **Tên ZIP dưới đây là tên code tạo hoặc mẫu tên cần tìm; chưa xác nhận toàn bộ ZIP hiện có trên Google Drive.** Không đánh dấu hoàn tất chỉ vì notebook có output.

| Mục người đóng gói cần điền | Trạng thái |
|---|---|
| Link tải ZIP code hoặc permalink GitHub tại commit nộp | **Chưa điền** — chốt sau khi commit bản bàn giao. |
| Mã commit đầy đủ của bộ code nộp | **Chưa điền** — lấy bằng `git rev-parse HEAD` khi xuất gói. |
| Link Drive chứa dữ liệu đã xử lý | **Chưa cung cấp/xác minh quyền tải**. |
| Link Drive chứa ZIP kết quả TN0–TN4 | **Chưa cung cấp/xác minh quyền tải**. |
| Người kiểm tra, ngày kiểm tra và các gói còn thiếu | **Chưa điền**. |

`/content/drive/MyDrive/mobivital/` là đường dẫn làm việc trong Colab, **không phải link tải gửi hội đồng**. Người sở hữu Drive cần điền link chia sẻ thật và thử mở bằng tài khoản được cấp quyền trước khi nộp. Không sửa bảng trạng thái thành “đủ” nếu chưa kiểm nội dung ZIP.

## 2. Bộ code nộp gồm gì?

| Thành phần | Vai trò |
|---|---|
| [README.md](README.md) | Điểm bắt đầu, setup và đường dẫn đọc tiếp. |
| [docs/THESIS.md](docs/THESIS.md) | Quy trình setup → dữ liệu → train/test → kết quả, cấu hình và lý do lựa chọn. |
| [docs/README.md](docs/README.md) | Danh mục tài liệu được giữ. |
| [docs/SUBMISSION_NOTEBOOKS.json](docs/SUBMISSION_NOTEBOOKS.json) | Danh mục **17 notebook**, có TN0, không có hai notebook chưa có output đã loại. |
| `src/`, `scripts/` | Logic model, loss, train, chọn ứng viên; chuẩn bị và đóng gói kết quả. |
| `notebooks/` | Lệnh và output thực nghiệm; lưu output để người chấm đọc mà không phải train lại. |
| [data/checksums.txt](data/checksums.txt) | Mốc mã băm nội dung dữ liệu `by_user`; không phải checksum ZIP hay checkpoint. |
| `runs/tn0/` và kết quả nhỏ đang được Git theo dõi | Bằng chứng đã có trong repo; không thay thế toàn bộ ZIP kết quả. |
| `ARTIFACTS.md` | Hướng dẫn bàn giao này; điền link và trạng thái trước khi chốt gói. |

**Xuất đúng code đã commit**, chạy ở thư mục gốc repo sau khi kiểm tra đang ở nhánh `submission`:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
git archive --format=zip --prefix=THESIS_EXPERIMENTS/ --output=/tmp/THESIS_EXPERIMENTS_code.zip HEAD
```

Lệnh `git archive` lấy cây tệp của commit HEAD, không lấy sửa đổi chưa commit, `.git/`, dữ liệu bị ignore hoặc `.submission_archive/`. Nó cũng không tải artifact từ Drive. Ghi commit vừa xuất vào thông tin bàn giao cùng tên ZIP. Không zip toàn bộ Desktop/project vì sẽ lẫn dữ liệu và bản khảo sát ngoài phạm vi.

## 3. Dữ liệu — lấy ở đâu, giải nén vào đâu?

Nguồn raw theo script: Zenodo record **15022885**, `tripod.zip`; tải bằng [scripts/download_dataset.py](scripts/download_dataset.py). Mã MobiVital được [setup_colab.py](scripts/setup_colab.py) clone riêng tại commit `4319731d2769d4134c92088dd846666e262f18e9`.

| Gói cần tìm trong Drive `MyDrive/mobivital/` | Nội dung | Sau khi khôi phục, tính từ gốc repo |
|---|---|---|
| `by_user.tar` | 12 NPZ A–L; mỗi file có `uwb`, `gt`, `files` | `data/processed/by_user/<người>.npz` |
| `windows.tar.gz` | Windows dev theo người và train gộp | `data/processed/windows/dev_cv/` và `data/processed/windows/final_train/` |

Sau mount Drive và setup, chạy:

```bash
python scripts/restore_processed_data_on_drive.py
```

Nếu giải nén thủ công, chạy từ gốc repo:

```bash
mkdir -p data/processed
tar -xf /content/drive/MyDrive/mobivital/by_user.tar -C data/processed
tar -xzf /content/drive/MyDrive/mobivital/windows.tar.gz -C data/processed
```

Kiểm đủ 12 file `by_user`, 8 file windows dev A–F/K/L và `windows/final_train/train_corr0.9_h200_f25.npz`. GT trong `by_user` đã chuẩn hóa; radar I/Q chưa chuẩn hóa, được biến đổi/chuẩn hóa khi tạo ứng viên.

TN1–TN4 dùng được các gói xử lý này. **TN0 phía tác giả còn cần CSV và `external/mobivital/data_final/*.npy`**, không nằm trong hai gói trên; làm đúng bước chuẩn bị trong THESIS.md. Khi kiểm checksum, giữ bản mốc trước vì `scripts/checksums.py` ghi đè `data/checksums.txt`.

## 4. TN0 — ưu tiên kiểm tra đủ bằng chứng

Notebook nguồn: [TN0.ipynb](notebooks/TN0.ipynb). Hai bên được đối chiếu trên **537 phiên GHIJ**, điểm **micro theo phiên**. Code hiện có bản vá công khai `model.eval()` phía inference tác giả; tái chạy clone mới cần làm bước áp dụng bản vá được ghi trong THESIS.md.

| Phép | Bằng chứng cần có | Mốc output đang lưu |
|---|---|---|
| TN0a — cùng lựa chọn tác giả | `TN0a.txt`, `scores_TN0a.csv`, `scores_ours_a.csv` | Hai phía 0,819481. |
| TN0b — cùng checkpoint | `TN0b.txt`, `ours_b.txt`, `scores_TN0b.csv`, `scores_ours_b.csv`; checkpoint công bố hoặc nguồn tải đúng bản | Hai phía 0,822175; trùng 537/537 cặp `(bin, method)`. |
| TN0c — train lại | `TN0c.txt`, `ours_c.txt`, `scores_TN0c.csv`, `scores_ours_c.csv`; checkpoint và log của từng phía nếu bàn giao khả năng chấm lại | Phía tác giả 0,812839, phía project 0,822642; kết quả tham khảo. |
| Đối chiếu tổng | `compare.csv`, output notebook, `README.txt` | Kết quả kiểm tra a/b và tham khảo c. |
| Kiểm dữ liệu | Output `check_data.py`, checksum | Đối chiếu 1.289 dev và 537 test giữa dữ liệu tác giả/project. |

**Có thật trong `runs/tn0/` của repo khi lập tài liệu:** `README.txt`, `TN0a.txt`, `TN0b.txt`, `scores_TN0a.csv`, `scores_TN0b.csv`, `scores_ours_a.csv`.

**Cần tìm trong `tn0.zip` trên Drive:** các TXT/CSV còn lại, `compare.csv`, checkpoint train lại và curve. Output notebook có kết quả không có nghĩa những tệp này đã được đưa lên Git.

Checkpoint tác giả công bố nằm tại `external/mobivital/checkpoints/lstm_pred_tripod_0.9.pth`; checkpoint TN0c phía project dự kiến ở `runs/tn0/ours_c/final.pth`. Checkpoint TN0c phía tác giả được tạo bên trong repo tác giả: phải kiểm và sao lưu riêng nếu không có trong ZIP. `save_results.py tn0` chỉ đóng gói **`runs/tn0/`**, không tự lấy checkpoint từ `external/`.

Không dùng TN0 micro để trừ trực tiếp với TN4 macro. Tài liệu [notebooks/TN0.md](notebooks/TN0.md) có ghi chú lịch sử đã gắn nhãn outdated; dùng bảng hiện tại trong THESIS.md khi viết báo cáo.

## 5. Danh mục ZIP thực nghiệm cần tìm

Tất cả tên dưới theo ô lưu notebook hoặc quy tắc runner, nằm trong Drive `MyDrive/mobivital/` nếu sao lưu thành công.

| Nhóm | ZIP tổng hợp / mẫu tên | Phạm vi cần kiểm |
|---|---|---|
| TN0 | `tn0.zip` | a/b/c và các file đối chiếu nêu ở mục 4. |
| TN1 nền | `tn1_tcn_weightnorm.zip`, hoặc các ZIP `tn1_<config_id>.zip` | TCN BN, TCN WN, DS-TCN nền; mỗi cấu hình 3 seed × 4 fold. |
| TN1 RF61 | `tn1_ds_tcn_rf61_c64.zip`, `tn1_ds_tcn_rf61_c192.zip` | C64/C192, mỗi cấu hình 3 seed × 4 fold. |
| TN2 RF | `tn2_rf_c64_4fold.zip`, `tn2_rf_c192_4fold.zip` | k5/k7/k9 mỗi mức seed 0 × 4 fold. RF61 dùng lại TN1. |
| TN2 RevIN | `tn2_ds_tcn_revin.zip` | DS-TCN nền + RevIN, 3 seed × 4 fold. |
| TN3 RF61 | `tn3_ds_tcn_c64.zip` | C64/k3, alpha 0–0,9 bước 0,1, mỗi alpha seed 0 × 4 fold. |
| TN3 RF121 | `tn3_ds_tcn_c64_k5.zip` | C64/k5, cùng dải alpha và giao thức. |
| TN3 đối chứng | `tn3_ds_tcn_c192.zip` | C192/k5, cùng dải alpha và giao thức. |
| TN4 | `tn4_<run_id>.zip` | Bốn tổ hợp tại mục 6; mỗi tổ hợp 3 seed. |
| Test DS-TCN nền | `tn1_ghij.zip` hoặc ZIP tự động của experiment `tn1_ghij` | C64/RF253, MSE, 3 seed GHIJ. |
| Sàng lọc phụ | `tn2_rf_ds_tcn_c64.zip` | C64 trên riêng KL, không trộn với CV bốn fold. |
| Phép thử phụ BN/dropout | `tn_test_ds_tcn_c192.zip` | C192/RF121, BatchNorm + dropout theo kênh, seed 0 × 4 fold. |

Không cần giữ mọi ZIP trùng nếu một gói đã bao phủ đủ bằng chứng, nhưng phải đối chiếu run_id/commit/nội dung trước khi loại bản trùng. `--out` chỉ đổi tên ZIP; không lọc theo cấu hình. Một ZIP có thể chứa kết quả của nhiều cấu hình cùng experiment ở runtime đó.

**Thiếu hụt đã ghi nhận:** bảng từng seed ghi C192/RF241 có điểm từ output notebook nhưng thiếu trong ZIP đã đối chiếu. Cần lấy lại artifact đúng lượt nếu còn; nếu không tìm được, giữ trạng thái “chỉ có output notebook”, không tự tạo CSV hoặc gán checkpoint của lượt khác.

## 6. Checkpoint test cuối — ưu tiên RF121 rồi RF61

Mỗi tổ hợp cần **seed 0, 1, 2**, mỗi seed được train đủ ABCDEFKL rồi chấm 537 phiên GHIJ. Tổng bốn tổ hợp dưới là **12 lượt**, không tính mốc DS-TCN nền.

| Vai trò | Channels / kernel / RF | Loss | Tham số |
|---|---|---|---:|
| **Ưu tiên 1** | **64 / 5 / 121** | **`mse_pearson`, alpha 0** | **38.105** |
| **Ưu tiên 2** | **64 / 3 / 61** | **`mse_pearson`, alpha 0,6** | **37.081** |
| Đối chứng loss | 64 / 5 / 121 | `mse` | 38.105 |
| Đối chứng dung lượng | 192 / 5 / 121 | `mse_pearson`, alpha 0,2 | 310.873 |

Cả bốn dùng 4 block, dilation 1/2/4/8, không norm, dropout **theo phần tử 0,2**, RevIN tắt, input/output 200→25. Chi tiết train trong THESIS.md.

**Mẫu run_id, thay `<s>` bằng 0, 1 hoặc 2:**

```text
ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed<s>
ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed<s>
ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed<s>
ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed<s>
```

Đường dẫn checkpoint sau khi giải nén là `runs/tn4/<run_id>/final.pth`; ZIP tự động là `tn4_<run_id>.zip`. Giữ nguyên tên để biết model, RF, loss và seed; không gom thành các file `best.pth` không rõ nguồn.

`final.pth` là state_dict, cần dựng đúng kiến trúc bằng `src/models.py` để load. Không dùng checkpoint CV của fold KL thay cho checkpoint TN4 train đủ dev. Không chọn seed có điểm GHIJ cao nhất làm điểm đại diện duy nhất; kết quả chính là trung bình ± độ lệch chuẩn ba seed.

## 7. Kiểm tra bên trong ZIP trước khi đánh dấu đủ

ZIP kết quả do `save_results.py` tạo có thư mục gốc là tên experiment:

```text
tn4/                                 ví dụ experiment tn4
├── README.txt                        thông tin đóng gói
├── summary.csv                       metric các lượt có trong gói
├── scores_<run_id>.csv               một dòng mỗi phiên
├── <run_id>.txt                      lựa chọn bin/method, runner test cuối
└── <run_id>/
    ├── final.pth                    trọng số sau epoch cuối
    └── curve.csv                    đường cong train
```

CV tương tự nhưng run_id có hậu tố `val_AB`, `val_CE`, `val_DF`, `val_KL`; không yêu cầu TXT test cuối ở mọi lượt CV. `last.pth` phục vụ resume khi train dở, không thay `final.pth` của lượt hoàn tất.

- Kiểm ZIP mở được và thư mục gốc đúng; có đủ run_id, seed/fold dự kiến.
- Đối chiếu `summary.csv` với `scores_*.csv`: macro theo người, micro theo phiên, số phiên và tập người khớp giao thức. Mỗi lượt TN4 đủ 537 phiên GHIJ.
- Đọc `git_commit` của từng dòng summary. Commit trong `README.txt` là lúc đóng gói, không nhất thiết là commit train của các lượt đã khôi phục.
- Giữ curve, checkpoint và scores của cùng lượt; file thiếu phải ghi rõ thiếu, không thay bằng artifact khác có cùng tên cấu hình.
- Nếu có `environment.json` / `requirements_runtime.txt` thì đính kèm; các lượt cũ chưa lưu đầy đủ phiên bản môi trường phải ghi đúng tình trạng.
- Với bản bàn giao, lập danh sách tên gói, kích thước, SHA-256 và link tải. SHA-256 của ZIP phục vụ kiểm tải nguyên vẹn, khác MD5 nội dung mảng trong `data/checksums.txt`.

Ví dụ kiểm và băm một ZIP đã tải, thay đường dẫn ví dụ bằng tệp thật:

```bash
unzip -t /path/to/tn0.zip
shasum -a 256 /path/to/tn0.zip
```

Mẫu sổ bàn giao để người đóng gói điền **sau khi kiểm tệp thật**:

| Tên gói | Link tải | Dung lượng byte | SHA-256 | Run/seed/fold đã kiểm | Còn thiếu | Người/ngày kiểm |
|---|---|---:|---|---|---|---|
| `tn0.zip` | Chưa điền | — | — | Chưa kiểm Drive | Xem mục 4 | — |
| `by_user.tar` | Chưa điền | — | — | Chưa kiểm Drive | — | — |
| `windows.tar.gz` | Chưa điền | — | — | Chưa kiểm Drive | — | — |
| ZIP TN1–TN4: thêm một dòng cho mỗi gói thực tế | Chưa điền | — | — | Chưa kiểm Drive | Xem mục 5–6 | — |

## 8. Khôi phục để đọc/chạy lại

Sau clone, setup và mount Drive, làm theo [THESIS.md mục 0](docs/THESIS.md). Với một ZIP, giải nén về `runs/` ở thư mục gốc project:

```bash
unzip /path/to/tn0.zip -d runs/
```

Gói chứa `tn0/` sẽ tạo `runs/tn0/`. Không giải nén vào `runs/tn0/` lần nữa. Khi có nhiều ZIP, dùng ô khôi phục trong notebook tương ứng: giải riêng vào thư mục tạm và gộp summary theo `(experiment, run_id)`. Runner tra bảng chung `runs/summary.csv`; chỉ giải ZIP không tự tạo bảng chung này. Không để summary của ZIP cuối đè mất metric trước; nếu cùng run_id nhưng khác commit/kết quả thì giữ riêng và xác định lượt nguồn, không gộp tùy ý.

Sau khi đã khôi phục bảng chung và scores:

```bash
python scripts/run_tn0.py --compare
python scripts/compare_cv.py --experiment tn1
python scripts/compare_cv.py --experiment tn2_rf
python scripts/compare_cv.py --experiment tn3
python scripts/compare_cv.py --experiment tn4
```

Lệnh compare không thay thế artifact còn thiếu. Không tự train lại để lấp một file lịch sử bị mất mà vẫn gán cho lượt cũ; nếu cần chạy lại, ghi đó là lượt tái lập mới với commit/môi trường riêng.

## 9. Điều kiện chốt phần thực nghiệm để nộp

- [ ] ZIP code đúng commit `submission`, đủ 17 notebook trong manifest; giữ output.
- [ ] Điền link tải thật, người nhận có quyền truy cập; ghi commit code nộp.
- [ ] Kiểm TN0 a/b/c; liệt kê rõ file có và file chưa tìm được.
- [ ] Kiểm hai gói dữ liệu xử lý và nguồn raw/tải lại.
- [ ] Kiểm ZIP TN1–TN3; phân biệt sàng lọc một fold và CV bốn fold.
- [ ] Kiểm TN4: hai mô hình cuối, hai đối chứng, mỗi tổ hợp ba seed; kèm mốc nền nếu dùng trong báo cáo.
- [ ] Ghi checksum ZIP, cấu hình, nguồn commit và giới hạn môi trường của lượt cũ.
- [ ] Điểm trong thesis truy về notebook/CSV tương ứng; không trộn TN0 micro với TN4 macro.

Chỉ phần đã kiểm thật mới đánh dấu hoàn tất. File này chuẩn bị cho thành viên đóng gói; chưa phải xác nhận tất cả dữ liệu/checkpoint đã được chuyển giao.
