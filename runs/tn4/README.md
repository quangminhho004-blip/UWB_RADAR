# TN4 — kết quả test cuối trên GHIJ

**Chỉ TN4 được nhập trong đợt này.** Train đủ ABCDEFKL, test GHIJ; bốn tổ hợp, mỗi tổ hợp seed 0/1/2. Không phải checkpoint tiếp tục từ CV.

## Kết quả tổng hợp

Macro: trung bình điểm phiên trong mỗi người rồi trung bình bốn người. Std dưới đây là sample std giữa ba seed (`ddof=1`); Pearson không phải phần trăm chính xác.

| Cấu hình | Seed 0 | Seed 1 | Seed 2 | Macro trung bình ± std |
|---|---:|---:|---:|---:|
| 64/RF121 — Pearson (ưu tiên 1) | 0.805832 | 0.790377 | 0.809010 | **0.801739 ± 0.009968** |
| 64/RF61 — hybrid 0,6 (ưu tiên 2) | 0.786906 | 0.817115 | 0.806751 | **0.803590 ± 0.015350** |
| 64/RF121 — MSE đối chứng | 0.781448 | 0.766026 | 0.739099 | **0.762191 ± 0.021433** |
| 192/RF121 — hybrid 0,2 đối chứng | 0.796041 | 0.793154 | 0.812969 | **0.800721 ± 0.010705** |

Ưu tiên sử dụng RF121 rồi RF61; RF61 hybrid có điểm trung bình cao hơn một chút. Bảng giữ nguyên số đo, không diễn giải thứ tự ưu tiên thành thứ hạng độ chính xác.

## Mở kết quả từng lượt

- [summary.csv](summary.csv): 12 dòng metric gốc đã gộp, không tính lại/ghi đè số nguồn.
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json): thư mục nguồn, SHA-256 từng tệp, số liệu kiểm lại và các bản trùng đã gộp.

Mỗi scores có 537 phiên GHIJ; mỗi curve có 20 epoch. TXT và CSV được kiểm trùng lựa chọn `(bin, method)` theo tên phiên.

| Cấu hình | Seed | Điểm từng phiên | Lựa chọn ứng viên | Đường cong train | Commit train |
|---|---:|---|---|---|---|
| 64/RF121 — Pearson (ưu tiên 1) | 0 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed0.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed0.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed0/curve.csv) | `b6829a5` |
| 64/RF121 — Pearson (ưu tiên 1) | 1 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed1.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed1.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed1/curve.csv) | `b6829a5` |
| 64/RF121 — Pearson (ưu tiên 1) | 2 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed2.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed2.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_pearson_a0_corr0.9_seed2/curve.csv) | `b6829a5` |
| 64/RF61 — hybrid 0,6 (ưu tiên 2) | 0 | [CSV](scores_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed0.csv) | [TXT](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed0.txt) | [curve.csv](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed0/curve.csv) | `bf1ac10` |
| 64/RF61 — hybrid 0,6 (ưu tiên 2) | 1 | [CSV](scores_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed1.csv) | [TXT](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed1.txt) | [curve.csv](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed1/curve.csv) | `bf1ac10` |
| 64/RF61 — hybrid 0,6 (ưu tiên 2) | 2 | [CSV](scores_ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed2.csv) | [TXT](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed2.txt) | [curve.csv](ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_pearson_a0.6_corr0.9_seed2/curve.csv) | `bf1ac10` |
| 64/RF121 — MSE đối chứng | 0 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed0.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed0.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed0/curve.csv) | `b6829a5` |
| 64/RF121 — MSE đối chứng | 1 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed1.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed1.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed1/curve.csv) | `b6829a5` |
| 64/RF121 — MSE đối chứng | 2 | [CSV](scores_ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed2.csv) | [TXT](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed2.txt) | [curve.csv](ds_tcn_c64_k5_n4_none_do0.2_dpel_mse_corr0.9_seed2/curve.csv) | `b6829a5` |
| 192/RF121 — hybrid 0,2 đối chứng | 0 | [CSV](scores_ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed0.csv) | [TXT](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed0.txt) | [curve.csv](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed0/curve.csv) | `bf1ac10` |
| 192/RF121 — hybrid 0,2 đối chứng | 1 | [CSV](scores_ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed1.csv) | [TXT](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed1.txt) | [curve.csv](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed1/curve.csv) | `bf1ac10` |
| 192/RF121 — hybrid 0,2 đối chứng | 2 | [CSV](scores_ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed2.csv) | [TXT](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed2.txt) | [curve.csv](ds_tcn_c192_k5_n4_none_do0.2_dpel_mse_pearson_a0.2_corr0.9_seed2/curve.csv) | `bf1ac10` |

## Nguồn, kiểm tra và giới hạn

Nguồn local: thư mục `Downloads/ket_qua_thuc_nghiem`, các thư mục con `tn4_*/tn4/`. Đây là dữ liệu đã giải nén; tên thư mục được ghi trong manifest, chưa xác minh ZIP gốc hoặc link Drive. Các bản chụp lặp được gộp sau khi kiểm nội dung giống nhau; không lấy bản cuối ghi đè tùy ý.

Đã kiểm: đủ 12 run, cùng tập 537 tên phiên GHIJ ở mọi run, không trùng phiên trong run; macro/micro tính lại khớp summary trong sai số 1e-12; số phiên âm khớp; TXT khớp CSV; mỗi curve đủ epoch 0–19.

Các cột `run_id` trong scores và `timestamp` trong summary có chỗ trống từ nguồn, được giữ nguyên. Run được nhận dạng qua tên file và manifest. Tên phiên trong nguồn được giữ nguyên; không đổi sang hệ tên CSV khác để tránh ghép nhầm.

**Chưa có checkpoint trong các thư mục TN4 được nhập.** Bộ này cho phép đọc/tính lại metric từ scores và xem đường cong, chưa đủ chạy lại inference từ trọng số. Cần bổ sung link checkpoint/ZIP đầy đủ qua [ARTIFACTS.md](../../ARTIFACTS.md). Không đưa raw radar, NPZ, ZIP hoặc checkpoint lên Git trong đợt này.

## Notebook nguồn

- [64/RF121 — Pearson và MSE](../../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb)
- [64/RF61 — hybrid](../../notebooks/TN4_final_test_ds_tcn_c64.ipynb)
- [192/RF121 — hybrid](../../notebooks/TN4_final_test_ds_tcn_c192.ipynb)

Đọc trực tiếp [THESIS.md](../../docs/THESIS.md) để xem thiết kế. Không trộn micro TN0 với macro TN4. `compare_cv.py` đọc bảng chung `runs/summary.csv`; nhập thư mục này không tự sửa bảng chung hoặc kết quả TN1–TN3.
