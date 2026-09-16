# runs/tn1 — chọn kiến trúc (4 fold · 3 seed)

Layout: `<cấu hình>/seed<N>/<fold>/{curve.csv, scores.csv, final.pth}`.
`summary.csv` giữ cột `run_id` đầy đủ để truy nguồn; dòng `fold=TONG` là cv_score.
`final.pth` là trọng số cuối mỗi fold (không kèm trạng thái Adam), giữ để chạy lại inference.

| thư mục | notebook sinh ra | config_id | cv_score (3 seed) |
|---|---|---|---:|
| `c64-rf61/` | `TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb` | ds_tcn_c64_k3_n4_none_do0.2_dpel_mse_corr0.9 | 0.760878 ± 0.003095 |
| `c192-rf61/` | `TN1_DS_TCN_RF61_no_norm_do02_c192.ipynb` | ds_tcn_c192_k3_n4_none_do0.2_dpel_mse_corr0.9 | 0.747955 ± 0.012189 |
| `DS-TCN-nen/` | `TN1_TCN_DSTCN_model_selection.ipynb` | ds_tcn_c64_mse_corr0.9 | 0.742110 ± 0.000673 |
| `TCN-64-BatchNorm/` | `TN1_TCN_DSTCN_model_selection.ipynb` | tcn_c64_mse_corr0.9 | 0.742338 ± 0.004369 |
| `TCN-64-WeightNorm/` | `TN1_TCN_DSTCN_model_selection.ipynb` | tcn_c64_weight_mse_corr0.9 | 0.746251 ± 0.002997 |
| `DS-TCN-nen+RevIN/` | `TN1_DS_TCN_RevIN.ipynb` | ds_tcn_c64_revin_mse_corr0.9 | 0.729715 ± 0.007270 |

`c64-rf61` (Ours-64/61) là cấu hình được chọn, mang xuống TN2.
Ba dòng TONG nền (BatchNorm seed 0, DS-TCN-nen seed 0 và 2) tính lại từ
`scores.csv` per-fold — khớp đầu ra `TN1_TCN_DSTCN_model_selection.ipynb`.

`DS-TCN-nen+RevIN/` gắn RevIN vào DS-TCN nền: đây là phép so kiến trúc "có RevIN
vs không", nên thuộc TN1 (chọn kiến trúc), không phải TN2 (trường tiếp nhận).
RevIN tệ hơn nền 0,0124 (0,729715 so với 0,742110) — nhánh cụt, không mang tiếp.
Layout `DS-TCN-nen+RevIN/seed<N>/<fold>/{curve.csv, scores.csv, final.pth}`.

## Tám tệp `final.pth` mang tên khoá cũ

`DS-TCN-nen/seed0/` và `TCN-64-BatchNorm/seed0/` (4 fold mỗi cấu hình) train
trước commit `d5ba646` — commit đổi tên biến trong `src/models.py` từ tiếng Việt
sang tiếng Anh. `state_dict` lưu theo tên thuộc tính, nên tám tệp này còn ghi
`tang_mot` / `tang_hai`, trong khi mã hiện tại tra `layer_one` / `layer_two`.

Trọng số nguyên vẹn: vẫn đủ 112 khoá, shape khớp từng cái. Chỉ khác cái nhãn.
Không mã nào trong đồ án nạp `final.pth`, nên chuyện này không chặn gì; điểm
trong `summary.csv` chấm lúc train, không phụ thuộc tệp trọng số.

Nếu sau này cần nạp:

```python
sd = torch.load(path, map_location="cpu", weights_only=True)
sd = {k.replace("tang_mot", "layer_one").replace("tang_hai", "layer_two"): v
      for k, v in sd.items()}
model.load_state_dict(sd)
```

seed1 và seed2 của cùng hai cấu hình nạp bình thường, không cần bước này.
