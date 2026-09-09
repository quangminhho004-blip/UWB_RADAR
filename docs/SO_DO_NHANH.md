# Kiến trúc nào đi tiếp qua bốn thực nghiệm

Chỉ vẽ **đường đi**, không có điểm số. Điểm xem `docs/BANG_NHANH_TCN.md`.

`──►` được chọn, đi tiếp.   `╳` dừng lại.


## Sơ đồ

```
        TN1                  TN2                TN3              TN4
   chọn kiến trúc        chọn tầm nhìn       chọn hàm loss     test GHIJ
═══════════════════════════════════════════════════════════════════════════

   TCN-64 BatchNorm    ╳

   TCN-64 WeightNorm   ╳

   ModernTCN-32        ╳

   DS-TCN-nền          ──────────────────────────────────►  mốc so

   DS-TCN-nền + RevIN                 ╳   (nhánh phụ TN2)

   TCN-64 + RevIN                     ╳   (chưa chạy)

   Ours-192/61         ──►  đối chứng      ╳
                           kích cỡ khác

   Ours-64/61  ★       ──►  nền TN2   ──►  alpha 0,6   ──►  ĐÃ CHẠY
                    (không chạy lại,
                     chỉ đổi kernel)
                           │
                           │  kernel 3 → 5, 7, 9
                           ▼
            ┌──────────────┴───────────────┐
            │                              │
      64 kênh:                        192 kênh:

      Ours-64/121  ★ ──►               Ours-192/121 ──►
      Ours-64/181    ╳                 Ours-192/181   ╳
      Ours-64/241    ╳                 Ours-192/241   ╳
            │                              │
            │  alpha 0,0                   │  alpha 0,2
            ▼                              ▼
         ĐÃ CHẠY  ★ ĐỀ XUẤT            ĐÃ CHẠY
```


## Ba bậc, đọc gọn

**TN1 — sáu kiến trúc họ TCN, chọn `Ours-64/61`**

```
   TCN-64 BatchNorm    ╳       ModernTCN-32   ╳
   TCN-64 WeightNorm   ╳       Ours-192/61    ╳  (xuống TN2 làm đối chứng rồi dừng)
   DS-TCN-nền          ╳  (đi thẳng tới TN4 làm mốc so)
   Ours-64/61     ★    ──►  đi tiếp
```

**TN2 — quét tầm nhìn từ `Ours-64/61`, chỉ đổi kernel**

```
   64 kênh:    Ours-64/121  ★ ──►     Ours-64/181  ╳     Ours-64/241  ╳
   192 kênh:   Ours-192/121   ──►     Ours-192/181 ╳     Ours-192/241 ╳
   nhánh phụ:  DS-TCN-nền + RevIN  ╳
```

→ **Hai cấu hình vào TN3:** `Ours-64/121`, `Ours-192/121`
   Cộng `Ours-64/61` mang thẳng từ TN1 → **TN3 có ba đầu vào**

**TN3 — quét hàm loss cho ba cấu hình, cả ba đi tiếp**

```
   Ours-64/61    ──►  alpha 0,6   ──►  TN4
   Ours-64/121   ──►  alpha 0,0   ──►  TN4   ★ ĐỀ XUẤT
   Ours-192/121  ──►  alpha 0,2   ──►  TN4
```

**TN4 — test GHIJ, ba cấu hình, ba seed mỗi cái, tất cả ĐÃ CHẠY**


## Bảng tên notebook

| tên trong sơ đồ | notebook |
|---|---|
| TCN-64 BatchNorm · WeightNorm | `TN1_TCN_DSTCN_model_selection` |
| ModernTCN-32 | `TN1_ModernTCN` |
| DS-TCN-nền | `TN1_TCN_DSTCN_model_selection` → `TN1_final_evaluation` |
| DS-TCN-nền + RevIN | `TN2_DS_TCN_RevIN` |
| TCN-64 + RevIN | `TN2_TCN_RevIN` |
| Ours-64/61 | `TN1_DS_TCN_RF61_no_norm_do02_c64` |
| Ours-192/61 | `TN1_DS_TCN_RF61_no_norm_do02_c192` |
| Ours-64/121 · 64/181 · 64/241 | `TN2_ReceptiveField_DS_TCN_c64_4fold` |
| Ours-192/121 · 192/181 · 192/241 | `TN2_ReceptiveField_DS_TCN_c192_4fold` |
| TN3 Ours-64/61 | `TN3_HybridLoss_DS_TCN_c64` |
| TN3 Ours-64/121 | `TN3_HybridLoss_DS_TCN_c64_rf121` |
| TN3 Ours-192/121 | `TN3_HybridLoss_DS_TCN_c192` |
| TN4 Ours-64/61 | `TN4_final_test_ds_tcn_c64` |
| TN4 Ours-64/121 | `TN4_final_test_ds_tcn_c64_rf121` |
| TN4 Ours-192/121 | `TN4_final_test_ds_tcn_c192` |


## Đường của cấu hình đề xuất

```
   TN1                TN2               TN3              TN4
   Ours-64/61   ──►   kernel 3→5   ──►  alpha 0,0   ──►  test GHIJ
   (kernel 3)         Ours-64/121                        Ours-64/121  ★
```
