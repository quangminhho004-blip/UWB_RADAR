# Sơ đồ các nhánh thực nghiệm

Đọc **từ trái sang phải**. Một cấu hình có thể tách thành nhiều RF; một cấu hình/RF có thể giữ nhiều loss để đi tiếp.

**Hai mô hình cuối:** ưu tiên **64/RF121 + Pearson**, sau đó **64/RF61 + hybrid α = 0,6**. Vị trí các hàng trong sơ đồ thể hiện đường đi thực nghiệm, không phải thứ tự ưu tiên.

```text
TN1                          TN2                          TN3                          TN4
Chọn cấu hình                Khảo sát RF                  Khảo sát loss                Đánh giá cuối

                             ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
                             │       64/RF61       │      │     Hybrid loss     │      │  64/RF61 + Hybrid   │
                         ┌──▶│      Kernel 3       ├─────▶│     alpha = 0,6     ├─────▶│    Đã test GHIJ     │
                         │   │  3 seed · dùng TN1  │      │   1 seed · 4 fold   │      │   3 seed: 0, 1, 2   │
                         │   └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                         │
┌─────────────────────┐  │
│   DS-TCN 64/RF61    │  │
│    Nền được giữ     ├──┤                                ┌─────────────────────┐      ┌─────────────────────┐
│   3 seed · 4 fold   │  │                                │    Pearson loss     │      │ 64/RF121 + Pearson  │
└─────────────────────┘  │                            ┌──▶│      alpha = 0      ├─────▶│    Đã test GHIJ     │
                         │                            │   │   1 seed · 4 fold   │      │   3 seed: 0, 1, 2   │
                         │   ┌─────────────────────┐  │   └─────────────────────┘      └─────────────────────┘
                         │   │      64/RF121       │  │
                         └──▶│      Kernel 5       ├──┤
                             │   1 seed · 4 fold   │  │
                             └─────────────────────┘  │   ┌─────────────────────┐      ┌─────────────────────┐
                                                      │   │         MSE         │      │   64/RF121 + MSE    │
                                                      └──▶│      Đối chứng      ├─────▶│    Đã test GHIJ     │
                                                          │  1 seed · dùng TN2  │      │   3 seed: 0, 1, 2   │
                                                          └─────────────────────┘      └─────────────────────┘



┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   DS-TCN 192/RF61   │      │      192/RF121      │      │     Hybrid loss     │      │ 192/RF121 + Hybrid  │
│    Nền được giữ     ├─────▶│      Kernel 5       ├─────▶│     alpha = 0,2     ├─────▶│    Đã test GHIJ     │
│   3 seed · 4 fold   │      │   1 seed · 4 fold   │      │   1 seed · 4 fold   │      │   3 seed: 0, 1, 2   │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
```

**Cách đọc:** `64/RF121` = DS-TCN có **64 kênh đặc trưng**, receptive field **121 mẫu**. Các mũi tên thể hiện đường đi của cấu hình qua thực nghiệm.

**Quy ước seed/fold trong các ô:**

- **TN1:** 3 seed `0, 1, 2`, mỗi seed chạy 4 fold.
- **TN2 — RF121:** 1 seed `0`, chạy 4 fold cho mỗi cấu hình. **64/RF61 dùng lại kết quả nền TN1: 3 seed × 4 fold**, không phải chạy thêm 3 seed ở TN2.
- **TN3 — khảo sát loss:** 1 seed `0` × 4 fold cho mỗi alpha của từng cấu hình/RF. **MSE đối chứng dùng mốc TN2: 1 seed × 4 fold**, không tính thành một lượt khảo sát loss mới.
- **TN4:** mỗi tổ hợp train/test với 3 seed `0, 1, 2`. Train trên ABCDEFKL, test GHIJ; **không chạy 4 fold ở TN4**.

## Mỗi bước làm gì?

| Bước | Việc thực hiện | Các nhánh được giữ |
|---|---|---|
| **TN1** | Khảo sát các cấu hình TCN/DS-TCN để chọn nền nghiên cứu tiếp. | DS-TCN 64/RF61 và 192/RF61. |
| **TN2** | Thay kernel để khảo sát receptive field trong từng nhóm số kênh. | 64/RF61, 64/RF121 và 192/RF121. |
| **TN3** | Khảo sát loss riêng trên từng cấu hình/RF; giữ loss phù hợp và đối chứng cần thiết. | Hybrid α = 0,6; Pearson thuần; MSE đối chứng; Hybrid α = 0,2, theo các đường trên. |
| **TN4** | Train trên ABCDEFKL, test trên GHIJ với seed 0, 1, 2. | Bốn tổ hợp đã có output trong notebook. |

Trong TN2, RF61 là mốc nền; các notebook 4 fold khảo sát thêm kernel 5, 7, 9 tương ứng RF121, RF181, RF241. **Sơ đồ chỉ vẽ các RF thực sự được đưa sang TN3.**

Trong TN3, hybrid loss có dạng `α × MSE + (1 − α) × (1 − Pearson)`. Khi α = 0, đây là Pearson loss thuần. **MSE của 64/RF121 được giữ làm đối chứng**, không gọi là loss thắng TN3.

TN1–TN3 sử dụng validation để khảo sát và chọn cấu hình. TN4 đánh giá cuối trên GHIJ. Mũi tên không hàm ý tiếp tục train từ checkpoint của bước trước.

## Notebook nguồn

| Đường đi | TN1 — nền | TN2 — RF | TN3 — loss | TN4 — test |
|---|---|---|---|---|
| **64/RF61** | [64/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c64.ipynb) | Giữ RF nền | [Loss 64/RF61](../notebooks/TN3_HybridLoss_DS_TCN_c64.ipynb) | [Hybrid α = 0,6](../notebooks/TN4_final_test_ds_tcn_c64.ipynb) |
| **64/RF121** | Chung nền 64/RF61 | [RF nhóm 64](../notebooks/TN2_ReceptiveField_DS_TCN_c64_4fold.ipynb) | [Loss 64/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c64_rf121.ipynb) | [Pearson và MSE](../notebooks/TN4_final_test_ds_tcn_c64_rf121.ipynb) |
| **192/RF121** | [192/RF61](../notebooks/TN1_DS_TCN_RF61_no_norm_do02_c192.ipynb) | [RF nhóm 192](../notebooks/TN2_ReceptiveField_DS_TCN_c192_4fold.ipynb) | [Loss 192/RF121](../notebooks/TN3_HybridLoss_DS_TCN_c192.ipynb) | [Hybrid α = 0,2](../notebooks/TN4_final_test_ds_tcn_c192.ipynb) |

Các cấu hình khảo sát khác và điểm số xem [BANG_TCN.md](BANG_TCN.md).
