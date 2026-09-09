# Mã nguồn có gì — nhìn từ bên ngoài

Đọc file này là biết mở file nào khi cần sửa gì. Không đi vào chi tiết bên
trong hàm.


## Ba tầng, mỗi tầng một việc

    notebooks/    vài lệnh !python dễ đọc, chạy trên Colab
         |        Không có logic. Đọc là biết thí nghiệm làm gì.
         v
    scripts/      xử lý thứ tự bước, đối số dòng lệnh, lỗi, ghi kết quả
         |        Mỗi script chạy được độc lập bằng python scripts/<tên>.py
         v
    src/          model, vòng train, chấm điểm
                  Không tự chạy. Chỉ được scripts/ gọi.

Quy tắc: **logic đi xuống, không đi lên.** `src/` không biết gì về Colab hay
Drive; `scripts/` không biết gì về notebook nào gọi nó.


## src/ — sáu file

| file | làm gì |
|---|---|
| `models.py` | Mọi kiến trúc: LSTM, BiLSTM, GRU, CNN-LSTM, TCN, DS-TCN, ModernTCN, MixLinear và các biến thể. Tất cả cùng một giao diện: vào `(batch, 200)`, ra `(batch, 25)`. `build_model(tên, ...)` dựng theo tên. |
| `training.py` | Vòng train: chia batch, tính loss, Adam, in tiến độ. Lưu checkpoint mỗi epoch để Colab ngắt phiên còn chạy tiếp được. |
| `scoring.py` | Bộ chọn kênh và hàm chấm điểm. Cho mỗi buổi ghi: chạy model trên khoảng 126 ứng viên còn lại sau bộ lọc, chọn ứng viên tự dự báo chính nó tốt nhất, rồi tính Pearson với nhịp thở thật. |
| `losses.py` | MSE, Pearson, và hàm lai `alpha × MSE + (1 − alpha) × (1 − Pearson)`. |
| `results.py` | Ghi một dòng vào `runs/summary.csv`. Chỉ ghi, không tính. Có hàm tra xem một lần chạy đã có kết quả chưa. |
| `mobivital_reference.py` | Mượn lại các thành phần ổn định của MobiVital (hằng số, bộ lọc, phép biến đổi). Không sửa file nào của tác giả. |


## scripts/ — mười bốn file, chia bốn nhóm

### Chuẩn bị dữ liệu — chạy một lần

| file | làm gì |
|---|---|
| `download_dataset.py` | Tải bộ dữ liệu tripod từ Zenodo, giải nén vào thư mục MobiVital. |
| `make_npz.py` | Đọc 1874 file CSV, gom theo từng người, lưu ra `by_user/A.npz` … `L.npz`. |
| `make_windows.py` | Cắt sẵn cửa sổ 200 vào / 25 ra, lưu ra `windows/`. Cắt trước một lần cho nhanh, khỏi cắt lại mỗi lần train. |
| `restore_processed_data_on_drive.py` | Lấy `by_user/` và `windows/` từ Google Drive về, thay vì dựng lại từ đầu mỗi phiên Colab. |

### Kiểm tra — chạy trước khi tin kết quả

| file | làm gì |
|---|---|
| `check_data.py` | Chứng minh dữ liệu của đồ án **trùng từng byte** với dữ liệu MobiVital dùng. Không khớp thì mọi so sánh về sau vô nghĩa. |
| `check_model.py` | Kiểm một kiến trúc trước khi train: đếm tham số, thử một lô dữ liệu giả, kiểm gradient tới đủ mọi tham số, kiểm lưu và nạp lại khớp. |
| `checksums.py` | Băm nội dung mảng để đối chiếu dữ liệu giữa hai máy. |

### Chạy thí nghiệm

| file | làm gì |
|---|---|
| `run_cv.py` | **Chọn cấu hình.** Chạy 4 fold trên tám người ABCDEFKL. Không bao giờ đụng G H I J. Fold nào đã xong thì bỏ qua, không train lại. |
| `run_final_test.py` | **Số công bố.** Train đủ ABCDEFKL rồi chấm đúng một lần trên GHIJ. Chỉ chạy sau khi đã chốt cấu hình bằng `run_cv.py`. |
| `run_tn0.py` | Chạy pipeline đồ án và pipeline MobiVital trên cùng dữ liệu rồi đối chiếu. Đây là bước chứng minh hai bên tương đương. |

### Đọc và cất kết quả

| file | làm gì |
|---|---|
| `compare_cv.py` | In bảng so các cấu hình trong một thực nghiệm: `cv_mean`, `seed_std`, `fold_std`, điểm từng seed. |
| `save_results.py` | Nén toàn bộ kết quả một thực nghiệm thành `.zip` rồi chép sang Drive. |
| `analyze_oracle_dev.py` | Tính trần trên: điểm đạt được nếu luôn chọn đúng kênh tốt nhất. Không train, không chạy model. |
| `setup_colab.py` | Chuẩn bị môi trường Colab sau khi clone repo: cài gói, ghim đúng commit MobiVital, in thiết bị đang có. |


## notebooks/ — đặt tên theo thực nghiệm

| nhóm | nội dung |
|---|---|
| `DATA_PREPARE` | Dựng dữ liệu từ đầu. Chạy một lần. |
| `TN0`, `tn0_reproduce` | Tái lập MobiVital, chứng minh hai pipeline tương đương. |
| `TN1_*` | So chín kiến trúc ở cùng ngân sách tham số. |
| `TN2_ReceptiveField_*`, `TN2_*_RevIN` | Khảo sát tầm nhìn, và khảo sát RevIN. |
| `TN3_HybridLoss_*` | Quét mức alpha của hàm loss lai. |
| `TN4_final_test_*` | Test cuối trên GHIJ. Số công bố. |
| `TN_MixLinear_C0/C2/C3` | Khảo sát phụ về model cực nhỏ. Không cùng ngân sách với TN1. |
| `XEM_KET_QUA_TREN_DRIVE`, `CHAN_DOAN_TN3` | Công cụ: xem tệp nén trên Drive, chẩn đoán khi khôi phục thiếu dòng. |

Notebook nào cũng cùng một khuôn: gắn Drive → clone mã → lấy dữ liệu → khôi
phục kết quả đã chạy → kiểm số tham số → các ô train → bảng kết quả → ngắt phiên.


## Dữ liệu chảy từ đâu tới đâu

    external/mobivital/dataset/.../*.csv        1874 file, bản duy nhất
            |
            |  make_npz.py
            v
    data/processed/by_user/A.npz … L.npz        gom theo người
            |
            |  make_windows.py
            v
    data/processed/windows/                     cửa sổ 200 vào / 25 ra
            |
            +--> dev_cv/       -> run_cv.py         -> chọn cấu hình
            |
            +--> final_train/  -> run_final_test.py -> số công bố
                                        |
                                        v
                              runs/<thực nghiệm>/   checkpoint, điểm từng buổi ghi
                              runs/summary.csv      mỗi lần chạy một dòng


## Thư mục còn lại

| | |
|---|---|
| `data/` | Chỉ chứa thứ pipeline đồ án sinh ra. Dữ liệu thô 13 GB không đưa lên GitHub. |
| `external/mobivital/` | Mã và dữ liệu của tác giả, ghim đúng một commit. Không sửa file nào trong đây. |
| `runs/` | Kết quả từng thực nghiệm, và `summary.csv` chung của cả đồ án. |
| `docs/` | Giao thức, bảng điểm, cấu hình từng thực nghiệm, tài liệu tham chiếu. |
