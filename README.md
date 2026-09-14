# Phân đoạn Tín hiệu Tiếng nói và Khoảng lặng (Speech / Silence Segmentation)

Dự án Xử lý tín hiệu số: Phân đoạn tín hiệu thu âm thành **tiếng nói (speech)** và **khoảng lặng (silence)**.
Triển khai dựa trên 2 thuật toán tiêu chuẩn:
1. **Thuật toán 1**: Năng lượng ngắn hạn ($\text{MA}/\log\text{MA}$) kết hợp **Tìm kiếm nhị phân (Binary Search)** theo *Hodgkinson (2012, Sec 2.1)*.
2. **Thuật toán 2**: Phân đoạn dựa trên **2 đặc trưng Histogram: Năng lượng (Energy) & Trọng tâm phổ (Spectral Centroid qua DFT)** theo chuẩn *Theodoros Giannakopoulos (2014)*.

---

## 1. Cấu trúc Dự án (Clean Architecture)

```
SP/
├── main.py                       # Điểm chạy chính 1 lần duy nhất cho Giảng viên chấm thi
├── requirements.txt              # Danh sách thư viện phụ thuộc (numpy, scipy, matplotlib)
├── README.md                     # Tài liệu hướng dẫn sử dụng
│
├── src/                          # Gói mã nguồn chính (tự cài đặt trên NumPy thuần)
│   ├── config.py                 # Cấu hình khung (20ms/10ms), lọc 300ms, dataclass
│   ├── audio/                    # Xử lý âm thanh vào/ra và tính SNR
│   │   ├── io.py                 # load_audio, parse_lab_file, load_ground_truth
│   │   └── snr.py                # compute_snr_db
│   ├── features/                 # Trích xuất đặc trưng ngắn hạn
│   │   ├── short_time.py         # compute_short_time_feature (MA, logMA, STE, logSTE)
│   │   └── spectral.py           # compute_energy_and_spectral_centroid (Energy & Spectral Centroid DFT)
│   ├── algorithms/               # Các thuật toán phân đoạn
│   │   ├── base.py               # Lớp cơ sở trừu tượng BaseSegmenter
│   │   ├── binary_search.py      # Thuật toán 1: Hodgkinson 2012 (Binary Search)
│   │   └── histogram.py          # Thuật toán 2: Giannakopoulos 2014 (Histogram 2-Feature)
│   ├── segmentation/             # Hậu xử lý và đánh giá
│   │   ├── postprocess.py        # frames_to_segments, remove_short_silence (300ms)
│   │   └── metrics.py            # evaluate_boundaries (MAE, RMSE, F1-Score)
│   └── visualization/            # Trực quan hóa đồ thị
│       ├── plots.py              # Vẽ dạng sóng, hàm đặc trưng, biên đỏ (GT) & xanh
│       └── layout.py             # Sắp xếp 4 figure vào 4 góc màn hình
│
├── scripts/                      # Kịch bản huấn luyện và khảo sát
│   ├── train_algo1.py            # Huấn luyện & khảo sát Thuật toán 1
│   ├── train_algo2.py            # Huấn luyện & khảo sát Thuật toán 2 (Energy & Centroid)
│   └── export_summary.py         # Xuất bảng tổng hợp kết quả ra output/evaluation_summary.csv
│
├── tests/                        # Kiểm thử tự động (Unit Tests)
│   ├── test_audio_io.py          # Test đọc WAV, parse file .lab
│   ├── test_features.py          # Test framing, tính MA/STE/logMA
│   ├── test_spectral.py          # Test tính Energy và Spectral Centroid qua DFT
│   ├── test_binary_search.py     # Test hội tụ thuật toán tìm kiếm nhị phân
│   └── test_histogram.py         # Test lọc trung vị và phân tích histogram
│
├── TinHieuHuanLuyen/             # 4 cặp file .wav và .lab huấn luyện
├── TinHieuKiemThu/               # Dữ liệu kiểm thử khi chấm thi
└── output/                       # Kết quả xuất ra
    ├── figures/                  # Lưu ảnh biểu đồ phân đoạn và phân phối
    ├── global_threshold.json     # Cấu hình ngưỡng tối ưu Thuật toán 1
    ├── histogram_threshold.json  # Cấu hình ngưỡng tối ưu Thuật toán 2 (T1 & T2)
    └── evaluation_summary.csv    # Bảng tổng hợp số liệu thực nghiệm
```

---

## 2. Hướng dẫn Cài đặt Môi trường

1. **Kích hoạt môi trường ảo Python (đã tạo sẵn trong `venv`)**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Cài đặt thư viện (nếu thiết lập môi trường mới)**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 3. Hướng dẫn Chạy Chương trình

### A. Chạy Demo chấm thi cho Giảng viên:
> Giảng viên yêu cầu bấm Run **chạy 01 lần duy nhất** từ `main.py`, duyệt qua 4 file kiểm thử và hiển thị 4 figure tại 4 góc màn hình:

```powershell
# Chạy Thuật toán 1 (Hodgkinson 2012 - Binary Search):
python main.py --algo 1

# Chạy Thuật toán 2 (Giannakopoulos 2014 - Histogram 2-Feature: Energy & Centroid):
python main.py --algo 2

# Chạy chế độ Ngưỡng toàn cục:
python main.py --algo 2 --mode global

# Chạy chế độ tự động lưu ảnh (không mở cửa sổ GUI):
python main.py --algo 1 --no-gui
python main.py --algo 2 --no-gui
```

### B. Chạy Huấn luyện & Khảo sát Tham số Tối ưu:
```powershell
# Huấn luyện Thuật toán 1 (Hodgkinson 2012 - Binary Search):
python scripts/train_algo1.py

# Huấn luyện Thuật toán 2 (Giannakopoulos 2014 - Energy & Centroid):
python scripts/train_algo2.py

# Xuất bảng tổng hợp kết quả định lượng ra CSV:
python scripts/export_summary.py
```

### C. Chạy Bộ kiểm thử tự động (Unit Tests):
```powershell
python -m unittest discover tests
```

---

## 4. Kết quả Thực nghiệm & So sánh Giữa 2 Thuật Toán

### A. Chế độ Ngưỡng Toàn Cục (`--mode global`)

| File Âm Thanh | Kênh / Môi trường | SNR (dB) | Thuật toán 1 (Hodgkinson 2012)<br>$T_{\text{global}} = -5.2847$ ($\log\text{MA}$) | Thuật toán 2 (Giannakopoulos 2014)<br>$T_1 = 0.01193, T_2 = 32.4$ |
| :--- | :--- | :---: | :---: | :---: |
| **`phone_F1`** | Điện thoại (Nữ) | **23.8** | F1: `0.0%` *(Nhiễu nền lớn)* | **MAE: 5.0 ms \| RMSE: 5.0 ms \| F1: 100.0%** |
| **`phone_M1`** | Điện thoại (Nam) | **23.3** | **MAE: 10.0 ms \| RMSE: 11.2 ms \| F1: 100.0%** | **MAE: 45.0 ms \| RMSE: 46.1 ms \| F1: 50.0%** |
| **`studio_F1`**| Phòng thu (Nữ) | **38.8** | **MAE: 5.0 ms \| RMSE: 5.0 ms \| F1: 100.0%** | **MAE: 15.0 ms \| RMSE: 18.0 ms \| F1: 100.0%** |
| **`studio_M1`**| Phòng thu (Nam) | **37.8** | **MAE: 10.0 ms \| RMSE: 11.2 ms \| F1: 100.0%** | **MAE: 55.0 ms \| RMSE: 62.6 ms \| F1: 66.7%** |

### B. Chế độ Ngưỡng Động Thích Nghi (`--mode dynamic` - Giannakopoulos 2014 Chuẩn)
> Tính 2 ngưỡng động độc lập $T_1$ (Energy) và $T_2$ (Spectral Centroid) trực tiếp cho từng file bằng phân tích 2 đỉnh Histogram ($M_1, M_2$), không cần nhãn Ground Truth:

| File Âm Thanh | Kênh / Môi trường | SNR (dB) | Ngưỡng động ($T_1, T_2$) | MAE (ms) | RMSE (ms) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`phone_F1`** | Điện thoại (Nữ) | **23.8** | $T_1=0.00361, T_2=18.9$ | **15.0 ms** | **15.0 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`phone_M1`** | Điện thoại (Nam) | **23.3** | $T_1=0.00041, T_2=29.5$ | **25.0 ms** | **26.9 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`studio_F1`**| Phòng thu (Nữ) | **38.8** | $T_1=0.00029, T_2=34.3$ | **35.0 ms** | **35.0 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`studio_M1`**| Phòng thu (Nam) | **37.8** | $T_1=0.00036, T_2=36.8$ | **10.0 ms** | **11.2 ms** | **100.0%** | **100.0%** | **100.0%** |


