# Phân đoạn Tín hiệu Tiếng nói và Khoảng lặng (Speech / Silence Segmentation)

Dự án Xử lý tín hiệu số: Phân đoạn tín hiệu thu âm thành **tiếng nói (speech)** và **khoảng lặng (silence)**.
Triển khai dựa trên 2 thuật toán tiêu chuẩn:
1. **Thuật toán 1**: Năng lượng ngắn hạn ($\text{MA}/\log\text{MA}$) kết hợp **Tìm kiếm nhị phân (Binary Search)** theo *Hodgkinson (2012, Sec 2.1)*.
2. **Thuật toán 2**: Phân đoạn dựa trên **Histogram** theo *Theodoros Giannakopoulos (2014)*.

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
│   │   └── short_time.py         # compute_short_time_feature (MA, logMA, STE, logSTE)
│   ├── algorithms/               # Các thuật toán phân đoạn
│   │   ├── base.py               # Lớp cơ sở trừu tượng BaseSegmenter
│   │   ├── binary_search.py      # Thuật toán 1: Hodgkinson 2012 (Binary Search)
│   │   └── histogram.py          # Thuật toán 2: Giannakopoulos 2014 (Histogram)
│   ├── segmentation/             # Hậu xử lý và đánh giá
│   │   ├── postprocess.py        # frames_to_segments, remove_short_silence (300ms)
│   │   └── metrics.py            # evaluate_boundaries (MAE, RMSE, F1-Score)
│   └── visualization/            # Trực quan hóa đồ thị
│       ├── plots.py              # Vẽ dạng sóng, hàm đặc trưng, biên đỏ (GT) & xanh
│       └── layout.py             # Sắp xếp 4 figure vào 4 góc màn hình
│
├── scripts/                      # Kịch bản huấn luyện và khảo sát
│   ├── train_algo1.py            # Huấn luyện & khảo sát Thuật toán 1
│   └── train_algo2.py            # Huấn luyện & khảo sát Thuật toán 2
│
├── tests/                        # Kiểm thử tự động (Unit Tests)
│   ├── test_audio_io.py          # Test đọc WAV, parse file .lab
│   ├── test_features.py          # Test framing, tính MA/STE/logMA
│   └── test_binary_search.py     # Test hội tụ thuật toán tìm kiếm nhị phân
│
├── TinHieuHuanLuyen/             # 4 cặp file .wav và .lab huấn luyện
├── TinHieuKiemThu/               # Dữ liệu kiểm thử khi chấm thi
└── output/                       # Kết quả xuất ra
    ├── figures/                  # Lưu ảnh biểu đồ phân đoạn và phân phối
    └── global_threshold.json     # Cấu hình ngưỡng tối ưu
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
# Chạy chế độ Ngưỡng toàn cục chuẩn (mặc định --mode global):
python main.py --algo 1
python main.py --algo 2

# Chạy chế độ Ngưỡng động thích nghi theo file (--mode dynamic, F1 đạt 100% toàn bộ):
python main.py --algo 2 --mode dynamic
python main.py --algo 1 --mode dynamic

# Chạy chế độ tự động lưu ảnh (không mở cửa sổ GUI):
python main.py --algo 2 --no-gui
python main.py --algo 2 --mode dynamic --no-gui
```

### B. Chạy Huấn luyện & Khảo sát Tham số Tối ưu:
```powershell
# Huấn luyện Thuật toán 1 (Hodgkinson 2012 - Binary Search):
python scripts/train_algo1.py

# Huấn luyện Thuật toán 2 (Giannakopoulos 2014 - Histogram):
python scripts/train_algo2.py
```

### C. Chạy Bộ kiểm thử tự động (Unit Tests):
```powershell
python -m unittest discover tests
```

---

## 4. Kết quả Thực nghiệm & So sánh Giữa 2 Thuật Toán

### A. Chế độ Ngưỡng Toàn Cục (`--mode global`)

| File Âm Thanh | Kênh / Môi trường | SNR (dB) | Thuật toán 1 (Hodgkinson 2012)<br>$T_{\text{global}} = -5.2847$ ($\log\text{MA}$) | Thuật toán 2 (Giannakopoulos 2014)<br>$T_{\text{global}} = 0.0392$ ($\text{MA}$) |
| :--- | :--- | :---: | :---: | :---: |
| **`phone_F1`** | Điện thoại (Nữ) | **23.8** | F1: `0.0%` *(Nhiễu nền lớn)* | **MAE: 15.0 ms \| RMSE: 15.0 ms \| F1: 100.0%** |
| **`phone_M1`** | Điện thoại (Nam) | **23.3** | **MAE: 10.0 ms \| RMSE: 11.2 ms \| F1: 100.0%** | **MAE: 60.0 ms \| RMSE: 65.0 ms \| F1: 50.0%** |
| **`studio_F1`**| Phòng thu (Nữ) | **38.8** | **MAE: 5.0 ms \| RMSE: 5.0 ms \| F1: 100.0%** | **MAE: 20.0 ms \| RMSE: 25.0 ms \| F1: 100.0%** |
| **`studio_M1`**| Phòng thu (Nam) | **37.8** | **MAE: 10.0 ms \| RMSE: 11.2 ms \| F1: 100.0%** | **MAE: 55.0 ms \| RMSE: 55.0 ms \| F1: 50.0%** |

### B. Chế độ Ngưỡng Động Thích Nghi (`--mode dynamic` - Giannakopoulos 2014)
> Tính ngưỡng động trực tiếp theo từng file bằng phân tích 2 đỉnh Histogram ($M_1, M_2$), không giám sát:

| File Âm Thanh | Kênh / Môi trường | SNR (dB) | Ngưỡng động $T_{\text{dyn}}$ | MAE (ms) | RMSE (ms) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`phone_F1`** | Điện thoại (Nữ) | **23.8** | `0.03182` | **15.0 ms** | **15.0 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`phone_M1`** | Điện thoại (Nam) | **23.3** | `0.01066` | **10.0 ms** | **11.2 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`studio_F1`**| Phòng thu (Nữ) | **38.8** | `0.00759` | **5.0 ms** | **5.0 ms** | **100.0%** | **100.0%** | **100.0%** |
| **`studio_M1`**| Phòng thu (Nam) | **37.8** | `0.01290` | **40.0 ms** | **42.7 ms** | **100.0%** | **100.0%** | **100.0%** |

---

## 5. Kịch bản Huấn luyện Tham số (Scripts)

- **Thuật toán 1 (Binary Search)**:
  ```powershell
  python scripts/train_algo1.py
  ```
  Xuất cấu hình ngưỡng tối ưu ra `output/global_threshold.json` và biểu đồ phân phối overlap ra `output/figures/global_distribution_overlap.png`.

- **Thuật toán 2 (Histogram & Median Filter)**:
  ```powershell
  python scripts/train_algo2.py
  ```
  Xuất cấu hình ngưỡng histogram ra `output/histogram_threshold.json` và biểu đồ 2 đỉnh phân bố ra `output/figures/global_histogram_analysis.png`.

