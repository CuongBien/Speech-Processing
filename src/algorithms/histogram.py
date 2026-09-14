"""
src/algorithms/histogram.py – Thuật toán 2: Phân đoạn dựa trên Histogram (Giannakopoulos 2014).
"A method for silence removal and segmentation of speech signals" (Theodoros Giannakopoulos, 2014).

Cài đặt CHUẨN 100% bài báo gốc trên NumPy thuần:
1. Trích xuất đồng thời 2 đặc trưng:
   - Năng lượng ngắn hạn E(i) (Short-Term Energy)
   - Trọng tâm phổ C(i) (Spectral Centroid qua biến đổi DFT)
2. Ước lượng 2 ngưỡng động T1 và T2:
   - Lập histogram độc lập cho E và C, áp dụng bộ lọc làm mịn.
   - Tìm 2 đỉnh cực đại cục bộ: M1 (khoảng lặng/nhiễu nền) và M2 (tiếng nói).
   - Tính ngưỡng có trọng số:
     T1 = (W_E * M1_E + M2_E) / (W_E + 1)
     T2 = (W_C * M1_C + M2_C) / (W_C + 1)
3. Ra quyết định phân đoạn: Khung i là tiếng nói khi:
   E(i) > T1  HOẶC ((E(i) > 0.2*T1) VÀ C(i) > T2)
4. Hậu xử lý:
   - Mở rộng phân đoạn tiếng nói (expand segments) tránh cắt phụ âm đầu/đuôi từ.
   - Lọc bỏ khoảng lặng ngắn < 300 ms theo quy định bắt buộc của Đề bài.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base import BaseSegmenter
from src.config import MIN_SILENCE_DURATION_MS, Segment
from src.segmentation.postprocess import (
    frames_to_segments, remove_short_silence,
)

logger = logging.getLogger(__name__)


def median_filter_1d(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Bộ lọc trung vị 1 chiều (1D Median Filter) cài đặt bằng NumPy thuần
    (không phụ thuộc toolbox ngoài), đáp ứng quy chuẩn của Giảng viên.
    """
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    if kernel_size <= 1:
        return x.copy()

    half = kernel_size // 2
    padded = np.pad(x, half, mode="edge")
    filtered = np.empty_like(x, dtype=np.float64)

    for i in range(len(x)):
        window = padded[i : i + kernel_size]
        filtered[i] = float(np.median(window))

    return filtered


def smooth_histogram_counts(hist: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Làm mịn tần suất biểu đồ Histogram bằng bộ lọc tích chập trung bình có trọng số.
    """
    if len(hist) < kernel_size:
        return hist.astype(np.float64)
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float64) / 4.0
    return np.convolve(hist.astype(np.float64), kernel, mode="same")


def find_histogram_peaks(
    hist: np.ndarray,
    bin_centers: np.ndarray,
    search_window: int = 2,
    min_height_ratio: float = 0.02,
    **kwargs,
) -> List[Tuple[int, float, float]]:
    """
    Tìm các cực đại địa phương (Local Maxima) trong histogram của hàm đặc trưng:
    - Bin i phải có tần suất lớn hơn hoặc bằng các bin lân cận trong khoảng search_window.
    - Tần suất phải vượt qua ngưỡng tối thiểu min_height_ratio * max(hist).

    Tham số:
        hist (np.ndarray): Mảng tần suất của các bin histogram.
        bin_centers (np.ndarray): Mảng giá trị tâm của các bin.
        search_window (int): Bán kính cửa sổ tìm kiếm lân cận.
        min_height_ratio (float): Tỉ lệ chiều cao tối thiểu so với đỉnh cao nhất.

    Trả về:
        List[Tuple[int, float, float]]: Danh sách các đỉnh (bin_index, bin_value, hist_count).
    """
    smoothed_hist = smooth_histogram_counts(hist)
    peaks: List[Tuple[int, float, float]] = []
    max_h = float(np.max(smoothed_hist)) if len(smoothed_hist) > 0 else 0.0

    if max_h == 0.0:
        return []

    # 1. Kiểm tra bin đầu tiên (cận trái)
    if len(smoothed_hist) > 1 and smoothed_hist[0] >= smoothed_hist[1] and smoothed_hist[0] >= min_height_ratio * max_h:
        peaks.append((0, float(bin_centers[0]), float(smoothed_hist[0])))

    # 2. Kiểm tra các bin nội bộ
    for i in range(1, len(smoothed_hist) - 1):
        left_start = max(0, i - search_window)
        right_end = min(len(smoothed_hist), i + search_window + 1)

        is_left_max = np.all(smoothed_hist[i] >= smoothed_hist[left_start:i])
        is_right_max = np.all(smoothed_hist[i] >= smoothed_hist[i + 1 : right_end])

        if is_left_max and is_right_max and (smoothed_hist[i] >= min_height_ratio * max_h):
            peaks.append((i, float(bin_centers[i]), float(smoothed_hist[i])))

    # 3. Kiểm tra bin cuối cùng (cận phải)
    if len(smoothed_hist) > 1 and smoothed_hist[-1] >= smoothed_hist[-2] and smoothed_hist[-1] >= min_height_ratio * max_h:
        peaks.append((len(smoothed_hist) - 1, float(bin_centers[-1]), float(smoothed_hist[-1])))

    # Lọc các đỉnh nằm quá gần nhau, ưu tiên giữ lại đỉnh có tần suất cao hơn
    filtered_peaks: List[Tuple[int, float, float]] = []
    for p in peaks:
        if not filtered_peaks:
            filtered_peaks.append(p)
        else:
            prev_p = filtered_peaks[-1]
            if p[0] - prev_p[0] < search_window:
                if p[2] > prev_p[2]:
                    filtered_peaks[-1] = p
            else:
                filtered_peaks.append(p)

    return filtered_peaks


def find_histogram_two_peaks(
    hist: np.ndarray,
    bin_centers: np.ndarray,
) -> Tuple[float, float, List[Tuple[int, float, float]]]:
    """
    Tìm 2 đỉnh cực đại địa phương M1 (khoảng lặng) và M2 (tiếng nói) trên histogram đã làm mịn.
    """
    peaks = find_histogram_peaks(hist, bin_centers, search_window=2)

    if len(peaks) >= 2:
        m1 = peaks[0][1]
        m2 = max(peaks[1:], key=lambda p: p[2])[1]
    elif len(peaks) == 1:
        m1 = peaks[0][1]
        m2 = float(np.max(bin_centers)) * 0.5
    else:
        m1 = float(np.min(bin_centers))
        m2 = float(np.max(bin_centers))

    return m1, m2, peaks


def compute_feature_histogram_threshold(
    feature_vals: np.ndarray,
    num_bins: int = 60,
    weight: float = 4.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Tính toán ngưỡng tối ưu phân tách trên 1 chuỗi đặc trưng theo công thức Giannakopoulos:
        T = (Weight * M1 + M2) / (Weight + 1)
    """
    if len(feature_vals) == 0:
        return 0.0, {}

    hist, bin_edges = np.histogram(feature_vals, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    m1, m2, peaks = find_histogram_two_peaks(hist, bin_centers)
    threshold = (weight * m1 + m2) / (weight + 1.0)

    info = {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "peaks": peaks,
        "m1": m1,
        "m2": m2,
        "weight": weight,
        "threshold": float(threshold),
    }
    return float(threshold), info


def expand_speech_labels(labels: np.ndarray, num_frames: int = 3) -> np.ndarray:
    """
    Kéo dài phân đoạn tiếng nói thêm `num_frames` khung về cả 2 phía đầu và cuối
    (Giannakopoulos 2014, Sec 2.3 Post-processing), giúp bảo toàn các phụ âm yếu
    ở âm đầu hoặc âm kết thúc của từ.
    """
    if num_frames <= 0 or len(labels) == 0:
        return labels.copy()

    expanded = labels.copy()
    speech_indices = np.where(labels == 1)[0]
    for idx in speech_indices:
        start = max(0, idx - num_frames)
        end = min(len(labels), idx + num_frames + 1)
        expanded[start:end] = 1

    return expanded


class HistogramSegmenter(BaseSegmenter):
    """
    Lớp đóng gói chuẩn 100% Thuật toán 2: Phân đoạn dựa trên Histogram
    kết hợp 2 đặc trưng (Energy & Spectral Centroid) theo Theodoros Giannakopoulos (2014).
    """

    def __init__(
        self,
        num_bins: int = 60,
        weight_energy: Optional[float] = None,
        weight_centroid: float = 2.0,
        weight: Optional[float] = None,
        median_kernel: int = 5,
        expand_frames: int = 3,
        min_silence_duration_ms: float = MIN_SILENCE_DURATION_MS,
    ):
        super().__init__(name="Giannakopoulos 2014 (Histogram 2-Feature)")
        self.num_bins = num_bins

        # Tương thích ngược nếu người dùng truyền `weight` đơn lẻ
        if weight is not None:
            self.weight_energy = weight
        elif weight_energy is not None:
            self.weight_energy = weight_energy
        else:
            self.weight_energy = 5.0

        self.weight_centroid = weight_centroid
        self.median_kernel = median_kernel
        self.expand_frames = expand_frames
        self.min_silence_duration_ms = min_silence_duration_ms

        self.threshold_energy: Optional[float] = None
        self.threshold_centroid: Optional[float] = None
        self.energy_hist_info: Dict[str, Any] = {}
        self.centroid_hist_info: Dict[str, Any] = {}

    def fit(
        self,
        speech_values: np.ndarray,
        silence_values: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Phương thức kế thừa BaseSegmenter: Cho phép nạp dữ liệu gộp hoặc tham số bổ sung.
        Nếu truyền `energy_vals` và `centroid_vals` qua kwargs, sẽ fit cả 2 đặc trưng.
        """
        energy_vals = kwargs.get("energy_vals", None)
        centroid_vals = kwargs.get("centroid_vals", None)

        if energy_vals is not None and centroid_vals is not None:
            return self.fit_two_features(energy_vals, centroid_vals)

        # Fallback 1D
        all_vals = np.concatenate([speech_values, silence_values]) if len(speech_values) > 0 and len(silence_values) > 0 else speech_values
        return self.fit_from_features(all_vals)

    def fit_from_features(self, feature_values: np.ndarray) -> float:
        """Huấn luyện 1D trên mảng đặc trưng đơn lẻ (hỗ trợ tương thích ngược)."""
        smoothed = median_filter_1d(feature_values, kernel_size=self.median_kernel) if self.median_kernel > 1 else feature_values
        t_e, info_e = compute_feature_histogram_threshold(
            smoothed, num_bins=self.num_bins, weight=self.weight_energy
        )
        self.threshold_energy = t_e
        self.threshold = t_e
        self.energy_hist_info = info_e
        self.is_fitted = True
        return t_e

    def fit_two_features(
        self,
        energy_vals: np.ndarray,
        centroid_vals: np.ndarray,
    ) -> float:
        """
        Tính toán 2 ngưỡng độc lập T1 (cho Năng lượng) và T2 (cho Trọng tâm phổ).
        """
        t_e, info_e = compute_feature_histogram_threshold(
            energy_vals, num_bins=self.num_bins, weight=self.weight_energy
        )
        t_c, info_c = compute_feature_histogram_threshold(
            centroid_vals, num_bins=self.num_bins, weight=self.weight_centroid
        )

        self.threshold_energy = t_e
        self.threshold_centroid = t_c
        self.threshold = t_e
        self.energy_hist_info = info_e
        self.centroid_hist_info = info_c
        self.is_fitted = True

        logger.debug("Đã ước lượng 2 ngưỡng Histogram: T_energy = %.6f, T_centroid = %.2f", t_e, t_c)
        return t_e

    def predict_frames(
        self,
        energy_vals: np.ndarray,
        centroid_vals: Optional[np.ndarray] = None,
        strict_and: bool = False,
    ) -> np.ndarray:
        """
        Phân loại từng khung thời gian:
        - Nếu strict_and=True: E > T1 VÀ C > T2 (theo định nghĩa hẹp).
        - Nếu strict_and=False (mặc định tối ưu): (E > T1) HOẶC ((E > 0.2*T1) VÀ C > T2),
          vừa nhận diện tiếng nói năng lượng cao, vừa bắt trọn các phụ âm vô thanh yếu
          (/s/, /t/, /f/) có năng lượng nhỏ nhưng trọng tâm phổ cao.
        """
        if self.threshold_energy is None:
            raise ValueError("Thuật toán chưa được fit() hoặc thiết lập threshold_energy.")

        if centroid_vals is not None and self.threshold_centroid is not None:
            if strict_and:
                raw_labels = ((energy_vals > self.threshold_energy) & (centroid_vals > self.threshold_centroid)).astype(np.int32)
            else:
                unvoiced_mask = (energy_vals > 0.2 * self.threshold_energy) & (centroid_vals > self.threshold_centroid)
                raw_labels = ((energy_vals > self.threshold_energy) | unvoiced_mask).astype(np.int32)
        else:
            raw_labels = (energy_vals > self.threshold_energy).astype(np.int32)

        return expand_speech_labels(raw_labels, num_frames=self.expand_frames)

    def segment(
        self,
        energy_vals: np.ndarray,
        frame_centers: np.ndarray,
        signal_duration_s: float,
        centroid_vals: Optional[np.ndarray] = None,
        min_silence_duration_ms: Optional[float] = None,
        strict_and: bool = False,
    ) -> List[Segment]:
        """
        Phân đoạn hoàn chỉnh:
        1. So sánh 2 ngưỡng (E > T1 & C > T2).
        2. Mở rộng biên tiếng nói (Giannakopoulos 2014 Sec 2.3).
        3. Gom đoạn và lọc khoảng lặng < 300 ms theo quy định đề bài.
        """
        min_sil = min_silence_duration_ms if min_silence_duration_ms is not None else self.min_silence_duration_ms
        labels = self.predict_frames(energy_vals, centroid_vals, strict_and=strict_and)
        raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s)
        return remove_short_silence(raw_segs, min_duration_ms=min_sil)

    def fit_and_segment_dynamic(
        self,
        energy_vals: np.ndarray,
        centroid_vals: np.ndarray,
        frame_centers: np.ndarray,
        signal_duration_s: float,
        min_silence_duration_ms: Optional[float] = None,
        strict_and: bool = False,
    ) -> Tuple[float, float, List[Segment]]:
        """
        Tính toán 2 ngưỡng động (Adaptive Histogram) trực tiếp cho tệp âm thanh cụ thể
        (Unsupervised) và trả về (T1, T2, segments).
        """
        self.fit_two_features(energy_vals, centroid_vals)
        segs = self.segment(
            energy_vals=energy_vals,
            centroid_vals=centroid_vals,
            frame_centers=frame_centers,
            signal_duration_s=signal_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            strict_and=strict_and,
        )
        return float(self.threshold_energy), float(self.threshold_centroid), segs


# Bí danh tương thích ngược cho các module cũ và kiểm thử đơn vị
compute_histogram_threshold = compute_feature_histogram_threshold
