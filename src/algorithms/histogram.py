"""
src/algorithms/histogram.py – Thuật toán 2: Phân đoạn dựa trên Histogram (Giannakopoulos 2014).
"A method for silence removal and segmentation of speech signals" (Theodoros Giannakopoulos, 2014).

Thuật toán gồm các bước:
1. Lọc trung vị (Median Filter) 1D làm mịn chuỗi đặc trưng ngắn hạn (bằng NumPy thuần).
2. Tính toán Histogram của đặc trưng đã làm mịn.
3. Tìm các cực đại địa phương (Local Maxima) nổi bật:
   - Đỉnh M1 (giá trị V1): Phân phối của khoảng lặng / nhiễu nền.
   - Đỉnh M2 (giá trị V2): Phân phối của tiếng nói.
4. Tính toán ngưỡng phân tách có trọng số:
   T = (Weight * V1 + V2) / (Weight + 1)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base import BaseSegmenter
from src.config import Segment
from src.segmentation.postprocess import (
    classify_frames, frames_to_segments, remove_short_silence,
)

logger = logging.getLogger(__name__)


def median_filter_1d(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Bộ lọc trung vị 1 chiều (1D Median Filter) cài đặt bằng NumPy thuần
    (không phụ thuộc toolbox ngoài), đáp ứng quy chuẩn của Giảng viên.

    Tham số:
        x (np.ndarray): Mảng tín hiệu 1D đầu vào.
        kernel_size (int): Kích thước cửa sổ lọc (số lẻ, mặc định 5).

    Trả về:
        np.ndarray: Tín hiệu sau khi đã làm mịn.
    """
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    if kernel_size <= 1:
        return x.copy()

    half = kernel_size // 2
    # Đệm 2 đầu mảng bằng giá trị biên (edge mode)
    padded = np.pad(x, half, mode="edge")
    filtered = np.empty_like(x, dtype=np.float64)

    for i in range(len(x)):
        window = padded[i : i + kernel_size]
        filtered[i] = float(np.median(window))

    return filtered


def find_histogram_peaks(
    hist: np.ndarray,
    bin_centers: np.ndarray,
    search_window: int = 2,
    min_height_ratio: float = 0.02,
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
    peaks: List[Tuple[int, float, float]] = []
    max_h = float(np.max(hist)) if len(hist) > 0 else 0.0

    if max_h == 0.0:
        return []

    for i in range(1, len(hist) - 1):
        left_start = max(0, i - search_window)
        right_end = min(len(hist), i + search_window + 1)

        is_left_max = np.all(hist[i] >= hist[left_start:i])
        is_right_max = np.all(hist[i] >= hist[i + 1 : right_end])

        if is_left_max and is_right_max and (hist[i] >= min_height_ratio * max_h):
            peaks.append((i, float(bin_centers[i]), float(hist[i])))

    # Lọc các đỉnh nằm quá gần nhau, ưu tiên giữ lại đỉnh có tần suất cao hơn
    filtered_peaks: List[Tuple[int, float, float]] = []
    for p in peaks:
        if not filtered_peaks:
            filtered_peaks.append(p)
        else:
            prev_p = filtered_peaks[-1]
            if p[0] - prev_p[0] < search_window:
                # Trùng lân cận -> giữ đỉnh cao hơn
                if p[2] > prev_p[2]:
                    filtered_peaks[-1] = p
            else:
                filtered_peaks.append(p)

    return filtered_peaks


def compute_histogram_threshold(
    feature_vals: np.ndarray,
    num_bins: int = 60,
    weight: float = 4.0,
    search_window: int = 2,
) -> Tuple[float, Dict[str, Any]]:
    """
    Tính toán ngưỡng tối ưu phân tách tiếng nói và khoảng lặng theo Giannakopoulos (2014).

    Công thức:
        T = (Weight * V1 + V2) / (Weight + 1)
        với V1 là đỉnh khoảng lặng (năng lượng thấp), V2 là đỉnh tiếng nói.

    Tham số:
        feature_vals (np.ndarray): Chuỗi đặc trưng ngắn hạn (đã làm mịn).
        num_bins (int): Số lượng cột phân chia histogram.
        weight (float): Trọng số của đỉnh khoảng lặng (mặc định 4.0).
        search_window (int): Bán kính lân cận tìm đỉnh.

    Trả về:
        Tuple[float, Dict[str, Any]]: (ngưỡng T, thông tin bổ trợ histogram).
    """
    if len(feature_vals) == 0:
        return 0.0, {}

    # Tính toán histogram
    hist, bin_edges = np.histogram(feature_vals, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Tìm các đỉnh
    peaks = find_histogram_peaks(
        hist=hist,
        bin_centers=bin_centers,
        search_window=search_window,
    )

    if len(peaks) >= 2:
        # Lấy 2 đỉnh đầu tiên (M1 là silence, M2 là speech)
        m1_idx, v1, h1 = peaks[0]
        m2_idx, v2, h2 = peaks[1]
        threshold = (weight * v1 + v2) / (weight + 1.0)
    elif len(peaks) == 1:
        # Trường hợp 1 đỉnh: ước tính ngưỡng dựa trên vị trí đỉnh và giá trị trung bình
        m1_idx, v1, h1 = peaks[0]
        threshold = float(np.mean(feature_vals)) * 0.6
        v2 = threshold
    else:
        # Trường hợp không phát hiện đỉnh rõ rệt
        threshold = float(np.mean(feature_vals)) * 0.5
        v1, v2 = threshold, threshold

    info = {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "peaks": peaks,
        "v1": v1 if len(peaks) >= 1 else None,
        "v2": v2 if len(peaks) >= 2 else None,
        "weight": weight,
        "threshold": threshold,
    }

    return float(threshold), info


class HistogramSegmenter(BaseSegmenter):
    """
    Lớp đóng gói hoàn chỉnh Thuật toán 2: Phân đoạn dựa trên Histogram
    (Theodoros Giannakopoulos, 2014).
    """

    def __init__(
        self,
        num_bins: int = 60,
        weight: float = 4.0,
        median_kernel: int = 5,
        search_window: int = 2,
    ):
        super().__init__(name="Giannakopoulos 2014 (Histogram-based)")
        self.num_bins = num_bins
        self.weight = weight
        self.median_kernel = median_kernel
        self.search_window = search_window
        self.histogram_info: Dict[str, Any] = {}

    def fit(self, speech_values: np.ndarray, silence_values: np.ndarray, **kwargs) -> float:
        """
        Huấn luyện xác định ngưỡng tối ưu từ dữ liệu đặc trưng.
        """
        if len(speech_values) > 0 and len(silence_values) > 0:
            all_vals = np.concatenate([speech_values, silence_values])
        elif len(speech_values) > 0:
            all_vals = speech_values
        else:
            all_vals = silence_values

        smoothed = median_filter_1d(all_vals, kernel_size=self.median_kernel)
        threshold, info = compute_histogram_threshold(
            feature_vals=smoothed,
            num_bins=self.num_bins,
            weight=self.weight,
            search_window=self.search_window,
        )
        self.threshold = threshold
        self.histogram_info = info
        self.is_fitted = True
        return threshold

    def fit_from_features(self, feature_values: np.ndarray) -> float:
        """
        Huấn luyện không giám sát (Unsupervised) trực tiếp trên toàn bộ chuỗi đặc trưng.
        """
        smoothed = median_filter_1d(feature_values, kernel_size=self.median_kernel)
        threshold, info = compute_histogram_threshold(
            feature_vals=smoothed,
            num_bins=self.num_bins,
            weight=self.weight,
            search_window=self.search_window,
        )
        self.threshold = threshold
        self.histogram_info = info
        self.is_fitted = True
        return threshold

    def segment(
        self,
        feature_values: np.ndarray,
        frame_centers: np.ndarray,
        signal_duration_s: float,
        min_silence_duration_ms: float = 300.0,
    ) -> List[Segment]:
        """
        Thực hiện lọc trung vị và phân đoạn hoàn chỉnh, tuân thủ quy tắc 300ms.
        """
        if self.threshold is None:
            raise ValueError("HistogramSegmenter chưa được fit() hoặc thiết lập ngưỡng threshold.")

        # Lọc trung vị làm mịn trước khi so sánh ngưỡng
        smoothed = median_filter_1d(feature_values, kernel_size=self.median_kernel)
        labels = classify_frames(smoothed, self.threshold)
        raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s)
        return remove_short_silence(raw_segs, min_duration_ms=min_silence_duration_ms)
