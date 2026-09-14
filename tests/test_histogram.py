"""
tests/test_histogram.py – Kiểm thử đơn vị cho Thuật toán 2: Histogram (Giannakopoulos 2014).
"""

import unittest
import numpy as np

from src.algorithms.histogram import (
    HistogramSegmenter,
    compute_histogram_threshold,
    find_histogram_peaks,
    median_filter_1d,
)


class TestHistogram(unittest.TestCase):
    """Kiểm tra các hàm lọc trung vị và phân tích histogram."""

    def test_median_filter_1d(self):
        """Kiểm tra bộ lọc trung vị loại bỏ được xung nhiễu (impulse noise)."""
        x = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
        # Với cửa sổ 3 hoặc 5, giá trị xung 10.0 sẽ bị triệt tiêu về 1.0
        filtered = median_filter_1d(x, kernel_size=3)
        self.assertEqual(filtered[2], 1.0)
        self.assertEqual(len(filtered), len(x))

    def test_find_histogram_peaks(self):
        """Kiểm tra phát hiện chính xác 2 đỉnh phân phối bimodal."""
        np.random.seed(42)
        # Tạo phân phối 2 cụm: cụm 1 quanh 0.01 (silence), cụm 2 quanh 0.08 (speech)
        mode1 = np.random.normal(0.01, 0.002, 500)
        mode2 = np.random.normal(0.08, 0.005, 500)
        data = np.clip(np.concatenate([mode1, mode2]), 0.0, None)

        hist, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        peaks = find_histogram_peaks(hist, bin_centers, search_window=2)
        self.assertGreaterEqual(len(peaks), 2)
        # Đỉnh đầu tiên quanh 0.01, đỉnh thứ 2 quanh 0.08
        self.assertAlmostEqual(peaks[0][1], 0.01, delta=0.01)
        self.assertAlmostEqual(peaks[1][1], 0.08, delta=0.02)

    def test_compute_histogram_threshold(self):
        """Kiểm tra ngưỡng tính toán có trọng số T = (W*V1 + V2)/(W+1)."""
        np.random.seed(42)
        mode1 = np.random.normal(0.01, 0.002, 500)
        mode2 = np.random.normal(0.08, 0.005, 500)
        data = np.clip(np.concatenate([mode1, mode2]), 0.0, None)

        weight = 4.0
        smoothed = median_filter_1d(data, kernel_size=5)
        threshold, info = compute_histogram_threshold(smoothed, num_bins=50, weight=weight)
        self.assertGreater(threshold, 0.01)
        self.assertLess(threshold, 0.08)

        # Kiểm tra class HistogramSegmenter
        segmenter = HistogramSegmenter(num_bins=50, weight=weight, median_kernel=5)
        t2 = segmenter.fit_from_features(data)
        self.assertTrue(segmenter.is_fitted)
        self.assertAlmostEqual(threshold, t2, places=4)


if __name__ == "__main__":
    unittest.main()
