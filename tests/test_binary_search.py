"""
tests/test_binary_search.py – Kiểm thử đơn vị cho thuật toán tìm kiếm nhị phân.
"""

import unittest
import numpy as np

from src.algorithms.binary_search import (
    BinarySearchSegmenter,
    compute_confusion,
    find_optimal_threshold_binary_search,
    find_overlap_region,
)


class TestBinarySearch(unittest.TestCase):
    """Kiểm tra logic tìm kiếm nhị phân và hội tụ cân bằng độ nhầm lẫn."""

    def test_find_overlap(self):
        """Kiểm tra xác định đúng vùng giao thoa giữa 2 phân phối."""
        silence = np.array([-7.0, -6.0, -5.0, -4.5])
        speech = np.array([-5.5, -4.0, -3.0, -2.0])

        overlap = find_overlap_region(speech, silence)
        self.assertTrue(overlap["has_overlap"])
        self.assertAlmostEqual(overlap["overlap_min"], -5.5)
        self.assertAlmostEqual(overlap["overlap_max"], -4.5)

    def test_convergence(self):
        """Kiểm tra thuật toán tìm kiếm nhị phân hội tụ tới điểm cân bằng."""
        np.random.seed(42)
        # Phân phối silence: N(-6, 0.5), speech: N(-3, 0.5)
        silence = np.random.normal(-6.0, 0.5, 500)
        speech = np.random.normal(-3.0, 0.5, 500)

        threshold, history, _ = find_optimal_threshold_binary_search(speech, silence)
        self.assertGreater(threshold, -6.0)
        self.assertLess(threshold, -3.0)
        self.assertLess(len(history), 100)

        # Kiểm tra qua class BinarySearchSegmenter
        segmenter = BinarySearchSegmenter()
        t2 = segmenter.fit(speech, silence)
        self.assertAlmostEqual(threshold, t2, places=5)
        self.assertTrue(segmenter.is_fitted)


if __name__ == "__main__":
    unittest.main()
