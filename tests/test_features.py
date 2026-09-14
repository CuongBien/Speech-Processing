"""
tests/test_features.py – Kiểm thử đơn vị cho việc trích xuất đặc trưng ngắn hạn.
"""

import unittest
import numpy as np

from src.config import FeatureType
from src.features import compute_short_time_feature


class TestFeatures(unittest.TestCase):
    """Kiểm tra các hàm tính đặc trưng MA, logMA, STE, logSTE."""

    def setUp(self):
        # Tạo tín hiệu kiểm thử dạng sin nhân tạo: 1 giây, 16000 Hz
        self.sr = 16000
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_compute_feature_dimensions(self):
        """Kiểm tra số lượng khung được tính toán chính xác theo frame length và hop size."""
        frame_len_ms = 20
        frame_shift_ms = 10
        feat, centers = compute_short_time_feature(
            self.signal, self.sr, frame_length_ms=frame_len_ms, frame_shift_ms=frame_shift_ms,
            feature_type=FeatureType.MA
        )
        self.assertGreater(len(feat), 0)
        self.assertEqual(len(feat), len(centers))
        # Với 16000 mẫu, frame 320, step 160 -> số khung = 1 + (16000 - 320) // 160 = 99 khung
        self.assertEqual(len(feat), 99)

    def test_feature_values_consistency(self):
        """Kiểm tra giá trị MA, STE > 0 và logMA < 0 đối với tín hiệu trong khoảng [-1, 1]."""
        ma, _ = compute_short_time_feature(self.signal, self.sr, feature_type=FeatureType.MA)
        ste, _ = compute_short_time_feature(self.signal, self.sr, feature_type=FeatureType.STE)
        log_ma, _ = compute_short_time_feature(self.signal, self.sr, feature_type=FeatureType.LOG_MA)

        self.assertTrue(np.all(ma >= 0.0))
        self.assertTrue(np.all(ste >= 0.0))
        self.assertTrue(np.all(log_ma <= 0.0))


if __name__ == "__main__":
    unittest.main()
