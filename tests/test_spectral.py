"""
tests/test_spectral.py – Kiểm thử đơn vị cho trích xuất đặc trưng Spectral Centroid và Energy.
"""

import unittest
import numpy as np

from src.features.spectral import compute_energy_and_spectral_centroid
from src.algorithms.histogram import HistogramSegmenter


class TestSpectralFeatures(unittest.TestCase):
    """Kiểm tra tính đúng đắn của hàm tính Năng lượng ngắn hạn và Trọng tâm phổ."""

    def test_energy_computation(self):
        """Kiểm tra tính năng lượng ngắn hạn trên tín hiệu nhân tạo."""
        sr = 16000
        # Tín hiệu biên độ 2.0 -> Năng lượng E = 4.0
        signal = np.full(1600, 2.0, dtype=np.float32)
        E, C, centers = compute_energy_and_spectral_centroid(
            signal, sample_rate=sr, frame_length_ms=20, frame_shift_ms=10
        )
        self.assertGreater(len(E), 0)
        self.assertAlmostEqual(E[0], 4.0, places=4)

    def test_spectral_centroid_synthetic(self):
        """Kiểm tra sóng tần số cao có Spectral Centroid lớn hơn sóng tần số thấp."""
        sr = 16000
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        # Sóng tần số thấp (300 Hz)
        low_tone = np.sin(2 * np.pi * 300 * t)
        # Sóng tần số cao (3000 Hz)
        high_tone = np.sin(2 * np.pi * 3000 * t)

        _, c_low, _ = compute_energy_and_spectral_centroid(low_tone, sr, frame_length_ms=20, frame_shift_ms=10)
        _, c_high, _ = compute_energy_and_spectral_centroid(high_tone, sr, frame_length_ms=20, frame_shift_ms=10)

        # Trọng tâm phổ của âm tần số cao phải lớn hơn âm tần số thấp
        self.assertGreater(float(np.mean(c_high)), float(np.mean(c_low)))

    def test_dual_feature_segmenter(self):
        """Kiểm tra HistogramSegmenter hoạt động chính xác với 2 đặc trưng."""
        np.random.seed(42)
        sr = 16000
        # 2.0s tổng cộng:
        # 0.0 -> 0.6s: Khoảng lặng (nhiễu biên độ nhỏ 0.001) > 300ms
        # 0.6 -> 1.4s: Tiếng nói (sóng hình sin biên độ 0.5)
        # 1.4 -> 2.0s: Khoảng lặng > 300ms
        sig = np.random.normal(0, 0.001, 32000)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        sig[9600:22400] += 0.5 * np.sin(2 * np.pi * 500 * t[9600:22400])

        E, C, fc = compute_energy_and_spectral_centroid(sig, sr)
        segmenter = HistogramSegmenter(weight_energy=4.0, weight_centroid=1.5, expand_frames=2)
        t1, t2, segs = segmenter.fit_and_segment_dynamic(E, C, fc, signal_duration_s=2.0)

        self.assertTrue(segmenter.is_fitted)
        self.assertGreater(t1, 0.0)
        self.assertGreater(t2, 0.0)

        speech_segs = [s for s in segs if s.label == "speech"]
        self.assertEqual(len(speech_segs), 1)
        # Ranh giới tiếng nói phải xấp xỉ 0.6s và 1.4s
        self.assertAlmostEqual(speech_segs[0].start, 0.6, delta=0.08)
        self.assertAlmostEqual(speech_segs[0].end, 1.4, delta=0.08)


if __name__ == "__main__":
    unittest.main()
