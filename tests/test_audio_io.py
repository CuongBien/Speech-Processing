"""
tests/test_audio_io.py – Kiểm thử đơn vị cho module đọc/ghi audio và file .lab.
"""

import os
import unittest
from pathlib import Path

import numpy as np

from src.audio import load_audio, load_ground_truth, parse_lab_file
from src.config import TRAINING_DIR


class TestAudioIO(unittest.TestCase):
    """Kiểm tra các hàm đọc audio và nhãn ground truth."""

    def setUp(self):
        self.wav_path = os.path.join(TRAINING_DIR, "phone_F1.wav")
        self.lab_path = os.path.join(TRAINING_DIR, "phone_F1.lab")

    def test_load_audio(self):
        """Kiểm tra đọc file WAV hợp lệ, mảng chuẩn hóa [-1, 1] và tần số lấy mẫu."""
        if not os.path.exists(self.wav_path):
            self.skipTest(f"Không tìm thấy file: {self.wav_path}")

        signal, sr = load_audio(self.wav_path)
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.ndim, 1)
        self.assertGreater(sr, 0)
        self.assertGreaterEqual(float(np.min(signal)), -1.0)
        self.assertLessEqual(float(np.max(signal)), 1.0)

    def test_parse_lab_file(self):
        """Kiểm tra đọc file .lab, trích xuất đúng các đoạn nhãn hợp lệ."""
        if not os.path.exists(self.lab_path):
            self.skipTest(f"Không tìm thấy file: {self.lab_path}")

        lab_segs = parse_lab_file(self.lab_path)
        self.assertGreater(len(lab_segs), 0)
        for seg in lab_segs:
            self.assertGreaterEqual(seg.start, 0.0)
            self.assertGreater(seg.end, seg.start)
            self.assertIn(seg.label, ("sil", "v", "uv"))

    def test_load_ground_truth(self):
        """Kiểm tra gộp đoạn nhãn nhị phân và tính liên tục thời gian."""
        if not os.path.exists(self.lab_path):
            self.skipTest(f"Không tìm thấy file: {self.lab_path}")

        lab_segs = parse_lab_file(self.lab_path)
        gt_segs = load_ground_truth(lab_segs)
        self.assertGreater(len(gt_segs), 0)
        # Các đoạn kề nhau không được trùng nhãn
        for i in range(1, len(gt_segs)):
            self.assertNotEqual(gt_segs[i].label, gt_segs[i - 1].label)
            self.assertAlmostEqual(gt_segs[i].start, gt_segs[i - 1].end, places=4)


if __name__ == "__main__":
    unittest.main()
