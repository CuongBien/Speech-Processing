"""
src/features/spectral.py – Trích xuất đặc trưng Trọng tâm phổ (Spectral Centroid) và Năng lượng ngắn hạn.
Triển khai hoàn toàn bằng NumPy thuần (không dùng thư viện ngoài) theo bài báo
Theodoros Giannakopoulos (2014): "A method for silence removal and segmentation of speech signals".
"""

import logging
from typing import Tuple

import numpy as np

from src.config import FRAME_LENGTH_MS, FRAME_SHIFT_MS

logger = logging.getLogger(__name__)


def compute_energy_and_spectral_centroid(
    signal: np.ndarray,
    sample_rate: int,
    frame_length_ms: int = FRAME_LENGTH_MS,
    frame_shift_ms: int = FRAME_SHIFT_MS,
    use_hamming: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trích xuất đồng thời 2 đặc trưng cơ bản theo Giannakopoulos (2014):
    1. Năng lượng ngắn hạn (Short-time Energy - E):
       E(i) = (1 / N) * sum(x_i[n]^2)
    2. Trọng tâm phổ (Spectral Centroid - C):
       C(i) = sum((k + 1) * |X_i(k)|) / sum(|X_i(k)|)
       với X_i(k) là phổ biên độ DFT của khung thứ i.

    Tham số:
        signal (np.ndarray): Tín hiệu âm thanh 1 chiều (float32 hoặc float64).
        sample_rate (int): Tần số lấy mẫu (Hz).
        frame_length_ms (int): Độ dài khung (mili-giây).
        frame_shift_ms (int): Bước nhảy khung (mili-giây).
        use_hamming (bool): Có nhân cửa sổ Hamming trước khi biến đổi phổ hay không.

    Trả về:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - energy_vals (np.ndarray): Mảng năng lượng ngắn hạn E(i).
            - centroid_vals (np.ndarray): Mảng trọng tâm phổ C(i).
            - frame_centers (np.ndarray): Mốc thời gian trung tâm của từng khung (giây).
    """
    frame_len = int(sample_rate * frame_length_ms / 1000)
    frame_step = int(sample_rate * frame_shift_ms / 1000)

    if frame_len <= 0 or frame_step <= 0:
        raise ValueError(f"Độ dài khung ({frame_len}) hoặc bước nhảy ({frame_step}) không hợp lệ.")

    num_samples = len(signal)
    if num_samples < frame_len:
        logger.warning("Độ dài tín hiệu ngắn hơn một khung đơn lẻ.")
        return np.array([]), np.array([]), np.array([])

    num_frames = 1 + (num_samples - frame_len) // frame_step

    energy_vals = np.empty(num_frames, dtype=np.float64)
    centroid_vals = np.empty(num_frames, dtype=np.float64)
    frame_centers = np.empty(num_frames, dtype=np.float64)

    # Cửa sổ Hamming làm mịn phổ biên độ
    window = np.hamming(frame_len) if use_hamming else np.ones(frame_len, dtype=np.float64)

    # Hệ số chỉ số tần số k = 1, 2, ..., K
    # Với np.fft.rfft, số bin phổ là frame_len // 2 + 1
    num_fft_bins = frame_len // 2 + 1
    k_indices = np.arange(1, num_fft_bins + 1, dtype=np.float64)

    for i in range(num_frames):
        start_idx = i * frame_step
        end_idx = start_idx + frame_len
        frame = signal[start_idx:end_idx].astype(np.float64)

        # 1. Năng lượng ngắn hạn E(i)
        energy_vals[i] = float(np.mean(frame ** 2))

        # 2. Trọng tâm phổ Spectral Centroid C(i) qua biến đổi Fourier rời rạc DFT
        windowed_frame = frame * window
        fft_mag = np.abs(np.fft.rfft(windowed_frame))

        sum_mag = float(np.sum(fft_mag))
        if sum_mag > 1e-12:
            centroid_vals[i] = float(np.sum(k_indices * fft_mag) / sum_mag)
        else:
            centroid_vals[i] = 0.0

        # Mốc thời gian tâm khung
        frame_centers[i] = (start_idx + frame_len / 2.0) / sample_rate

    return energy_vals, centroid_vals, frame_centers
