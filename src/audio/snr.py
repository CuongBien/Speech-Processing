"""
src/audio/snr.py – Đo lường tỉ số tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR).
"""

from typing import List

import numpy as np

from src.config import LabSegment


def compute_snr_db(
    signal: np.ndarray,
    sample_rate: int,
    lab_segments: List[LabSegment],
) -> float:
    """
    Ước tính tỉ số tín hiệu trên nhiễu (SNR) theo thang dB
    dựa vào công suất trung bình tại các vùng tiếng nói và vùng khoảng lặng.

    Công thức:
        P_speech = mean(x^2 [speech])
        P_noise  = mean(x^2 [silence])
        SNR (dB) = 10 * log10(P_speech / (P_noise + eps))

    Tham số:
        signal (np.ndarray): Mảng mẫu tín hiệu âm thanh.
        sample_rate (int): Tần số lấy mẫu (Hz).
        lab_segments (List[LabSegment]): Danh sách các đoạn chuẩn.

    Trả về:
        float: Giá trị SNR đo bằng dB.
    """
    speech_samples: List[np.ndarray] = []
    silence_samples: List[np.ndarray] = []

    # Thu thập các mẫu âm thanh tương ứng từng vùng
    for seg in lab_segments:
        s_idx = int(seg.start * sample_rate)
        e_idx = min(int(seg.end * sample_rate), len(signal))
        if s_idx < e_idx:
            chunk = signal[s_idx:e_idx]
            if seg.is_silence():
                silence_samples.append(chunk)
            else:
                speech_samples.append(chunk)

    if not speech_samples or not silence_samples:
        return 0.0

    # Ghép các mẫu và tính công suất trung bình
    all_speech = np.concatenate(speech_samples)
    all_silence = np.concatenate(silence_samples)

    power_speech = float(np.mean(all_speech.astype(np.float64) ** 2))
    power_noise = float(np.mean(all_silence.astype(np.float64) ** 2))

    if power_noise <= 1e-12:
        return 100.0

    snr_db = 10.0 * np.log10((power_speech + 1e-12) / (power_noise + 1e-12))
    return float(snr_db)
