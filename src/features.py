"""
src/features.py – Trích xuất đặc trưng ngắn hạn và gán nhãn khung thời gian.
Tự cài đặt các phép toán trên Numpy thuần (không dùng toolbox ngoài).
Bao gồm tính STE, MA, logMA, logSTE, gán nhãn Ground Truth và ước tính tỉ số tín hiệu trên nhiễu (SNR).
"""

import logging
from typing import List, Tuple

import numpy as np

from src.config import (
    DEFAULT_FEATURE_TYPE, FRAME_LENGTH_MS, FRAME_SHIFT_MS, LOG_EPSILON,
    FeatureType, LabSegment,
)

logger = logging.getLogger(__name__)


def compute_short_time_feature(
    signal: np.ndarray,
    sample_rate: int,
    frame_length_ms: int = FRAME_LENGTH_MS,
    frame_shift_ms: int = FRAME_SHIFT_MS,
    feature_type: FeatureType = DEFAULT_FEATURE_TYPE,
    log_epsilon: float = LOG_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chia tín hiệu âm thanh thành các khung ngắn và tính đặc trưng ngắn hạn tương ứng.
    Hỗ trợ MA, logMA, STE, logSTE.

    Tham số:
        signal (np.ndarray): Mảng 1 chiều chứa các mẫu âm thanh (float32).
        sample_rate (int): Tần số lấy mẫu (Hz).
        frame_length_ms (int): Độ dài mỗi khung (mili-giây), mặc định 20ms.
        frame_shift_ms (int): Bước dịch khung (mili-giây), mặc định 10ms.
        feature_type (FeatureType): Loại đặc trưng ngắn hạn cần tính toán.
        log_epsilon (float): Hằng số nhỏ để tránh lỗi logarit của số 0.

    Trả về:
        Tuple[np.ndarray, np.ndarray]:
            - feature_values (np.ndarray): Mảng giá trị đặc trưng ngắn hạn theo từng khung.
            - frame_centers (np.ndarray): Mốc thời gian trung tâm của mỗi khung (tính bằng giây).
    """
    # Tính số mẫu (samples) trong một khung và bước nhảy giữa 2 khung kề nhau
    frame_len_samples = int(sample_rate * frame_length_ms / 1000)
    frame_step_samples = int(sample_rate * frame_shift_ms / 1000)

    if frame_len_samples <= 0 or frame_step_samples <= 0:
        raise ValueError(f"Độ dài khung ({frame_len_samples}) hoặc bước nhảy ({frame_step_samples}) không hợp lệ.")

    num_samples = len(signal)
    if num_samples < frame_len_samples:
        logger.warning("Độ dài tín hiệu ngắn hơn một khung đơn lẻ.")
        return np.array([]), np.array([])

    # Xác định tổng số khung có thể tạo ra
    num_frames = 1 + (num_samples - frame_len_samples) // frame_step_samples
    feature_values = np.empty(num_frames, dtype=np.float64)
    frame_centers = np.empty(num_frames, dtype=np.float64)

    # Duyệt qua từng khung và tính đặc trưng tương ứng trên numpy thuần
    for i in range(num_frames):
        start_idx = i * frame_step_samples
        end_idx = start_idx + frame_len_samples
        frame = signal[start_idx:end_idx]

        # Tính độ lớn ngắn hạn (MA) hoặc năng lượng ngắn hạn (STE)
        if feature_type in (FeatureType.MA, FeatureType.LOG_MA):
            # Biên độ trung bình của khung: MA = mean(|x|)
            val = float(np.mean(np.abs(frame)))
        else:
            # Năng lượng trung bình của khung: STE = mean(x^2)
            val = float(np.mean(frame.astype(np.float64) ** 2))

        # Áp dụng hàm log nếu chọn logMA hoặc logSTE
        if feature_type in (FeatureType.LOG_MA, FeatureType.LOG_STE):
            val = float(np.log(val + log_epsilon))

        feature_values[i] = val
        # Điểm mốc thời gian tại tâm của khung (giây)
        frame_centers[i] = (start_idx + frame_len_samples / 2.0) / sample_rate

    return feature_values, frame_centers


def assign_frame_labels(
    frame_centers: np.ndarray,
    lab_segments: List[LabSegment],
) -> np.ndarray:
    """
    Gán nhãn cho từng khung thời gian dựa trên tâm của khung và danh sách đoạn chuẩn .lab.

    Tham số:
        frame_centers (np.ndarray): Mảng mốc thời gian trung tâm của các khung.
        lab_segments (List[LabSegment]): Danh sách các đoạn chuẩn từ file .lab.

    Trả về:
        np.ndarray: Mảng nhãn nhị phân: 1 (tiếng nói - speech), 0 (khoảng lặng - silence).
    """
    # Khởi tạo nhãn mặc định là -1 (chưa gán)
    labels = np.full(len(frame_centers), -1, dtype=np.int32)

    # Duyệt qua từng đoạn trong file .lab để gán nhãn
    for seg in lab_segments:
        # Khung có tâm nằm trong khoảng nửa mở [start, end) được gán nhãn của đoạn đó
        mask = (frame_centers >= seg.start) & (frame_centers < seg.end)
        if seg.is_silence():
            labels[mask] = 0
        else:
            labels[mask] = 1

    # Nếu có khung ở ngoài cùng chưa thuộc đoạn nào, gán về khoảng lặng
    labels[labels == -1] = 0
    return labels


def compute_snr_db(
    signal: np.ndarray,
    sample_rate: int,
    lab_segments: List[LabSegment],
) -> float:
    """
    Ước tính tỉ số tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR) theo thang dB
    dựa vào năng lượng trung bình tại các vùng tiếng nói và vùng khoảng lặng.

    Công thức:
        P_speech = mean(x^2 [speech])
        P_noise  = mean(x^2 [silence])
        SNR (dB) = 10 * log10(P_speech / (P_noise + eps))

    Tham số:
        signal (np.ndarray): Mảng mẫu tín hiệu âm thanh đã chuẩn hóa.
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
