"""
src/features/short_time.py – Trích xuất đặc trưng ngắn hạn và gán nhãn khung thời gian.
Tự cài đặt các phép toán trên Numpy thuần (không dùng toolbox ngoài).
Bao gồm tính MA, logMA, STE, logSTE và gán nhãn Ground Truth cho từng khung.
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
    frame_len_samples = int(sample_rate * frame_length_ms / 1000)
    frame_step_samples = int(sample_rate * frame_shift_ms / 1000)

    if frame_len_samples <= 0 or frame_step_samples <= 0:
        raise ValueError(f"Độ dài khung ({frame_len_samples}) hoặc bước nhảy ({frame_step_samples}) không hợp lệ.")

    num_samples = len(signal)
    if num_samples < frame_len_samples:
        logger.warning("Độ dài tín hiệu ngắn hơn một khung đơn lẻ.")
        return np.array([]), np.array([])

    num_frames = 1 + (num_samples - frame_len_samples) // frame_step_samples
    feature_values = np.empty(num_frames, dtype=np.float64)
    frame_centers = np.empty(num_frames, dtype=np.float64)

    # Duyệt qua từng khung và tính đặc trưng tương ứng trên numpy thuần
    for i in range(num_frames):
        start_idx = i * frame_step_samples
        end_idx = start_idx + frame_len_samples
        frame = signal[start_idx:end_idx]

        if feature_type in (FeatureType.MA, FeatureType.LOG_MA):
            # Độ lớn ngắn hạn: MA = mean(|x|)
            val = float(np.mean(np.abs(frame)))
        else:
            # Năng lượng ngắn hạn: STE = mean(x^2)
            val = float(np.mean(frame.astype(np.float64) ** 2))

        if feature_type in (FeatureType.LOG_MA, FeatureType.LOG_STE):
            val = float(np.log(val + log_epsilon))

        feature_values[i] = val
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
    labels = np.full(len(frame_centers), -1, dtype=np.int32)

    for seg in lab_segments:
        mask = (frame_centers >= seg.start) & (frame_centers < seg.end)
        if seg.is_silence():
            labels[mask] = 0
        else:
            labels[mask] = 1

    labels[labels == -1] = 0
    return labels
