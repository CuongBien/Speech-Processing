"""
src/algorithms/base.py – Lớp cơ sở trừu tượng cho các thuật toán phân đoạn tiếng nói / khoảng lặng.
Cung cấp interface chuẩn để thống nhất cách gọi giữa Thuật toán 1 (Binary Search) và Thuật toán 2 (Histogram).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import Segment
from src.segmentation.postprocess import (
    classify_frames, frames_to_segments, remove_short_silence,
)


class BaseSegmenter(ABC):
    """Lớp giao diện trừu tượng cho mọi thuật toán phân đoạn âm thanh."""

    def __init__(self, name: str):
        self.name = name
        self.threshold: Optional[float] = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, speech_values: np.ndarray, silence_values: np.ndarray, **kwargs) -> float:
        """
        Huấn luyện xác định tham số hoặc ngưỡng phân biệt từ dữ liệu huấn luyện.

        Tham số:
            speech_values (np.ndarray): Mảng đặc trưng của các khung tiếng nói.
            silence_values (np.ndarray): Mảng đặc trưng của các khung khoảng lặng.

        Trả về:
            float: Ngưỡng tối ưu tìm được.
        """
        pass

    def predict_frames(self, feature_values: np.ndarray) -> np.ndarray:
        """
        Phân loại từng khung thời gian thành tiếng nói (1) hoặc khoảng lặng (0).

        Tham số:
            feature_values (np.ndarray): Mảng giá trị đặc trưng ngắn hạn.

        Trả về:
            np.ndarray: Mảng nhãn nhị phân.
        """
        if self.threshold is None:
            raise ValueError(f"Thuật toán {self.name} chưa được fit() hoặc thiết lập threshold.")
        return classify_frames(feature_values, self.threshold)

    def segment(
        self,
        feature_values: np.ndarray,
        frame_centers: np.ndarray,
        signal_duration_s: float,
        min_silence_duration_ms: float = 300.0,
    ) -> List[Segment]:
        """
        Phân đoạn đầy đủ từ đặc trưng khung thành danh sách các đoạn (Segment),
        tự động áp dụng quy tắc lọc khoảng lặng ngắn.

        Tham số:
            feature_values (np.ndarray): Mảng giá trị đặc trưng.
            frame_centers (np.ndarray): Mốc thời gian trung tâm các khung.
            signal_duration_s (float): Tổng thời lượng tín hiệu.
            min_silence_duration_ms (float): Độ dài khoảng lặng tối thiểu (mặc định 300ms).

        Trả về:
            List[Segment]: Danh sách các đoạn đã phân tách.
        """
        labels = self.predict_frames(feature_values)
        raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s)
        return remove_short_silence(raw_segs, min_duration_ms=min_silence_duration_ms)
