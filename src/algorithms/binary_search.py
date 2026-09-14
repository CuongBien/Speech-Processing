"""
src/algorithms/binary_search.py – Thuật toán 1: Tìm kiếm nhị phân (Hodgkinson 2012, Sec 2.1).
Cân bằng hàm độ nhầm lẫn giữa vùng tiếng nói và khoảng lặng trên miền giao thoa.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base import BaseSegmenter
from src.config import (
    MAX_ITERATIONS, THRESHOLD_TOLERANCE, BinarySearchState,
)

logger = logging.getLogger(__name__)


def find_overlap_region(
    speech_values: np.ndarray,
    silence_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Xác định khoảng giao thoa (overlap region) giữa phân phối đặc trưng của Tiếng nói và Khoảng lặng:
        overlap_min = max(min(speech), min(silence))
        overlap_max = min(max(speech), max(silence))
    """
    if len(speech_values) == 0 or len(silence_values) == 0:
        raise ValueError("Cần ít nhất một khung tiếng nói và một khung khoảng lặng để tìm overlap.")

    min_spch = float(np.min(speech_values))
    max_spch = float(np.max(speech_values))
    min_sil = float(np.min(silence_values))
    max_sil = float(np.max(silence_values))

    ov_min = max(min_spch, min_sil)
    ov_max = min(max_spch, max_sil)
    has_overlap = ov_min < ov_max

    if not has_overlap:
        logger.warning(
            "Hai phân phối không có vùng giao thoa: Tiếng nói=[%.4f, %.4f], Khoảng lặng=[%.4f, %.4f]",
            min_spch, max_spch, min_sil, max_sil,
        )
        ov_min = ov_max = (min(max_spch, max_sil) + max(min_spch, min_sil)) / 2.0

    spch_overlap = speech_values[(speech_values >= ov_min) & (speech_values <= ov_max)]
    sil_overlap = silence_values[(silence_values >= ov_min) & (silence_values <= ov_max)]

    return {
        "overlap_min": ov_min,
        "overlap_max": ov_max,
        "has_overlap": has_overlap,
        "speech_overlap_vals": spch_overlap,
        "silence_overlap_vals": sil_overlap,
        "speech_all_vals": speech_values,
        "silence_all_vals": silence_values,
    }


def compute_confusion(
    silence_vals: np.ndarray,
    speech_vals: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Tính độ nhầm lẫn (Confusion measure) theo Hodgkinson (2012):
        C_sil  = mean(max(silence - T, 0))
        C_spch = mean(max(T - speech,  0))
        diff   = C_sil - C_spch
    """
    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return 0.0, 0.0, 0.0

    c_sil = float(np.mean(np.maximum(silence_vals - threshold, 0.0)))
    c_spch = float(np.mean(np.maximum(threshold - speech_vals, 0.0)))
    diff = c_sil - c_spch
    return c_sil, c_spch, diff


def find_optimal_threshold_binary_search(
    speech_values: np.ndarray,
    silence_values: np.ndarray,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = THRESHOLD_TOLERANCE,
) -> Tuple[float, List[BinarySearchState], Dict[str, Any]]:
    """
    Thực hiện thuật toán tìm kiếm nhị phân (Binary Search) trên vùng giao thoa.
    """
    overlap = find_overlap_region(speech_values, silence_values)
    t_min = overlap["overlap_min"]
    t_max = overlap["overlap_max"]
    sil_ov = overlap["silence_overlap_vals"]
    spch_ov = overlap["speech_overlap_vals"]

    if not overlap["has_overlap"] or len(sil_ov) == 0 or len(spch_ov) == 0:
        logger.warning("Không có dữ liệu giao thoa hợp lệ. Dùng midpoint T = %.6f", t_min)
        dummy_state = BinarySearchState(0, t_min, t_max, t_min, 0.0, 0.0, 0.0, 0, 0)
        return t_min, [dummy_state], overlap

    current_threshold = (t_min + t_max) / 2.0
    history: List[BinarySearchState] = []

    for it in range(1, max_iterations + 1):
        c_sil, c_spch, diff = compute_confusion(sil_ov, spch_ov, current_threshold)
        spch_err = int(np.sum(spch_ov < current_threshold))
        sil_err = int(np.sum(sil_ov > current_threshold))

        state = BinarySearchState(
            iteration=it,
            t_min=t_min,
            t_max=t_max,
            threshold=current_threshold,
            c_sil=c_sil,
            c_spch=c_spch,
            diff=diff,
            speech_below_t=spch_err,
            silence_above_t=sil_err,
        )
        history.append(state)

        if abs(diff) < tolerance or (t_max - t_min) < tolerance:
            break

        if diff > 0:
            t_min = current_threshold
        else:
            t_max = current_threshold

        current_threshold = (t_min + t_max) / 2.0

    return current_threshold, history, overlap


class BinarySearchSegmenter(BaseSegmenter):
    """Lớp đóng gói Thuật toán 1: Tìm kiếm nhị phân cân bằng độ nhầm lẫn."""

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        tolerance: float = THRESHOLD_TOLERANCE,
    ):
        super().__init__(name="Hodgkinson 2012 (Binary Search)")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history: List[BinarySearchState] = []
        self.overlap_info: Dict[str, Any] = {}

    def fit(self, speech_values: np.ndarray, silence_values: np.ndarray, **kwargs) -> float:
        """Huấn luyện tìm ngưỡng tối ưu bằng binary search."""
        threshold, history, overlap = find_optimal_threshold_binary_search(
            speech_values=speech_values,
            silence_values=silence_values,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        self.threshold = threshold
        self.history = history
        self.overlap_info = overlap
        self.is_fitted = True
        return threshold
