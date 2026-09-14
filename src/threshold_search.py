"""
src/threshold_search.py – Thuật toán tìm kiếm nhị phân xác định ngưỡng tối ưu.
Cài đặt chính xác theo lý thuyết của Hodgkinson (2012, Sec 2.1) "Energy-based Speech/Silence discrimination".
Cân bằng hàm độ nhầm lẫn giữa vùng tiếng nói và khoảng lặng trên miền giao thoa.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

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

    Tham số:
        speech_values (np.ndarray): Mảng giá trị đặc trưng của các khung tiếng nói.
        silence_values (np.ndarray): Mảng giá trị đặc trưng của các khung khoảng lặng.

    Trả về:
        Dict[str, Any]: Từ điển chứa thông tin vùng overlap và các giá trị nằm trong vùng này.
    """
    if len(speech_values) == 0 or len(silence_values) == 0:
        raise ValueError("Cần ít nhất một khung tiếng nói và một khung khoảng lặng để tìm overlap.")

    # Tìm giá trị cực trị của hai phân phối
    min_spch = float(np.min(speech_values))
    max_spch = float(np.max(speech_values))
    min_sil = float(np.min(silence_values))
    max_sil = float(np.max(silence_values))

    # Tính toán vùng giao nhau
    ov_min = max(min_spch, min_sil)
    ov_max = min(max_spch, max_sil)
    has_overlap = ov_min < ov_max

    # Trường hợp đặc biệt: hai phân phối tách biệt hoàn toàn
    if not has_overlap:
        logger.warning(
            "Hai phân phối không có vùng giao thoa: Tiếng nói=[%.4f, %.4f], Khoảng lặng=[%.4f, %.4f]",
            min_spch, max_spch, min_sil, max_sil,
        )
        ov_min = ov_max = (min(max_spch, max_sil) + max(min_spch, min_sil)) / 2.0

    # Lọc lấy các giá trị nằm trong vùng overlap để phục vụ tìm kiếm nhị phân
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
    Tính độ nhầm lẫn (Confusion measure) theo công thức của Hodgkinson (2012):
        C_sil  = mean(max(silence - T, 0))  --> Khoảng lặng bị nhận nhầm thành tiếng nói (vượt ngưỡng)
        C_spch = mean(max(T - speech,  0))  --> Tiếng nói bị nhận nhầm thành khoảng lặng (dưới ngưỡng)
        diff   = C_sil - C_spch

    Tham số:
        silence_vals (np.ndarray): Mảng đặc trưng của các khung khoảng lặng trong vùng overlap.
        speech_vals (np.ndarray): Mảng đặc trưng của các khung tiếng nói trong vùng overlap.
        threshold (float): Ngưỡng kiểm tra T.

    Trả về:
        Tuple[float, float, float]: (C_sil, C_spch, diff).
    """
    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return 0.0, 0.0, 0.0

    # Độ nhầm lẫn khoảng lặng vượt ngưỡng T
    c_sil = float(np.mean(np.maximum(silence_vals - threshold, 0.0)))

    # Độ nhầm lẫn tiếng nói rơi xuống dưới ngưỡng T
    c_spch = float(np.mean(np.maximum(threshold - speech_vals, 0.0)))

    # Chênh lệch độ nhầm lẫn
    diff = c_sil - c_spch
    return c_sil, c_spch, diff


def find_optimal_threshold_binary_search(
    speech_values: np.ndarray,
    silence_values: np.ndarray,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = THRESHOLD_TOLERANCE,
) -> Tuple[float, List[BinarySearchState], Dict[str, Any]]:
    """
    Thực hiện thuật toán tìm kiếm nhị phân (Binary Search) trên vùng giao thoa
    để tìm ngưỡng T* sao cho chênh lệch độ nhầm lẫn diff xấp xỉ 0.

    Quy tắc cập nhật:
        Nếu diff > 0: C_sil > C_spch (ngưỡng T quá thấp) -> t_min = T
        Nếu diff < 0: C_spch > C_sil (ngưỡng T quá cao)  -> t_max = T

    Tham số:
        speech_values (np.ndarray): Toàn bộ đặc trưng các khung tiếng nói.
        silence_values (np.ndarray): Toàn bộ đặc trưng các khung khoảng lặng.
        max_iterations (int): Số bước lặp tối đa.
        tolerance (float): Sai số hội tụ cho phép.

    Trả về:
        Tuple[float, List[BinarySearchState], Dict[str, Any]]:
            - optimal_threshold (float): Ngưỡng phân tách tối ưu tìm được.
            - history (List[BinarySearchState]): Lịch sử các bước lặp.
            - overlap_info (Dict[str, Any]): Thông tin vùng giao thoa.
    """
    # 1. Xác định vùng giao thoa giữa 2 phân phối
    overlap = find_overlap_region(speech_values, silence_values)
    t_min = overlap["overlap_min"]
    t_max = overlap["overlap_max"]
    sil_ov = overlap["silence_overlap_vals"]
    spch_ov = overlap["speech_overlap_vals"]

    # Kiểm tra nếu không có vùng giao thoa khả dụng
    if not overlap["has_overlap"] or len(sil_ov) == 0 or len(spch_ov) == 0:
        logger.warning("Không có dữ liệu giao thoa hợp lệ. Sử dụng điểm chính giữa: T = %.6f", t_min)
        dummy_state = BinarySearchState(0, t_min, t_max, t_min, 0.0, 0.0, 0.0, 0, 0)
        return t_min, [dummy_state], overlap

    current_threshold = (t_min + t_max) / 2.0
    history: List[BinarySearchState] = []

    logger.info("Bắt đầu Binary Search: Khoảng overlap = [%.6f, %.6f]", t_min, t_max)

    # 2. Vòng lặp chia đôi khoảng tìm kiếm
    for it in range(1, max_iterations + 1):
        c_sil, c_spch, diff = compute_confusion(sil_ov, spch_ov, current_threshold)

        # Đếm số khung nhầm lẫn tại ngưỡng này
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

        # Điều kiện dừng: chênh lệch nhỏ hơn tolerance hoặc khoảng tìm kiếm quá hẹp
        if abs(diff) < tolerance or (t_max - t_min) < tolerance:
            logger.info("Thuật toán đã hội tụ thành công tại vòng lặp %d (diff=%.2e)", it, diff)
            break

        # Quy tắc cập nhật cận trên hoặc cận dưới
        if diff > 0:
            # Khoảng lặng bị nhầm nhiều hơn -> Cần tăng ngưỡng
            t_min = current_threshold
        else:
            # Tiếng nói bị nhầm nhiều hơn -> Cần giảm ngưỡng
            t_max = current_threshold

        current_threshold = (t_min + t_max) / 2.0

    return current_threshold, history, overlap
