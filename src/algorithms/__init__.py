"""
src/algorithms – Gói chứa các thuật toán phân đoạn tiếng nói / khoảng lặng.
"""

from src.algorithms.base import BaseSegmenter
from src.algorithms.binary_search import (
    BinarySearchSegmenter,
    compute_confusion,
    find_optimal_threshold_binary_search,
    find_overlap_region,
)
from src.algorithms.histogram import (
    HistogramSegmenter,
    compute_histogram_threshold,
    find_histogram_peaks,
    median_filter_1d,
)

__all__ = [
    "BaseSegmenter",
    "BinarySearchSegmenter",
    "HistogramSegmenter",
    "find_overlap_region",
    "compute_confusion",
    "find_optimal_threshold_binary_search",
    "median_filter_1d",
    "find_histogram_peaks",
    "compute_histogram_threshold",
]
