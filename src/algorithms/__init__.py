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
from src.algorithms.histogram import HistogramSegmenter

__all__ = [
    "BaseSegmenter",
    "BinarySearchSegmenter",
    "HistogramSegmenter",
    "find_overlap_region",
    "compute_confusion",
    "find_optimal_threshold_binary_search",
]
