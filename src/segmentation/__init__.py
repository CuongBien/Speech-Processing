"""
src/segmentation – Gói hậu xử lý phân đoạn và đánh giá sai số.
"""

from src.segmentation.metrics import (
    evaluate_boundaries,
    extract_boundaries,
    match_boundaries,
)
from src.segmentation.postprocess import (
    classify_frames,
    frames_to_segments,
    remove_short_silence,
)

__all__ = [
    "classify_frames",
    "frames_to_segments",
    "remove_short_silence",
    "extract_boundaries",
    "match_boundaries",
    "evaluate_boundaries",
]
