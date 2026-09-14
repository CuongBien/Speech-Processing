"""
src/segmentation/postprocess.py – Chuyển đổi nhãn khung thành đoạn và lọc khoảng lặng ngắn.
"""

from typing import List

import numpy as np

from src.config import (
    FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS, Segment,
)


def classify_frames(
    feature_values: np.ndarray,
    threshold: float,
    speech_is_above: bool = True,
) -> np.ndarray:
    """Phân loại từng khung: 1 (speech), 0 (silence)."""
    if speech_is_above:
        return (feature_values >= threshold).astype(np.int32)
    return (feature_values <= threshold).astype(np.int32)


def frames_to_segments(
    labels: np.ndarray,
    frame_centers: np.ndarray,
    signal_duration_s: float = 0.0,
    frame_shift_ms: int = FRAME_SHIFT_MS,
) -> List[Segment]:
    """
    Chuyển đổi chuỗi nhãn theo khung thành danh sách các đoạn liên tục (Segment).
    Ranh giới nội bộ: (center[i-1] + center[i]) / 2.0.
    """
    if len(labels) == 0:
        return []

    segments: List[Segment] = []
    current_label = labels[0]
    current_start = 0.0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            boundary_time = (frame_centers[i - 1] + frame_centers[i]) / 2.0
            seg_name = "speech" if current_label == 1 else "silence"
            segments.append(Segment(start=current_start, end=boundary_time, label=seg_name))
            current_label = labels[i]
            current_start = boundary_time

    last_end = signal_duration_s if signal_duration_s > 0.0 else frame_centers[-1] + (frame_shift_ms / 2000.0)
    seg_name = "speech" if current_label == 1 else "silence"
    segments.append(Segment(start=current_start, end=last_end, label=seg_name))

    return segments


def remove_short_silence(
    segments: List[Segment],
    min_duration_ms: float = MIN_SILENCE_DURATION_MS,
) -> List[Segment]:
    """
    Loại bỏ các khoảng lặng ngắn (ảo) có độ dài < min_duration_ms (mặc định 300ms)
    theo đúng quy định của Đề bài: gộp khoảng lặng ngắn vào đoạn tiếng nói bao quanh nó.
    """
    if not segments:
        return []

    min_sec = min_duration_ms / 1000.0
    current_list = list(segments)
    has_changed = True

    while has_changed:
        has_changed = False
        filtered: List[Segment] = []
        idx = 0

        while idx < len(current_list):
            seg = current_list[idx]

            if seg.label == "silence" and seg.duration_s < min_sec:
                has_changed = True
                has_prev = len(filtered) > 0
                has_next = idx < len(current_list) - 1

                if has_prev and has_next:
                    prev_seg = filtered.pop()
                    next_seg = current_list[idx + 1]
                    merged = Segment(start=prev_seg.start, end=next_seg.end, label="speech")
                    filtered.append(merged)
                    idx += 2
                elif has_prev:
                    prev_seg = filtered.pop()
                    merged = Segment(start=prev_seg.start, end=seg.end, label=prev_seg.label)
                    filtered.append(merged)
                    idx += 1
                elif has_next:
                    next_seg = current_list[idx + 1]
                    merged = Segment(start=seg.start, end=next_seg.end, label=next_seg.label)
                    filtered.append(merged)
                    idx += 2
                else:
                    filtered.append(seg)
                    idx += 1
            else:
                filtered.append(seg)
                idx += 1

        current_list = filtered

    final_segments: List[Segment] = []
    for seg in current_list:
        if final_segments and final_segments[-1].label == seg.label:
            final_segments[-1] = Segment(start=final_segments[-1].start, end=seg.end, label=seg.label)
        else:
            final_segments.append(Segment(start=seg.start, end=seg.end, label=seg.label))

    return final_segments
