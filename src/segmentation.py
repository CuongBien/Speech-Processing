"""
src/segmentation.py – Phân loại khung thời gian, tạo đoạn, lọc khoảng lặng ngắn và tính sai số ranh giới.
Cài đặt quy tắc loại bỏ khoảng lặng ảo < 300ms theo đề bài và tính toán MAE, RMSE (đơn vị ms).
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import (
    BOUNDARY_MATCH_TOLERANCE_MS, FRAME_SHIFT_MS,
    MIN_SILENCE_DURATION_MS, Segment,
)

logger = logging.getLogger(__name__)


def classify_frames(
    feature_values: np.ndarray,
    threshold: float,
    speech_is_above: bool = True,
) -> np.ndarray:
    """
    Phân loại từng khung dựa trên ngưỡng T:
    - speech_is_above=True: đặc trưng >= T là tiếng nói (1), < T là khoảng lặng (0).

    Tham số:
        feature_values (np.ndarray): Mảng giá trị đặc trưng ngắn hạn.
        threshold (float): Ngưỡng phân tách T.
        speech_is_above (bool): Tiếng nói có giá trị năng lượng cao hơn khoảng lặng.

    Trả về:
        np.ndarray: Mảng nhãn nhị phân kiểu int32 (1: speech, 0: silence).
    """
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
    Ranh giới nội bộ giữa 2 khung đổi nhãn được lấy tại điểm giữa của 2 tâm khung:
        boundary = (center[i-1] + center[i]) / 2.0

    Tham số:
        labels (np.ndarray): Chuỗi nhãn của các khung (0 hoặc 1).
        frame_centers (np.ndarray): Mảng thời gian tâm của từng khung.
        signal_duration_s (float): Tổng thời lượng tín hiệu (giây).
        frame_shift_ms (int): Bước dịch khung.

    Trả về:
        List[Segment]: Danh sách các đoạn phân chia thời gian.
    """
    if len(labels) == 0:
        return []

    segments: List[Segment] = []
    current_label = labels[0]
    current_start = 0.0

    # Duyệt qua các khung và tạo ranh giới khi có sự đổi nhãn
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            boundary_time = (frame_centers[i - 1] + frame_centers[i]) / 2.0
            seg_name = "speech" if current_label == 1 else "silence"
            segments.append(Segment(start=current_start, end=boundary_time, label=seg_name))
            current_label = labels[i]
            current_start = boundary_time

    # Điểm kết thúc của đoạn cuối cùng
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

    Tham số:
        segments (List[Segment]): Danh sách các đoạn ban đầu.
        min_duration_ms (float): Ngưỡng độ dài tối thiểu của khoảng lặng (mili-giây).

    Trả về:
        List[Segment]: Danh sách các đoạn đã được lọc sạch khoảng lặng ảo.
    """
    if not segments:
        return []

    min_sec = min_duration_ms / 1000.0
    current_list = list(segments)
    has_changed = True

    # Lặp lại quá trình lọc cho tới khi không còn khoảng lặng nào < min_duration_ms
    while has_changed:
        has_changed = False
        filtered: List[Segment] = []
        idx = 0

        while idx < len(current_list):
            seg = current_list[idx]

            # Kiểm tra nếu là khoảng lặng và thời lượng ngắn hơn 300ms
            if seg.label == "silence" and seg.duration_s < min_sec:
                has_changed = True
                has_prev = len(filtered) > 0
                has_next = idx < len(current_list) - 1

                if has_prev and has_next:
                    # Khoảng lặng ngắn kẹp giữa 2 đoạn tiếng nói -> Nối liền cả 3 thành 1 đoạn tiếng nói
                    prev_seg = filtered.pop()
                    next_seg = current_list[idx + 1]
                    merged = Segment(start=prev_seg.start, end=next_seg.end, label="speech")
                    filtered.append(merged)
                    idx += 2
                elif has_prev:
                    # Khoảng lặng ngắn ở cuối tín hiệu -> Gộp vào đoạn trước
                    prev_seg = filtered.pop()
                    merged = Segment(start=prev_seg.start, end=seg.end, label=prev_seg.label)
                    filtered.append(merged)
                    idx += 1
                elif has_next:
                    # Khoảng lặng ngắn ở đầu tín hiệu -> Gộp vào đoạn kế tiếp
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

    # Ghép nối lại các đoạn liên tiếp cùng nhãn sau khi gộp
    final_segments: List[Segment] = []
    for seg in current_list:
        if final_segments and final_segments[-1].label == seg.label:
            final_segments[-1] = Segment(start=final_segments[-1].start, end=seg.end, label=seg.label)
        else:
            final_segments.append(Segment(start=seg.start, end=seg.end, label=seg.label))

    return final_segments


def extract_boundaries(segments: List[Segment]) -> List[float]:
    """
    Trích xuất các mốc thời gian chuyển đổi ranh giới nội bộ (bỏ qua mốc 0 và kết thúc).

    Tham số:
        segments (List[Segment]): Danh sách các đoạn.

    Trả về:
        List[float]: Danh sách các mốc thời gian ranh giới (giây).
    """
    if not segments:
        return []
    return [round(s.end, 4) for s in segments[:-1]]


def match_boundaries(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
    tolerance_s: float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """
    Ghép cặp ranh giới dự đoán với ranh giới chuẩn (Ground Truth) theo nguyên tắc lân cận gần nhất
    trong khoảng sai số dung sai cho phép (tolerance_s).

    Tham số:
        pred_boundaries (List[float]): Các mốc ranh giới thuật toán tìm được.
        gt_boundaries (List[float]): Các mốc ranh giới chuẩn từ file .lab.
        tolerance_s (float): Khoảng cách tối đa cho phép ghép cặp (giây).

    Trả về:
        Tuple:
            - matched_pairs: Danh sách cặp (pred_boundary, gt_boundary).
            - unmatched_pred: Ranh giới dự đoán không ghép được với GT nào.
            - unmatched_gt: Ranh giới GT không được dự đoán khớp.
    """
    used_pred = [False] * len(pred_boundaries)
    matched: List[Tuple[float, float]] = []
    unmatched_gt: List[float] = []

    # Với mỗi mốc GT, tìm mốc dự đoán gần nhất chưa được ghép
    for g in gt_boundaries:
        best_idx = -1
        min_dist = float("inf")
        for j, p in enumerate(pred_boundaries):
            dist = abs(p - g)
            if not used_pred[j] and dist < min_dist:
                min_dist = dist
                best_idx = j

        # Nếu khoảng cách nằm trong dung sai cho phép thì ghép cặp thành công
        if best_idx >= 0 and min_dist <= tolerance_s:
            matched.append((pred_boundaries[best_idx], g))
            used_pred[best_idx] = True
        else:
            unmatched_gt.append(g)

    unmatched_pred = [pred_boundaries[j] for j in range(len(pred_boundaries)) if not used_pred[j]]
    return matched, unmatched_pred, unmatched_gt


def evaluate_boundaries(
    pred_segments: List[Segment],
    gt_segments: List[Segment],
    tolerance_ms: float = BOUNDARY_MATCH_TOLERANCE_MS,
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số định lượng đánh giá độ chính xác của thuật toán:
    - MAE (Mean Absolute Error) tính theo mili-giây (ms).
    - RMSE (Root Mean Squared Error) tính theo mili-giây (ms).
    - Precision, Recall, F1-Score.

    Tham số:
        pred_segments (List[Segment]): Các đoạn phân tách do thuật toán xuất ra.
        gt_segments (List[Segment]): Các đoạn phân tách Ground Truth.
        tolerance_ms (float): Dung sai ghép cặp (mili-giây).

    Trả về:
        Dict[str, Any]: Từ điển chứa toàn bộ các chỉ số đo lường.
    """
    pred_bounds = extract_boundaries(pred_segments)
    gt_bounds = extract_boundaries(gt_segments)

    tol_s = tolerance_ms / 1000.0
    matched_pairs, unmatched_pred, unmatched_gt = match_boundaries(pred_bounds, gt_bounds, tol_s)

    n_pred = len(pred_bounds)
    n_gt = len(gt_bounds)
    n_matched = len(matched_pairs)

    # Tính Precision, Recall và F1
    precision = (n_matched / n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    recall = (n_matched / n_gt) if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    # Nếu không ghép được cặp nào
    if n_matched == 0:
        return {
            "mae_ms": float("nan"),
            "rmse_ms": float("nan"),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "n_matched": 0,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "pred_boundaries": pred_bounds,
            "gt_boundaries": gt_bounds,
            "errors_ms": [],
        }

    # Tính sai số tuyệt đối (ms) cho từng cặp ranh giới đã ghép
    errors_ms = [abs(p - g) * 1000.0 for p, g in matched_pairs]
    err_array = np.array(errors_ms, dtype=np.float64)

    mae_val = float(np.mean(err_array))
    rmse_val = float(np.sqrt(np.mean(err_array ** 2)))

    return {
        "mae_ms": mae_val,
        "rmse_ms": rmse_val,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_matched": n_matched,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "pred_boundaries": pred_bounds,
        "gt_boundaries": gt_bounds,
        "errors_ms": errors_ms,
    }
