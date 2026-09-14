"""
src/segmentation/metrics.py – Đo lường sai số định lượng (MAE, RMSE, F1) giữa ranh giới dự đoán và Ground Truth.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import BOUNDARY_MATCH_TOLERANCE_MS, Segment


def extract_boundaries(segments: List[Segment]) -> List[float]:
    """Trích xuất các mốc thời gian chuyển đổi ranh giới nội bộ (giây)."""
    if not segments:
        return []
    return [round(s.end, 4) for s in segments[:-1]]


def match_boundaries(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
    tolerance_s: float = BOUNDARY_MATCH_TOLERANCE_MS / 1000.0,
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """Ghép cặp ranh giới dự đoán với Ground Truth trong khoảng sai số cho phép."""
    used_pred = [False] * len(pred_boundaries)
    matched: List[Tuple[float, float]] = []
    unmatched_gt: List[float] = []

    for g in gt_boundaries:
        best_idx = -1
        min_dist = float("inf")
        for j, p in enumerate(pred_boundaries):
            dist = abs(p - g)
            if not used_pred[j] and dist < min_dist:
                min_dist = dist
                best_idx = j

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
    """
    pred_bounds = extract_boundaries(pred_segments)
    gt_bounds = extract_boundaries(gt_segments)

    tol_s = tolerance_ms / 1000.0
    matched_pairs, unmatched_pred, unmatched_gt = match_boundaries(pred_bounds, gt_bounds, tol_s)

    n_pred = len(pred_bounds)
    n_gt = len(gt_bounds)
    n_matched = len(matched_pairs)

    precision = (n_matched / n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    recall = (n_matched / n_gt) if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

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
