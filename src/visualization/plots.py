"""
src/visualization/plots.py – Các hàm vẽ đồ thị trực quan hóa tín hiệu, hàm năng lượng và phân đoạn.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.config import Segment
from src.segmentation.metrics import extract_boundaries

logger = logging.getLogger(__name__)


def plot_single_file_result(
    signal: np.ndarray,
    sample_rate: int,
    feature_vals: np.ndarray,
    frame_centers: np.ndarray,
    pred_segments: List[Segment],
    gt_segments: List[Segment],
    threshold: float,
    feature_name: str,
    title_text: str,
    metrics: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    fig_num: int = 1,
) -> plt.Figure:
    """
    Vẽ kết quả phân đoạn cho 1 file tín hiệu gồm 2 subplot:
    - Subplot 1: Dạng sóng tín hiệu gốc + ranh giới GT (đỏ) + ranh giới dự đoán (xanh).
    - Subplot 2: Đường đặc trưng ngắn hạn + ngưỡng T (xanh lá) + ranh giới.
    """
    total_time_s = len(signal) / sample_rate
    time_axis = np.linspace(0.0, total_time_s, len(signal), endpoint=False)

    gt_bounds = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    fig, (ax_wave, ax_feat) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, num=fig_num, clear=True)

    # 1. Subplot trên: Dạng sóng tín hiệu
    ax_wave.plot(time_axis, signal, color="#2c3e50", linewidth=0.6, alpha=0.85, label="Waveform")
    ax_wave.set_ylabel("Biên độ (Amplitude)", fontsize=10, fontweight="bold")

    if metrics:
        mae_str = f"{metrics.get('mae_ms', float('nan')):.1f} ms"
        rmse_str = f"{metrics.get('rmse_ms', float('nan')):.1f} ms"
        f1_str = f"{metrics.get('f1_score', 0.0) * 100:.1f}%"
        main_title = f"{title_text} | MAE: {mae_str} | RMSE: {rmse_str} | F1: {f1_str}"
    else:
        main_title = f"{title_text} - Phân đoạn Tiếng nói / Khoảng lặng"

    ax_wave.set_title(main_title, fontsize=11, fontweight="bold", pad=8)
    ax_wave.grid(True, linestyle=":", alpha=0.6)

    for seg in pred_segments:
        if seg.label == "speech":
            ax_wave.axvspan(seg.start, seg.end, color="#27ae60", alpha=0.12)

    for i, b in enumerate(gt_bounds):
        ax_wave.axvline(b, color="#e74c3c", linewidth=1.8, linestyle="-",
                        label="Biên chuẩn (GT)" if i == 0 else "")

    for i, b in enumerate(pred_bounds):
        ax_wave.axvline(b, color="#2980b9", linewidth=1.8, linestyle="--",
                        label="Biên dự đoán (Predicted)" if i == 0 else "")

    ax_wave.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # 2. Subplot dưới: Hàm đặc trưng ngắn hạn
    ax_feat.plot(frame_centers, feature_vals, color="#e67e22", linewidth=1.2, label=f"Đặc trưng {feature_name}")
    ax_feat.axhline(threshold, color="#27ae60", linewidth=1.8, linestyle="-.",
                    label=f"Ngưỡng T = {threshold:.4f}")

    ax_feat.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    ax_feat.set_ylabel(f"Hàm {feature_name}", fontsize=10, fontweight="bold")
    ax_feat.set_xlim(0, total_time_s)
    ax_feat.grid(True, linestyle=":", alpha=0.6)

    for b in gt_bounds:
        ax_feat.axvline(b, color="#e74c3c", linewidth=1.5, linestyle="-")
    for b in pred_bounds:
        ax_feat.axvline(b, color="#2980b9", linewidth=1.5, linestyle="--")

    ax_feat.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_distribution_and_overlap(
    speech_vals: np.ndarray,
    silence_vals: np.ndarray,
    overlap_info: Dict[str, Any],
    threshold: float,
    feature_name: str,
    title_text: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ phân phối (Histogram / Density) của đặc trưng giữa Tiếng nói và Khoảng lặng,
    làm nổi bật dải giao thoa (overlap) và ngưỡng tối ưu T tìm được.
    """
    fig, ax = plt.subplots(figsize=(9, 5), clear=True)

    bins = np.linspace(
        min(np.min(speech_vals), np.min(silence_vals)),
        max(np.max(speech_vals), np.max(silence_vals)),
        80,
    )
    ax.hist(silence_vals, bins=bins, color="#3498db", alpha=0.6, density=True, label="Khoảng lặng (Silence)")
    ax.hist(speech_vals, bins=bins, color="#e67e22", alpha=0.6, density=True, label="Tiếng nói (Speech)")

    ov_min = overlap_info.get("overlap_min", 0.0)
    ov_max = overlap_info.get("overlap_max", 0.0)
    ax.axvspan(ov_min, ov_max, color="#95a5a6", alpha=0.25, label=f"Vùng overlap [{ov_min:.3f}, {ov_max:.3f}]")
    ax.axvline(threshold, color="#27ae60", linewidth=2.0, linestyle="--", label=f"Ngưỡng T = {threshold:.4f}")

    ax.set_title(f"Phân phối đặc trưng {feature_name} - {title_text}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Giá trị đặc trưng {feature_name}", fontsize=10)
    ax.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
