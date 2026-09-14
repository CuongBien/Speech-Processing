"""
src/visualization/plots.py – Các hàm vẽ đồ thị trực quan hóa tín hiệu, hàm năng lượng và phân đoạn.
Hỗ trợ cả Thuật toán 1 (1 đặc trưng logMA/MA) và Thuật toán 2 (2 đặc trưng Energy & Spectral Centroid).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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
    corner_label: str = "",
    snr_db: Optional[float] = None,
    centroid_vals: Optional[np.ndarray] = None,
    threshold_centroid: Optional[float] = None,
) -> plt.Figure:
    """
    Vẽ kết quả phân đoạn cho 1 file tín hiệu:
    - Nếu chỉ có 1 đặc trưng: Vẽ 2 subplot (Waveform + Feature).
    - Nếu có 2 đặc trưng (Thuật toán 2 Giannakopoulos): Vẽ 3 subplot (Waveform + Energy + Spectral Centroid).
    """
    total_time_s = len(signal) / sample_rate
    time_axis = np.linspace(0.0, total_time_s, len(signal), endpoint=False)

    gt_bounds = extract_boundaries(gt_segments)
    pred_bounds = extract_boundaries(pred_segments)

    has_dual = centroid_vals is not None and threshold_centroid is not None
    num_subplots = 3 if has_dual else 2
    figsize = (10, 7.5) if has_dual else (10, 6)

    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True, num=fig_num, clear=True)
    if num_subplots == 2:
        ax_wave, ax_feat = axes[0], axes[1]
    else:
        ax_wave, ax_feat, ax_cent = axes[0], axes[1], axes[2]

    # Tiêu đề chung
    snr_str = f" | SNR: {snr_db:.1f} dB" if snr_db is not None else ""
    if metrics:
        mae_str = f"{metrics.get('mae_ms', float('nan')):.1f} ms"
        rmse_str = f"{metrics.get('rmse_ms', float('nan')):.1f} ms"
        f1_str = f"{metrics.get('f1_score', 0.0) * 100:.1f}%"
        main_title = f"{title_text}{snr_str} | MAE: {mae_str} | RMSE: {rmse_str} | F1: {f1_str}"
        prefix = f"{corner_label} " if corner_label else ""
        window_title = f"{prefix}{title_text}{snr_str} | MAE: {mae_str} | F1: {f1_str}".strip()
    else:
        main_title = f"{title_text}{snr_str} - Phân đoạn Tiếng nói / Khoảng lặng"
        prefix = f"{corner_label} " if corner_label else ""
        window_title = f"{prefix}{title_text}{snr_str}".strip()

    if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
        if hasattr(fig.canvas.manager, "set_window_title"):
            try:
                fig.canvas.manager.set_window_title(window_title)
            except Exception:
                pass

    # 1. Subplot 1: Dạng sóng tín hiệu gốc
    ax_wave.plot(time_axis, signal, color="#2c3e50", linewidth=0.6, alpha=0.85, label="Waveform")
    ax_wave.set_ylabel("Biên độ", fontsize=9, fontweight="bold")
    ax_wave.set_title(main_title, fontsize=10, fontweight="bold", pad=6)
    ax_wave.grid(True, linestyle=":", alpha=0.6)

    for seg in pred_segments:
        if seg.label == "speech":
            ax_wave.axvspan(seg.start, seg.end, color="#27ae60", alpha=0.15)

    for i, b in enumerate(gt_bounds):
        ax_wave.axvline(b, color="#e74c3c", linewidth=1.8, linestyle="-",
                        label="Biên chuẩn (GT)" if i == 0 else "")

    for i, b in enumerate(pred_bounds):
        ax_wave.axvline(b, color="#2980b9", linewidth=1.8, linestyle="--",
                        label="Biên dự đoán" if i == 0 else "")

    ax_wave.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # 2. Subplot 2: Đặc trưng năng lượng / MA
    feat_label = f"Năng lượng {feature_name}" if has_dual else f"Đặc trưng {feature_name}"
    ax_feat.plot(frame_centers, feature_vals, color="#e67e22", linewidth=1.2, label=feat_label)
    ax_feat.axhline(threshold, color="#27ae60", linewidth=1.8, linestyle="-.",
                    label=f"Ngưỡng T1 = {threshold:.6f}" if has_dual else f"Ngưỡng T = {threshold:.4f}")

    ax_feat.set_ylabel(feature_name, fontsize=9, fontweight="bold")
    ax_feat.set_xlim(0, total_time_s)
    ax_feat.grid(True, linestyle=":", alpha=0.6)

    for b in gt_bounds:
        ax_feat.axvline(b, color="#e74c3c", linewidth=1.4, linestyle="-")
    for b in pred_bounds:
        ax_feat.axvline(b, color="#2980b9", linewidth=1.4, linestyle="--")

    ax_feat.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # 3. Subplot 3 (nếu có): Trọng tâm phổ Spectral Centroid
    if has_dual:
        ax_cent.plot(frame_centers, centroid_vals, color="#8e44ad", linewidth=1.2, label="Trọng tâm phổ (Spectral Centroid)")
        ax_cent.axhline(threshold_centroid, color="#d35400", linewidth=1.8, linestyle="-.",
                        label=f"Ngưỡng T2 = {threshold_centroid:.1f}")
        ax_cent.set_ylabel("Spectral Centroid", fontsize=9, fontweight="bold")
        ax_cent.set_xlim(0, total_time_s)
        ax_cent.grid(True, linestyle=":", alpha=0.6)

        for b in gt_bounds:
            ax_cent.axvline(b, color="#e74c3c", linewidth=1.4, linestyle="-")
        for b in pred_bounds:
            ax_cent.axvline(b, color="#2980b9", linewidth=1.4, linestyle="--")

        ax_cent.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_cent.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
    else:
        ax_feat.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")

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


def plot_dual_histogram_analysis(
    energy_info: Dict[str, Any],
    centroid_info: Dict[str, Any],
    title_text: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ phân tích 2 biểu đồ Histogram (Energy & Spectral Centroid) theo Giannakopoulos (2014):
    - Cột trái: Histogram Năng lượng E với đỉnh M1, M2 và ngưỡng T1.
    - Cột phải: Histogram Trọng tâm phổ C với đỉnh M1, M2 và ngưỡng T2.
    """
    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(13, 5), clear=True)

    # 1. Histogram Năng lượng E
    h_e = energy_info.get("hist", np.array([]))
    c_e = energy_info.get("bin_centers", np.array([]))
    t1 = energy_info.get("threshold", 0.0)
    m1_e = energy_info.get("m1", 0.0)
    m2_e = energy_info.get("m2", 0.0)
    w_e = energy_info.get("weight", 4.0)

    if len(c_e) > 1:
        w = (c_e[1] - c_e[0]) * 0.9
        ax_e.bar(c_e, h_e, width=w, color="#e67e22", alpha=0.65, edgecolor="#d35400", label="Hist Năng lượng (E)")
        ax_e.axvline(m1_e, color="#2980b9", linestyle=":", linewidth=2, label=f"M1_E (Sil) = {m1_e:.6f}")
        ax_e.axvline(m2_e, color="#27ae60", linestyle=":", linewidth=2, label=f"M2_E (Spch) = {m2_e:.6f}")
        ax_e.axvline(t1, color="#c0392b", linestyle="--", linewidth=2.2, label=f"T1 = {t1:.6f} (W={w_e})")

    ax_e.set_title("Biểu đồ Histogram Năng lượng (Energy)", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("Năng lượng E", fontsize=10)
    ax_e.set_ylabel("Số lượng khung", fontsize=10)
    ax_e.grid(True, linestyle=":", alpha=0.5)
    ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # 2. Histogram Trọng tâm phổ C
    h_c = centroid_info.get("hist", np.array([]))
    c_c = centroid_info.get("bin_centers", np.array([]))
    t2 = centroid_info.get("threshold", 0.0)
    m1_c = centroid_info.get("m1", 0.0)
    m2_c = centroid_info.get("m2", 0.0)
    w_c = centroid_info.get("weight", 1.5)

    if len(c_c) > 1:
        w = (c_c[1] - c_c[0]) * 0.9
        ax_c.bar(c_c, h_c, width=w, color="#8e44ad", alpha=0.65, edgecolor="#6c3483", label="Hist Trọng tâm phổ (C)")
        ax_c.axvline(m1_c, color="#2980b9", linestyle=":", linewidth=2, label=f"M1_C (Sil) = {m1_c:.1f}")
        ax_c.axvline(m2_c, color="#27ae60", linestyle=":", linewidth=2, label=f"M2_C (Spch) = {m2_c:.1f}")
        ax_c.axvline(t2, color="#c0392b", linestyle="--", linewidth=2.2, label=f"T2 = {t2:.1f} (W={w_c})")

    ax_c.set_title("Biểu đồ Histogram Trọng tâm phổ (Spectral Centroid)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Chỉ số Trọng tâm phổ C", fontsize=10)
    ax_c.set_ylabel("Số lượng khung", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"Phân tích 2 Histogram (Giannakopoulos 2014) - {title_text}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_histogram_analysis(
    hist: np.ndarray,
    bin_centers: np.ndarray,
    peaks: List[Tuple[int, float, float]],
    threshold: float,
    feature_name: str,
    title_text: str,
    weight: float = 4.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ phân tích 1 Histogram đơn biến (tương thích ngược).
    """
    fig, ax = plt.subplots(figsize=(9, 5), clear=True)

    width = (bin_centers[1] - bin_centers[0]) * 0.9 if len(bin_centers) > 1 else 0.01
    ax.bar(bin_centers, hist, width=width, color="#3498db", alpha=0.65, label=f"Histogram {feature_name}", edgecolor="#2980b9")

    if len(peaks) >= 1:
        ax.plot(peaks[0][1], peaks[0][2], "ro", markersize=9, label=f"Đỉnh M1 (Khoảng lặng) = {peaks[0][1]:.4f}")
    if len(peaks) >= 2:
        ax.plot(peaks[1][1], peaks[1][2], "go", markersize=9, label=f"Đỉnh M2 (Tiếng nói) = {peaks[1][1]:.4f}")

    ax.axvline(threshold, color="#e74c3c", linewidth=2.0, linestyle="--",
               label=f"Ngưỡng T = {threshold:.4f} (W={weight})")

    ax.set_title(f"Phân tích Histogram - {title_text}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Giá trị đặc trưng {feature_name}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Số lượng khung (Counts)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

