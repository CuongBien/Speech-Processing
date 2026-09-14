"""
scripts/train_algo2.py – Kịch bản huấn luyện & khảo sát Thuật toán 2 (Giannakopoulos 2014 - Histogram).
Thực hiện:
1. Đọc và phân tích 4 file tín hiệu trong TinHieuHuanLuyen.
2. Rút trích hàm độ lớn ngắn hạn (MA) và lọc trung vị (Median Filter 1D).
3. Phân tích Histogram để xác định 2 cực đại địa phương: Đỉnh khoảng lặng M1 và đỉnh tiếng nói M2.
4. Tính toán ngưỡng phân tách có trọng số: T = (Weight * V1 + V2) / (Weight + 1).
5. Đánh giá sai số MAE, RMSE (đơn vị ms) và F1-Score so với Ground Truth .lab.
6. Lưu kết quả ra output/histogram_threshold.json và xuất biểu đồ vào output/figures/.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Cấu hình UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.audio import (
    compute_snr_db, load_audio, load_ground_truth, parse_lab_file,
    save_threshold_json,
)
from src.config import (
    FIGURES_DIR, FRAME_LENGTH_MS, FRAME_SHIFT_MS,
    MIN_SILENCE_DURATION_MS, OUTPUT_DIR, TRAINING_DIR, FeatureType,
)
from src.features import compute_short_time_feature
from src.algorithms.histogram import (
    compute_histogram_threshold, median_filter_1d,
)
from src.segmentation import (
    classify_frames, evaluate_boundaries, frames_to_segments,
    remove_short_silence,
)
from src.visualization import (
    plot_histogram_analysis, plot_single_file_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_algo2")

HIST_FEATURE_TYPE = FeatureType.MA
DEFAULT_NUM_BINS = 60
DEFAULT_WEIGHT = 4.0


def run_training_algorithm2():
    """Hàm chính điều phối quy trình huấn luyện Thuật toán 2 (Histogram)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    training_path = Path(TRAINING_DIR)
    wav_files = sorted(training_path.glob("*.wav"))

    if not wav_files:
        logger.error("Không tìm thấy file .wav nào trong: %s", TRAINING_DIR)
        return

    logger.info("=" * 80)
    logger.info("HUẤN LUYỆN THUẬT TOÁN 2: GIANNAKOPOULOS 2014 (HISTOGRAM-BASED)")
    logger.info("Đặc trưng: %s | Lọc trung vị: K=5 | Trọng số: W=%.1f | Bins: %d | Lọc silence: >= %d ms",
                HIST_FEATURE_TYPE.value, DEFAULT_WEIGHT, DEFAULT_NUM_BINS, int(MIN_SILENCE_DURATION_MS))
    logger.info("=" * 80)

    file_data_list: List[Dict[str, Any]] = []
    all_smoothed_features: List[np.ndarray] = []

    # 1. Quét và tính đặc trưng trên từng file
    for wav_f in wav_files:
        stem = wav_f.stem
        lab_f = wav_f.with_suffix(".lab")

        if not lab_f.exists():
            continue

        signal, sr = load_audio(str(wav_f))
        lab_segs = parse_lab_file(str(lab_f))
        gt_segments = load_ground_truth(lab_segs)
        snr_val = compute_snr_db(signal, sr, lab_segs)

        feat_vals, frame_centers = compute_short_time_feature(
            signal=signal,
            sample_rate=sr,
            frame_length_ms=FRAME_LENGTH_MS,
            frame_shift_ms=FRAME_SHIFT_MS,
            feature_type=HIST_FEATURE_TYPE,
        )

        # Lọc trung vị làm mịn
        smoothed_feat = median_filter_1d(feat_vals, kernel_size=5)
        all_smoothed_features.append(smoothed_feat)

        # Tìm ngưỡng riêng theo histogram của file này
        per_file_threshold, hist_info = compute_histogram_threshold(
            feature_vals=smoothed_feat,
            num_bins=DEFAULT_NUM_BINS,
            weight=DEFAULT_WEIGHT,
        )

        n_peaks = len(hist_info.get("peaks", []))
        logger.info(
            "File: %-12s | SNR: %5.1f dB | Khung: %4d | Số đỉnh phát hiện: %2d | Ngưỡng riêng T: %8.5f",
            stem, snr_val, len(feat_vals), n_peaks, per_file_threshold,
        )

        file_data_list.append({
            "stem": stem,
            "signal": signal,
            "sr": sr,
            "duration_s": len(signal) / sr,
            "gt_segments": gt_segments,
            "snr_db": snr_val,
            "feat_vals": feat_vals,
            "smoothed_feat": smoothed_feat,
            "frame_centers": frame_centers,
            "per_file_threshold": per_file_threshold,
            "hist_info": hist_info,
        })

    # 2. Tìm ngưỡng dùng chung gộp (Global Histogram Threshold)
    pooled_smoothed = np.concatenate(all_smoothed_features)
    global_threshold, global_hist_info = compute_histogram_threshold(
        feature_vals=pooled_smoothed,
        num_bins=DEFAULT_NUM_BINS,
        weight=DEFAULT_WEIGHT,
    )

    logger.info("-" * 80)
    logger.info(">>> NGƯỠNG DÙNG CHUNG HISTOGRAM T_global = %.5f (Số đỉnh: %d) <<<",
                global_threshold, len(global_hist_info.get("peaks", [])))
    if global_hist_info.get("v1") and global_hist_info.get("v2"):
        logger.info("  Đỉnh M1 (Khoảng lặng) = %.5f | Đỉnh M2 (Tiếng nói) = %.5f | Trọng số W = %.1f",
                    global_hist_info["v1"], global_hist_info["v2"], DEFAULT_WEIGHT)
    logger.info("-" * 80)

    # Vẽ và lưu biểu đồ phân tích Histogram toàn cục
    fig_hist = plot_histogram_analysis(
        hist=global_hist_info["hist"],
        bin_centers=global_hist_info["bin_centers"],
        peaks=global_hist_info["peaks"],
        threshold=global_threshold,
        feature_name=HIST_FEATURE_TYPE.value,
        title_text="Dữ liệu gộp 4 file huấn luyện",
        weight=DEFAULT_WEIGHT,
        save_path=os.path.join(FIGURES_DIR, "global_histogram_analysis.png"),
    )
    plt.close(fig_hist)

    # 3. Đánh giá định lượng
    logger.info("KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG VỚI GLOBAL HISTOGRAM THRESHOLD (T=%.5f):", global_threshold)
    logger.info("%-12s | %-8s | %-10s | %-10s | %-10s | %-10s | %-10s",
                "File", "SNR (dB)", "MAE (ms)", "RMSE (ms)", "Precision", "Recall", "F1-Score")
    logger.info("-" * 85)

    eval_summary = []
    all_maes, all_rmses = [], []

    for idx, item in enumerate(file_data_list, start=1):
        stem = item["stem"]
        smoothed_feat = item["smoothed_feat"]
        frame_centers = item["frame_centers"]
        signal = item["signal"]
        sr = item["sr"]
        gt_segments = item["gt_segments"]
        duration_s = item["duration_s"]

        # Phân loại và lọc khoảng lặng ngắn
        labels = classify_frames(smoothed_feat, global_threshold)
        raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s=duration_s)
        final_segs = remove_short_silence(raw_segs, min_duration_ms=MIN_SILENCE_DURATION_MS)

        metrics = evaluate_boundaries(final_segs, gt_segments)

        mae_str = f"{metrics['mae_ms']:10.1f}" if not np.isnan(metrics['mae_ms']) else "       nan"
        rmse_str = f"{metrics['rmse_ms']:10.1f}" if not np.isnan(metrics['rmse_ms']) else "       nan"

        logger.info(
            "%-12s | %8.1f | %s | %s | %9.1f%% | %9.1f%% | %9.1f%%",
            stem, item["snr_db"], mae_str, rmse_str,
            metrics["precision"] * 100, metrics["recall"] * 100, metrics["f1_score"] * 100,
        )

        if not np.isnan(metrics["mae_ms"]):
            all_maes.append(metrics["mae_ms"])
            all_rmses.append(metrics["rmse_ms"])

        # Xuất biểu đồ phân đoạn cho file này
        save_path = os.path.join(FIGURES_DIR, f"{stem}_segmentation_algo2.png")
        fig_single = plot_single_file_result(
            signal=signal,
            sample_rate=sr,
            feature_vals=smoothed_feat,
            frame_centers=frame_centers,
            pred_segments=final_segs,
            gt_segments=gt_segments,
            threshold=global_threshold,
            feature_name=f"{HIST_FEATURE_TYPE.value} (Smoothed)",
            title_text=f"{stem} - Thuật toán 2 (Histogram T={global_threshold:.5f})",
            metrics=metrics,
            save_path=save_path,
            fig_num=idx + 20,
        )
        plt.close(fig_single)

        eval_summary.append({
            "stem": stem,
            "snr_db": round(item["snr_db"], 2),
            "per_file_threshold": round(item["per_file_threshold"], 6),
            "mae_ms": round(metrics["mae_ms"], 2) if not np.isnan(metrics["mae_ms"]) else None,
            "rmse_ms": round(metrics["rmse_ms"], 2) if not np.isnan(metrics["rmse_ms"]) else None,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
        })

    avg_mae = float(np.mean(all_maes)) if all_maes else float("nan")
    avg_rmse = float(np.mean(all_rmses)) if all_rmses else float("nan")

    logger.info("-" * 85)
    logger.info("TRUNG BÌNH TOÀN BỘ TẬP HUẤN LUYỆN (THUẬT TOÁN 2): MAE = %.2f ms | RMSE = %.2f ms", avg_mae, avg_rmse)
    logger.info("=" * 85)

    config_data = {
        "algorithm": "Giannakopoulos 2014 (Histogram-based)",
        "feature_type": HIST_FEATURE_TYPE.value,
        "frame_length_ms": FRAME_LENGTH_MS,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "min_silence_duration_ms": MIN_SILENCE_DURATION_MS,
        "num_bins": DEFAULT_NUM_BINS,
        "weight": DEFAULT_WEIGHT,
        "global_threshold": float(global_threshold),
        "v1_silence_peak": global_hist_info.get("v1"),
        "v2_speech_peak": global_hist_info.get("v2"),
        "average_mae_ms": round(avg_mae, 2) if not np.isnan(avg_mae) else None,
        "average_rmse_ms": round(avg_rmse, 2) if not np.isnan(avg_rmse) else None,
        "files": eval_summary,
    }

    json_dest = os.path.join(OUTPUT_DIR, "histogram_threshold.json")
    save_threshold_json(config_data, json_dest)
    logger.info("Đã lưu cấu hình ngưỡng Histogram vào: %s", json_dest)


if __name__ == "__main__":
    run_training_algorithm2()
