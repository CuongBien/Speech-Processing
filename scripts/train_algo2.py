"""
scripts/train_algo2.py – Kịch bản huấn luyện & khảo sát Thuật toán 2 (Giannakopoulos 2014).
Chuẩn 100% bài báo gốc: 2 đặc trưng (Short-Term Energy & Spectral Centroid).

Thực hiện:
1. Đọc và phân tích 4 file tín hiệu trong TinHieuHuanLuyen.
2. Rút trích đồng thời Năng lượng ngắn hạn E(i) và Trọng tâm phổ C(i) (qua DFT trên NumPy thuần).
3. Phân tích 2 Histogram độc lập: Tìm các cực đại địa phương M1 (khoảng lặng) và M2 (tiếng nói).
4. Tính toán 2 ngưỡng tối ưu có trọng số:
   T1 = (W_E * M1_E + M2_E) / (W_E + 1)
   T2 = (W_C * M1_C + M2_C) / (W_C + 1)
5. Đánh giá sai số MAE, RMSE (đơn vị ms), Precision, Recall và F1-Score so với Ground Truth .lab.
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
    MIN_SILENCE_DURATION_MS, OUTPUT_DIR, TRAINING_DIR,
)
from src.features import compute_energy_and_spectral_centroid
from src.algorithms.histogram import (
    HistogramSegmenter, compute_feature_histogram_threshold,
)
from src.segmentation import evaluate_boundaries
from src.visualization import (
    plot_dual_histogram_analysis, plot_single_file_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_algo2")

DEFAULT_NUM_BINS = 60
DEFAULT_WEIGHT_ENERGY = 5.0
DEFAULT_WEIGHT_CENTROID = 2.0
EXPAND_FRAMES = 3


def run_training_algorithm2():
    """Hàm chính điều phối quy trình huấn luyện Thuật toán 2 (Giannakopoulos 2014)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    training_path = Path(TRAINING_DIR)
    wav_files = sorted(training_path.glob("*.wav"))

    if not wav_files:
        logger.error("Không tìm thấy file .wav nào trong: %s", TRAINING_DIR)
        return

    logger.info("=" * 85)
    logger.info("HUẤN LUYỆN THUẬT TOÁN 2: GIANNAKOPOULOS 2014 (ENERGY & SPECTRAL CENTROID)")
    logger.info("Đặc trưng: Energy E & Spectral Centroid C (DFT) | Bins: %d | W_E: %.1f | W_C: %.1f | Lọc silence: >= %d ms",
                DEFAULT_NUM_BINS, DEFAULT_WEIGHT_ENERGY, DEFAULT_WEIGHT_CENTROID, int(MIN_SILENCE_DURATION_MS))
    logger.info("=" * 85)

    file_data_list: List[Dict[str, Any]] = []
    all_energy: List[np.ndarray] = []
    all_centroid: List[np.ndarray] = []

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

        energy_vals, centroid_vals, frame_centers = compute_energy_and_spectral_centroid(
            signal=signal,
            sample_rate=sr,
            frame_length_ms=FRAME_LENGTH_MS,
            frame_shift_ms=FRAME_SHIFT_MS,
            use_hamming=True,
        )

        all_energy.append(energy_vals)
        all_centroid.append(centroid_vals)

        # Khởi tạo segmenter để tìm ngưỡng động cho riêng file này
        seg_file = HistogramSegmenter(
            num_bins=DEFAULT_NUM_BINS,
            weight_energy=DEFAULT_WEIGHT_ENERGY,
            weight_centroid=DEFAULT_WEIGHT_CENTROID,
            expand_frames=EXPAND_FRAMES,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        )
        t_e_dyn, t_c_dyn, dyn_segs = seg_file.fit_and_segment_dynamic(
            energy_vals=energy_vals,
            centroid_vals=centroid_vals,
            frame_centers=frame_centers,
            signal_duration_s=len(signal) / sr,
        )
        dyn_metrics = evaluate_boundaries(dyn_segs, gt_segments)

        logger.info(
            "File: %-12s | SNR: %5.1f dB | Khung: %4d | T1_Energy: %8.6f | T2_Centroid: %5.1f | F1: %5.1f%%",
            stem, snr_val, len(energy_vals), t_e_dyn, t_c_dyn, dyn_metrics.get("f1_score", 0.0) * 100,
        )

        file_data_list.append({
            "stem": stem,
            "signal": signal,
            "sr": sr,
            "duration_s": len(signal) / sr,
            "gt_segments": gt_segments,
            "snr_db": snr_val,
            "energy_vals": energy_vals,
            "centroid_vals": centroid_vals,
            "frame_centers": frame_centers,
            "t_e_dyn": t_e_dyn,
            "t_c_dyn": t_c_dyn,
            "dyn_metrics": dyn_metrics,
            "dyn_segs": dyn_segs,
            "hist_info_energy": seg_file.energy_hist_info,
            "hist_info_centroid": seg_file.centroid_hist_info,
        })

    # 2. Tìm ngưỡng dùng chung gộp (Global Histogram Thresholds)
    pooled_energy = np.concatenate(all_energy)
    pooled_centroid = np.concatenate(all_centroid)

    global_t1, global_hist_e = compute_feature_histogram_threshold(
        feature_vals=pooled_energy,
        num_bins=DEFAULT_NUM_BINS,
        weight=DEFAULT_WEIGHT_ENERGY,
    )
    global_t2, global_hist_c = compute_feature_histogram_threshold(
        feature_vals=pooled_centroid,
        num_bins=DEFAULT_NUM_BINS,
        weight=DEFAULT_WEIGHT_CENTROID,
    )

    logger.info("-" * 85)
    logger.info(">>> NGƯỠNG DÙNG CHUNG TOÀN CỤC: T1_Energy = %.6f | T2_Centroid = %.2f <<<", global_t1, global_t2)
    logger.info("-" * 85)

    # Xuất đồ thị 2 Histogram toàn cục
    fig_hist = plot_dual_histogram_analysis(
        energy_info=global_hist_e,
        centroid_info=global_hist_c,
        title_text="Dữ liệu gộp 4 file huấn luyện (Thuật toán 2)",
        save_path=os.path.join(FIGURES_DIR, "global_histogram_dual_features.png"),
    )
    plt.close(fig_hist)

    # 3. Đánh giá và xuất biểu đồ từng file (ở chế độ Ngưỡng động thích nghi - chuẩn Giannakopoulos)
    logger.info("KẾT QUẢ ĐÁNH GIÁ (CHẾ ĐỘ NGƯỠNG ĐỘNG ADAPTIVE HISTOGRAM 2-FEATURE):")
    logger.info("%-12s | %-8s | %-12s | %-12s | %-10s | %-10s | %-10s",
                "File", "SNR (dB)", "T1 (Energy)", "T2 (Centroid)", "MAE (ms)", "RMSE (ms)", "F1-Score")
    logger.info("-" * 95)

    eval_summary = []
    valid_maes, valid_rmses = [], []

    for idx, item in enumerate(file_data_list, start=1):
        stem = item["stem"]
        metrics = item["dyn_metrics"]
        t_e = item["t_e_dyn"]
        t_c = item["t_c_dyn"]

        mae_str = f"{metrics['mae_ms']:10.1f}" if not np.isnan(metrics.get("mae_ms", float("nan"))) else "       nan"
        rmse_str = f"{metrics['rmse_ms']:10.1f}" if not np.isnan(metrics.get("rmse_ms", float("nan"))) else "       nan"

        logger.info(
            "%-12s | %8.1f | %12.6f | %12.2f | %s | %s | %9.1f%%",
            stem, item["snr_db"], t_e, t_c, mae_str, rmse_str, metrics.get("f1_score", 0.0) * 100,
        )

        if not np.isnan(metrics.get("mae_ms", float("nan"))):
            valid_maes.append(metrics["mae_ms"])
            valid_rmses.append(metrics["rmse_ms"])

        # Xuất đồ thị 3 subplot cho từng file
        fig_single = plot_single_file_result(
            signal=item["signal"],
            sample_rate=item["sr"],
            feature_vals=item["energy_vals"],
            frame_centers=item["frame_centers"],
            pred_segments=item["dyn_segs"],
            gt_segments=item["gt_segments"],
            threshold=t_e,
            feature_name="Energy",
            title_text=f"{stem} (Giannakopoulos 2014)",
            metrics=metrics,
            save_path=os.path.join(FIGURES_DIR, f"{stem}_segmentation_algo2.png"),
            fig_num=idx,
            corner_label=f"[File {idx}]",
            snr_db=item["snr_db"],
            centroid_vals=item["centroid_vals"],
            threshold_centroid=t_c,
        )
        plt.close(fig_single)

        eval_summary.append({
            "stem": stem,
            "snr_db": round(item["snr_db"], 2),
            "threshold_energy": round(t_e, 6),
            "threshold_centroid": round(t_c, 2),
            "mae_ms": round(metrics["mae_ms"], 2) if not np.isnan(metrics.get("mae_ms", float("nan"))) else None,
            "rmse_ms": round(metrics["rmse_ms"], 2) if not np.isnan(metrics.get("rmse_ms", float("nan"))) else None,
            "precision": round(metrics.get("precision", 0.0), 4),
            "recall": round(metrics.get("recall", 0.0), 4),
            "f1_score": round(metrics.get("f1_score", 0.0), 4),
        })

    avg_mae = float(np.mean(valid_maes)) if valid_maes else float("nan")
    avg_rmse = float(np.mean(valid_rmses)) if valid_rmses else float("nan")

    logger.info("-" * 95)
    logger.info("TRUNG BÌNH TOÀN BỘ CÁC FILE: MAE = %.2f ms | RMSE = %.2f ms", avg_mae, avg_rmse)
    logger.info("=" * 95)

    config_data = {
        "algorithm": "Giannakopoulos 2014 (Histogram 2-Feature: Energy & Spectral Centroid)",
        "features": ["Short-Time Energy (E)", "Spectral Centroid (C)"],
        "frame_length_ms": FRAME_LENGTH_MS,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "min_silence_duration_ms": MIN_SILENCE_DURATION_MS,
        "weight_energy": DEFAULT_WEIGHT_ENERGY,
        "weight_centroid": DEFAULT_WEIGHT_CENTROID,
        "expand_frames": EXPAND_FRAMES,
        "global_threshold_energy": float(global_t1),
        "global_threshold_centroid": float(global_t2),
        "global_threshold": float(global_t1),
        "average_mae_ms": round(avg_mae, 2) if not np.isnan(avg_mae) else None,
        "average_rmse_ms": round(avg_rmse, 2) if not np.isnan(avg_rmse) else None,
        "files": eval_summary,
    }

    json_dest = os.path.join(OUTPUT_DIR, "histogram_threshold.json")
    save_threshold_json(config_data, json_dest)
    logger.info("Đã lưu cấu hình ngưỡng tối ưu vào: %s", json_dest)


if __name__ == "__main__":
    run_training_algorithm2()
