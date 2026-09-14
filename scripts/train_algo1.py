"""
scripts/train_algo1.py – Kịch bản huấn luyện & khảo sát Thuật toán 1 (Hodgkinson 2012).
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

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.audio import (
    compute_snr_db, load_audio, load_ground_truth, parse_lab_file,
    save_threshold_json,
)
from src.config import (
    DEFAULT_FEATURE_TYPE, FIGURES_DIR, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS, OUTPUT_DIR, TRAINING_DIR,
)
from src.features import (
    assign_frame_labels, compute_short_time_feature,
)
from src.algorithms.binary_search import (
    find_optimal_threshold_binary_search,
)
from src.segmentation import (
    classify_frames, evaluate_boundaries, frames_to_segments,
    remove_short_silence,
)
from src.visualization import (
    plot_distribution_and_overlap, plot_single_file_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_algo1")


def run_training_algorithm1():
    """Hàm chính điều phối quy trình huấn luyện Thuật toán 1."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    training_path = Path(TRAINING_DIR)
    wav_files = sorted(training_path.glob("*.wav"))

    if not wav_files:
        logger.error("Không tìm thấy file .wav nào trong: %s", TRAINING_DIR)
        return

    logger.info("=" * 75)
    logger.info("HUẤN LUYỆN THUẬT TOÁN 1: HODGKINSON 2012 (BINARY SEARCH)")
    logger.info("Đặc trưng: %s | Khung: %d ms | Bước: %d ms | Lọc khoảng lặng: >= %d ms",
                DEFAULT_FEATURE_TYPE.value, FRAME_LENGTH_MS, FRAME_SHIFT_MS, int(MIN_SILENCE_DURATION_MS))
    logger.info("=" * 75)

    file_data_list: List[Dict[str, Any]] = []
    all_speech_features: List[np.ndarray] = []
    all_silence_features: List[np.ndarray] = []
    phone_speech, phone_silence = [], []
    studio_speech, studio_silence = [], []

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
            feature_type=DEFAULT_FEATURE_TYPE,
        )

        frame_labels = assign_frame_labels(frame_centers, lab_segs)
        speech_vals = feat_vals[frame_labels == 1]
        silence_vals = feat_vals[frame_labels == 0]

        all_speech_features.append(speech_vals)
        all_silence_features.append(silence_vals)

        if "phone" in stem:
            phone_speech.append(speech_vals)
            phone_silence.append(silence_vals)
        else:
            studio_speech.append(speech_vals)
            studio_silence.append(silence_vals)

        file_threshold, history, overlap_info = find_optimal_threshold_binary_search(
            speech_values=speech_vals,
            silence_values=silence_vals,
        )

        logger.info(
            "File: %-12s | SNR: %5.1f dB | Khung: %4d (Spch: %4d, Sil: %4d) | Ngưỡng riêng T: %8.4f",
            stem, snr_val, len(feat_vals), len(speech_vals), len(silence_vals), file_threshold,
        )

        file_data_list.append({
            "stem": stem,
            "wav_path": str(wav_f),
            "signal": signal,
            "sr": sr,
            "duration_s": len(signal) / sr,
            "lab_segs": lab_segs,
            "gt_segments": gt_segments,
            "snr_db": snr_val,
            "feat_vals": feat_vals,
            "frame_centers": frame_centers,
            "speech_vals": speech_vals,
            "silence_vals": silence_vals,
            "per_file_threshold": file_threshold,
            "overlap_info": overlap_info,
        })

    pooled_speech = np.concatenate(all_speech_features)
    pooled_silence = np.concatenate(all_silence_features)

    logger.info("-" * 75)
    global_threshold, _, global_overlap = find_optimal_threshold_binary_search(
        speech_values=pooled_speech,
        silence_values=pooled_silence,
    )

    phone_t, _, _ = find_optimal_threshold_binary_search(np.concatenate(phone_speech), np.concatenate(phone_silence))
    studio_t, _, _ = find_optimal_threshold_binary_search(np.concatenate(studio_speech), np.concatenate(studio_silence))

    logger.info(">>> NGƯỠNG DÙNG CHUNG TỐI ƯU T_global     = %.6f <<<", global_threshold)
    logger.info(">>> NGƯỠNG MÔI TRƯỜNG PHONE (SNR thấp)   = %.6f <<<", phone_t)
    logger.info(">>> NGƯỠNG MÔI TRƯỜNG STUDIO (SNR cao)   = %.6f <<<", studio_t)
    logger.info("-" * 75)

    # Biểu đồ phân phối
    fig_dist = plot_distribution_and_overlap(
        speech_vals=pooled_speech,
        silence_vals=pooled_silence,
        overlap_info=global_overlap,
        threshold=global_threshold,
        feature_name=DEFAULT_FEATURE_TYPE.value,
        title_text="Dữ liệu gộp 4 file huấn luyện (Thuật toán 1)",
        save_path=os.path.join(FIGURES_DIR, "global_distribution_overlap.png"),
    )
    plt.close(fig_dist)

    logger.info("KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG VỚI GLOBAL THRESHOLD (T=%.4f):", global_threshold)
    logger.info("%-12s | %-8s | %-10s | %-10s | %-10s | %-10s | %-10s",
                "File", "SNR (dB)", "MAE (ms)", "RMSE (ms)", "Precision", "Recall", "F1-Score")
    logger.info("-" * 85)

    eval_summary = []
    valid_maes, valid_rmses = [], []

    for idx, item in enumerate(file_data_list, start=1):
        stem = item["stem"]
        feat_vals = item["feat_vals"]
        frame_centers = item["frame_centers"]
        signal = item["signal"]
        sr = item["sr"]
        gt_segments = item["gt_segments"]
        duration_s = item["duration_s"]

        labels = classify_frames(feat_vals, global_threshold)
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
            valid_maes.append(metrics["mae_ms"])
            valid_rmses.append(metrics["rmse_ms"])

        fig_single = plot_single_file_result(
            signal=signal,
            sample_rate=sr,
            feature_vals=feat_vals,
            frame_centers=frame_centers,
            pred_segments=final_segs,
            gt_segments=gt_segments,
            threshold=global_threshold,
            feature_name=DEFAULT_FEATURE_TYPE.value,
            title_text=f"{stem} (Ngưỡng chung T={global_threshold:.4f})",
            metrics=metrics,
            save_path=os.path.join(FIGURES_DIR, f"{stem}_segmentation_global.png"),
            fig_num=idx,
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

    avg_mae = float(np.mean(valid_maes)) if valid_maes else float("nan")
    avg_rmse = float(np.mean(valid_rmses)) if valid_rmses else float("nan")

    logger.info("-" * 85)
    logger.info("TRUNG BÌNH CÁC FILE KHỚP ĐƯỢC RANH GIỚI: MAE = %.2f ms | RMSE = %.2f ms", avg_mae, avg_rmse)
    logger.info("=" * 85)

    config_data = {
        "algorithm": "Hodgkinson 2012 (Binary Search)",
        "feature_type": DEFAULT_FEATURE_TYPE.value,
        "frame_length_ms": FRAME_LENGTH_MS,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "min_silence_duration_ms": MIN_SILENCE_DURATION_MS,
        "global_threshold": float(global_threshold),
        "phone_threshold": float(phone_t),
        "studio_threshold": float(studio_t),
        "average_mae_ms": round(avg_mae, 2) if not np.isnan(avg_mae) else None,
        "average_rmse_ms": round(avg_rmse, 2) if not np.isnan(avg_rmse) else None,
        "files": eval_summary,
    }

    json_dest = os.path.join(OUTPUT_DIR, "global_threshold.json")
    save_threshold_json(config_data, json_dest)
    logger.info("Đã lưu kết quả cấu hình tối ưu vào: %s", json_dest)


if __name__ == "__main__":
    run_training_algorithm1()
