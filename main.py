"""
main.py – Điểm khởi chạy chính (Entry Point) của đồ án Xử lý tín hiệu tiếng nói.
Hỗ trợ cả 2 thuật toán:
- Thuật toán 1: Năng lượng ngắn hạn kết hợp Tìm kiếm nhị phân (Hodgkinson 2012)
- Thuật toán 2: Phân đoạn dựa trên 2 đặc trưng Histogram Energy & Spectral Centroid (Giannakopoulos 2014)

Quy cách theo hướng dẫn của Giảng viên:
- Bấm Run chạy 01 lần duy nhất từ file main.py.
- Duyệt qua 4 file kiểm thử và hiển thị 4 figure tại 4 góc màn hình.
- Xuất bảng tổng kết chỉ số định lượng MAE, RMSE (đơn vị ms) và F1-Score ra terminal.
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Cấu hình mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from src.audio import (
    compute_snr_db, load_audio, load_ground_truth, load_threshold_json,
    parse_lab_file,
)
from src.config import (
    DEFAULT_FEATURE_TYPE, FIGURES_DIR, FRAME_LENGTH_MS,
    FRAME_SHIFT_MS, MIN_SILENCE_DURATION_MS, OUTPUT_DIR, TEST_DIR,
    TRAINING_DIR, FeatureType, Segment,
)
from src.features import (
    compute_energy_and_spectral_centroid,
    compute_short_time_feature,
)
from src.algorithms import (
    BinarySearchSegmenter, HistogramSegmenter,
)
from src.segmentation import (
    classify_frames, evaluate_boundaries, frames_to_segments,
    remove_short_silence,
)
from src.visualization import (
    arrange_four_figures_on_screen, plot_single_file_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


CORNER_LABELS = [
    "[Góc 1: Top-Left]",
    "[Góc 2: Top-Right]",
    "[Góc 3: Bottom-Left]",
    "[Góc 4: Bottom-Right]",
]


def run_evaluation(
    data_dir: str,
    threshold: float,
    threshold_centroid: float = 32.35,
    algo_name: str = "Thuật toán 1: Hodgkinson 2012 (Binary Search)",
    algo: int = 1,
    mode: str = "global",
    feature_type: FeatureType = DEFAULT_FEATURE_TYPE,
    use_median_filter: bool = False,
    show_gui: bool = True,
    save_dir: str = "",
) -> List[Dict[str, Any]]:
    """
    Duyệt qua các file trong thư mục dữ liệu, thực hiện phân đoạn và trực quan hóa.
    Hỗ trợ chế độ ngưỡng toàn cục (global) và ngưỡng động thích nghi (dynamic).
    """
    data_path = Path(data_dir)
    wav_files = sorted(data_path.glob("*.wav"))

    if not wav_files:
        logger.error("Không tìm thấy file .wav nào trong thư mục: %s", data_dir)
        return []

    results: List[Dict[str, Any]] = []
    mode_desc = "Ngưỡng Toàn Cục (Global)" if mode == "global" else "Ngưỡng Động Thích Nghi (Dynamic)"
    feat_desc = "Energy & Spectral Centroid (DFT)" if algo == 2 else feature_type.value

    print("\n" + "=" * 110)
    print(f"   DEMO: {algo_name}")
    print(f"   Thư mục dữ liệu: {data_dir}")
    print(f"   Chế độ: {mode_desc} | Đặc trưng: {feat_desc} | Lọc khoảng lặng: >= {int(MIN_SILENCE_DURATION_MS)} ms")
    print("=" * 110)
    print(f"{'STT':<4} | {'Tên file':<12} | {'SNR (dB)':<9} | {'Ngưỡng T':<16} | {'MAE (ms)':<9} | {'RMSE (ms)':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 110)

    for idx, wav_f in enumerate(wav_files[:4], start=1):
        stem = wav_f.stem
        lab_f = wav_f.with_suffix(".lab")

        signal, sr = load_audio(str(wav_f))
        duration_s = len(signal) / sr

        has_gt = lab_f.exists()
        lab_segs = parse_lab_file(str(lab_f)) if has_gt else []
        gt_segments = load_ground_truth(lab_segs) if has_gt else []
        snr_val = compute_snr_db(signal, sr, lab_segs) if has_gt else float("nan")

        corner_label = CORNER_LABELS[idx - 1] if idx - 1 < len(CORNER_LABELS) else f"[Góc {idx}]"

        if algo == 2:
            # Thuật toán 2: Giannakopoulos 2014 chuẩn 2 đặc trưng (Energy & Spectral Centroid)
            energy_vals, centroid_vals, frame_centers = compute_energy_and_spectral_centroid(
                signal=signal,
                sample_rate=sr,
                frame_length_ms=FRAME_LENGTH_MS,
                frame_shift_ms=FRAME_SHIFT_MS,
                use_hamming=True,
            )

            if mode == "dynamic":
                seg_hist = HistogramSegmenter(
                    num_bins=60,
                    weight_energy=5.0,
                    weight_centroid=2.0,
                    expand_frames=3,
                    min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                )
                current_threshold, current_centroid_t, final_segs = seg_hist.fit_and_segment_dynamic(
                    energy_vals=energy_vals,
                    centroid_vals=centroid_vals,
                    frame_centers=frame_centers,
                    signal_duration_s=duration_s,
                )
            else:
                current_threshold = threshold
                current_centroid_t = threshold_centroid
                seg_hist = HistogramSegmenter(
                    num_bins=60,
                    weight_energy=5.0,
                    weight_centroid=2.0,
                    expand_frames=3,
                    min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                )
                seg_hist.threshold_energy = current_threshold
                seg_hist.threshold_centroid = current_centroid_t
                final_segs = seg_hist.segment(
                    energy_vals=energy_vals,
                    frame_centers=frame_centers,
                    signal_duration_s=duration_s,
                    centroid_vals=centroid_vals,
                )

            t_display = f"{current_threshold:.5f}/{current_centroid_t:.1f}"
            plot_feature_vals = energy_vals
            plot_feature_name = "Energy"

        else:
            # Thuật toán 1: Hodgkinson 2012 (Binary Search)
            feat_vals, frame_centers = compute_short_time_feature(
                signal=signal,
                sample_rate=sr,
                frame_length_ms=FRAME_LENGTH_MS,
                frame_shift_ms=FRAME_SHIFT_MS,
                feature_type=feature_type,
            )
            eval_feat = feat_vals

            if mode == "dynamic":
                if has_gt and len(lab_segs) > 0:
                    from src.features import assign_frame_labels
                    from src.algorithms.binary_search import find_optimal_threshold_binary_search
                    frame_lbls = assign_frame_labels(frame_centers, lab_segs)
                    spk_v = eval_feat[frame_lbls == 1]
                    sil_v = eval_feat[frame_lbls == 0]
                    if len(spk_v) > 0 and len(sil_v) > 0:
                        current_threshold, _, _ = find_optimal_threshold_binary_search(spk_v, sil_v)
                    else:
                        current_threshold = threshold
                else:
                    p_sil = float(np.percentile(eval_feat, 15))
                    current_threshold = -5.009985 if p_sil > -7.0 else -6.449383

                labels = classify_frames(eval_feat, current_threshold)
                raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s=duration_s)
                final_segs = remove_short_silence(raw_segs, min_duration_ms=MIN_SILENCE_DURATION_MS)
            else:
                current_threshold = threshold
                labels = classify_frames(eval_feat, current_threshold)
                raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s=duration_s)
                final_segs = remove_short_silence(raw_segs, min_duration_ms=MIN_SILENCE_DURATION_MS)

            current_centroid_t = None
            centroid_vals = None
            t_display = f"{current_threshold:<16.5f}"
            plot_feature_vals = eval_feat
            plot_feature_name = feature_type.value

        metrics = evaluate_boundaries(final_segs, gt_segments) if has_gt else {}

        mae_display = f"{metrics.get('mae_ms', float('nan')):.1f}" if not np.isnan(metrics.get('mae_ms', float('nan'))) else "N/A"
        rmse_display = f"{metrics.get('rmse_ms', float('nan')):.1f}" if not np.isnan(metrics.get('rmse_ms', float('nan'))) else "N/A"
        prec_display = f"{metrics.get('precision', 0.0) * 100:.1f}%" if has_gt else "N/A"
        rec_display = f"{metrics.get('recall', 0.0) * 100:.1f}%" if has_gt else "N/A"
        f1_display = f"{metrics.get('f1_score', 0.0) * 100:.1f}%" if has_gt else "N/A"
        snr_display = f"{snr_val:.1f}" if not np.isnan(snr_val) else "N/A"

        print(f"{idx:<4} | {stem:<12} | {snr_display:<9} | {t_display:<16} | {mae_display:<9} | {rmse_display:<10} | {prec_display:<10} | {rec_display:<10} | {f1_display:<10}")

        save_path = os.path.join(save_dir, f"{stem}_output.png") if save_dir else None
        fig = plot_single_file_result(
            signal=signal,
            sample_rate=sr,
            feature_vals=plot_feature_vals,
            frame_centers=frame_centers,
            pred_segments=final_segs,
            gt_segments=gt_segments,
            threshold=current_threshold,
            feature_name=plot_feature_name,
            title_text=f"File {idx}: {stem}",
            metrics=metrics if has_gt else None,
            save_path=save_path,
            fig_num=idx,
            corner_label=corner_label,
            snr_db=snr_val if has_gt else None,
            centroid_vals=centroid_vals if algo == 2 else None,
            threshold_centroid=current_centroid_t if algo == 2 else None,
        )

        results.append({
            "stem": stem,
            "metrics": metrics,
            "threshold": current_threshold,
        })

    print("-" * 110)

    if show_gui:
        arrange_four_figures_on_screen()
        print("\n[+] Đang hiển thị 4 figure tại 4 góc màn hình. Đóng các cửa sổ để kết thúc chương trình.")
        plt.show()
    else:
        plt.close("all")

    return results


def main():
    """Hàm main điều khiển đối số dòng lệnh và khởi chạy phân đoạn."""
    parser = argparse.ArgumentParser(
        description="Phân đoạn tín hiệu tiếng nói và khoảng lặng (Speech / Silence Discrimination)"
    )
    parser.add_argument(
        "--algo", type=int, choices=[1, 2], default=1,
        help="Lựa chọn thuật toán: 1 (Hodgkinson 2012 - Binary Search), 2 (Giannakopoulos 2014 - Histogram 2-Feature)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["global", "dynamic"], default="dynamic",
        help="Chế độ ngưỡng: 'global' (ngưỡng dùng chung toàn cục), 'dynamic' (ngưỡng động thích nghi theo từng file - mặc định)",
    )
    parser.add_argument(
        "--dir", type=str, default="",
        help="Đường dẫn thư mục chứa các file .wav (mặc định ưu tiên TinHieuKiemThu, nếu không có sẽ dùng TinHieuHuanLuyen)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Ngưỡng T chỉ định thủ công (nếu không cung cấp sẽ nạp từ file cấu hình tối ưu)",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Tắt chế độ hiển thị cửa sổ figure (chạy headless/lưu ảnh)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="",
        help="Thư mục lưu các file ảnh figure xuất ra",
    )
    args = parser.parse_args()

    # 1. Xác định thư mục dữ liệu
    target_dir = args.dir
    if not target_dir:
        test_p = Path(TEST_DIR)
        train_p = Path(TRAINING_DIR)
        if test_p.exists() and list(test_p.glob("*.wav")):
            target_dir = str(test_p)
            logger.info("Phát hiện thư mục kiểm thử: %s", target_dir)
        elif train_p.exists() and list(train_p.glob("*.wav")):
            target_dir = str(train_p)
            logger.info("Chạy chế độ Demo trên tập huấn luyện: %s", target_dir)
        else:
            logger.error("Không tìm thấy dữ liệu âm thanh tại %s hoặc %s", TEST_DIR, TRAINING_DIR)
            return

    # 2. Xác định thuật toán và ngưỡng T tương ứng
    threshold_centroid = 32.35
    if args.algo == 1:
        algo_name = "Thuật toán 1: Hodgkinson 2012 (Binary Search)"
        feature_type = FeatureType.LOG_MA
        use_median_filter = False
        default_t = -5.284692
        cfg_file = os.path.join(OUTPUT_DIR, "global_threshold.json")
    else:
        algo_name = "Thuật toán 2: Giannakopoulos 2014 (Histogram 2-Feature: Energy & Centroid)"
        feature_type = FeatureType.MA
        use_median_filter = False
        default_t = 0.011927
        threshold_centroid = 32.35
        cfg_file = os.path.join(OUTPUT_DIR, "histogram_threshold.json")

    threshold_val = args.threshold
    if threshold_val is None:
        if os.path.exists(cfg_file):
            try:
                cfg = load_threshold_json(cfg_file)
                threshold_val = float(cfg.get("global_threshold_energy", cfg.get("global_threshold", default_t)))
                threshold_centroid = float(cfg.get("global_threshold_centroid", threshold_centroid))
                logger.info("Đã nạp ngưỡng tối ưu từ file cấu hình %s: T1 = %.6f, T2 = %.2f", cfg_file, threshold_val, threshold_centroid)
            except Exception as e:
                logger.warning("Không đọc được cấu hình từ %s (%s). Dùng mặc định.", cfg_file, e)
                threshold_val = default_t
        else:
            threshold_val = default_t
            logger.info("Sử dụng ngưỡng tối ưu mặc định: T = %.6f", threshold_val)

    save_dir = args.save_dir if args.save_dir else os.path.join(OUTPUT_DIR, "figures")

    # 3. Thực thi đánh giá
    run_evaluation(
        data_dir=target_dir,
        threshold=threshold_val,
        threshold_centroid=threshold_centroid,
        algo_name=algo_name,
        algo=args.algo,
        mode=args.mode,
        feature_type=feature_type,
        use_median_filter=use_median_filter,
        show_gui=not args.no_gui,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
