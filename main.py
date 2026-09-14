"""
main.py – Điểm khởi chạy chính (Entry Point) của đồ án Xử lý tín hiệu tiếng nói.
Hỗ trợ cả 2 thuật toán:
- Thuật toán 1: Năng lượng ngắn hạn kết hợp Tìm kiếm nhị phân (Hodgkinson 2012)
- Thuật toán 2: Phân đoạn dựa trên Histogram (Giannakopoulos 2014)

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


def run_evaluation(
    data_dir: str,
    threshold: float,
    algo_name: str = "Thuật toán 1: Hodgkinson 2012 (Binary Search)",
    show_gui: bool = True,
    save_dir: str = "",
) -> List[Dict[str, Any]]:
    """
    Duyệt qua các file trong thư mục dữ liệu, thực hiện phân đoạn và trực quan hóa.
    """
    data_path = Path(data_dir)
    wav_files = sorted(data_path.glob("*.wav"))

    if not wav_files:
        logger.error("Không tìm thấy file .wav nào trong thư mục: %s", data_dir)
        return []

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(f"   DEMO: {algo_name}")
    print(f"   Thư mục dữ liệu: {data_dir}")
    print(f"   Đặc trưng: {DEFAULT_FEATURE_TYPE.value} | Ngưỡng T = {threshold:.6f} | Lọc khoảng lặng: >= {int(MIN_SILENCE_DURATION_MS)} ms")
    print("=" * 90)
    print(f"{'STT':<4} | {'Tên file':<12} | {'SNR (dB)':<9} | {'MAE (ms)':<9} | {'RMSE (ms)':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 90)

    for idx, wav_f in enumerate(wav_files[:4], start=1):
        stem = wav_f.stem
        lab_f = wav_f.with_suffix(".lab")

        signal, sr = load_audio(str(wav_f))
        duration_s = len(signal) / sr

        has_gt = lab_f.exists()
        lab_segs = parse_lab_file(str(lab_f)) if has_gt else []
        gt_segments = load_ground_truth(lab_segs) if has_gt else []
        snr_val = compute_snr_db(signal, sr, lab_segs) if has_gt else float("nan")

        feat_vals, frame_centers = compute_short_time_feature(
            signal=signal,
            sample_rate=sr,
            frame_length_ms=FRAME_LENGTH_MS,
            frame_shift_ms=FRAME_SHIFT_MS,
            feature_type=DEFAULT_FEATURE_TYPE,
        )

        labels = classify_frames(feat_vals, threshold)
        raw_segs = frames_to_segments(labels, frame_centers, signal_duration_s=duration_s)
        final_segs = remove_short_silence(raw_segs, min_duration_ms=MIN_SILENCE_DURATION_MS)

        metrics = evaluate_boundaries(final_segs, gt_segments) if has_gt else {}

        mae_display = f"{metrics.get('mae_ms', float('nan')):.1f}" if not np.isnan(metrics.get('mae_ms', float('nan'))) else "N/A"
        rmse_display = f"{metrics.get('rmse_ms', float('nan')):.1f}" if not np.isnan(metrics.get('rmse_ms', float('nan'))) else "N/A"
        prec_display = f"{metrics.get('precision', 0.0) * 100:.1f}%" if has_gt else "N/A"
        rec_display = f"{metrics.get('recall', 0.0) * 100:.1f}%" if has_gt else "N/A"
        f1_display = f"{metrics.get('f1_score', 0.0) * 100:.1f}%" if has_gt else "N/A"
        snr_display = f"{snr_val:.1f}" if not np.isnan(snr_val) else "N/A"

        print(f"{idx:<4} | {stem:<12} | {snr_display:<9} | {mae_display:<9} | {rmse_display:<10} | {prec_display:<10} | {rec_display:<10} | {f1_display:<10}")

        save_path = os.path.join(save_dir, f"{stem}_output.png") if save_dir else None
        fig = plot_single_file_result(
            signal=signal,
            sample_rate=sr,
            feature_vals=feat_vals,
            frame_centers=frame_centers,
            pred_segments=final_segs,
            gt_segments=gt_segments,
            threshold=threshold,
            feature_name=DEFAULT_FEATURE_TYPE.value,
            title_text=f"File {idx}: {stem}",
            metrics=metrics if has_gt else None,
            save_path=save_path,
            fig_num=idx,
        )

        results.append({
            "stem": stem,
            "metrics": metrics,
        })

    print("-" * 90)

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
        help="Lựa chọn thuật toán: 1 (Hodgkinson 2012 - Binary Search), 2 (Giannakopoulos 2014 - Histogram)",
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
    if args.algo == 1:
        algo_name = "Thuật toán 1: Hodgkinson 2012 (Binary Search)"
        cfg_file = os.path.join(OUTPUT_DIR, "global_threshold.json")
    else:
        algo_name = "Thuật toán 2: Giannakopoulos 2014 (Histogram)"
        cfg_file = os.path.join(OUTPUT_DIR, "histogram_threshold.json")

    threshold_val = args.threshold
    if threshold_val is None:
        if os.path.exists(cfg_file):
            try:
                cfg = load_threshold_json(cfg_file)
                threshold_val = float(cfg.get("global_threshold", -5.284692))
                logger.info("Đã nạp ngưỡng tối ưu từ file cấu hình %s: T = %.6f", cfg_file, threshold_val)
            except Exception as e:
                logger.warning("Không đọc được cấu hình từ %s (%s). Dùng mặc định.", cfg_file, e)
                threshold_val = -5.284692
        else:
            threshold_val = -5.284692
            logger.info("Sử dụng ngưỡng tối ưu mặc định: T = %.6f", threshold_val)

    save_dir = args.save_dir if args.save_dir else os.path.join(OUTPUT_DIR, "figures")

    # 3. Thực thi đánh giá
    run_evaluation(
        data_dir=target_dir,
        threshold=threshold_val,
        algo_name=algo_name,
        show_gui=not args.no_gui,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
