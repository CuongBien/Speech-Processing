"""
scripts/export_summary.py – Xuất bảng tổng hợp kết quả thực nghiệm ra output/evaluation_summary.csv.
"""

import csv
import glob
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Đảm bảo import được src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algorithms.histogram import HistogramSegmenter, median_filter_1d
from src.audio import compute_snr_db, load_audio, load_ground_truth, parse_lab_file
from src.config import FeatureType
from src.features import compute_short_time_feature
from src.segmentation import (
    classify_frames, evaluate_boundaries, frames_to_segments,
    remove_short_silence,
)


def generate_evaluation_summary():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TinHieuHuanLuyen")
    files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    rows = []

    # 1. Algo 1 Global (T = -5.284692)
    t1_global = -5.284692
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        sig, sr = load_audio(f)
        lab = load_ground_truth(parse_lab_file(f.replace(".wav", ".lab")))
        snr = compute_snr_db(sig, sr, parse_lab_file(f.replace(".wav", ".lab")))
        fv, fc = compute_short_time_feature(sig, sr, feature_type=FeatureType.LOG_MA)
        segs = remove_short_silence(frames_to_segments(classify_frames(fv, t1_global), fc, len(sig) / sr))
        m = evaluate_boundaries(segs, lab)
        rows.append({
            "File": stem,
            "Kenh": "Dien thoai" if "phone" in stem else "Phong thu",
            "SNR_dB": round(snr, 1),
            "ThuatToan": "Hodgkinson 2012 (Binary Search)",
            "CheDo": "Global",
            "Nguong_T": round(t1_global, 6),
            "MAE_ms": round(m.get("mae_ms", float("nan")), 1),
            "RMSE_ms": round(m.get("rmse_ms", float("nan")), 1),
            "Precision": f"{m.get('precision', 0.0) * 100:.1f}%",
            "Recall": f"{m.get('recall', 0.0) * 100:.1f}%",
            "F1_Score": f"{m.get('f1_score', 0.0) * 100:.1f}%",
        })

    # 2. Algo 2 Global (T = 0.039178)
    t2_global = 0.039178
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        sig, sr = load_audio(f)
        lab = load_ground_truth(parse_lab_file(f.replace(".wav", ".lab")))
        snr = compute_snr_db(sig, sr, parse_lab_file(f.replace(".wav", ".lab")))
        fv, fc = compute_short_time_feature(sig, sr, feature_type=FeatureType.MA)
        smoothed = median_filter_1d(fv, kernel_size=5)
        segs = remove_short_silence(frames_to_segments(classify_frames(smoothed, t2_global), fc, len(sig) / sr))
        m = evaluate_boundaries(segs, lab)
        rows.append({
            "File": stem,
            "Kenh": "Dien thoai" if "phone" in stem else "Phong thu",
            "SNR_dB": round(snr, 1),
            "ThuatToan": "Giannakopoulos 2014 (Histogram)",
            "CheDo": "Global",
            "Nguong_T": round(t2_global, 6),
            "MAE_ms": round(m.get("mae_ms", float("nan")), 1),
            "RMSE_ms": round(m.get("rmse_ms", float("nan")), 1),
            "Precision": f"{m.get('precision', 0.0) * 100:.1f}%",
            "Recall": f"{m.get('recall', 0.0) * 100:.1f}%",
            "F1_Score": f"{m.get('f1_score', 0.0) * 100:.1f}%",
        })

    # 3. Algo 2 Dynamic (Adaptive Histogram)
    seg_h = HistogramSegmenter()
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        sig, sr = load_audio(f)
        lab = load_ground_truth(parse_lab_file(f.replace(".wav", ".lab")))
        snr = compute_snr_db(sig, sr, parse_lab_file(f.replace(".wav", ".lab")))
        fv, fc = compute_short_time_feature(sig, sr, feature_type=FeatureType.MA)
        smoothed = median_filter_1d(fv, kernel_size=5)
        t_dyn, segs = seg_h.fit_and_segment_dynamic(smoothed, fc, len(sig) / sr)
        m = evaluate_boundaries(segs, lab)
        rows.append({
            "File": stem,
            "Kenh": "Dien thoai" if "phone" in stem else "Phong thu",
            "SNR_dB": round(snr, 1),
            "ThuatToan": "Giannakopoulos 2014 (Histogram)",
            "CheDo": "Dynamic",
            "Nguong_T": round(t_dyn, 6),
            "MAE_ms": round(m.get("mae_ms", float("nan")), 1),
            "RMSE_ms": round(m.get("rmse_ms", float("nan")), 1),
            "Precision": f"{m.get('precision', 0.0) * 100:.1f}%",
            "Recall": f"{m.get('recall', 0.0) * 100:.1f}%",
            "F1_Score": f"{m.get('f1_score', 0.0) * 100:.1f}%",
        })

    out_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "evaluation_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Đã xuất {len(rows)} hàng kết quả thực nghiệm ra: {out_csv}")


if __name__ == "__main__":
    generate_evaluation_summary()
