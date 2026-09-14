"""
src/audio_io.py – Module đọc/ghi file âm thanh WAV và file nhãn chuẩn .lab.
Tuân thủ quy chuẩn tự xử lý trên numpy, có chú thích đầy đủ cho từng hàm
và từng khối mã lệnh.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

from src.config import LabSegment, Segment

logger = logging.getLogger(__name__)


def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Đọc file âm thanh WAV và chuẩn hóa về dạng mono mảng float32 trong khoảng [-1.0, 1.0].

    Tham số:
        wav_path (str): Đường dẫn tuyệt đối hoặc tương đối tới file .wav.

    Trả về:
        Tuple[np.ndarray, int]:
            - signal (np.ndarray): Tín hiệu âm thanh 1 chiều kiểu float32.
            - sample_rate (int): Tần số lấy mẫu (Hz).
    """
    # Kiểm tra sự tồn tại của file âm thanh
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {wav_path}")

    # Đọc dữ liệu thô và tần số lấy mẫu bằng scipy.io.wavfile
    sample_rate, raw_data = wavfile.read(wav_path)

    # Chuẩn hóa kiểu dữ liệu nguyên thành float32 trong đoạn [-1.0, 1.0]
    # Dựa vào số bit mã hóa (int16, int32 hoặc uint8)
    if raw_data.dtype == np.int16:
        signal = raw_data.astype(np.float32) / 32768.0
    elif raw_data.dtype == np.int32:
        signal = raw_data.astype(np.float32) / 2147483648.0
    elif raw_data.dtype == np.uint8:
        signal = (raw_data.astype(np.float32) - 128.0) / 128.0
    else:
        signal = raw_data.astype(np.float32)

    # Nếu file âm thanh có 2 kênh (stereo), tính trung bình 2 kênh để chuyển thành mono
    if signal.ndim == 2:
        logger.info("Chuyển đổi tín hiệu 2 kênh (stereo) sang 1 kênh (mono)")
        signal = np.mean(signal, axis=1)

    return signal, int(sample_rate)


def parse_lab_file(lab_path: str) -> List[LabSegment]:
    """
    Đọc và phân tích cấu trúc file nhãn .lab.
    Định dạng mỗi dòng: biên_trái<TAB>biên_phải<TAB>nhãn
    Bỏ qua 2 dòng metadata cuối (F0mean, F0std).

    Tham số:
        lab_path (str): Đường dẫn đến file .lab.

    Trả về:
        List[LabSegment]: Danh sách các đoạn nhãn chuẩn.
    """
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Không tìm thấy file nhãn .lab: {lab_path}")

    valid_labels = {"sil", "v", "uv"}
    ignore_keys = {"f0mean", "f0std"}
    segments: List[LabSegment] = []

    # Mở và đọc từng dòng trong file .lab với mã hóa UTF-8
    with open(lab_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line_clean = line.strip()
            if not line_clean:
                continue

            # Tách các cột theo dấu tab hoặc khoảng trắng
            parts = [p.strip() for p in line_clean.split("\t") if p.strip()]
            if not parts:
                continue

            # Bỏ qua các thông số F0mean và F0std ở cuối file
            if parts[0].lower() in ignore_keys:
                continue

            # Kiểm tra đủ 3 trường: start, end, label
            if len(parts) < 3:
                # Thử tách theo khoảng trắng nếu không có ký tự tab
                parts = line_clean.split()
                if len(parts) < 3:
                    logger.warning("Bỏ qua dòng không đúng định dạng tại dòng %d: %s", line_num, line_clean)
                    continue

            # Chuyển đổi mốc thời gian sang float và kiểm tra nhãn hợp lệ
            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
                label = parts[2].lower()
            except ValueError:
                logger.warning("Không phân tích được số liệu tại dòng %d: %s", line_num, line_clean)
                continue

            if label in valid_labels and start_time < end_time:
                segments.append(LabSegment(start=start_time, end=end_time, label=label))

    return segments


def load_ground_truth(lab_segments: List[LabSegment]) -> List[Segment]:
    """
    Gộp các đoạn nhãn chi tiết (sil, v, uv) thành 2 lớp nhị phân:
    'silence' (từ sil) và 'speech' (từ v và uv).
    Tự động ghép nối các đoạn liên tiếp có cùng nhãn.

    Tham số:
        lab_segments (List[LabSegment]): Danh sách các đoạn đọc từ file .lab.

    Trả về:
        List[Segment]: Danh sách các đoạn chuẩn Ground Truth phân loại nhị phân.
    """
    if not lab_segments:
        return []

    # Ánh xạ nhãn: sil -> silence; v, uv -> speech
    def map_to_binary(label: str) -> str:
        return "silence" if label == "sil" else "speech"

    merged: List[Segment] = []
    current_start = lab_segments[0].start
    current_label = map_to_binary(lab_segments[0].label)
    current_end = lab_segments[0].end

    # Duyệt qua các đoạn kế tiếp để gộp nếu cùng loại nhãn nhị phân
    for seg in lab_segments[1:]:
        binary_lbl = map_to_binary(seg.label)
        if binary_lbl == current_label:
            current_end = seg.end
        else:
            merged.append(Segment(start=current_start, end=current_end, label=current_label))
            current_start = seg.start
            current_end = seg.end
            current_label = binary_lbl

    # Thêm đoạn cuối cùng vào danh sách
    merged.append(Segment(start=current_start, end=current_end, label=current_label))
    return merged


def save_threshold_json(threshold_data: Dict[str, Any], json_path: str):
    """
    Lưu thông tin cấu hình ngưỡng tối ưu ra file định dạng JSON.

    Tham số:
        threshold_data (Dict[str, Any]): Từ điển chứa thông tin ngưỡng.
        json_path (str): Đường dẫn file JSON đích.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4, ensure_ascii=False)
    logger.info("Đã lưu cấu hình ngưỡng vào: %s", json_path)


def load_threshold_json(json_path: str) -> Dict[str, Any]:
    """
    Đọc thông tin cấu hình ngưỡng tối ưu từ file JSON.

    Tham số:
        json_path (str): Đường dẫn file JSON.

    Trả về:
        Dict[str, Any]: Từ điển chứa thông tin ngưỡng.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình ngưỡng: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
