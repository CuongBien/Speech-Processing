"""
src/config.py – Tham số cấu hình toàn cục và các định nghĩa cấu trúc dữ liệu.
Chứa các hằng số phân tích khung thời gian ngắn, thông số tìm kiếm nhị phân
và các lớp đại diện cho phân đoạn âm thanh.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class FeatureType(Enum):
    """Các loại đặc trưng ngắn hạn hỗ trợ."""
    MA = "MA"          # Short-Time Magnitude: mean(|x|)
    LOG_MA = "logMA"   # Log Short-Time Magnitude: log(mean(|x|) + eps)
    STE = "STE"        # Short-Time Energy: mean(x^2)
    LOG_STE = "logSTE" # Log Short-Time Energy: log(mean(x^2) + eps)


# ── Thông số phân tích khung (Framing parameters) ───────────────────────────
# Độ dài khung chuẩn 20ms và độ dịch khung 10ms (chồng chập 50%)
FRAME_LENGTH_MS: int = 20
FRAME_SHIFT_MS: int = 10

# ── Thông số đặc trưng ngắn hạn ─────────────────────────────────────────────
# Mặc định sử dụng logMA vì mở rộng dải động ở mức năng lượng thấp
DEFAULT_FEATURE_TYPE: FeatureType = FeatureType.LOG_MA
LOG_EPSILON: float = 1e-10

# ── Thông số thuật toán tìm kiếm nhị phân (Hodgkinson 2012) ──────────────────
MAX_ITERATIONS: int = 100
THRESHOLD_TOLERANCE: float = 1e-9

# ── Ràng buộc hậu xử lý (Post-processing) ──────────────────────────────────
# Độ dài tối thiểu của 1 khoảng lặng chuẩn theo yêu cầu đề bài là 300ms
MIN_SILENCE_DURATION_MS: float = 300.0

# ── Thông số đánh giá ranh giới (Boundary Evaluation) ────────────────────────
# Dung sai khi ghép cặp biên dự đoán với biên groundtruth (100ms)
BOUNDARY_MATCH_TOLERANCE_MS: float = 100.0

# ── Đường dẫn thư mục dự án ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR: str = str(BASE_DIR / "TinHieuHuanLuyen")
TEST_DIR: str = str(BASE_DIR / "TinHieuKiemThu")
OUTPUT_DIR: str = str(BASE_DIR / "output")
FIGURES_DIR: str = str(BASE_DIR / "output" / "figures")


# ── Cấu trúc dữ liệu ─────────────────────────────────────────────────────────

@dataclass
class LabSegment:
    """Đoạn nhãn đọc từ file .lab."""
    start: float       # Thời điểm bắt đầu (giây)
    end: float         # Thời điểm kết thúc (giây)
    label: str         # "sil" (khoảng lặng), "v" (hữu thanh), "uv" (vô thanh)

    @property
    def duration(self) -> float:
        """Độ dài đoạn (giây)."""
        return self.end - self.start

    def is_silence(self) -> bool:
        """Kiểm tra có phải khoảng lặng không."""
        return self.label.lower() == "sil"

    def is_speech(self) -> bool:
        """Kiểm tra có phải tiếng nói (hữu thanh hoặc vô thanh) không."""
        return self.label.lower() in ("v", "uv")


@dataclass
class Segment:
    """Đoạn phân loại tiếng nói hoặc khoảng lặng."""
    start: float       # Thời điểm bắt đầu (giây)
    end: float         # Thời điểm kết thúc (giây)
    label: str         # "speech" hoặc "silence"

    @property
    def duration_s(self) -> float:
        """Thời lượng tính bằng giây."""
        return self.end - self.start

    @property
    def duration_ms(self) -> float:
        """Thời lượng tính bằng mili-giây."""
        return (self.end - self.start) * 1000.0


@dataclass
class BinarySearchState:
    """Lưu lại trạng thái tại từng bước lặp của thuật toán tìm kiếm nhị phân."""
    iteration: int         # Vòng lặp thứ i
    t_min: float           # Cận dưới của khoảng tìm kiếm
    t_max: float           # Cận trên của khoảng tìm kiếm
    threshold: float       # Giá trị ngưỡng thử nghiệm T = (t_min + t_max) / 2
    c_sil: float           # Lượng nhầm lẫn khoảng lặng vượt ngưỡng T
    c_spch: float          # Lượng nhầm lẫn tiếng nói dưới ngưỡng T
    diff: float            # Hiệu số c_sil - c_spch
    speech_below_t: int    # Số khung tiếng nói bị nhầm dưới ngưỡng
    silence_above_t: int   # Số khung khoảng lặng bị nhầm trên ngưỡng
