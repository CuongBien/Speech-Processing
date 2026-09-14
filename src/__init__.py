"""
Package xử lý tín hiệu tiếng nói và khoảng lặng (Speech / Silence Discrimination).
Clean Architecture chuẩn hóa cho các thuật toán phân đoạn âm thanh.
"""

from src.config import (
    DEFAULT_FEATURE_TYPE,
    FIGURES_DIR,
    FRAME_LENGTH_MS,
    FRAME_SHIFT_MS,
    LOG_EPSILON,
    MAX_ITERATIONS,
    MIN_SILENCE_DURATION_MS,
    OUTPUT_DIR,
    TEST_DIR,
    THRESHOLD_TOLERANCE,
    TRAINING_DIR,
    BinarySearchState,
    FeatureType,
    LabSegment,
    Segment,
)
from src.audio import (
    compute_snr_db,
    load_audio,
    load_ground_truth,
    load_threshold_json,
    parse_lab_file,
    save_threshold_json,
)
from src.features import (
    assign_frame_labels,
    compute_short_time_feature,
)
from src.algorithms import (
    BaseSegmenter,
    BinarySearchSegmenter,
    HistogramSegmenter,
    compute_confusion,
    compute_histogram_threshold,
    find_histogram_peaks,
    find_optimal_threshold_binary_search,
    find_overlap_region,
    median_filter_1d,
)
from src.segmentation import (
    classify_frames,
    evaluate_boundaries,
    extract_boundaries,
    frames_to_segments,
    match_boundaries,
    remove_short_silence,
)
from src.visualization import (
    arrange_four_figures_on_screen,
    plot_distribution_and_overlap,
    plot_histogram_analysis,
    plot_single_file_result,
)

__version__ = "1.1.0"

__all__ = [
    # Config & Data structures
    "FeatureType",
    "LabSegment",
    "Segment",
    "BinarySearchState",
    "FRAME_LENGTH_MS",
    "FRAME_SHIFT_MS",
    "DEFAULT_FEATURE_TYPE",
    "MIN_SILENCE_DURATION_MS",
    "TRAINING_DIR",
    "TEST_DIR",
    "OUTPUT_DIR",
    "FIGURES_DIR",
    # Audio
    "load_audio",
    "parse_lab_file",
    "load_ground_truth",
    "compute_snr_db",
    # Features
    "compute_short_time_feature",
    "assign_frame_labels",
    # Algorithms
    "BaseSegmenter",
    "BinarySearchSegmenter",
    "HistogramSegmenter",
    "find_overlap_region",
    "compute_confusion",
    "find_optimal_threshold_binary_search",
    # Segmentation
    "classify_frames",
    "frames_to_segments",
    "remove_short_silence",
    "extract_boundaries",
    "match_boundaries",
    "evaluate_boundaries",
    # Visualization
    "plot_single_file_result",
    "plot_distribution_and_overlap",
    "arrange_four_figures_on_screen",
]
