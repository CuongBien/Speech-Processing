"""
src/visualization – Gói trực quan hóa dữ liệu và hiển thị kết quả phân đoạn.
"""

from src.visualization.layout import arrange_four_figures_on_screen
from src.visualization.plots import (
    plot_distribution_and_overlap,
    plot_dual_histogram_analysis,
    plot_histogram_analysis,
    plot_single_file_result,
)

__all__ = [
    "plot_single_file_result",
    "plot_distribution_and_overlap",
    "plot_histogram_analysis",
    "plot_dual_histogram_analysis",
    "arrange_four_figures_on_screen",
]

