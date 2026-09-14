"""
src/visualization/layout.py – Tiện ích sắp xếp cửa sổ figure hiển thị trên màn hình.
"""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def arrange_four_figures_on_screen():
    """
    Sắp xếp 4 cửa sổ Figure của Matplotlib vào 4 góc của màn hình:
    - Góc trên trái: (0, 0)
    - Góc trên phải: (W/2, 0)
    - Góc dưới trái: (0, H/2)
    - Góc dưới phải: (W/2, H/2)
    """
    try:
        fig_nums = plt.get_fignums()
        if len(fig_nums) < 4:
            return

        first_manager = plt.figure(fig_nums[0]).canvas.manager
        if hasattr(first_manager, "window"):
            win = first_manager.window
            if hasattr(win, "winfo_screenwidth"):
                screen_w = win.winfo_screenwidth()
                screen_h = win.winfo_screenheight()
                half_w = screen_w // 2
                half_h = screen_h // 2 - 30

                positions = [
                    (0, 0),
                    (half_w, 0),
                    (0, half_h),
                    (half_w, half_h),
                ]

                for i, num in enumerate(fig_nums[:4]):
                    mgr = plt.figure(num).canvas.manager
                    w = mgr.window
                    x, y = positions[i]
                    if hasattr(w, "geometry"):
                        w.geometry(f"{half_w}x{half_h}+{x}+{y}")
    except Exception as e:
        logger.debug("Không tự động điều chỉnh tọa độ cửa sổ: %s", e)
