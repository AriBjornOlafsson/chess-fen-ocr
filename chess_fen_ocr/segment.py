"""Slice a warped, top-down board image into 64 square crops."""
from __future__ import annotations

import cv2
import numpy as np

from .dataset import SQUARE_PX


def split_squares(board_bgr: np.ndarray) -> list[list[np.ndarray]]:
    """Return an 8x8 grid of square crops, row 0 = top of the image."""
    h, w = board_bgr.shape[:2]
    cell = h / 8.0
    grid: list[list[np.ndarray]] = []
    for row in range(8):
        cols = []
        y0, y1 = int(row * cell), int((row + 1) * cell)
        for col in range(8):
            x0, x1 = int(col * cell), int((col + 1) * cell)
            crop = board_bgr[y0:y1, x0:x1]
            crop = cv2.resize(crop, (SQUARE_PX, SQUARE_PX), interpolation=cv2.INTER_AREA)
            cols.append(crop)
        grid.append(cols)
    return grid
