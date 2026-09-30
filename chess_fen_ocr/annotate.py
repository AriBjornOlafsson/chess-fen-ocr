"""Draw grid lines and predicted labels onto a warped board image, for
visually sanity-checking predictions in the web UI."""
from __future__ import annotations

import cv2
import numpy as np


def annotate_board(
    board_bgr: np.ndarray,
    labels: list[list[str]],
    highlighted_squares: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    img = board_bgr.copy()
    h, w = img.shape[:2]
    cell = h / 8.0

    for i in range(9):
        pos = int(i * cell)
        cv2.line(img, (pos, 0), (pos, h), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (0, pos), (w, pos), (0, 0, 0), 1, cv2.LINE_AA)

    for row, col in highlighted_squares or []:
        x0, y0 = int(col * cell), int(row * cell)
        x1, y1 = int((col + 1) * cell), int((row + 1) * cell)
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 215, 255), 3)

    for row in range(8):
        for col in range(8):
            symbol = labels[row][col]
            if symbol == "empty":
                continue
            x = int(col * cell) + 4
            y = int(row * cell) + int(cell) - 6
            color = (40, 40, 220) if symbol.isupper() else (220, 40, 40)
            cv2.putText(
                img, symbol, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA
            )
            cv2.putText(
                img, symbol, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )

    return img
