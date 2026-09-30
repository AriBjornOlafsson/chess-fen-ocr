"""Detect last-move highlighted squares and use them to infer side-to-move.

Highlighted squares are found theme-agnostically: for each square we sample
its corner patches (almost never covered by piece art) as a proxy for the
square's background color, then flag squares whose color is a statistical
outlier relative to the other squares of the same checkerboard parity. This
works regardless of the board's actual color theme or the highlight's hue.
"""
from __future__ import annotations

import cv2
import numpy as np

from .segment import split_squares

CORNER_FRACTION = 0.15
MIN_ABS_DISTANCE = 20.0
MAD_MULTIPLIER = 6.0


def _corner_patch_pixels(cell_bgr: np.ndarray) -> np.ndarray:
    h, w = cell_bgr.shape[:2]
    s = max(2, int(min(h, w) * CORNER_FRACTION))
    patches = [
        cell_bgr[0:s, 0:s],
        cell_bgr[0:s, w - s : w],
        cell_bgr[h - s : h, 0:s],
        cell_bgr[h - s : h, w - s : w],
    ]
    return np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)


def _square_bg_color(cell_bgr: np.ndarray) -> np.ndarray:
    return np.median(_corner_patch_pixels(cell_bgr), axis=0)


def detect_highlighted_squares(
    board_bgr: np.ndarray, grid_cells: list[list[np.ndarray]] | None = None
) -> list[tuple[int, int]]:
    """Return (row, col) of squares whose background tint stands out from the
    rest of the board -- i.e. the last-move highlight, in image row order."""
    if grid_cells is None:
        grid_cells = split_squares(board_bgr)

    colors = [[_square_bg_color(grid_cells[r][c]) for c in range(8)] for r in range(8)]

    light = np.array([colors[r][c] for r in range(8) for c in range(8) if (r + c) % 2 == 0])
    dark = np.array([colors[r][c] for r in range(8) for c in range(8) if (r + c) % 2 == 1])
    light_ref = np.median(light, axis=0)
    dark_ref = np.median(dark, axis=0)

    dists = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            ref = light_ref if (r + c) % 2 == 0 else dark_ref
            dists[r, c] = float(np.linalg.norm(colors[r][c].astype(np.float32) - ref))

    median_d = float(np.median(dists))
    mad = float(np.median(np.abs(dists - median_d))) + 1e-6
    threshold = max(MIN_ABS_DISTANCE, median_d + MAD_MULTIPLIER * mad)

    return [(r, c) for r in range(8) for c in range(8) if dists[r, c] > threshold]


def infer_active_color(
    labels: list[list[str]], highlighted_squares: list[tuple[int, int]]
) -> str | None:
    """Given predicted piece labels (image row order) and the detected
    highlighted squares, return the side to move next ('w'/'b'), or None if
    it can't be determined (no highlight found, or highlighted squares are
    all empty).
    """
    occupied = [(r, c) for (r, c) in highlighted_squares if labels[r][c] != "empty"]
    if not occupied:
        return None

    white_count = sum(1 for (r, c) in occupied if labels[r][c].isupper())
    black_count = len(occupied) - white_count
    mover_is_white = white_count >= black_count
    return "b" if mover_is_white else "w"
