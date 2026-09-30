"""Shared image -> FEN pipeline, used by both the CLI and the web app so the
Core ML classifier is loaded once and reused across calls."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board_detect import detect_and_warp
from .classify import SquareClassifier
from .fen import grid_to_fen
from .highlight import detect_highlighted_squares, infer_active_color
from .segment import split_squares


@dataclass
class OcrResult:
    fen: str
    board_img: np.ndarray
    labels: list[list[str]]  # image row order (row 0 = top of board_img)
    active_color: str
    active_color_source: str  # "highlight" | "manual" | "default"
    highlighted_squares: list[tuple[int, int]]


def image_bgr_to_result(
    image_bgr: np.ndarray,
    classifier: SquareClassifier,
    active_color: str | None = None,
    flipped: bool = False,
) -> OcrResult:
    """Run the full pipeline. `active_color` of None means auto-detect the
    side to move from the last-move highlight (falling back to 'w' if no
    highlight is found); pass 'w'/'b' to override.
    """
    board_img = detect_and_warp(image_bgr)
    grid = split_squares(board_img)
    image_order_labels = classifier.predict_grid(grid)

    highlighted_squares = detect_highlighted_squares(board_img, grid)

    if active_color is not None:
        resolved_color = active_color
        source = "manual"
    else:
        inferred = infer_active_color(image_order_labels, highlighted_squares)
        if inferred is not None:
            resolved_color, source = inferred, "highlight"
        else:
            resolved_color, source = "w", "default"

    fen_labels = image_order_labels
    if flipped:
        fen_labels = [list(reversed(row)) for row in reversed(image_order_labels)]

    fen = grid_to_fen(fen_labels, active_color=resolved_color)

    return OcrResult(
        fen=fen,
        board_img=board_img,
        labels=image_order_labels,
        active_color=resolved_color,
        active_color_source=source,
        highlighted_squares=highlighted_squares,
    )
