"""chess-fen-ocr: screenshot -> FEN

Usage:
    python -m chess_fen_ocr.cli path/to/screenshot.png
"""
from __future__ import annotations

import argparse
import sys

import cv2

from .board_detect import detect_and_warp
from .classify import SquareClassifier
from .fen import grid_to_fen
from .segment import split_squares


def image_to_fen(
    image_path: str,
    active_color: str = "w",
    flipped: bool = False,
) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    board_img = detect_and_warp(image)
    grid = split_squares(board_img)

    classifier = SquareClassifier()
    labels = classifier.predict_grid(grid)

    if flipped:
        labels = [list(reversed(row)) for row in reversed(labels)]

    return grid_to_fen(labels, active_color=active_color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a chessboard screenshot")
    parser.add_argument(
        "--active-color",
        choices=["w", "b"],
        default="w",
        help="Side to move (not visually determinable; defaults to 'w')",
    )
    parser.add_argument(
        "--flipped",
        action="store_true",
        help="Pass this if the screenshot is from Black's perspective (rank 1 at top)",
    )
    args = parser.parse_args()

    fen = image_to_fen(args.image, active_color=args.active_color, flipped=args.flipped)
    print(fen)


if __name__ == "__main__":
    sys.exit(main())
