"""chess-fen-ocr: screenshot -> FEN

Usage:
    python -m chess_fen_ocr.cli path/to/screenshot.png
"""
from __future__ import annotations

import argparse
import sys

import cv2

from .classify import SquareClassifier
from .pipeline import OcrResult, image_bgr_to_result


def image_to_result(
    image_path: str,
    active_color: str | None = None,
    flipped: bool = False,
) -> OcrResult:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    classifier = SquareClassifier()
    return image_bgr_to_result(image, classifier, active_color=active_color, flipped=flipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a chessboard screenshot")
    parser.add_argument(
        "--active-color",
        choices=["w", "b"],
        default=None,
        help=(
            "Side to move. By default this is auto-detected from the last-move "
            "highlight (the highlighted destination square's piece color tells us "
            "who just moved); pass this to override."
        ),
    )
    parser.add_argument(
        "--flipped",
        action="store_true",
        help="Pass this if the screenshot is from Black's perspective (rank 1 at top)",
    )
    args = parser.parse_args()

    result = image_to_result(args.image, active_color=args.active_color, flipped=args.flipped)

    if result.active_color_source == "highlight":
        note = f"active color auto-detected from last-move highlight: {result.active_color}"
    elif result.active_color_source == "default":
        note = "no last-move highlight detected; defaulted active color to 'w'"
    else:
        note = f"active color set explicitly: {result.active_color}"
    print(note, file=sys.stderr)

    print(result.fen)


if __name__ == "__main__":
    sys.exit(main())
