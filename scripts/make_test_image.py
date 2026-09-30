"""Render a synthetic 'screenshot' (board + last-move highlight) from a FEN,
with a known ground-truth answer, to sanity-check the OCR pipeline end to end.

Usage:
    python scripts/make_test_image.py "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R" \
        --lastmove g1f3 --out /tmp/test_board.png
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess_fen_ocr  # noqa: F401,E402  (sets DYLD_FALLBACK_LIBRARY_PATH before cairosvg loads)

import cairosvg  # noqa: E402
import chess  # noqa: E402
import chess.svg  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fen_placement", help="Board part of a FEN, e.g. 'rnbqkbnr/pppppppp/...'")
    parser.add_argument("--lastmove", default=None, help="UCI move, e.g. e2e4")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--out", default="/tmp/test_board.png")
    args = parser.parse_args()

    fen = f"{args.fen_placement} w - - 0 1"
    board = chess.Board(fen)

    lastmove = chess.Move.from_uci(args.lastmove) if args.lastmove else None
    svg_data = chess.svg.board(board, size=args.size, coordinates=False, lastmove=lastmove)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    Image.open(io.BytesIO(png_bytes)).convert("RGB").save(args.out)
    print(f"Wrote {args.out}")
    print(f"Ground-truth FEN placement: {board.board_fen()}")


if __name__ == "__main__":
    main()
