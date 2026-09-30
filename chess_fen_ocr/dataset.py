"""Synthetic training data: render randomized boards using a variety of
freely-licensed piece sets (see assets/pieces/, fully offline once fetched)
and slice them into labeled squares.
"""
from __future__ import annotations

import random

import chess
import cv2
import numpy as np

from .board_render import render_board_custom
from .labels import CLASS_TO_IDX
from .piece_assets import available_piece_sets
from .preprocess import square_bgr_to_tensor

SQUARE_PX = 64  # model input size per square

# A handful of board-color themes loosely modeled on real sites, plus we
# additionally randomize hue/alpha per render for broader coverage.
THEMES = [
    ("#f0d9b5", "#b58863"),  # classic brown
    ("#eeeed2", "#769656"),  # lichess green
    ("#dee3e6", "#8ca2ad"),  # lichess blue-gray
    ("#e8ebef", "#7d87a3"),  # blue
    ("#f4f4f4", "#9f9f9f"),  # gray
    ("#eeeeee", "#b48fca"),  # purple
    ("#ffffff", "#999999"),  # flat gray/white, e.g. En Croissant-style apps
    ("#f5deb0", "#6b6b6b"),
]

PIECE_SETS = available_piece_sets()
if not PIECE_SETS:
    raise RuntimeError(
        "No piece sets found under assets/pieces/. Fetch at least one set "
        "(e.g. cburnett) before generating training data."
    )


def _rand_bgr() -> tuple[int, int, int]:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def random_board(min_pieces: int = 6, max_pieces: int = 30) -> chess.Board:
    board = chess.Board.empty()
    squares = list(chess.SQUARES)
    random.shuffle(squares)

    n_pieces = random.randint(min_pieces, max_pieces)
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ]

    # Always place both kings so the render looks like a real (if illegal) position.
    board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.BLACK))

    for _ in range(max(0, n_pieces - 2)):
        if not squares:
            break
        sq = squares.pop()
        ptype = random.choice(piece_types)
        color = random.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(sq, chess.Piece(ptype, color))

    return board


def render_board_png(
    board: chess.Board,
    size: int,
    with_lastmove: bool,
) -> np.ndarray:
    light, dark = random.choice(THEMES)
    piece_set = random.choice(PIECE_SETS)

    lastmove = None
    hl_light = hl_dark = None
    if with_lastmove:
        occupied = [s for s in chess.SQUARES if board.piece_at(s)]
        empties = [s for s in chess.SQUARES if not board.piece_at(s)]
        if occupied and empties:
            frm = random.choice(occupied)
            to = random.choice(empties)
            lastmove = chess.Move(frm, to)
            alpha = random.uniform(0.3, 0.65)
            hl_light = (_rand_bgr(), alpha)
            hl_dark = (_rand_bgr(), alpha)

    show_coords = random.random() < 0.3

    return render_board_custom(
        board,
        size=size,
        piece_set=piece_set,
        light_hex=light,
        dark_hex=dark,
        lastmove=lastmove,
        hl_light=hl_light,
        hl_dark=hl_dark,
        show_coords=show_coords,
    )


def _augment(square_bgr: np.ndarray) -> np.ndarray:
    img = square_bgr.astype(np.float32)

    # brightness / contrast jitter
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20, 20)
    img = img * alpha + beta
    img = np.clip(img, 0, 255)

    img = img.astype(np.uint8)

    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.5:
        quality = random.randint(35, 90)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def _resize_jitter(board_img_bgr: np.ndarray) -> np.ndarray:
    """Simulate the extra whole-board resize that board_detect.py's crop step
    introduces at inference time (native render size -> canvas_size), so the
    classifier isn't only ever trained on a single clean downsample.
    """
    h, w = board_img_bgr.shape[:2]
    target = random.randint(400, 700)
    interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    resized = cv2.resize(board_img_bgr, (target, target), interpolation=interp)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_AREA)


def board_to_samples(
    board: chess.Board, board_img_bgr: np.ndarray, augment: bool
) -> tuple[list[np.ndarray], list[int]]:
    if augment and random.random() < 0.6:
        board_img_bgr = _resize_jitter(board_img_bgr)

    h, w = board_img_bgr.shape[:2]
    cell = h / 8.0
    samples = []
    labels = []

    for rank in range(8):  # rank 0 = top row of the rendered image = rank 8
        for file in range(8):
            y0, y1 = int(rank * cell), int((rank + 1) * cell)
            x0, x1 = int(file * cell), int((file + 1) * cell)
            crop = board_img_bgr[y0:y1, x0:x1]
            crop = cv2.resize(crop, (SQUARE_PX, SQUARE_PX), interpolation=cv2.INTER_AREA)
            if augment:
                crop = _augment(crop)

            square = chess.square(file, 7 - rank)
            piece = board.piece_at(square)
            label = CLASS_TO_IDX[piece.symbol() if piece else "empty"]

            samples.append(square_bgr_to_tensor(crop))
            labels.append(label)

    return samples, labels


def generate_dataset(
    num_boards: int, augment: bool = True, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_x: list[np.ndarray] = []
    all_y: list[int] = []

    for i in range(num_boards):
        board = random_board()
        size = random.choice([320, 400, 512, 640, 720, 800])
        with_lastmove = random.random() < 0.7
        img = render_board_png(board, size=size, with_lastmove=with_lastmove)
        samples, labels = board_to_samples(board, img, augment=augment)
        all_x.extend(samples)
        all_y.extend(labels)

    x = np.stack(all_x).astype(np.float32)  # (N, 4, SQUARE_PX, SQUARE_PX)
    y = np.array(all_y, dtype=np.int64)
    return x, y
