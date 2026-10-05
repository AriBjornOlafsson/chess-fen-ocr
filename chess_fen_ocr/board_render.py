"""Composite a chess.Board onto a canvas using a chosen piece set, with
optional last-move highlight and in-cell coordinate labels (some real chess
apps draw rank/file labels inside the outer squares rather than in a margin,
which otherwise confuses a classifier that never saw that during training).
"""
from __future__ import annotations

import random

import chess
import cv2
import numpy as np

from .piece_assets import get_piece_rgba


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _alpha_paste(canvas_bgr: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> None:
    h, w = rgba.shape[:2]
    ch, cw = canvas_bgr.shape[:2]
    h, w = min(h, ch - y0), min(w, cw - x0)
    if h <= 0 or w <= 0:
        return
    roi = canvas_bgr[y0 : y0 + h, x0 : x0 + w]
    alpha = rgba[:h, :w, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:h, :w, :3][:, :, ::-1].astype(np.float32)
    blended = rgb * alpha + roi.astype(np.float32) * (1 - alpha)
    canvas_bgr[y0 : y0 + h, x0 : x0 + w] = blended.astype(np.uint8)


def _make_board_texture_params(size: int) -> dict | None:
    """Pick one set of wood-grain-like texture parameters for an entire
    board render (grain direction/frequency/phase must be shared across all
    64 squares -- real wood grain flows continuously across a board, it
    doesn't reset at square boundaries)."""
    if random.random() < 0.15:
        return None  # keep some boards flat, so "perfectly uniform" stays in-distribution too
    return {
        "angle": random.uniform(0, 2 * np.pi),
        "freq": random.uniform(6, 40) / size,  # grain stripes per pixel
        "phase": random.uniform(0, 2 * np.pi),
        "grain_amp": random.uniform(6, 30),
        "gradient_amp": random.uniform(5, 20),
        "fine_noise_amp": random.uniform(0, 6),
    }


def _add_square_texture(
    canvas: np.ndarray, x0: int, y0: int, cell: int, params: dict | None
) -> None:
    """Overlay a shared-across-the-board gradient + periodic grain texture
    onto one square's background, so the classifier can't shortcut "empty"
    as "perfectly uniform flat color" -- real boards (wood grain, lighting)
    rarely render a square as one exact solid color, and the synthetic
    THEMES above are otherwise always flat.

    The grain is a genuine periodic stripe pattern (modulated by smoothed
    noise for natural irregularity), not plain per-pixel noise: a real
    wood-grain board produces a persistent, spread-out edge signal across
    an entire empty square, which plain Gaussian noise doesn't reproduce,
    and the model needs to see that exact failure mode during training to
    not mistake it for a piece silhouette.
    """
    if params is None:
        return

    region = canvas[y0 : y0 + cell, x0 : x0 + cell].astype(np.float32)
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return

    # Global (board-level) pixel coordinates so the pattern is continuous
    # across square boundaries.
    xs, ys = np.meshgrid(np.arange(x0, x0 + w), np.arange(y0, y0 + h))
    gx, gy = np.cos(params["angle"]), np.sin(params["angle"])
    proj = xs * gx + ys * gy

    gradient = proj / (np.abs(proj).max() + 1e-6)
    region += gradient[..., None] * params["gradient_amp"]

    grain = np.sin(proj * params["freq"] * 2 * np.pi + params["phase"])
    amp_mod = np.random.normal(0, 1, (h, w)).astype(np.float32)
    amp_mod = cv2.GaussianBlur(amp_mod, (0, 0), sigmaX=max(h, w) * 0.08)
    amp_mod = 0.4 + 0.6 * (amp_mod / (np.abs(amp_mod).max() + 1e-6))
    region += (grain * amp_mod)[..., None] * params["grain_amp"]

    if params["fine_noise_amp"] > 0.5:
        region += np.random.normal(0, params["fine_noise_amp"], region.shape)

    canvas[y0 : y0 + cell, x0 : x0 + cell] = np.clip(region, 0, 255).astype(np.uint8)


def render_board_custom(
    board: chess.Board,
    size: int,
    piece_set: str,
    light_hex: str,
    dark_hex: str,
    lastmove: chess.Move | None = None,
    hl_light: tuple[tuple[int, int, int], float] | None = None,
    hl_dark: tuple[tuple[int, int, int], float] | None = None,
    show_coords: bool = False,
    piece_scale_range: tuple[float, float] = (0.78, 0.96),
) -> np.ndarray:
    """Returns a `size`x`size` BGR uint8 image."""
    cell = size // 8
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    light_bgr = _hex_to_bgr(light_hex)
    dark_bgr = _hex_to_bgr(dark_hex)

    hl_squares = {lastmove.from_square, lastmove.to_square} if lastmove is not None else set()
    texture_params = _make_board_texture_params(size)

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            is_light = (row + col) % 2 == 0
            base = light_bgr if is_light else dark_bgr
            x0, y0 = col * cell, row * cell

            color = base
            if square in hl_squares:
                hl = hl_light if is_light else hl_dark
                if hl is not None:
                    hl_bgr, alpha = hl
                    color = tuple(int(b * (1 - alpha) + h * alpha) for b, h in zip(base, hl_bgr))

            canvas[y0 : y0 + cell, x0 : x0 + cell] = color
            _add_square_texture(canvas, x0, y0, cell, texture_params)

    if show_coords:
        files = "abcdefgh"
        for row in range(8):
            color = (20, 20, 20) if random.random() < 0.5 else (235, 235, 235)
            cv2.putText(
                canvas, str(8 - row), (int(cell * 0.04), int(row * cell + cell * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.32, 0.5), color, 1, cv2.LINE_AA,
            )
        for col in range(8):
            color = (20, 20, 20) if random.random() < 0.5 else (235, 235, 235)
            cv2.putText(
                canvas, files[col], (int(col * cell + cell * 0.7), int(7 * cell + cell * 0.94)),
                cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.32, 0.5), color, 1, cv2.LINE_AA,
            )

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece is None:
                continue
            scale = random.uniform(*piece_scale_range)
            px = max(8, int(cell * scale))
            rgba = get_piece_rgba(piece_set, piece.symbol(), px)
            max_offset = max(0, cell - px)
            ox = random.randint(0, max_offset)
            oy = random.randint(0, max_offset)
            _alpha_paste(canvas, rgba, col * cell + ox, row * cell + oy)

    return canvas
