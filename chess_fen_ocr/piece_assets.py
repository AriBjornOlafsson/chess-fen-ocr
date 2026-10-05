"""Load and rasterize piece SVGs from assets/pieces/<set>/<code>.svg.

Several visually distinct, freely-licensed piece sets (sourced from lichess's
open-source repo: MIT/CC0/CC-BY/GPL/Apache licensed subsets) are bundled so
the classifier trains on more than one art style and generalizes to whatever
a real chess app happens to use.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "pieces"

PIECE_CODES = ["P", "N", "B", "R", "Q", "K"]

# Sets excluded from training: the classifier is trained on grayscale (no
# color channel, see preprocess.py), so a set must distinguish white from
# black by luminance alone. "firi" renders white pieces as mid-gray (~92,92,92)
# and black pieces as dark gold (~88,72,10) -- a gray-vs-gold theme, not
# black-vs-white -- so the two are nearly indistinguishable once color is
# dropped (luminance gap of just ~0.09 on a 0-1 scale).
EXCLUDED_SETS = {"firi"}


def available_piece_sets() -> list[str]:
    if not ASSETS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in ASSETS_DIR.iterdir() if p.is_dir() and p.name not in EXCLUDED_SETS
    )


@lru_cache(maxsize=None)
def _find_piece_file(piece_set: str, color: str, ptype: str) -> Path:
    stem = ASSETS_DIR / piece_set / f"{color}{ptype}"
    for ext in (".svg", ".webp", ".png"):
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No piece asset found for {piece_set}/{color}{ptype}")


@lru_cache(maxsize=None)
def _load_piece_rgba(piece_set: str, color: str, ptype: str, px: int) -> np.ndarray:
    """Return an RGBA uint8 array of shape (px, px, 4) for one piece."""
    path = _find_piece_file(piece_set, color, ptype)
    if path.suffix == ".svg":
        png_bytes = cairosvg.svg2png(
            url=str(path), output_width=px, output_height=px, background_color=None
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    else:
        img = Image.open(path).convert("RGBA").resize((px, px), Image.LANCZOS)
    return np.array(img)


def get_piece_rgba(piece_set: str, symbol: str, px: int) -> np.ndarray:
    """`symbol` is a python-chess piece symbol, e.g. 'P', 'k'."""
    color = "w" if symbol.isupper() else "b"
    ptype = symbol.upper()
    return _load_piece_rgba(piece_set, color, ptype, px)
