"""Build a FEN string from an 8x8 grid of predicted piece symbols."""
from __future__ import annotations


def grid_to_fen(
    grid: list[list[str]],
    active_color: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    """`grid[0]` is rank 8 (top of a normally-oriented board image), `grid[7]`
    is rank 1, matching how a screenshot is read top-to-bottom.
    """
    rows = []
    for row in grid:
        fen_row = ""
        empty_run = 0
        for symbol in row:
            if symbol == "empty" or symbol == "":
                empty_run += 1
            else:
                if empty_run:
                    fen_row += str(empty_run)
                    empty_run = 0
                fen_row += symbol
        if empty_run:
            fen_row += str(empty_run)
        rows.append(fen_row)

    placement = "/".join(rows)
    return f"{placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"
