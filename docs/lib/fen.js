// Build a FEN string from an 8x8 grid of predicted piece symbols.
// Port of chess_fen_ocr/fen.py.

// grid[0] is rank 8 (top of a normally-oriented board image), grid[7] is
// rank 1, matching how a screenshot is read top-to-bottom.
export function gridToFen(
  grid,
  { activeColor = "w", castling = "-", enPassant = "-", halfmove = 0, fullmove = 1 } = {}
) {
  const rows = grid.map((row) => {
    let fenRow = "";
    let emptyRun = 0;
    for (const symbol of row) {
      if (symbol === "empty" || symbol === "") {
        emptyRun += 1;
      } else {
        if (emptyRun) {
          fenRow += String(emptyRun);
          emptyRun = 0;
        }
        fenRow += symbol;
      }
    }
    if (emptyRun) fenRow += String(emptyRun);
    return fenRow;
  });

  const placement = rows.join("/");
  return `${placement} ${activeColor} ${castling} ${enPassant} ${halfmove} ${fullmove}`;
}
