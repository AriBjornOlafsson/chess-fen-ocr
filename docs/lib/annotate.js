// Draw grid lines and predicted labels onto a warped board image, for
// visually sanity-checking predictions in the browser UI.
// Port of chess_fen_ocr/annotate.py, using a canvas 2D context.

// ctx must already have the board image drawn at (0, 0, size, size).
export function annotateBoard(ctx, size, labels, highlightedSquares = []) {
  const cell = size / 8.0;

  ctx.save();
  ctx.strokeStyle = "black";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const pos = Math.floor(i * cell) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(size, pos);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgb(255, 215, 0)";
  ctx.lineWidth = 3;
  for (const [row, col] of highlightedSquares) {
    const x0 = Math.floor(col * cell) + 2;
    const y0 = Math.floor(row * cell) + 2;
    const x1 = Math.floor((col + 1) * cell) - 2;
    const y1 = Math.floor((row + 1) * cell) - 2;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }

  ctx.font = "bold 20px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "alphabetic";
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const symbol = labels[row][col];
      if (symbol === "empty") continue;
      const x = Math.floor(col * cell) + 4;
      const y = Math.floor(row * cell) + Math.floor(cell) - 6;
      const isWhite = symbol === symbol.toUpperCase();
      ctx.lineWidth = 4;
      ctx.strokeStyle = "white";
      ctx.strokeText(symbol, x, y);
      ctx.fillStyle = isWhite ? "rgb(40, 40, 220)" : "rgb(220, 40, 40)";
      ctx.fillText(symbol, x, y);
    }
  }
  ctx.restore();
}
