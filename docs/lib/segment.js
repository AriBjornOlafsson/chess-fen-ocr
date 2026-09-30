// Slice a warped, top-down board image into 64 square crops.
// Port of chess_fen_ocr/segment.py.

export const SQUARE_PX = 64; // model input size per square, matches dataset.py

// Returns an 8x8 grid of cv.Mat crops, row 0 = top of the image. Caller owns
// (must .delete()) each returned Mat.
export function splitSquares(cv, boardMat) {
  const h = boardMat.rows;
  const w = boardMat.cols;
  const cell = h / 8.0;
  const grid = [];
  for (let row = 0; row < 8; row++) {
    const cols = [];
    const y0 = Math.floor(row * cell);
    const y1 = Math.floor((row + 1) * cell);
    for (let col = 0; col < 8; col++) {
      const x0 = Math.floor(col * cell);
      const x1 = Math.floor((col + 1) * cell);
      const rect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
      const crop = boardMat.roi(rect);
      const resized = new cv.Mat();
      cv.resize(crop, resized, new cv.Size(SQUARE_PX, SQUARE_PX), 0, 0, cv.INTER_AREA);
      crop.delete();
      cols.push(resized);
    }
    grid.push(cols);
  }
  return grid;
}
