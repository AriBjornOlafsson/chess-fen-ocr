// Detect last-move highlighted squares and use them to infer side-to-move.
// Port of chess_fen_ocr/highlight.py.
//
// Highlighted squares are found theme-agnostically: for each square we sample
// its corner patches (almost never covered by piece art) as a proxy for the
// square's background color, then flag squares whose color is a statistical
// outlier relative to the other squares of the same checkerboard parity.

const CORNER_FRACTION = 0.15;
const MIN_ABS_DISTANCE = 20.0;
const MAD_MULTIPLIER = 6.0;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function medianColor(pixels) {
  // pixels: array of [r,g,b]
  return [0, 1, 2].map((ch) => median(pixels.map((p) => p[ch])));
}

function cornerPatchPixels(cellMat) {
  const h = cellMat.rows;
  const w = cellMat.cols;
  const s = Math.max(2, Math.floor(Math.min(h, w) * CORNER_FRACTION));
  const data = cellMat.data; // RGBA, row-major
  const channels = cellMat.channels();
  const pixels = [];
  const regions = [
    [0, s, 0, s],
    [0, s, w - s, w],
    [h - s, h, 0, s],
    [h - s, h, w - s, w],
  ];
  for (const [y0, y1, x0, x1] of regions) {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const idx = (y * w + x) * channels;
        pixels.push([data[idx], data[idx + 1], data[idx + 2]]);
      }
    }
  }
  return pixels;
}

function squareBgColor(cellMat) {
  return medianColor(cornerPatchPixels(cellMat));
}

function norm(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// grid: 8x8 array of cv.Mat cells (from splitSquares). Returns [[row, col], ...].
export function detectHighlightedSquares(grid) {
  const colors = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (let c = 0; c < 8; c++) {
      row.push(squareBgColor(grid[r][c]));
    }
    colors.push(row);
  }

  const light = [];
  const dark = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      ((r + c) % 2 === 0 ? light : dark).push(colors[r][c]);
    }
  }
  const lightRef = medianColor(light);
  const darkRef = medianColor(dark);

  const dists = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (let c = 0; c < 8; c++) {
      const ref = (r + c) % 2 === 0 ? lightRef : darkRef;
      row.push(norm(colors[r][c], ref));
    }
    dists.push(row);
  }

  const flatDists = dists.flat();
  const medianD = median(flatDists);
  const mad = median(flatDists.map((d) => Math.abs(d - medianD))) + 1e-6;
  const threshold = Math.max(MIN_ABS_DISTANCE, medianD + MAD_MULTIPLIER * mad);

  const squares = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (dists[r][c] > threshold) squares.push([r, c]);
    }
  }
  return squares;
}

// labels: 8x8 array of predicted piece symbols (image row order).
// highlightedSquares: [[row, col], ...]. Returns 'w' | 'b' | null.
export function inferActiveColor(labels, highlightedSquares) {
  const occupied = highlightedSquares.filter(([r, c]) => labels[r][c] !== "empty");
  if (occupied.length === 0) return null;

  const whiteCount = occupied.filter(([r, c]) => labels[r][c] === labels[r][c].toUpperCase()).length;
  const blackCount = occupied.length - whiteCount;
  const moverIsWhite = whiteCount >= blackCount;
  return moverIsWhite ? "b" : "w";
}
