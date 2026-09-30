// Locate an 8x8 chessboard inside a screenshot and crop it to a square canvas.
// Port of chess_fen_ocr/board_detect.py, using opencv.js (global `cv`).

const CANVAS_SIZE = 512;

function orderCorners(pts) {
  // pts: [[x,y], [x,y], [x,y], [x,y]] -> [tl, tr, br, bl]
  const sums = pts.map(([x, y]) => x + y);
  const diffs = pts.map(([x, y]) => y - x);
  const tl = pts[sums.indexOf(Math.min(...sums))];
  const br = pts[sums.indexOf(Math.max(...sums))];
  const tr = pts[diffs.indexOf(Math.min(...diffs))];
  const bl = pts[diffs.indexOf(Math.max(...diffs))];
  return [tl, tr, br, bl];
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function rotatedRectBoxPoints(rect) {
  const { center, size, angle } = rect;
  const theta = (angle * Math.PI) / 180;
  const cosA = Math.cos(theta);
  const sinA = Math.sin(theta);
  const hw = size.width / 2;
  const hh = size.height / 2;
  const corners = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ];
  return corners.map(([x, y]) => [
    center.x + x * cosA - y * sinA,
    center.y + x * sinA + y * cosA,
  ]);
}

function findBoardQuad(cv, src) {
  const h = src.rows;
  const w = src.cols;
  const imageArea = h * w;

  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
  const edges = new cv.Mat();
  cv.Canny(blurred, edges, 40, 120);
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 2);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  let best = null;
  let bestArea = 0;

  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt);
    if (area < imageArea * 0.15) {
      cnt.delete();
      continue;
    }

    const peri = cv.arcLength(cnt, true);
    let approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

    let pts;
    if (approx.rows === 4 && cv.isContourConvex(approx)) {
      pts = [];
      for (let r = 0; r < 4; r++) {
        pts.push([approx.intAt(r, 0), approx.intAt(r, 1)]);
      }
    } else {
      const rect = cv.minAreaRect(cnt);
      pts = rotatedRectBoxPoints(rect);
    }
    approx.delete();

    const ordered = orderCorners(pts);
    const [tl, tr, br, bl] = ordered;
    const sideTop = dist(tr, tl);
    const sideBottom = dist(br, bl);
    const sideLeft = dist(bl, tl);
    const sideRight = dist(br, tr);
    const sides = [sideTop, sideBottom, sideLeft, sideRight];
    const minSide = Math.min(...sides);
    const maxSide = Math.max(...sides);
    if (minSide === 0) {
      cnt.delete();
      continue;
    }
    const squareness = minSide / maxSide;
    if (squareness < 0.95) {
      cnt.delete();
      continue;
    }

    const topVec = [tr[0] - tl[0], tr[1] - tl[1]];
    const leftVec = [bl[0] - tl[0], bl[1] - tl[1]];
    const topAngle = Math.abs((Math.atan2(topVec[1], topVec[0]) * 180) / Math.PI);
    const leftAngle = Math.abs((Math.atan2(leftVec[0], leftVec[1]) * 180) / Math.PI);
    if (topAngle > 2 || leftAngle > 2) {
      cnt.delete();
      continue;
    }

    if (area > bestArea) {
      bestArea = area;
      best = ordered;
    }
    cnt.delete();
  }

  gray.delete();
  blurred.delete();
  edges.delete();
  kernel.delete();
  contours.delete();
  hierarchy.delete();

  return best;
}

// Returns a new cv.Mat, canvasSize x canvasSize, RGBA.
export function detectAndWarp(cv, src, canvasSize = CANVAS_SIZE) {
  const h = src.rows;
  const w = src.cols;
  const quad = findBoardQuad(cv, src);

  let x0, y0, x1, y1;
  if (quad === null) {
    const side = Math.min(h, w);
    y0 = Math.floor((h - side) / 2);
    x0 = Math.floor((w - side) / 2);
    x1 = x0 + side;
    y1 = y0 + side;
  } else {
    const xs = quad.map((p) => p[0]);
    const ys = quad.map((p) => p[1]);
    x0 = Math.round(Math.min(...xs));
    x1 = Math.round(Math.max(...xs));
    y0 = Math.round(Math.min(...ys));
    y1 = Math.round(Math.max(...ys));
  }

  const rect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
  const cropped = src.roi(rect);
  const resized = new cv.Mat();
  cv.resize(cropped, resized, new cv.Size(canvasSize, canvasSize), 0, 0, cv.INTER_AREA);
  cropped.delete();
  return resized;
}
