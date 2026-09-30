// Client-side image -> FEN pipeline. Mirrors chess_fen_ocr/pipeline.py's
// image_bgr_to_result, but runs entirely in the browser via opencv.js
// (board detection/segmentation) and onnxruntime-web/WASM (classification).

import { detectAndWarp } from "./lib/boardDetect.js";
import { splitSquares } from "./lib/segment.js";
import { detectHighlightedSquares, inferActiveColor } from "./lib/highlight.js";
import { gridToFen } from "./lib/fen.js";
import { annotateBoard } from "./lib/annotate.js";
import { SquareClassifier } from "./lib/classify.js";

let classifierPromise = null;

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = (async () => {
      const classifier = new SquareClassifier();
      await classifier.load();
      return classifier;
    })();
  }
  return classifierPromise;
}

function waitForOpenCv() {
  return new Promise((resolve) => {
    if (typeof cv !== "undefined" && cv.Mat) {
      resolve();
      return;
    }
    const check = () => {
      if (typeof cv !== "undefined" && cv.Mat) {
        resolve();
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });
}

// Kicks off loading opencv.js and the ONNX model as soon as the page loads,
// so the first "Get FEN" click doesn't pay the full cold-start cost.
export function warmUp() {
  return Promise.all([waitForOpenCv(), getClassifier()]);
}

// imageElement: an HTMLImageElement already loaded with the source screenshot.
// Returns { fen, activeColor, activeColorSource, previewCanvas }.
export async function runPipeline(imageElement, { activeColor = null, flipped = false } = {}) {
  await waitForOpenCv();
  const classifier = await getClassifier();

  const src = cv.imread(imageElement);
  const boardMat = detectAndWarp(cv, src);
  src.delete();

  const grid = splitSquares(cv, boardMat);
  const imageOrderLabels = await classifier.predictGrid(grid);
  const highlightedSquares = detectHighlightedSquares(grid);

  let resolvedColor;
  let source;
  if (activeColor) {
    resolvedColor = activeColor;
    source = "manual";
  } else {
    const inferred = inferActiveColor(imageOrderLabels, highlightedSquares);
    if (inferred !== null) {
      resolvedColor = inferred;
      source = "highlight";
    } else {
      resolvedColor = "w";
      source = "default";
    }
  }

  let fenLabels = imageOrderLabels;
  if (flipped) {
    fenLabels = [...imageOrderLabels].reverse().map((row) => [...row].reverse());
  }
  const fen = gridToFen(fenLabels, { activeColor: resolvedColor });

  const previewCanvas = document.createElement("canvas");
  previewCanvas.width = boardMat.cols;
  previewCanvas.height = boardMat.rows;
  cv.imshow(previewCanvas, boardMat);
  const ctx = previewCanvas.getContext("2d");
  annotateBoard(ctx, boardMat.cols, imageOrderLabels, highlightedSquares);

  boardMat.delete();
  for (const row of grid) {
    for (const cell of row) cell.delete();
  }

  return { fen, activeColor: resolvedColor, activeColorSource: source, previewCanvas };
}
