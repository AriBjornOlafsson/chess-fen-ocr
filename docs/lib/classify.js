// Run the ONNX-exported SquareNet classifier (onnxruntime-web, WASM) over 64
// square crops in a single batched inference call.

import { CLASSES } from "./labels.js";
import { SQUARE_PX } from "./segment.js";

const MODEL_URL = new URL("../model/square_classifier.onnx", import.meta.url).href;

// Divisor bringing typical Sobel gradient magnitudes into roughly [0, 1].
// Must match EDGE_MAG_SCALE in chess_fen_ocr/preprocess.py exactly.
const EDGE_MAG_SCALE = 255.0;

// Fills `out` (a Float32Array view into one square's slot, length SQUARE_PX*SQUARE_PX)
// with a normalized Sobel gradient-magnitude map of `mat` (an RGBA cv.Mat cell) --
// an explicit shape/silhouette signal, in addition to RGB, that's far less
// sensitive to a square's raw color/texture than RGB alone. Mirrors
// chess_fen_ocr/preprocess.py's square_bgr_to_tensor exactly.
function computeEdgeMagnitude(mat, out) {
  const gray = new cv.Mat();
  const gx = new cv.Mat();
  const gy = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  cv.Sobel(gray, gx, cv.CV_32F, 1, 0, 3);
  cv.Sobel(gray, gy, cv.CV_32F, 0, 1, 3);

  const gxData = gx.data32F;
  const gyData = gy.data32F;
  for (let i = 0; i < out.length; i++) {
    const mag = Math.sqrt(gxData[i] * gxData[i] + gyData[i] * gyData[i]) / EDGE_MAG_SCALE;
    out[i] = Math.min(mag, 1.0);
  }

  gray.delete();
  gx.delete();
  gy.delete();
}

export class SquareClassifier {
  constructor() {
    this.session = null;
  }

  async load() {
    this.session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }

  // grid: 8x8 array of cv.Mat cells (SQUARE_PX x SQUARE_PX, RGBA), image row order.
  // Returns an 8x8 array of predicted class labels, same order.
  async predictGrid(grid) {
    const n = 64;
    const chw = SQUARE_PX * SQUARE_PX;
    const numInputChannels = 4; // RGB + Sobel edge magnitude, matches model.py's IN_CHANNELS
    const data = new Float32Array(n * numInputChannels * chw);

    let cellIndex = 0;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const mat = grid[row][col];
        const pixels = mat.data; // RGBA, row-major
        const channels = mat.channels();
        const base = cellIndex * numInputChannels * chw;
        for (let y = 0; y < SQUARE_PX; y++) {
          for (let x = 0; x < SQUARE_PX; x++) {
            const srcIdx = (y * SQUARE_PX + x) * channels;
            const dstIdx = y * SQUARE_PX + x;
            data[base + 0 * chw + dstIdx] = pixels[srcIdx] / 255.0;
            data[base + 1 * chw + dstIdx] = pixels[srcIdx + 1] / 255.0;
            data[base + 2 * chw + dstIdx] = pixels[srcIdx + 2] / 255.0;
          }
        }
        computeEdgeMagnitude(mat, data.subarray(base + 3 * chw, base + 4 * chw));
        cellIndex += 1;
      }
    }

    const tensor = new ort.Tensor("float32", data, [n, numInputChannels, SQUARE_PX, SQUARE_PX]);
    const feeds = { square: tensor };
    const output = await this.session.run(feeds);
    const logits = output.logits.data; // [n, numClasses], flat
    const numClasses = CLASSES.length;

    const labels = [];
    let idx = 0;
    for (let row = 0; row < 8; row++) {
      const rowLabels = [];
      for (let col = 0; col < 8; col++) {
        const base = idx * numClasses;
        let best = 0;
        for (let k = 1; k < numClasses; k++) {
          if (logits[base + k] > logits[base + best]) best = k;
        }
        rowLabels.push(CLASSES[best]);
        idx += 1;
      }
      labels.push(rowLabels);
    }
    return labels;
  }
}
