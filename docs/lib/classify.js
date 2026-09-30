// Run the ONNX-exported SquareNet classifier (onnxruntime-web, WASM) over 64
// square crops in a single batched inference call.

import { CLASSES } from "./labels.js";
import { SQUARE_PX } from "./segment.js";

const MODEL_URL = new URL("../model/square_classifier.onnx", import.meta.url).href;

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
    const data = new Float32Array(n * 3 * chw);

    let cellIndex = 0;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const mat = grid[row][col];
        const pixels = mat.data; // RGBA, row-major
        const channels = mat.channels();
        const base = cellIndex * 3 * chw;
        for (let y = 0; y < SQUARE_PX; y++) {
          for (let x = 0; x < SQUARE_PX; x++) {
            const srcIdx = (y * SQUARE_PX + x) * channels;
            const dstIdx = y * SQUARE_PX + x;
            data[base + 0 * chw + dstIdx] = pixels[srcIdx] / 255.0;
            data[base + 1 * chw + dstIdx] = pixels[srcIdx + 1] / 255.0;
            data[base + 2 * chw + dstIdx] = pixels[srcIdx + 2] / 255.0;
          }
        }
        cellIndex += 1;
      }
    }

    const tensor = new ort.Tensor("float32", data, [n, 3, SQUARE_PX, SQUARE_PX]);
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
