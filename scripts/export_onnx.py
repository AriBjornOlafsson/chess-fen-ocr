"""Export the trained SquareNet weights to ONNX, for in-browser inference via
onnxruntime-web (WASM), and verify parity against the PyTorch model.

Usage:
    python scripts/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch

from chess_fen_ocr.labels import CLASSES
from chess_fen_ocr.model import SquareNet

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    weights_path = MODELS_DIR / "squarenet.pt"
    onnx_path = MODELS_DIR / "square_classifier.onnx"

    model = SquareNet(len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    example = torch.rand(1, 2, 64, 64)

    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["square"],
        output_names=["logits"],
        dynamic_axes={"square": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {onnx_path}")

    with torch.no_grad():
        torch_out = model(example).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"square": example.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"max abs diff between torch and onnx outputs: {max_diff:.6f}")
    assert max_diff < 1e-4, "ONNX export diverges from PyTorch model"
    print("Parity check passed.")


if __name__ == "__main__":
    main()
