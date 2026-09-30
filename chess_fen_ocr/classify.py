"""Run the trained Core ML classifier (ANE-accelerated) over 64 square crops."""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np

from .preprocess import square_bgr_to_tensor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "SquareClassifier.mlpackage"


class SquareClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m chess_fen_ocr.train` first."
            )
        # compute_units=ALL lets Core ML schedule work onto the Neural Engine
        # when the model/ops support it; it falls back to GPU/CPU automatically.
        self.model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)

    def predict_square(self, square_bgr: np.ndarray) -> tuple[str, float]:
        arr = square_bgr_to_tensor(square_bgr)[None, ...]
        out = self.model.predict({"square": arr})
        label = out["classLabel"]
        class_probs = out["classLabel_probs"]
        logits = np.array(list(class_probs.values()))
        exp = np.exp(logits - logits.max())
        softmax = exp / exp.sum()
        confidence = float(softmax.max())
        return label, confidence

    def predict_grid(self, grid: list[list[np.ndarray]]) -> list[list[str]]:
        return [[self.predict_square(cell)[0] for cell in row] for row in grid]
