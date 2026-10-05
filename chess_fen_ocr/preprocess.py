"""Shared square preprocessing: a BGR uint8 crop -> a 2-channel float32
tensor (grayscale + a Sobel edge-magnitude channel), used identically at
training and inference time so a square is never seen with different
preprocessing than what the model was trained on.

Color is dropped entirely rather than merely de-emphasized: a board's hue
(brown, green, blue, purple, ...) is exactly what varies across themes, while
luminance contrast and piece silhouette shape stay stable regardless of
color scheme. Training only on grayscale + edges makes the classifier
invariant to board color by construction instead of hoping it generalizes.
"""
from __future__ import annotations

import cv2
import numpy as np

# Divisor bringing typical Sobel gradient magnitudes into roughly [0, 1].
EDGE_MAG_SCALE = 255.0


def square_bgr_to_tensor(square_bgr: np.ndarray) -> np.ndarray:
    """`square_bgr` is an HxWx3 uint8 BGR crop. Returns a 2xHxW float32
    array: channel 0 is grayscale in [0, 1], channel 1 is a normalized Sobel
    gradient-magnitude map (shape/silhouette signal).
    """
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) / EDGE_MAG_SCALE
    mag = np.clip(mag, 0.0, 1.0)

    return np.stack([gray_norm, mag], axis=0).astype(np.float32)  # (2, H, W)
