"""Shared square preprocessing: a BGR uint8 crop -> a 4-channel float32
tensor (RGB + a Sobel edge-magnitude channel), used identically at training
and inference time so a square is never seen with different preprocessing
than what the model was trained on.
"""
from __future__ import annotations

import cv2
import numpy as np

# Divisor bringing typical Sobel gradient magnitudes into roughly [0, 1].
EDGE_MAG_SCALE = 255.0


def square_bgr_to_tensor(square_bgr: np.ndarray) -> np.ndarray:
    """`square_bgr` is an HxWx3 uint8 BGR crop. Returns a 4xHxW float32
    array: channels 0-2 are RGB in [0, 1], channel 3 is a normalized Sobel
    gradient-magnitude map.

    The edge channel is an explicit shape/silhouette signal that's far less
    sensitive to a square's raw color or texture than RGB alone -- important
    because real screenshots use board themes (wood grain, gradients,
    lighting) the synthetic training set can't fully enumerate by color.
    """
    rgb = square_bgr[:, :, ::-1].astype(np.float32) / 255.0

    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) / EDGE_MAG_SCALE
    mag = np.clip(mag, 0.0, 1.0)

    chw = rgb.transpose(2, 0, 1)  # (3, H, W)
    return np.concatenate([chw, mag[None, ...]], axis=0).astype(np.float32)  # (4, H, W)
