"""Locate an 8x8 chessboard inside a screenshot and warp it to a square canvas."""
from __future__ import annotations

import cv2
import numpy as np

CANVAS_SIZE = 512


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_board_quad(image: np.ndarray) -> np.ndarray | None:
    """Search contours for the largest roughly-square quadrilateral."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    image_area = float(h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.15:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.int32)

        pts = approx.reshape(-1, 2).astype(np.float32)
        ordered = _order_corners(pts)
        side_top = np.linalg.norm(ordered[1] - ordered[0])
        side_bottom = np.linalg.norm(ordered[2] - ordered[3])
        side_left = np.linalg.norm(ordered[3] - ordered[0])
        side_right = np.linalg.norm(ordered[2] - ordered[1])
        sides = [side_top, side_bottom, side_left, side_right]
        if min(sides) == 0:
            continue
        squareness = min(sides) / max(sides)
        if squareness < 0.95:
            continue

        # Screenshots are essentially never rotated: reject quads whose edges
        # aren't close to axis-aligned (this filters out false positives from
        # internal diagonal contours, e.g. a bishop's outline).
        top_vec = ordered[1] - ordered[0]
        left_vec = ordered[3] - ordered[0]
        top_angle = abs(np.degrees(np.arctan2(top_vec[1], top_vec[0])))
        left_angle = abs(np.degrees(np.arctan2(left_vec[0], left_vec[1])))
        if top_angle > 2 or left_angle > 2:
            continue

        if area > best_area:
            best_area = area
            best = ordered

    return best


def detect_and_warp(image: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """Return a `canvas_size`x`canvas_size` top-down crop of the chessboard.

    Falls back to treating the whole image as the board (center-cropped to a
    square) if no confident quadrilateral is found -- the common case for
    screenshots that are already tightly cropped to the board.
    """
    h, w = image.shape[:2]
    quad = _find_board_quad(image)

    if quad is None:
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        x1, y1 = x0 + side, y0 + side
    else:
        # We only accept axis-aligned quads (see _find_board_quad), so a plain
        # bounding-box crop is both simpler and more robust than a full
        # 4-point perspective warp, which amplifies small per-corner detection
        # noise into a slight projective skew across the image.
        xs, ys = quad[:, 0], quad[:, 1]
        x0, x1 = int(round(xs.min())), int(round(xs.max()))
        y0, y1 = int(round(ys.min())), int(round(ys.max()))

    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
