"""Local web wrapper: drop an image in, get a FEN back.

Usage:
    python -m chess_fen_ocr.web
    open http://127.0.0.1:5000
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from .annotate import annotate_board
from .classify import SquareClassifier
from .pipeline import image_bgr_to_result

app = Flask(__name__)

# Loaded once at process start so requests don't pay Core ML model-load cost.
_classifier: SquareClassifier | None = None


def get_classifier() -> SquareClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SquareClassifier()
    return _classifier


def _encode_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode preview image")
    return base64.b64encode(buf).decode("ascii")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/fen")
def api_fen():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    data = file.read()
    if not data:
        return jsonify({"error": "Empty file"}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    active_color = request.form.get("active_color") or None
    flipped = request.form.get("flipped", "false").lower() == "true"

    try:
        classifier = get_classifier()
        result = image_bgr_to_result(
            image, classifier, active_color=active_color, flipped=flipped
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    annotated = annotate_board(result.board_img, result.labels, result.highlighted_squares)

    return jsonify(
        {
            "fen": result.fen,
            "active_color": result.active_color,
            "active_color_source": result.active_color_source,
            "preview_png_b64": _encode_png_b64(annotated),
            "lichess_url": f"https://lichess.org/analysis/{result.fen.replace(' ', '_')}",
        }
    )


def main() -> None:
    get_classifier()  # fail fast / warm up before serving
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
