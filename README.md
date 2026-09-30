# chess-fen-ocr

Turn a chessboard screenshot (with the last move highlighted, as on lichess/chess.com)
into a FEN string. Piece recognition runs as a Core ML model on Apple's Neural Engine.

## How it works

1. **Board detection** (`board_detect.py`, OpenCV): finds the board's quadrilateral in
   the screenshot (falls back to a center square crop if the image is already tightly
   cropped) and perspective-warps it to a flat 512x512 canvas.
2. **Segmentation** (`segment.py`): slices the canvas into 64 square crops.
3. **Classification** (`classify.py`): a small CNN (`model.py`), trained and exported
   to Core ML (`.mlpackage`, `compute_units=ALL`), labels each square as empty or one
   of the 12 piece types. Core ML schedules supported ops onto the ANE automatically
   on Apple Silicon.
4. **FEN assembly** (`fen.py`): turns the 8x8 label grid into the board-placement
   field of a FEN string.

Because there's no ready-made labeled dataset of chessboard screenshots, the model is
trained entirely on **synthetic data** (`dataset.py`). python-chess ships its own
"cburnett-style" piece SVGs, which we rasterize with `cairosvg` into randomized boards
(random piece placement, random board-color themes, random last-move highlight
color/opacity, plus blur/JPEG/noise augmentation) so the classifier learns to ignore
the highlight tint rather than being confused by it.

**Caveat:** side-to-move, castling rights, and en-passant target aren't visually
recoverable from a single static image, so the CLI defaults them (`w`, `-`, `-`).
Override `--active-color` if you know better.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Cairo (used only at training time, to rasterize the synthetic boards) must be
installed via Homebrew: `brew install cairo`.

## Train the model

```bash
source venv/bin/activate
python -m chess_fen_ocr.train --boards 900 --epochs 14
```

This generates synthetic boards on the fly (no downloads), trains `SquareNet`, and
writes `models/SquareClassifier.mlpackage` (plus raw weights in `models/squarenet.pt`).
Training happens on GPU via PyTorch MPS if available. The *exported* model is what
runs on the ANE at inference time.

## Run it on a screenshot

```bash
source venv/bin/activate
python -m chess_fen_ocr.cli path/to/screenshot.png
# rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1
```

Use `--flipped` if the screenshot is from Black's point of view (rank 1 at the top).

## Web wrapper

```bash
./run.sh
```

or manually:

```bash
source venv/bin/activate
python -m chess_fen_ocr.web
open http://127.0.0.1:5000
```

Drag and drop (or paste) a screenshot in the browser, pick side-to-move and
orientation, and it returns the FEN, a lichess analysis link, and an annotated
debug preview showing exactly what was predicted on each square. Runs entirely
locally; the Core ML model is loaded once at startup and reused across
requests.

## Validate against a known position

`scripts/make_test_image.py` renders a synthetic screenshot from a FEN you supply, so
you can round-trip a known position through the whole pipeline and check the output
FEN matches:

```bash
source venv/bin/activate
python scripts/make_test_image.py "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R" \
    --lastmove g1f3 --out /tmp/test_board.png
python -m chess_fen_ocr.cli /tmp/test_board.png
```

## Limitations

- Trained on a python-chess-rendered piece style. Real screenshots using very
  different piece art (e.g. 3D sets) will need retraining/fine-tuning on real
  examples for best accuracy. The architecture (`dataset.py` + `train.py`) supports
  mixing in real labeled screenshots.
- Board detection assumes a roughly axis-aligned, unobstructed board. Overlapping UI
  chrome (clocks, move lists) inside the board's bounding box isn't handled.
