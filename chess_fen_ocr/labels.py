"""Shared piece label vocabulary."""

# Index 0 is "empty square". The rest follow python-chess piece symbols.
CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
