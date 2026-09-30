"""Generate synthetic training data, train SquareNet, and export to Core ML.

Usage:
    python -m chess_fen_ocr.train --boards 800 --epochs 12
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import generate_dataset
from .labels import CLASSES, NUM_CLASSES
from .model import SquareNet

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def to_tensor_xy(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    # x is already (N, 4, SQUARE_PX, SQUARE_PX) float32 from preprocess.py
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    return TensorDataset(xt, yt)


def train(num_boards: int, epochs: int, batch_size: int, lr: float) -> SquareNet:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Generating synthetic dataset from {num_boards} rendered boards...")
    t0 = time.time()
    val_boards = max(20, num_boards // 8)
    x_train, y_train = generate_dataset(num_boards, augment=True, seed=1)
    x_val, y_val = generate_dataset(val_boards, augment=True, seed=2)
    print(
        f"  train squares: {len(x_train)}, val squares: {len(x_val)} "
        f"({time.time() - t0:.1f}s)"
    )

    train_ds = to_tensor_xy(x_train, y_train)
    val_ds = to_tensor_xy(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SquareNet(NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        sched.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    return model


def export_coreml(model: SquareNet, out_path: Path) -> None:
    model = model.to("cpu").eval()
    example = torch.rand(1, 4, 64, 64)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="square", shape=example.shape)],
        outputs=[ct.TensorType(name="logits")],
        classifier_config=ct.ClassifierConfig(CLASSES),
        compute_units=ct.ComputeUnit.ALL,  # allow scheduling onto the ANE
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.author = "chess-fen-ocr"
    mlmodel.short_description = "Classifies a single chessboard square image into a piece type or empty."
    mlmodel.save(str(out_path))
    print(f"Saved Core ML model to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    model = train(args.boards, args.epochs, args.batch_size, args.lr)

    torch_path = MODELS_DIR / "squarenet.pt"
    torch.save(model.state_dict(), torch_path)
    print(f"Saved PyTorch weights to {torch_path}")

    export_coreml(model, MODELS_DIR / "SquareClassifier.mlpackage")


if __name__ == "__main__":
    main()
