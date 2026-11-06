"""Training plots from a saved EfficientNet history CSV."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history_path: Path, output_path: Path) -> Path:
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"}
    if not rows or not columns.issubset(rows[0]):
        raise ValueError(f"History CSV is empty or missing columns: {history_path}")
    epoch = [int(row["epoch"]) for row in rows]
    best = min(rows, key=lambda row: float(row["val_loss"]))
    best_epoch = int(best["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, train_key, val_key, title, scale in (
        (axes[0], "train_loss", "val_loss", "Loss", 1),
        (axes[1], "train_accuracy", "val_accuracy", "Accuracy (%)", 100),
    ):
        ax.plot(epoch, [float(row[train_key]) * scale for row in rows], label="Train")
        ax.plot(epoch, [float(row[val_key]) * scale for row in rows], label="Validation")
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"Best checkpoint: epoch {best_epoch}")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("EfficientNetV2-B0 / CIFAR-100 training history")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
