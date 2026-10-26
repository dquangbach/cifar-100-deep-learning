"""Export separate accuracy/loss figures from training CSV logs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(
    history_path: Path, output_dir: Path, *, monitor: str | None = None
) -> Path:
    history_path, output_dir = Path(history_path), Path(output_dir)
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"epoch", "accuracy", "val_accuracy", "loss", "val_loss"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing training history columns in {history_path}")
    if monitor is None:
        run_file = history_path.parent / "run.json"
        if run_file.is_file():
            monitor = json.loads(run_file.read_text(encoding="utf-8"))["checkpoint_monitor"]
        else:
            monitor = "val_accuracy" if "aug20_history" in history_path.name else "val_loss"
    if monitor not in {"val_accuracy", "val_loss"}:
        raise ValueError("monitor must be 'val_accuracy' or 'val_loss'")
    epochs = np.asarray([int(row["epoch"]) + 1 for row in rows])
    train_accuracy = np.asarray([float(row["accuracy"]) for row in rows]) * 100
    val_accuracy = np.asarray([float(row["val_accuracy"]) for row in rows]) * 100
    train_loss = np.asarray([float(row["loss"]) for row in rows])
    val_loss = np.asarray([float(row["val_loss"]) for row in rows])
    best = int(epochs[np.argmax(val_accuracy) if monitor == "val_accuracy"
                      else np.argmin(val_loss)])
    fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, train_values, val_values, title, ylabel in (
        (accuracy_ax, train_accuracy, val_accuracy, "Accuracy", "Accuracy (%)"),
        (loss_ax, train_loss, val_loss, "Loss", "Cross-entropy loss"),
    ):
        ax.plot(epochs, train_values, label="Train", color="#3678b9")
        ax.plot(epochs, val_values, label="Validation", color="#e58232")
        ax.axvline(best, color="#37966e", linestyle="--", alpha=0.8,
                   label=f"Checkpoint: epoch {best}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"WRN-16-4 / CIFAR-100: {history_path.stem}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{history_path.stem}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
