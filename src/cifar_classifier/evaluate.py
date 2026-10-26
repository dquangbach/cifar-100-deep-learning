"""Evaluate a WRN checkpoint on the untouched CIFAR-100 test set."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from .checkpoints import load_checkpoint
from .config import NUM_CLASSES
from .data import (
    check_data_dir, make_test_dataset, read_batch, read_class_names, split_train,
)


def _normalization(checkpoint: Path, data_dir: Path, seed: int, validation_size: int):
    run_file = checkpoint.parent / "run.json"
    if run_file.is_file():
        metadata = json.loads(run_file.read_text(encoding="utf-8"))
        preprocessing = metadata["preprocessing"]
        return np.asarray(preprocessing["mean"], dtype=np.float64), np.asarray(
            preprocessing["std"], dtype=np.float64
        )
    train_images, train_labels = read_batch(data_dir / "train")
    split = split_train(
        train_images, train_labels, seed=seed, validation_size=validation_size
    )
    return split.mean, split.std


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 64,
    seed: int = 42,
    validation_size: int = 5_000,
) -> dict:
    checkpoint, data_dir, output_dir = map(Path, (checkpoint, data_dir, output_dir))
    check_data_dir(data_dir)
    class_names = read_class_names(data_dir)
    mean, std = _normalization(checkpoint, data_dir, seed, validation_size)
    images, labels = read_batch(data_dir / "test")
    model = load_checkpoint(checkpoint)
    probabilities = model.predict(
        make_test_dataset(images, mean, std, batch_size), verbose=0
    )
    if probabilities.shape != (len(labels), NUM_CLASSES):
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")
    predictions = probabilities.argmax(axis=1)
    loss = float(tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
    ).numpy())
    report = classification_report(
        labels, predictions, labels=np.arange(NUM_CLASSES),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "test_samples": int(len(labels)),
        "test_loss": loss,
        "test_accuracy": float(np.mean(predictions == labels)),
        "test_top5_accuracy": float(top_k_accuracy_score(
            labels, probabilities, k=5, labels=np.arange(NUM_CLASSES)
        )),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "precision", "recall", "f1_score", "support"))
        for name in [*class_names, "macro avg", "weighted avg"]:
            row = report[name]
            writer.writerow((
                name, row["precision"], row["recall"], row["f1-score"], row["support"]
            ))
    cm = confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES))
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("true/predicted", *class_names))
        for name, row in zip(class_names, cm):
            writer.writerow((name, *row))
    fig, ax = plt.subplots(figsize=(12, 10))
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set(
        title="WRN-16-4 / CIFAR-100: test confusion matrix (row normalized)",
        xlabel="Predicted class index", ylabel="True class index",
    )
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics
