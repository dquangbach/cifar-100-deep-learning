"""Evaluate a saved EfficientNetV2-B0 checkpoint on CIFAR-100 test images."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .config import EfficientNetConfig, IMAGENET_MEAN, IMAGENET_STD, MODEL_NAME, NUM_CLASSES
from .data import make_test_loader
from .engine import choose_device, evaluate_loader, load_checkpoint


def _top5_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    top5 = np.argsort(probabilities, axis=1)[:, -5:]
    return float(np.mean(np.any(top5 == labels[:, None], axis=1)))


def evaluate(checkpoint_path: Path, data_dir: Path, output_dir: Path, *,
             device_name: str = "auto", batch_size: int | None = None,
             image_size: int | None = None, limit_test: int | None = None) -> dict:
    device = choose_device(device_name)
    model, saved_settings = load_checkpoint(checkpoint_path, device)
    config = EfficientNetConfig(**saved_settings) if saved_settings else EfficientNetConfig()
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)
    if image_size is not None:
        config = replace(config, image_size=image_size)
    test_loader, class_names = make_test_loader(
        data_dir, config, pin_memory=device.type == "cuda", limit_test=limit_test
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    result = evaluate_loader(model, test_loader, criterion, device, collect_predictions=True)
    labels = result.pop("y_true")
    probabilities = result.pop("y_prob")
    predictions = probabilities.argmax(1)
    report = classification_report(
        labels, predictions, labels=np.arange(NUM_CLASSES),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    metrics = {
        "model": MODEL_NAME,
        "checkpoint": str(checkpoint_path),
        "test_samples": result["samples"],
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
        "test_top5_accuracy": _top5_accuracy(labels, probabilities),
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
    matrix = confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES))
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("true/predicted", *class_names))
        for name, row in zip(class_names, matrix):
            writer.writerow((name, *row))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set(title="EfficientNetV2-B0 / CIFAR-100: test confusion matrix",
           xlabel="Predicted class index", ylabel="True class index")
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    count = min(9, len(labels))
    indices = np.random.default_rng(config.seed).choice(len(labels), size=count, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        image_tensor, _ = test_loader.dataset[int(index)]
        image_array = image_tensor.numpy().transpose(1, 2, 0)
        image_array = np.clip(
            image_array * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0, 1
        )
        ax.imshow(image_array)
        ax.set_title(
            f"True: {class_names[int(labels[index])]}\n"
            f"Pred: {class_names[int(predictions[index])]}",
            color="green" if predictions[index] == labels[index] else "red", fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("EfficientNetV2-B0 predictions on CIFAR-100 test", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics
