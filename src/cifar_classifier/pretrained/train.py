"""Train EfficientNetV2-B0 from ImageNet weights and save a reproducible run."""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import EfficientNetConfig, MODEL_NAME
from .data import make_training_loaders
from .engine import build_model, choose_device, evaluate_loader, train_batch


def train(data_dir: Path, run_dir: Path, config: EfficientNetConfig, *,
          device_name: str = "auto", pretrained: bool = True,
          limit_train: int | None = None, limit_val: int | None = None,
          pretrained_weights: Path | None = None,
          cache_dir: Path | None = None) -> dict:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    device = choose_device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_loader, val_loader, train_mean = make_training_loaders(
        data_dir, config, limit_train=limit_train, limit_val=limit_val,
        pin_memory=device.type == "cuda",
    )
    if pretrained_weights is not None and not pretrained:
        raise ValueError("--pretrained-weights and --no-pretrained cannot be combined")
    if cache_dir is not None and pretrained:
        cache_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        pretrained=pretrained, drop_rate=config.drop_rate,
        pretrained_weights=pretrained_weights, cache_dir=cache_dir,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-7
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pth"
    history_path = run_dir / "history.csv"
    metadata = {
        "model": MODEL_NAME,
        "pretrained_initialization": pretrained,
        "pretrained_weights_source": str(pretrained_weights) if pretrained_weights else MODEL_NAME,
        "config": config.to_dict(),
        "train_mean_32px": train_mean.tolist(),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "checkpoint_monitor": "val_loss",
        "torch_version": torch.__version__,
        "device": str(device),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    print(f"Device: {device} | Train: {len(train_loader.dataset)} | "
          f"Validation: {len(val_loader.dataset)} | Output: {run_dir}", flush=True)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"
        ])
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0.0
            train_count = 0
            for images, labels in train_loader:
                loss, correct = train_batch(
                    model, images, labels, criterion, optimizer, config, device
                )
                batch_count = len(labels)
                train_loss += loss * batch_count
                train_correct += correct
                train_count += batch_count
            if train_count == 0:
                raise ValueError("Training loader is empty")
            val = evaluate_loader(model, val_loader, criterion, device)
            scheduler.step(val["loss"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "train_accuracy": train_correct / train_count,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch}/{config.epochs} | train loss {row['train_loss']:.4f}, "
                  f"acc {row['train_accuracy']:.2%} | val loss {val['loss']:.4f}, "
                  f"acc {val['accuracy']:.2%} | lr {row['lr']:.2e}", flush=True)
            if val["loss"] < best_loss - 1e-4:
                best_loss = val["loss"]
                best_epoch = epoch
                stale_epochs = 0
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save({
                    "model": MODEL_NAME,
                    "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                    "config": config.to_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                }, temporary)
                os.replace(temporary, checkpoint_path)
                print(f"Saved best checkpoint: epoch {best_epoch}", flush=True)
            else:
                stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after {stale_epochs} epochs without improvement.")
                break
    metadata["best_epoch"] = best_epoch
    metadata["best_val_loss"] = best_loss
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
