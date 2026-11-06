"""Model loading, batch mixing and evaluation shared by train/test commands."""

import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn

from .config import EfficientNetConfig, MODEL_NAME, NUM_CLASSES


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def build_model(*, pretrained: bool, drop_rate: float,
                pretrained_weights: Path | None = None,
                cache_dir: Path | None = None) -> nn.Module:
    if pretrained_weights is not None and not pretrained_weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")
    try:
        return timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES, drop_rate=drop_rate,
            pretrained_cfg_overlay=(
                {"file": str(pretrained_weights)} if pretrained_weights is not None else None
            ),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as error:
        if pretrained:
            raise RuntimeError(
                f"Cannot download pretrained weights for {MODEL_NAME}. "
                "Connect to the internet, pass --pretrained-weights with a local ImageNet "
                "checkpoint, or use --no-pretrained for a pipeline check."
            ) from error
        raise


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        settings = checkpoint.get("config", {})
    else:
        # The original notebook saved the raw state_dict in a .pth file.
        state = checkpoint
        settings = {}
    model = build_model(pretrained=False, drop_rate=float(settings.get("drop_rate", 0.3)))
    model.load_state_dict(state, strict=True)
    return model.to(device), settings


def sample_beta(alpha: float) -> float:
    # Sample on CPU so the same augmentation works on CUDA, MPS, and CPU.
    concentration = torch.tensor(float(alpha))
    return float(torch.distributions.Beta(concentration, concentration).sample().item())


def apply_mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    indices = torch.randperm(images.size(0), device=images.device)
    lam = sample_beta(alpha)
    return lam * images + (1.0 - lam) * images[indices], labels, labels[indices], lam


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    indices = torch.randperm(images.size(0), device=images.device)
    lam = sample_beta(alpha)
    _, _, height, width = images.shape
    patch_h = round((1.0 - lam) ** 0.5 * height)
    patch_w = round((1.0 - lam) ** 0.5 * width)
    center_y, center_x = random.randrange(height), random.randrange(width)
    y1, y2 = max(0, center_y - patch_h // 2), min(height, center_y + (patch_h + 1) // 2)
    x1, x2 = max(0, center_x - patch_w // 2), min(width, center_x + (patch_w + 1) // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    kept = 1.0 - (y2 - y1) * (x2 - x1) / float(height * width)
    return mixed, labels, labels[indices], kept


@torch.inference_mode()
def evaluate_loader(model: nn.Module, loader, criterion: nn.Module,
                    device: torch.device, *, collect_predictions: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    labels_all = []
    probabilities_all = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if collect_predictions:
            # Preserve labels before the loader/model reuses any tensor storage.
            labels_all.append(labels.detach().cpu().numpy().copy())
        logits = model(images)
        batch_size = labels.size(0)
        total_loss += criterion(logits, labels).item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        count += batch_size
        if collect_predictions:
            probabilities_all.append(logits.softmax(1).cpu().numpy().copy())
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    result = {"loss": total_loss / count, "accuracy": correct / count, "samples": count}
    if collect_predictions:
        result["y_true"] = np.concatenate(labels_all)
        result["y_prob"] = np.concatenate(probabilities_all)
    return result


def train_batch(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                criterion: nn.Module, optimizer, config: EfficientNetConfig,
                device: torch.device) -> tuple[float, float]:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    choice = random.random()
    if choice < config.mixup_probability:
        images, labels_a, labels_b, lam = apply_mixup(images, labels, config.mixup_alpha)
    elif choice < config.mixup_probability + config.cutmix_probability:
        images, labels_a, labels_b, lam = apply_cutmix(images, labels, config.cutmix_alpha)
    else:
        labels_a, labels_b, lam = labels, labels, 1.0
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = lam * criterion(logits, labels_a)
    if lam < 1.0:
        loss = loss + (1.0 - lam) * criterion(logits, labels_b)
    loss.backward()
    optimizer.step()
    predictions = logits.argmax(1)
    weighted_correct = (
        lam * (predictions == labels_a).sum().item()
        + (1.0 - lam) * (predictions == labels_b).sum().item()
    )
    return float(loss.item()), float(weighted_correct)
