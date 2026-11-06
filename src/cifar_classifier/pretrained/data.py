"""CIFAR-100 loading and the 32-pixel augmentation used by the notebook."""

import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .config import EfficientNetConfig, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        entry = pickle.load(stream, encoding="latin1")
    data = entry.get("data", entry.get(b"data"))
    labels = entry.get("fine_labels", entry.get(b"fine_labels"))
    images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
    return images.transpose(0, 2, 3, 1).copy(), np.asarray(labels, dtype=np.int64)


def read_class_names(data_dir: Path) -> list[str]:
    with (data_dir / "meta").open("rb") as stream:
        meta = pickle.load(stream, encoding="latin1")
    names = meta.get("fine_label_names", meta.get(b"fine_label_names"))
    names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in names]
    if len(names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class names, found {len(names)}")
    return names


def validate_data_dir(data_dir: Path) -> None:
    missing = [name for name in ("train", "test", "meta") if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-100 files in {data_dir}: {', '.join(missing)}")


def split_indices(count: int, seed: int, validation_size: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_size < count:
        raise ValueError("validation_size must be smaller than the training set")
    permutation = np.random.default_rng(seed).permutation(count)
    return permutation[validation_size:], permutation[:validation_size]


class CIFAR100Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        if len(images) != len(labels):
            raise ValueError("Image and label counts differ")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])


class WRNImageAugment:
    """Jitter plus one of Cutout, Random Erasing, or unchanged pixels at 32x32."""

    def __init__(self, train_mean: np.ndarray):
        self.train_mean = torch.as_tensor(train_mean, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            image = image + random.uniform(-0.05, 0.05)
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, random.uniform(0.9, 1.1))
            image = TF.adjust_hue(image, random.uniform(-0.02, 0.02))
            image = image.clamp(0.0, 1.0)
        choice = random.random()
        if choice < 0.25:
            cy, cx = random.randrange(32), random.randrange(32)
            y1, y2 = max(0, cy - 4), min(32, cy + 4)
            x1, x2 = max(0, cx - 4), min(32, cx + 4)
            image = image.clone()
            image[:, y1:y2, x1:x2] = self.train_mean.to(image.dtype)
        elif choice < 0.50:
            area = random.uniform(0.02, 0.15) * 32 * 32
            aspect = random.uniform(0.5, 2.0)
            erase_h = max(1, min(32, round((area / aspect) ** 0.5)))
            erase_w = max(1, min(32, round((area * aspect) ** 0.5)))
            top = random.randint(0, 32 - erase_h)
            left = random.randint(0, 32 - erase_w)
            image = image.clone()
            image[:, top:top + erase_h, left:left + erase_w] = torch.rand(
                (3, erase_h, erase_w), dtype=image.dtype
            )
        return image


def train_transform(image_size: int, train_mean: np.ndarray):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        WRNImageAugment(train_mean),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, *, shuffle: bool, config: EfficientNetConfig,
                pin_memory: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=pin_memory, worker_init_fn=_seed_worker, generator=generator,
    )


def make_training_loaders(data_dir: Path, config: EfficientNetConfig, *,
                          limit_train: int | None = None, limit_val: int | None = None,
                          pin_memory: bool = False):
    validate_data_dir(data_dir)
    images, labels = read_batch(data_dir / "train")
    train_indices, val_indices = split_indices(len(images), config.seed, config.validation_size)
    train_mean = images[train_indices].mean(axis=(0, 1, 2), dtype=np.float64) / 255.0
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    if limit_val is not None:
        val_indices = val_indices[:limit_val]
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Training and validation subsets must be non-empty")
    train_data = CIFAR100Dataset(images[train_indices], labels[train_indices],
                                 train_transform(config.image_size, train_mean))
    val_data = CIFAR100Dataset(images[val_indices], labels[val_indices],
                               eval_transform(config.image_size))
    return (
        make_loader(train_data, config.batch_size, shuffle=True, config=config,
                    pin_memory=pin_memory),
        make_loader(val_data, config.batch_size, shuffle=False, config=config,
                    pin_memory=pin_memory),
        train_mean,
    )


def make_test_loader(data_dir: Path, config: EfficientNetConfig, *, pin_memory: bool = False,
                     limit_test: int | None = None):
    validate_data_dir(data_dir)
    images, labels = read_batch(data_dir / "test")
    if limit_test is not None:
        images, labels = images[:limit_test], labels[:limit_test]
    if not len(images):
        raise ValueError("Test subset must be non-empty")
    dataset = CIFAR100Dataset(images, labels, eval_transform(config.image_size))
    return make_loader(dataset, config.batch_size, shuffle=False, config=config,
                       pin_memory=pin_memory), read_class_names(data_dir)
