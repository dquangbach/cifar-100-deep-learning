"""CIFAR-100 input, split and augmentation matching the WRN notebook."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import IMAGE_SIZE, NUM_CLASSES, TrainConfig


@dataclass
class TrainSplit:
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_images: np.ndarray
    validation_labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1).copy()
    labels = np.asarray(batch[b"fine_labels"], dtype=np.int32)
    return images, labels


def read_class_names(data_dir: Path) -> list[str]:
    with (data_dir / "meta").open("rb") as stream:
        metadata = pickle.load(stream, encoding="bytes")
    names = [name.decode("utf-8") for name in metadata[b"fine_label_names"]]
    if len(names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class names, got {len(names)}")
    return names


def check_data_dir(data_dir: Path) -> None:
    missing = [str(data_dir / name) for name in ("train", "test", "meta")
               if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing CIFAR-100 files: " + ", ".join(missing))


def split_train(
    images: np.ndarray, labels: np.ndarray, *, seed: int, validation_size: int
) -> TrainSplit:
    if len(images) != len(labels) or not 0 < validation_size < len(images):
        raise ValueError("Invalid image/label count or validation_size")
    permutation = np.random.default_rng(seed).permutation(len(images))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    train_images = images[training_indices]
    mean = train_images.mean(axis=(0, 1, 2), dtype=np.float64) / 255.0
    std = train_images.std(axis=(0, 1, 2), dtype=np.float64) / 255.0
    if np.any(std == 0):
        raise ValueError("Training images have a zero RGB standard deviation")
    return TrainSplit(
        train_images,
        labels[training_indices],
        images[validation_indices],
        labels[validation_indices],
        mean,
        std,
    )


def load_train_split(data_dir: Path, config: TrainConfig) -> TrainSplit:
    check_data_dir(data_dir)
    images, labels = read_batch(data_dir / "train")
    return split_train(
        images, labels, seed=config.seed, validation_size=config.validation_size
    )


def _random_cutout(image: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
    center_y = tf.random.uniform([], 0, IMAGE_SIZE, dtype=tf.int32)
    center_x = tf.random.uniform([], 0, IMAGE_SIZE, dtype=tf.int32)
    yy = tf.range(IMAGE_SIZE)[:, None]
    xx = tf.range(IMAGE_SIZE)[None, :]
    mask = tf.logical_and(
        tf.logical_and(yy >= center_y - 4, yy < center_y + 4),
        tf.logical_and(xx >= center_x - 4, xx < center_x + 4),
    )
    return tf.where(mask[:, :, None], mean[None, None, :], image)


def _random_erasing(image: tf.Tensor) -> tf.Tensor:
    area = tf.random.uniform([], 0.02, 0.15) * (IMAGE_SIZE * IMAGE_SIZE)
    aspect = tf.random.uniform([], 0.5, 2.0)
    height = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area / aspect)), tf.int32), 1, IMAGE_SIZE)
    width = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area * aspect)), tf.int32), 1, IMAGE_SIZE)
    top = tf.random.uniform([], 0, IMAGE_SIZE - height + 1, dtype=tf.int32)
    left = tf.random.uniform([], 0, IMAGE_SIZE - width + 1, dtype=tf.int32)
    yy = tf.range(IMAGE_SIZE)[:, None]
    xx = tf.range(IMAGE_SIZE)[None, :]
    mask = tf.logical_and(
        tf.logical_and(yy >= top, yy < top + height),
        tf.logical_and(xx >= left, xx < left + width),
    )
    noise = tf.random.uniform(tf.shape(image), 0.0, 1.0)
    return tf.where(mask[:, :, None], noise, image)


def _sample_beta(alpha: float) -> tf.Tensor:
    a = tf.random.gamma([], alpha)
    b = tf.random.gamma([], alpha)
    return a / (a + b)


def _mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float):
    indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    lam = _sample_beta(alpha)
    return (
        lam * images + (1.0 - lam) * tf.gather(images, indices),
        lam * labels + (1.0 - lam) * tf.gather(labels, indices),
    )


def _cutmix(images: tf.Tensor, labels: tf.Tensor, alpha: float):
    indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    lam = _sample_beta(alpha)
    patch = tf.cast(tf.round(tf.sqrt(1.0 - lam) * IMAGE_SIZE), tf.int32)
    center_y = tf.random.uniform([], 0, IMAGE_SIZE, dtype=tf.int32)
    center_x = tf.random.uniform([], 0, IMAGE_SIZE, dtype=tf.int32)
    y1 = tf.maximum(0, center_y - patch // 2)
    y2 = tf.minimum(IMAGE_SIZE, center_y + (patch + 1) // 2)
    x1 = tf.maximum(0, center_x - patch // 2)
    x2 = tf.minimum(IMAGE_SIZE, center_x + (patch + 1) // 2)
    yy = tf.range(IMAGE_SIZE)[:, None]
    xx = tf.range(IMAGE_SIZE)[None, :]
    mask = tf.logical_and(
        tf.logical_and(yy >= y1, yy < y2),
        tf.logical_and(xx >= x1, xx < x2),
    )[None, :, :, None]
    mixed_images = tf.where(mask, tf.gather(images, indices), images)
    kept = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / (IMAGE_SIZE * IMAGE_SIZE)
    mixed_labels = kept * labels + (1.0 - kept) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def make_train_datasets(split: TrainSplit, config: TrainConfig):
    mean = tf.constant(split.mean, tf.float32)
    std = tf.constant(split.std, tf.float32)
    autotune = tf.data.AUTOTUNE

    def augment(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
        image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

        def jitter():
            x = tf.image.random_brightness(image, max_delta=0.05)
            x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
            x = tf.image.random_saturation(x, lower=0.9, upper=1.1)
            x = tf.image.random_hue(x, max_delta=0.02)
            return tf.clip_by_value(x, 0.0, 1.0)

        image = tf.cond(tf.random.uniform([]) < 0.8, jitter, lambda: image)
        choice = tf.random.uniform([])
        image = tf.cond(
            choice < config.cutout_probability,
            lambda: _random_cutout(image, mean),
            lambda: tf.cond(
                choice < config.cutout_probability + config.erasing_probability,
                lambda: _random_erasing(image),
                lambda: image,
            ),
        )
        return (image - mean) / std, tf.one_hot(label, NUM_CLASSES)

    def clean(image, label):
        image = (tf.cast(image, tf.float32) / 255.0 - mean) / std
        return image, tf.one_hot(label, NUM_CLASSES)

    def mix_batch(images, labels):
        choice = tf.random.uniform([])
        return tf.cond(
            choice < config.mix_probability,
            lambda: _mixup(images, labels, config.mix_alpha),
            lambda: tf.cond(
                choice < config.mix_probability + config.cutmix_probability,
                lambda: _cutmix(images, labels, config.cutmix_alpha),
                lambda: (images, labels),
            ),
        )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((split.train_images, split.train_labels))
        .shuffle(len(split.train_images), seed=config.seed + 1, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=autotune)
        .batch(config.batch_size)
        .map(mix_batch, num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    validation_ds = (
        tf.data.Dataset.from_tensor_slices(
            (split.validation_images, split.validation_labels)
        )
        .map(clean, num_parallel_calls=autotune)
        .batch(config.batch_size)
        .prefetch(autotune)
    )
    return train_ds, validation_ds


def make_test_dataset(
    images: np.ndarray, mean: np.ndarray, std: np.ndarray, batch_size: int
) -> tf.data.Dataset:
    mean_tf = tf.constant(mean, tf.float32)
    std_tf = tf.constant(std, tf.float32)
    return (
        tf.data.Dataset.from_tensor_slices(images)
        .map(
            lambda image: (tf.cast(image, tf.float32) / 255.0 - mean_tf) / std_tf,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
